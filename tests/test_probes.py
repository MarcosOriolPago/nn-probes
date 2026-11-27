import csv
import json
import urllib.request

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt
from layertrace.metrics import linear_cka


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qk = nn.Linear(d, 2 * d)

    def forward(self, x):
        return self.attention_weights(x) @ x

    @lt.probe(lt.AttentionEntropy(), lt.AttentionHeatmap(every=2))
    def attention_weights(self, x):
        q, k = self.qk(x).chunk(2, dim=-1)
        return torch.softmax(q @ k.transpose(1, 2), dim=-1)


class Model(nn.Module):
    def __init__(self, d=8, classes=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU())
        self.blocks = nn.ModuleList([Attention(16), Attention(16)])
        self.head = nn.Linear(16, classes)

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        return self.embedding(x.mean(dim=1))

    @lt.probe(lt.LinearProbe(), lt.SaveActivations(every=0), name="z")
    def embedding(self, x):
        return self.head(x)


@pytest.fixture
def loader():
    g = torch.Generator().manual_seed(0)
    y = torch.arange(90) % 3
    x = torch.randn(3, 1, 8, generator=g)[y] * 3 + torch.randn(90, 5, 8, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_added_layer_and_decorators(tmp_path, loader):
    model = Model()
    probes = lt.Probes(tmp_path).add(model.encoder, lt.EffectiveRank(), lt.Drift(), lt.PCA(every=2))
    for epoch in range(3):
        probes.run(model, loader, epoch)
    probes.run(model, loader, "test")

    rows = read_csv(tmp_path / "probes.csv")
    assert {r["probe"] for r in rows} == {"encoder", "z", "blocks.0.attention_weights", "blocks.1.attention_weights"}
    assert [r["step"] for r in rows if r["metric"] == "drift"] == ["1", "2"]
    assert sorted(p.stem for p in (tmp_path / "encoder" / "pca").glob("*.png")) == ["epoch000", "epoch002", "test"]
    assert sorted(p.stem for p in (tmp_path / "encoder" / "pca").glob("*.json")) == ["epoch000", "epoch002", "test"]
    assert (tmp_path / "encoder" / "curves.png").exists()
    assert (tmp_path / "blocks.1.attention_weights" / "attention_heatmap" / "epoch002.png").exists()
    saved = np.load(tmp_path / "z" / "activations" / "test.npz")
    assert saved["values"].shape == (90, 3) and len(saved["labels"]) == 90
    assert not (tmp_path / "z" / "activations" / "epoch000.npz").exists()


def test_run_returns_scalars_and_restores_the_model(tmp_path, loader):
    model = Model().train()
    results = lt.Probes(tmp_path).add(model.head, lt.ActivationStats(), name="logits").run(model, loader, 0)
    assert set(results["logits"]) == {"mean", "std", "zeros"}
    assert results["z"]["linear_probe_bacc"] > 0.5
    assert model.training
    assert all(not m._forward_hooks and "_probe_record" not in m.__dict__ for m in model.modules())


def test_probing_is_free_outside_run(loader):
    model = Model()
    x, _ = next(iter(loader))
    model(x)


def test_failing_metric_warns(tmp_path, loader):
    class Broken(lt.ScalarMetric):
        name = "broken"

        def compute(self, data):
            raise ValueError("boom")

    model = Model()
    probes = lt.Probes(tmp_path).add(model.head, Broken(), lt.EffectiveRank())
    with pytest.warns(UserWarning, match="boom"):
        results = probes.run(model, loader, 0)
    assert "effective_rank" in results["head"]


def test_unlabeled_data_and_max_samples(tmp_path, loader):
    model = Model()
    x = loader.dataset.tensors[0]
    probes = lt.Probes(tmp_path, max_samples=40).add(model.encoder[0], lt.SaveActivations())
    probes.run(model, [x[:50], x[50:]], "test")
    saved = np.load(tmp_path / "encoder.0" / "activations" / "test.npz")
    assert saved["values"].shape == (40, 5, 16) and saved["labels"].size == 0


def test_metrics():
    rng = np.random.default_rng(0)
    data = lambda v, **kw: lt.ProbeData("p", v, **kw)  # noqa: E731
    line = rng.normal(size=(200, 1)) @ rng.normal(size=(1, 6))
    assert lt.EffectiveRank().compute(data(line))["effective_rank"] == pytest.approx(1.0, abs=1e-3)
    assert lt.DeadNeurons().compute(data(np.array([[0.0, 1.0], [0.0, 2.0]])))["dead_units"] == 0.5
    uniform = np.full((4, 8), 1 / 8)
    assert lt.AttentionEntropy().compute(data(uniform))["attention_entropy"] == pytest.approx(1.0)
    x = rng.normal(size=(100, 5))
    drift = lt.Drift()
    assert drift.compute(data(x, step=0)) == {}
    assert drift.compute(data(2 * x, step=1))["drift"] == pytest.approx(0.0, abs=1e-9)
    assert drift.compute(data(rng.normal(size=(100, 5)), step="test")) == {}
    assert lt.LinearProbe().compute(data(x)) == {}
    assert lt.KNNProbe().compute(data(x)) == {}
    assert lt.NeuralCollapse().compute(data(x)) == {}
    assert lt.WeightStats().compute(data(x)) == {}
    assert lt.QuantSaturation().compute(data(x)) == {}


def test_cka_of_constant_rows_is_nan():
    rng = np.random.default_rng(0)
    constant = np.ones((50, 4)) + 1e-9 * rng.normal(size=(50, 4))
    assert np.isnan(linear_cka(constant, rng.normal(size=(50, 4))))
    assert np.isnan(linear_cka(constant, constant))
    assert linear_cka(constant + rng.normal(size=(50, 4)), rng.normal(size=(50, 4))) < 1


def test_drift_reference_follows_its_own_schedule():
    rng = np.random.default_rng(0)
    states = [rng.normal(size=(50, 4)) for _ in range(3)]
    first = lt.Drift(reference="first")
    for step, v in enumerate(states):
        result = first.compute(lt.ProbeData("p", v, step=step))
    assert set(result) == {"drift_from_first"}
    assert result["drift_from_first"] == pytest.approx(1 - linear_cka(states[0], states[2]))
    with pytest.raises(ValueError):
        lt.Drift(reference="init")


def test_decoding_metrics_with_classes_and_continuous_labels():
    rng = np.random.default_rng(0)
    y = np.arange(300) % 3
    x = np.eye(3)[y] * 4 + rng.normal(size=(300, 3))
    classes = lt.ProbeData("p", x, labels=y)
    assert lt.KNNProbe().compute(classes)["knn_bacc"] > 0.9
    assert lt.LinearProbe().compute(classes)["linear_probe_bacc"] > 0.9
    collapsed = lt.NeuralCollapse().compute(lt.ProbeData("p", np.eye(3)[y] * 4 + 0.01 * x, labels=y))["nc1"]
    assert collapsed < 1e-3 < lt.NeuralCollapse().compute(classes)["nc1"]

    t = rng.normal(size=300)
    regression = lt.ProbeData("p", np.stack([t, rng.normal(size=300)], axis=1), labels=t.astype(np.float32))
    assert regression.continuous and not regression.classes
    assert lt.KNNProbe().compute(regression)["knn_r2"] > 0.8
    assert lt.LinearProbe().compute(regression)["linear_probe_r2"] > 0.9
    spec = lt.PCA().compute(regression)
    assert len(spec["groups"]) == 1 and len(spec["groups"][0]["c"]) == 300
    lt.draw(spec)


def test_weight_stats_tracks_updates():
    layer = nn.Linear(4, 4)
    stats = lt.WeightStats()
    first = stats.compute(lt.ProbeData("w", np.zeros((1, 1)), step=0, module=layer))
    assert set(first) == {"weight_norm", "weight_stable_rank"} and 1 <= first["weight_stable_rank"] <= 4
    with torch.no_grad():
        layer.weight.mul_(2)
        layer.bias.mul_(2)
    assert stats.compute(lt.ProbeData("w", np.zeros((1, 1)), step=1, module=layer))["weight_update"] == pytest.approx(
        1.0
    )


def test_reduce_modes(tmp_path):
    conv = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
    x = torch.randn(10, 1, 6, 6)
    probes = lt.Probes(tmp_path).add(conv[1], lt.SaveActivations(), reduce="channels")
    probes.run(conv, [x], "test")
    assert np.load(tmp_path / "1" / "activations" / "test.npz")["values"].shape == (10, 4)

    model = Model()
    y = torch.arange(6) % 3
    probes = lt.Probes(tmp_path / "tokens").add(model.encoder, lt.SaveActivations(), reduce="tokens")
    probes.run(model, [(torch.randn(6, 5, 8), y)], "test")
    saved = np.load(tmp_path / "tokens" / "encoder" / "activations" / "test.npz")
    assert saved["values"].shape == (30, 16) and list(saved["labels"][:6]) == [0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        lt.Probes(tmp_path).add(conv, reduce="pool")


def test_dict_batches_outputs_and_custom_forward(tmp_path):
    class DictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(4, 3)

        def forward(self, input_ids, mask=None):
            return {"hidden": self.body(input_ids), "aux": None}

    model = DictModel()
    batch = {"input_ids": torch.randn(8, 4), "mask": torch.ones(8), "labels": torch.arange(8) % 2}
    results = lt.Probes(tmp_path).add(model, lt.ActivationStats(), lt.KNNProbe(k=2), name="m").run(model, [batch], 0)
    assert {"mean", "knn_bacc"} <= set(results["m"])

    calls = []
    custom = lt.Probes(tmp_path / "custom", forward=lambda m, b: calls.append(1) or m(b[0]))
    custom.add(model.body, lt.ActivationStats()).run(model, [(torch.randn(8, 4), torch.zeros(8))], 0)
    assert calls == [1]


def test_layer_cka_matrix(tmp_path, loader):
    model = Model()
    probes = lt.Probes(tmp_path, max_samples=50)
    probes.add(model.encoder[0], lt.LayerCKA(every=1), reduce="tokens")
    probes.add(model.encoder[1], lt.LayerCKA(every=1), reduce="tokens")
    probes.add(model.head, lt.LayerCKA(every=1))
    probes.run(model, loader, 0)
    spec = json.loads((tmp_path / "layers" / "layer_cka" / "epoch000.json").read_text())
    assert spec["rows"] == spec["cols"] == ["encoder.0", "encoder.1", "head"]
    z = spec["z"]
    assert z[0][0] == pytest.approx(1.0) and 0 < z[0][1] < 1
    assert z[0][2] is None and z[2][2] == pytest.approx(1.0)
    lt.draw(spec)


def test_quant_saturation_reads_the_grid():
    from layertrace.metrics import QuantInfo

    values = np.array([0.0, 0.1, 0.7, -0.7, 0.3])
    q = QuantInfo(np.array(0.1), np.array(0.0), bit_width=4, signed=True)
    result = lt.QuantSaturation().compute(lt.ProbeData("q", values, quant=q))
    assert result == {"quant_saturation": 0.4, "quant_zeros": 0.2}


def test_schedule():
    assert [s for s in range(7) if lt.PCA().is_due(s)] == [0, 5]
    assert not lt.EffectiveRank(every=0).is_due(4) and lt.EffectiveRank(every=0).is_due("best")


def test_figure_data_redraws(tmp_path, loader):
    model = Model()
    lt.Probes(tmp_path).add(model.encoder, lt.PCA(), lt.Histogram(), lt.SVDSpectrum()).run(model, loader, 0)
    for metric, kind in [("pca", "scatter"), ("histogram", "bars"), ("svd_spectrum", "line")]:
        spec = json.loads((tmp_path / "encoder" / metric / "epoch000.json").read_text())
        assert spec["kind"] == kind
        lt.draw(spec, "redrawn")
    heat = json.loads((tmp_path / "blocks.0.attention_weights" / "attention_heatmap" / "epoch000.json").read_text())
    assert len(heat["rows"]) == 3 and len(heat["z"][0]) == 5


def test_viewer_serves_the_run(tmp_path, loader):
    model = Model()
    probes = lt.Probes(tmp_path, serve=True).add(model.encoder, lt.EffectiveRank(), lt.PCA(every=1))
    for epoch in range(2):
        probes.run(model, loader, epoch)
    probes.run(model, loader, "test")
    url = f"http://127.0.0.1:{probes.viewer.server_address[1]}"
    try:
        assert b"layertrace" in urllib.request.urlopen(url).read()
        run = json.loads(urllib.request.urlopen(f"{url}/api/run").read())
        rank = run["scalars"]["encoder"]["effective_rank"]
        assert rank["x"] == [0, 1] and "test" in rank["marks"]
        assert [s for s, _ in run["figures"]["encoder"]["pca"]] == ["epoch000", "epoch001", "test"]
        spec = json.loads(urllib.request.urlopen(f"{url}/api/figure?probe=encoder&metric=pca&step=test").read())
        assert spec["kind"] == "scatter"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{url}/api/figure?probe=..&metric=encoder&step=pca")
    finally:
        probes.viewer.shutdown()
