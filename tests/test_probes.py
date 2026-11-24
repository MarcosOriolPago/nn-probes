import csv

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import nnprobes as nnp


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qk = nn.Linear(d, 2 * d)

    def forward(self, x):
        return self.attention_weights(x) @ x

    @nnp.probe(nnp.AttentionEntropy(), nnp.AttentionHeatmap(every=2))
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

    @nnp.probe(nnp.LinearProbe(), nnp.SaveActivations(every=0), name="z")
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
    probes = nnp.Probes(tmp_path).add(model.encoder, nnp.EffectiveRank(), nnp.Drift(), nnp.PCA(every=2))
    for epoch in range(3):
        probes.run(model, loader, epoch)
    probes.run(model, loader, "test")

    rows = read_csv(tmp_path / "probes.csv")
    assert {r["probe"] for r in rows} == {"encoder", "z", "blocks.0.attention_weights", "blocks.1.attention_weights"}
    assert [r["step"] for r in rows if r["metric"] == "drift"] == ["1", "2"]
    assert sorted(p.name for p in (tmp_path / "encoder" / "pca").iterdir()) == [
        "epoch000.png",
        "epoch002.png",
        "test.png",
    ]
    assert (tmp_path / "encoder" / "curves.png").exists()
    assert (tmp_path / "blocks.1.attention_weights" / "attention_heatmap" / "epoch002.png").exists()
    saved = np.load(tmp_path / "z" / "activations" / "test.npz")
    assert saved["values"].shape == (90, 3) and len(saved["labels"]) == 90
    assert not (tmp_path / "z" / "activations" / "epoch000.npz").exists()


def test_run_returns_scalars_and_restores_the_model(tmp_path, loader):
    model = Model().train()
    results = nnp.Probes(tmp_path).add(model.head, nnp.ActivationStats(), name="logits").run(model, loader, 0)
    assert set(results["logits"]) == {"mean", "std", "zeros"}
    assert results["z"]["linear_probe_bacc"] > 0.5
    assert model.training
    assert all(not m._forward_hooks and "_probe_record" not in m.__dict__ for m in model.modules())


def test_probing_is_free_outside_run(loader):
    model = Model()
    x, _ = next(iter(loader))
    model(x)


def test_failing_metric_warns(tmp_path, loader):
    class Broken(nnp.ScalarMetric):
        name = "broken"

        def compute(self, data):
            raise ValueError("boom")

    model = Model()
    probes = nnp.Probes(tmp_path).add(model.head, Broken(), nnp.EffectiveRank())
    with pytest.warns(UserWarning, match="boom"):
        results = probes.run(model, loader, 0)
    assert "effective_rank" in results["head"]


def test_unlabeled_data_and_max_samples(tmp_path, loader):
    model = Model()
    x = loader.dataset.tensors[0]
    probes = nnp.Probes(tmp_path, max_samples=40).add(model.encoder[0], nnp.SaveActivations())
    probes.run(model, [x[:50], x[50:]], "test")
    saved = np.load(tmp_path / "encoder.0" / "activations" / "test.npz")
    assert saved["values"].shape == (40, 5, 16) and saved["labels"].size == 0


def test_metrics():
    rng = np.random.default_rng(0)
    data = lambda v, **kw: nnp.ProbeData("p", v, **kw)  # noqa: E731
    line = rng.normal(size=(200, 1)) @ rng.normal(size=(1, 6))
    assert nnp.EffectiveRank().compute(data(line))["effective_rank"] == pytest.approx(1.0, abs=1e-3)
    assert nnp.DeadNeurons().compute(data(np.array([[0.0, 1.0], [0.0, 2.0]])))["dead_units"] == 0.5
    uniform = np.full((4, 8), 1 / 8)
    assert nnp.AttentionEntropy().compute(data(uniform))["attention_entropy"] == pytest.approx(1.0)
    x = rng.normal(size=(100, 5))
    assert nnp.Drift().compute(data(x, previous=2 * x))["drift"] == pytest.approx(0.0, abs=1e-9)
    assert nnp.Drift().compute(data(x)) == {}
    assert nnp.LinearProbe().compute(data(x)) == {}
    assert nnp.QuantSaturation().compute(data(x)) == {}


def test_quant_saturation_reads_the_grid():
    from nnprobes.metrics import QuantInfo

    values = np.array([0.0, 0.1, 0.7, -0.7, 0.3])
    q = QuantInfo(np.array(0.1), np.array(0.0), bit_width=4, signed=True)
    result = nnp.QuantSaturation().compute(nnp.ProbeData("q", values, quant=q))
    assert result == {"quant_saturation": 0.4, "quant_zeros": 0.2}


def test_schedule():
    assert [s for s in range(7) if nnp.PCA().is_due(s)] == [0, 5]
    assert not nnp.EffectiveRank(every=0).is_due(4) and nnp.EffectiveRank(every=0).is_due("best")
