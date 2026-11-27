import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt


class Sparsity(lt.ScalarMetric):
    """Share of values close to zero: a number per run, written to probes.csv and plotted over epochs."""

    name = "sparsity"

    def __init__(self, threshold=1e-3, every=None):
        super().__init__(every)
        self.threshold = threshold

    def compute(self, data):
        return {"sparsity": float((np.abs(data.values) < self.threshold).mean())}


class ClassSeparation(lt.ScalarMetric):
    """Mean distance between class means over mean distance to the own class mean (needs labels)."""

    name = "class_separation"

    def compute(self, data):
        if not data.classes:
            return {}
        rows = data.rows()
        classes = np.unique(data.labels)
        means = np.stack([rows[data.labels == c].mean(axis=0) for c in classes])
        spread = np.mean(
            [np.linalg.norm(rows[data.labels == c] - m, axis=1).mean() for c, m in zip(classes, means, strict=True)]
        )
        between = np.linalg.norm(means[:, None] - means[None], axis=-1)[np.triu_indices(len(classes), 1)].mean()
        return {"class_separation": float(between / spread)}


class UnitUsage(lt.FigureMetric):
    """How often each unit is active, sorted: a plot every few epochs, drawn to PNG and in the viewer."""

    name = "unit_usage"
    default_every = 3

    def compute(self, data):
        usage = np.sort((data.rows() > 0).mean(axis=0))[::-1]
        return {
            "kind": "line",
            "x": list(range(1, len(usage) + 1)),
            "y": usage.round(4).tolist(),
            "xlabel": "unit (most used first)",
            "ylabel": "share of samples active",
            "note": f"{(usage == 0).sum()} of {len(usage)} units never fire",
        }


def make_data(n, d=10, classes=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, classes, (n,), generator=g)
    centers = 2 * torch.randn(classes, d, generator=torch.Generator().manual_seed(42))
    return TensorDataset(centers[y] + torch.randn(n, d, generator=g), y)


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(make_data(3000, seed=0), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_data(600, seed=1), batch_size=200)
    model = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 5))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    probes = lt.Probes("runs/custom_metrics", class_names=list("ABCDE"))
    probes.add(
        model[1],
        Sparsity(),
        ClassSeparation(),
        UnitUsage(),
        lt.PCA(every=3),
        lt.SaveActivations(every=0),
        name="hidden",
    )

    for epoch in range(10):
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)["hidden"]
        print(epoch, {key: round(value, 3) for key, value in results.items()})
    probes.run(model, val_loader, "final")

    # afterwards: the raw activations and every figure's data are on disk
    saved = np.load("runs/custom_metrics/hidden/activations/final.npz")
    values, labels = saved["values"], saved["labels"]
    print("final activations", values.shape, "| mean activation per class:")
    for c in np.unique(labels):
        print(f"  {'ABCDE'[c]}: {values[labels == c].mean():.3f}")

    spec = json.loads(Path("runs/custom_metrics/hidden/pca/final.json").read_text())
    spec["xlabel"], spec["ylabel"] = "first component", "second component"
    lt.draw(spec, "hidden layer after training").savefig("runs/custom_metrics/pca_redrawn.png", dpi=150)
    print("view it with: layertrace view runs/custom_metrics")


if __name__ == "__main__":
    main()
