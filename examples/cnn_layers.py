import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt

CLASSES = ["horizontal", "vertical", "diagonal", "checker"]


def make_images(n, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, len(CLASSES), (n,), generator=g)
    axis = torch.arange(size, dtype=torch.float32)
    i, j = torch.meshgrid(axis, axis, indexing="ij")
    freq = 0.6 + 0.6 * torch.rand(n, 1, 1, generator=g)
    phase = 2 * math.pi * torch.rand(n, 1, 1, generator=g)
    patterns = torch.stack(
        [
            torch.sin(freq * i + phase),
            torch.sin(freq * j + phase),
            torch.sin(freq * (i + j) / math.sqrt(2) + phase),
            torch.sin(freq * i + phase) * torch.sin(freq * j + phase),
        ],
        dim=1,
    )
    x = patterns[torch.arange(n), y] + 2.0 * torch.randn(n, size, size, generator=g)
    return TensorDataset(x.unsqueeze(1), y)


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, len(CLASSES)))

    def forward(self, x):
        return self.head(self.blocks(x))


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(make_images(3000, seed=0), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_images(600, seed=1), batch_size=200)
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    probes = lt.Probes("runs/cnn_layers", class_names=CLASSES)
    for k, b in enumerate(model.blocks):
        probes.add(
            b[1],
            lt.LinearProbe(),
            lt.DeadNeurons(),
            lt.Drift(reference="first"),
            lt.LayerCKA(every=2),
            name=f"conv{k + 1}",
            reduce="channels",
        )
    probes.add(model.blocks[2][1], lt.SVDSpectrum(every=4), lt.PCA(every=4), name="conv3", reduce="channels")

    probes.run(model, val_loader, -1)
    for epoch in range(12):
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)
        print(epoch, {name: round(r["linear_probe_bacc"], 3) for name, r in results.items()})

    print("view it with: layertrace view runs/cnn_layers")


if __name__ == "__main__":
    main()
