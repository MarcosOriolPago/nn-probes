import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt


def make_data(n, d=20, seed=0):
    """A 2-D latent hidden in 20-D; the target is a non-linear function of the latent."""
    g = torch.Generator().manual_seed(seed)
    z = 2 * torch.rand(n, 2, generator=g) - 1
    mixing = torch.randn(2, d, generator=torch.Generator().manual_seed(42))
    x = z @ mixing + 0.1 * torch.randn(n, d, generator=g)
    y = torch.sin(3 * z[:, 0]) * torch.cos(2 * z[:, 1])
    return TensorDataset(x, y)


class MLP(nn.Module):
    def __init__(self, d=20, width=64):
        super().__init__()
        self.input = nn.Identity()
        self.hidden1 = nn.Sequential(nn.Linear(d, width), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        return self.out(self.hidden2(self.hidden1(self.input(x)))).squeeze(-1)


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(make_data(4000, seed=0), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_data(1000, seed=1), batch_size=250)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # float labels are continuous: the probes report R² and the scatters are colored by the target
    probes = lt.Probes("runs/regression")
    for name in ["input", "hidden1", "hidden2"]:
        layer = getattr(model, name)
        probes.add(layer, lt.LinearProbe(), lt.KNNProbe(), lt.PCA(every=5), lt.WeightStats(), name=name)

    for epoch in range(20):
        for x, y in train_loader:
            loss = nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)
        print(epoch, {name: round(r["linear_probe_r2"], 3) for name, r in results.items()})

    print("view it with: layertrace view runs/regression")


if __name__ == "__main__":
    main()
