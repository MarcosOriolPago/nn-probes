import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt

CLASSES = 6


def make_data(n=900, d=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % CLASSES
    centers = 1.5 * torch.randn(CLASSES, d, generator=g)
    return TensorDataset(centers[y] + torch.randn(n, d, generator=g), y)


class Classifier(nn.Module):
    def __init__(self, d=30, width=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 64), nn.ReLU()
        )
        self.head = nn.Linear(64, CLASSES)

    def forward(self, x):
        return self.head(self.features(x))

    @lt.probe(lt.NeuralCollapse(every=5), lt.EffectiveRank(every=5), lt.LinearProbe(every=20), lt.TSNE(every=20))
    def features(self, x):
        return self.body(x)


def main():
    torch.manual_seed(0)
    dataset = make_data()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    probe_loader = DataLoader(dataset, batch_size=300)  # the training set, unshuffled
    model = Classifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-3)

    # keep training after the training error reaches 0: the features keep collapsing to the class means
    probes = lt.Probes("runs/neural_collapse")
    for epoch in range(101):
        correct = 0
        for x, y in train_loader:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
        results = probes.run(model, probe_loader, epoch)
        if epoch % 10 == 0:
            r = results["features"]
            print(
                f"{epoch:3d} train acc {correct / len(dataset):.3f}  nc1 {r['nc1']:.4f}  rank {r['effective_rank']:.2f}"
            )

    print("view it with: layertrace view runs/neural_collapse")


if __name__ == "__main__":
    main()
