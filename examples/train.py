import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import nnprobes as nnp


class Classifier(nn.Module):
    def __init__(self, d_in=8, d=32, classes=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        return self.head(self.embedding(x))

    @nnp.probe(nnp.EffectiveRank(), nnp.LinearProbe(), nnp.PCA(every=1), nnp.TSNE(every=5))
    def embedding(self, x):
        return self.encoder(x)


def make_data(n=2000, d=8, classes=4):
    y = torch.randint(0, classes, (n,))
    x = torch.randn(classes, d)[y] + torch.randn(n, d)
    return TensorDataset(x, y)


def main():
    torch.manual_seed(0)
    train_ds, val_ds = random_split(make_data(), [0.8, 0.2])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    probes = nnp.Probes("runs/example", class_names=["up", "down", "left", "right"], serve=True)
    probes.add(model.encoder[1], nnp.DeadNeurons(), nnp.Drift(), nnp.Histogram(every=5), name="relu1")

    for epoch in range(15):
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)
        print(epoch, results["embedding"])

    probes.run(model, val_loader, "final")


if __name__ == "__main__":
    main()
