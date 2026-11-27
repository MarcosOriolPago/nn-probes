import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import layertrace as lt


class Classifier(nn.Module):
    def __init__(self, d_in=8, d=32, classes=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        return self.head(self.embedding(x))

    @lt.probe(
        lt.EffectiveRank(),
        lt.LinearProbe(),
        lt.KNNProbe(),
        lt.NeuralCollapse(),
        lt.PCA(every=1),
        lt.TSNE(every=5),
        lt.LayerCKA(every=1),
    )
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

    probes = lt.Probes("runs/quickstart", class_names=["up", "down", "left", "right"], serve=True)
    probes.add(
        model.encoder[1], lt.DeadNeurons(), lt.Drift(), lt.Histogram(every=5), lt.LayerCKA(every=1), name="relu1"
    )
    probes.add(model.encoder[0], lt.LayerCKA(every=1), name="linear1")
    probes.add(model.encoder, lt.WeightStats(), lt.Drift(reference="first"))

    probes.run(model, val_loader, -1)  # before training: the reference for Drift(reference="first")

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
