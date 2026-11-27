import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt

LENGTH, VOCAB, MARKER = 12, 16, 0


def make_sequences(n, seed=0):
    """Random tokens with one marker token; the label is the marker's position."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, VOCAB, (n, LENGTH), generator=g)
    y = torch.randint(0, LENGTH, (n,), generator=g)
    tokens[torch.arange(n), y] = MARKER
    return TensorDataset(tokens, y)


class SelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)

    def forward(self, x):
        v = self.qkv(x).chunk(3, dim=-1)[2]
        return x + self.weights(x) @ v

    @lt.probe(lt.AttentionEntropy(), lt.AttentionHeatmap(every=3), name="attn_weights")
    def weights(self, x):
        q, k, _ = self.qkv(x).chunk(3, dim=-1)
        return torch.softmax(q @ k.transpose(1, 2) / q.shape[-1] ** 0.5, dim=-1)


class Pooling(nn.Module):
    """Attention pooling: one learned query reads the sequence."""

    def __init__(self, d):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) / d**0.5)

    def forward(self, x):
        return (self.weights(x).unsqueeze(1) @ x).squeeze(1)

    @lt.probe(lt.AttentionEntropy(), lt.AttentionHeatmap(every=3), name="pool_weights")
    def weights(self, x):
        return torch.softmax(x @ self.query, dim=-1)


class Model(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.tokens = nn.Embedding(VOCAB, d)
        self.positions = nn.Parameter(torch.randn(LENGTH, d) / d**0.5)
        self.attention = SelfAttention(d)
        self.pool = Pooling(d)
        self.head = nn.Linear(d, LENGTH)

    def forward(self, tokens):
        x = self.attention(self.tokens(tokens) + self.positions)
        return self.head(self.pool(x))


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(make_sequences(4000, seed=0), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_sequences(800, seed=1), batch_size=200)
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    probes = lt.Probes("runs/attention", class_names=[f"pos {i}" for i in range(LENGTH)])
    probes.add(model.attention, lt.EffectiveRank(), lt.Histogram(every=3), name="tokens", reduce="tokens")
    probes.add(model.pool, lt.LinearProbe(), lt.KNNProbe(), lt.PCA(every=3), name="pooled")

    for epoch in range(10):
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)
        attn, pool = results["attn_weights"]["attention_entropy"], results["pool_weights"]["attention_entropy"]
        print(
            f"{epoch} entropy: self-attention {attn:.2f}, pooling {pool:.2f}  knn {results['pooled']['knn_bacc']:.3f}"
        )

    print("view it with: layertrace view runs/attention")


if __name__ == "__main__":
    main()
