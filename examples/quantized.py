import brevitas.nn as qnn
import torch
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import layertrace as lt


def make_data(n, d=16, classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, classes, (n,), generator=g)
    centers = torch.randn(classes, d, generator=torch.Generator().manual_seed(42))
    return TensorDataset(centers[y] + torch.randn(n, d, generator=g), y)


class QuantMLP(nn.Module):
    """Quantization-aware MLP: 4-bit weights, a 4-bit activation, and a 2-bit one whose range starts too small."""

    def __init__(self, d=16, width=64, classes=4):
        super().__init__()
        self.input = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(d, width, weight_bit_width=4, bias=True)
        self.act1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(width, width, weight_bit_width=4, bias=True)
        self.act2 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloatMaxInit, bit_width=2, max_val=0.25, return_quant_tensor=True
        )
        self.out = qnn.QuantLinear(width, classes, weight_bit_width=4, bias=True)

    def forward(self, x):
        return self.out(self.act2(self.fc2(self.act1(self.fc1(self.input(x))))))


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(make_data(3000, seed=0), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_data(600, seed=1), batch_size=200)
    model = QuantMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # QuantTensor outputs are detected automatically: QuantSaturation reads their grid
    probes = lt.Probes("runs/quantized")
    for name in ["act1", "act2"]:
        layer = getattr(model, name)
        probes.add(layer, lt.QuantSaturation(), lt.ActivationStats(), lt.LinearProbe(), lt.Histogram(every=2))

    for epoch in range(10):
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results = probes.run(model, val_loader, epoch)
        print(
            epoch, {name: (round(r["quant_saturation"], 3), round(r["quant_zeros"], 3)) for name, r in results.items()}
        )

    print("(saturation, zeros) per activation")
    print("view it with: layertrace view runs/quantized")


if __name__ == "__main__":
    main()
