# nn-probes

See how each layer of a PyTorch model evolves while it trains. Point at a layer, choose
metrics, and get a CSV of numbers plus plots every few epochs.

```bash
pip install nn-probes            # add [sklearn] for LinearProbe and TSNE
```

## Usage

Link layers from the training script:

```python
import nnprobes as nnp

probes = nnp.Probes("runs/exp1")
probes.add(model.encoder, nnp.EffectiveRank(), nnp.LinearProbe(), nnp.PCA(every=5))
probes.add(model.blocks[2].relu, nnp.DeadNeurons())

for epoch in range(epochs):
    train_one_epoch(model, train_loader)
    probes.run(model, val_loader, epoch)

probes.run(model, test_loader, "test")
```

Or decorate the methods to watch, in the model itself:

```python
class Encoder(nn.Module):
    @nnp.probe(nnp.EffectiveRank(), nnp.TSNE(every=5), name="z_enc")
    def forward(self, x):
        ...
```

`probes.run` finds decorated methods automatically, so you can mix both styles.

## Live viewer

```python
probes = nnp.Probes("runs/exp1", serve=True)   # prints http://127.0.0.1:8050
```

Open the address to watch the run while it trains: each metric over epochs, every probe side by
side, and each figure with a slider to step or play through the epochs. Hover for values, and click
a class in a scatter's legend to highlight it. The viewer runs in a background thread and stops when
the script ends. To open a run again later:

```bash
nnprobes view runs/exp1
```

It only listens on localhost. On a remote machine, forward the port with
`ssh -L 8050:localhost:8050 <host>`. It needs no internet and no extra dependencies.

## What you get

```
runs/exp1/
├── probes.csv                      step, probe, metric, value
└── encoder/
    ├── curves.png                  each metric over epochs ("test" as a dashed line)
    ├── pca/epoch005.json, epoch005.png, ..., test.json, test.png
    └── activations/epoch005.npz    raw values + labels, with SaveActivations()
```

Each figure's `.json` holds the data behind it, so you can redraw or restyle it later with
`nnp.draw(json.load(open(path)))`, which returns a matplotlib figure. `run` also returns the
numbers, `{probe: {metric: value}}`.

## Metrics

Each metric takes `every`, meaning it runs every `every` epochs. `every=0` runs it only on
named runs such as `"test"`. Named runs compute every metric.

| Numbers (→ CSV) | Plots (every 5 by default) | Data |
| --- | --- | --- |
| `EffectiveRank`: dimensions in use | `PCA`, `TSNE`: rows colored by label | `SaveActivations` → `.npz` |
| `LinearProbe`: linear decodability of labels | `SVDSpectrum`: singular values | |
| `Drift`: change since last epoch (1 − CKA) | `Histogram`: value distribution | |
| `DeadNeurons`: share of never-active units | `AttentionHeatmap`: weights per label | |
| `ActivationStats`: mean, std, zeros | | |
| `AttentionEntropy`: focus of attention weights | | |
| `QuantSaturation`: clipping on a QuantTensor grid | | |

Writing your own metric:

```python
class Sparsity(nnp.ScalarMetric):
    name = "sparsity"

    def compute(self, data):
        return {"sparsity": float((abs(data.values) < 1e-3).mean())}
```

`data.values` holds what the layer returned for each sample, and `data.labels` holds the labels when the
batches carry them. For a plot, subclass `nnp.FigureMetric` and return the plot's data from
`compute(data)`, for example `{"kind": "line", "x": [...], "y": [...]}`. The kinds are
`scatter`, `line`, `bars`, `heatmap` and `message`. The PNG and the viewer both draw from that data.

## Notes

- Batches can be `x` or `(x, y, ...)`. Labels are used for coloring and `LinearProbe`.
- Pass the same unshuffled loader every epoch so results compare. Each probe keeps at most
  `max_samples` (default 2000) rows, the same ones every time.
- Probes only record inside `run`. Training runs at full speed.
- A layer used several times in the model (e.g. repeated blocks) gives one probe per
  instance, named by its path.
- If a metric fails, it raises a warning and doesn't stop training.

## License

MIT
