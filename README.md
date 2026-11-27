# layertrace

See how each layer of a PyTorch model evolves while it trains. Point at a layer, choose
metrics, and get a CSV of numbers plus plots every few epochs.

```bash
pip install layertrace            # add [sklearn] for LinearProbe and TSNE
```

## Usage

Link layers from the training script:

```python
import layertrace as lt

probes = lt.Probes("runs/exp1")
probes.add(model.encoder, lt.EffectiveRank(), lt.LinearProbe(), lt.PCA(every=5))
probes.add(model.blocks[2].relu, lt.DeadNeurons())

for epoch in range(epochs):
    train_one_epoch(model, train_loader)
    probes.run(model, val_loader, epoch)

probes.run(model, test_loader, "test")
```

Or decorate the methods to watch, in the model itself:

```python
class Encoder(nn.Module):
    @lt.probe(lt.EffectiveRank(), lt.TSNE(every=5), name="z_enc")
    def forward(self, x): ...
```

`probes.run` finds decorated methods automatically, so you can mix both styles.

### Convolutions, sequences and other inputs

By default each sample becomes one row of features. For other shapes, pass `reduce` to `add` or `@probe`:

```python
probes.add(model.layer3, lt.DeadNeurons(), reduce="channels")  # [N, C, H, W] → [N, C]
probes.add(model.blocks[4], lt.EffectiveRank(), reduce="tokens")  # [N, T, D] → [N·T, D]
```

`reduce` also takes a function of the output tensor. Dict batches (e.g. Hugging Face) work as they
are: the model is called with `model(**batch)` and labels are read from `labels`, `label`, `y` or
`target`. For anything else, pass `Probes(..., forward=lambda model, batch: ...)`. When a layer
returns a tuple or a dict, its first tensor is probed.

More complete examples, one per kind of model and metric, are in [`examples/`](examples/README.md).

## Live viewer

```python
probes = lt.Probes("runs/exp1", serve=True)  # prints http://127.0.0.1:8050
```

Open the address to watch the run while it trains: each metric over epochs, every probe side by
side, and each figure with a slider to step or play through the epochs. Hover for values, and click
a class in a scatter's legend to highlight it. The viewer runs in a background thread and stops when
the script ends. To open a run again later:

```bash
layertrace view runs/exp1
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
`lt.draw(json.load(open(path)))`, which returns a matplotlib figure. `run` also returns the
numbers, `{probe: {metric: value}}`.

## Metrics

Each metric takes `every`, meaning it runs every `every` epochs. `every=0` runs it only on
named runs such as `"test"`. Named runs compute every metric.

| Numbers (→ CSV) | Plots (every 5 by default) | Data |
| --- | --- | --- |
| `EffectiveRank`: dimensions in use | `PCA`, `TSNE`: rows colored by label | `SaveActivations` → `.npz` |
| `LinearProbe`: linear decodability of labels | `SVDSpectrum`: singular values | |
| `KNNProbe`: non-linear decodability (k-NN) | `Histogram`: value distribution | |
| `Drift`: change since the last run, or `reference="first"` (1 − CKA) | `AttentionHeatmap`: weights per label | |
| `NeuralCollapse`: NC1, class collapse before the head | `LayerCKA`: similarity of every pair of layers | |
| `DeadNeurons`: share of never-active units | | |
| `WeightStats`: weight norm, stable rank, relative update | | |
| `ActivationStats`: mean, std, zeros | | |
| `AttentionEntropy`: focus of attention weights | | |
| `QuantSaturation`: clipping on a QuantTensor grid | | |

Integer labels are classes. Float labels are continuous targets: the probes then report R², and
the scatters are colored on a ramp. Add `LayerCKA()` to each probe to compare: the layer × layer
heatmap is saved under `layers/layer_cka/`. To measure drift from initialization, call
`probes.run(model, loader, -1)` once before training and use `Drift(reference="first")`.

Writing your own metric:

```python
class Sparsity(lt.ScalarMetric):
    name = "sparsity"

    def compute(self, data):
        return {"sparsity": float((abs(data.values) < 1e-3).mean())}
```

`data.values` holds what the layer returned for each sample, and `data.labels` holds the labels when the
batches carry them. For a plot, subclass `lt.FigureMetric` and return the plot's data from
`compute(data)`, for example `{"kind": "line", "x": [...], "y": [...]}`. The kinds are
`scatter`, `line`, `bars`, `heatmap` and `message`. The PNG and the viewer both draw from that data.

## Notes

- Batches can be `x`, `(x, y, ...)` or a dict. Labels are used for coloring and the decoding probes.
- Pass the same unshuffled loader every epoch so results compare. Each probe keeps at most
  `max_samples` (default 2000) rows, the same ones every time.
- Probes only record inside `run`. Training runs at full speed.
- A layer used several times in the model (e.g. repeated blocks) gives one probe per
  instance, named by its path.
- If a metric fails, it raises a warning and doesn't stop training.

## License

MIT
