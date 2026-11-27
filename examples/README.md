# Examples

Each example trains a small model on synthetic data, so it runs on a CPU in under a minute with
nothing to download. Run one from this folder, then open its run folder in the viewer:

```bash
python cnn_layers.py
layertrace view runs/cnn_layers
```

| Example | Shows | Metrics |
| --- | --- | --- |
| [`quickstart.py`](quickstart.py) | both ways to attach probes, the live viewer | `EffectiveRank`, `LinearProbe`, `KNNProbe`, `NeuralCollapse`, `PCA`, `TSNE`, `DeadNeurons`, `Drift`, `WeightStats`, `LayerCKA` |
| [`cnn_layers.py`](cnn_layers.py) | a CNN with `reduce="channels"`, layers compared | `LayerCKA`, `Drift(reference="first")`, `LinearProbe`, `DeadNeurons`, `SVDSpectrum`, `PCA` |
| [`attention.py`](attention.py) | attention learning to focus, `reduce="tokens"` | `AttentionEntropy`, `AttentionHeatmap`, `EffectiveRank`, `KNNProbe`, `Histogram` |
| [`regression.py`](regression.py) | continuous labels | `LinearProbe` and `KNNProbe` as R², `PCA` colored by the target, `WeightStats` |
| [`neural_collapse.py`](neural_collapse.py) | training past zero training error | `NeuralCollapse`, `EffectiveRank`, `TSNE` |
| [`quantized.py`](quantized.py) | quantization-aware training with brevitas (`pip install brevitas`) | `QuantSaturation`, `ActivationStats`, `Histogram` |
| [`huggingface.py`](huggingface.py) | a BERT with dict batches and a function as `reduce` (`pip install transformers`) | `LayerCKA`, `LinearProbe`, `Drift` |
| [`custom_metrics.py`](custom_metrics.py) | writing your own metrics, reading the saved data | `ScalarMetric`, `FigureMetric`, `SaveActivations`, `lt.draw` |

## What to look for

- **cnn_layers**: `drift_from_first` separates the layers. The first conv barely leaves its
  initialization (about 0.06), while the third moves far from it (about 0.9). `linear_probe_bacc`
  rises with depth, and the `layers` heatmap shows neighbouring blocks are similar while the
  first and last are not.
- **attention**: the self-attention entropy falls from about 1 (uniform) to about 0.2 as it learns
  to look at the marker, and its heatmap turns into a diagonal. The pooling stays broad, because by
  then every token already carries the answer.
- **regression**: the target is barely linear in the input (R² about 0.34) but almost exactly linear
  in the last hidden layer (about 0.99). The PCA plots, colored by the target, show the gradient
  forming.
- **neural_collapse**: training accuracy reaches 100% within 10 epochs, yet `nc1` keeps falling,
  and the effective rank shrinks toward the number of classes minus one.
- **quantized**: the 2-bit activation starts with too small a range and clips about 40% of its
  values (`quant_saturation`), while the 4-bit one clips none. This is how a badly ranged
  quantizer shows up.
- **huggingface**: the label becomes decodable from the [CLS] position in the deeper layers first.
  `LayerCKA` shows how similar the layers stay to each other.
- **custom_metrics**: two `ScalarMetric`s and a `FigureMetric` in a few lines each. After training,
  the script loads the saved activations and redraws a saved figure with new labels.

Each run writes into `runs/<example>/`. `probes.csv` is appended to, so delete the folder
before running an example again.
