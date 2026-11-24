from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class QuantInfo:
    """Quantization grid of a quantized output (a brevitas-style QuantTensor)."""

    scale: np.ndarray
    zero_point: np.ndarray
    bit_width: int
    signed: bool

    @property
    def qmax(self) -> int:
        return 2 ** (self.bit_width - 1) - 1 if self.signed else 2**self.bit_width - 1

    @property
    def qmin(self) -> int:
        return -self.qmax if self.signed else 0

    def levels(self, values: np.ndarray) -> np.ndarray:
        return np.rint(values / self.scale + self.zero_point)


@dataclass
class ProbeData:
    """What a probe saw in one run: `values` is [rows, ...] with one entry of `labels` per row."""

    name: str
    values: np.ndarray
    labels: np.ndarray | None = None
    previous: np.ndarray | None = None
    quant: QuantInfo | None = None
    class_names: Sequence[str] | None = None

    def rows(self) -> np.ndarray:
        return self.values.reshape(len(self.values), -1)

    def class_name(self, label) -> str:
        if self.class_names is not None:
            return str(self.class_names[int(label)])
        return str(label)


class Metric:
    """`every`: run every `every` epochs; 0 runs it only on named runs such as "test"."""

    name = "metric"
    default_every = 1

    def __init__(self, every: int | None = None):
        self.every = self.default_every if every is None else every

    def is_due(self, step: int | str) -> bool:
        return isinstance(step, str) or (self.every > 0 and step % self.every == 0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(every={self.every})"


class ScalarMetric(Metric):
    """Numbers tracked over epochs: written to probes.csv and plotted in <probe>/curves.png."""

    def compute(self, data: ProbeData) -> dict[str, float]:
        raise NotImplementedError


class SaveActivations(Metric):
    """Saves the raw values and labels to <probe>/activations/<step>.npz for later analysis."""

    name = "activations"
    default_every = 5


def effective_rank(rows: np.ndarray) -> float:
    """exp(entropy of the normalized singular values) of the centered rows (Roy & Vetterli, 2007)."""
    s = np.linalg.svd(rows - rows.mean(axis=0), compute_uv=False)
    p = s[s > 0] / s.sum()
    return float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment between two sets of rows (Kornblith et al., 2019)."""
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    norm = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(y.T @ x) ** 2 / norm) if norm > 0 else float("nan")


class ActivationStats(ScalarMetric):
    """Mean, standard deviation and share of exact zeros."""

    name = "activation_stats"

    def compute(self, data):
        v = data.values
        return {"mean": float(v.mean()), "std": float(v.std()), "zeros": float((v == 0).mean())}


class DeadNeurons(ScalarMetric):
    """Share of units (flattened columns) never above `threshold`."""

    name = "dead_neurons"

    def __init__(self, threshold: float = 0.0, every: int | None = None):
        super().__init__(every)
        self.threshold = threshold

    def compute(self, data):
        return {"dead_units": float(1.0 - (data.rows() > self.threshold).any(axis=0).mean())}


class EffectiveRank(ScalarMetric):
    """Effective number of dimensions used by the rows: detects a collapsing layer."""

    name = "effective_rank"

    def compute(self, data):
        return {"effective_rank": effective_rank(data.rows())}


class Drift(ScalarMetric):
    """How much the layer changed since the previous epoch run: 1 - CKA(previous, current)."""

    name = "drift"

    def compute(self, data):
        if data.previous is None or data.previous.shape != data.values.shape:
            return {}
        return {"drift": 1.0 - linear_cka(data.previous.reshape(len(data.previous), -1), data.rows())}


class LinearProbe(ScalarMetric):
    """Cross-validated balanced accuracy of a logistic regression on the rows (needs labels, scikit-learn).

    Rows wider than `max_dim` are first reduced by PCA. Compare it across epochs and layers.
    """

    name = "linear_probe"

    def __init__(self, folds: int = 5, max_dim: int = 64, every: int | None = None):
        super().__init__(every)
        self.folds, self.max_dim = folds, max_dim

    def compute(self, data):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if data.labels is None:
            return {}
        counts = np.unique(data.labels, return_counts=True)[1]
        folds = min(self.folds, counts.min())
        if folds < 2 or len(counts) < 2:
            return {}
        rows = data.rows()
        reduce = PCA(self.max_dim, random_state=0) if rows.shape[1] > self.max_dim else "passthrough"
        model = make_pipeline(StandardScaler(), reduce, LogisticRegression(max_iter=500))
        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, rows, data.labels, cv=cv, scoring="balanced_accuracy")
        return {"linear_probe_bacc": float(scores.mean())}


class AttentionEntropy(ScalarMetric):
    """Entropy of attention weights (distributions over the last axis), normalized by log T.

    1: uniform over the T steps; 0: a single step. Rows summing to < 0.5 count as dead.
    """

    name = "attention_entropy"

    def compute(self, data):
        w = data.values.reshape(-1, data.values.shape[-1])
        total = w.sum(axis=1)
        alive = total >= 0.5
        entropy = float("nan")
        if alive.any() and w.shape[1] > 1:
            p = w[alive] / total[alive, None]
            h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)
            entropy = float(h.mean() / np.log(w.shape[1]))
        return {"attention_entropy": entropy, "attention_dead": float(1.0 - alive.mean())}


class QuantSaturation(ScalarMetric):
    """Share of values clipped to the quantization grid edges, and share rounded to 0 (needs a QuantTensor)."""

    name = "quant_saturation"

    def compute(self, data):
        q = data.quant
        if q is None:
            return {}
        levels = q.levels(data.values)
        clipped = levels >= q.qmax
        if q.signed:
            clipped |= levels <= q.qmin
        return {"quant_saturation": float(clipped.mean()), "quant_zeros": float((levels == 0).mean())}
