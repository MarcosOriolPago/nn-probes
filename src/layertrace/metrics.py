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
    """What a probe saw in one run: `values` is [rows, ...] with one entry of `labels` per row.

    Integer labels are classes; float labels are continuous targets (regression).
    `module` is the layer (or the module owning the decorated method), for weight metrics.
    """

    name: str
    values: np.ndarray
    labels: np.ndarray | None = None
    step: int | str | None = None
    module: object = None
    quant: QuantInfo | None = None
    class_names: Sequence[str] | None = None

    def rows(self) -> np.ndarray:
        return self.values.reshape(len(self.values), -1)

    @property
    def classes(self) -> bool:
        return self.labels is not None and not np.issubdtype(self.labels.dtype, np.floating)

    @property
    def continuous(self) -> bool:
        return self.labels is not None and np.issubdtype(self.labels.dtype, np.floating)

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


class _Tracked(ScalarMetric):
    """Keeps a reference per probe from a previous epoch run ("last") or the first one ("first")."""

    def __init__(self, reference: str = "last", every: int | None = None):
        if reference not in ("last", "first"):
            raise ValueError(f"reference must be 'last' or 'first', not {reference!r}")
        super().__init__(every)
        self.reference = reference
        self._refs: dict[str, np.ndarray] = {}

    def _swap(self, data: ProbeData, current: np.ndarray) -> np.ndarray | None:
        """The reference for this probe, then remembers `current` if this epoch run should replace it."""
        ref = self._refs.get(data.name)
        if not isinstance(data.step, str) and (self.reference == "last" or ref is None):
            self._refs[data.name] = current
        return None if isinstance(data.step, str) else ref


class SaveActivations(Metric):
    """Saves the raw values and labels to <probe>/activations/<step>.npz for later analysis."""

    name = "activations"
    default_every = 5


def effective_rank(rows: np.ndarray) -> float:
    """exp(entropy of the normalized singular values) of the centered rows (Roy & Vetterli, 2007)."""
    s = np.linalg.svd(rows - rows.mean(axis=0), compute_uv=False)
    p = s[s > 0] / s.sum()
    return float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0


def balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean recall over the classes present in `truth`."""
    return float(np.mean([(pred[truth == c] == c).mean() for c in np.unique(truth)]))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment between two sets of rows (Kornblith et al., 2019).

    NaN when either set is constant (up to float noise): there is nothing to align.
    """
    centered = []
    for rows in (x, y):
        rows = np.asarray(rows, dtype=np.float64)
        c = rows - rows.mean(axis=0)
        if np.linalg.norm(c) <= 1e-6 * np.linalg.norm(rows):
            return float("nan")
        centered.append(c)
    x, y = centered
    return float(np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


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


class Drift(_Tracked):
    """How much the layer changed, 1 - CKA(reference, current), skipped on named runs.

    reference="last": since this metric last ran. reference="first": since its first run, so run
    once before training (e.g. step -1) to measure the distance from initialization.
    """

    name = "drift"

    def compute(self, data):
        rows = data.rows()
        ref = self._swap(data, rows)
        if ref is None or ref.shape != rows.shape:
            return {}
        key = "drift" if self.reference == "last" else "drift_from_first"
        return {key: 1.0 - linear_cka(ref, rows)}


class LinearProbe(ScalarMetric):
    """How linearly decodable the labels are (needs labels, scikit-learn), cross-validated.

    Classes: balanced accuracy of a logistic regression. Continuous labels: R² of a ridge regression.
    Rows wider than `max_dim` are first reduced by PCA. Compare it across epochs and layers.
    """

    name = "linear_probe"

    def __init__(self, folds: int = 5, max_dim: int = 64, every: int | None = None):
        super().__init__(every)
        self.folds, self.max_dim = folds, max_dim

    def compute(self, data):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if data.labels is None:
            return {}
        if data.classes:
            counts = np.unique(data.labels, return_counts=True)[1]
            folds = min(self.folds, counts.min())
            if folds < 2 or len(counts) < 2:
                return {}
            head, cv = LogisticRegression(max_iter=500), StratifiedKFold(folds, shuffle=True, random_state=0)
            scoring, key = "balanced_accuracy", "linear_probe_bacc"
        else:
            if len(data.labels) < 2 * self.folds:
                return {}
            head, cv = RidgeCV(alphas=np.logspace(-3, 3, 7)), KFold(self.folds, shuffle=True, random_state=0)
            scoring, key = "r2", "linear_probe_r2"
        rows = data.rows()
        reduce = PCA(self.max_dim, random_state=0) if rows.shape[1] > self.max_dim else "passthrough"
        model = make_pipeline(StandardScaler(), reduce, head)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, rows, data.labels, cv=cv, scoring=scoring)
        return {key: float(scores.mean())}


class KNNProbe(ScalarMetric):
    """Leave-one-out k-nearest-neighbour decodability of the labels: no linearity assumed, no scikit-learn.

    Classes: balanced accuracy of the majority vote. Continuous labels: R² of the neighbours' mean.
    """

    name = "knn_probe"

    def __init__(self, k: int = 10, every: int | None = None):
        super().__init__(every)
        self.k = k

    def compute(self, data):
        if data.labels is None or len(data.labels) <= self.k:
            return {}
        x = data.rows().astype(np.float64)
        sq = (x**2).sum(axis=1)
        dist = sq[:, None] + sq[None, :] - 2 * x @ x.T
        np.fill_diagonal(dist, np.inf)
        neighbours = np.argpartition(dist, self.k, axis=1)[:, : self.k]
        y = data.labels
        if data.continuous:
            pred = y[neighbours].mean(axis=1)
            total = ((y - y.mean()) ** 2).sum()
            return {"knn_r2": float(1 - ((y - pred) ** 2).sum() / total) if total > 0 else float("nan")}
        classes, codes = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            return {}
        votes = np.zeros((len(y), len(classes)))
        np.add.at(votes, (np.arange(len(y))[:, None], codes[neighbours]), 1)
        return {"knn_bacc": balanced_accuracy(codes, votes.argmax(axis=1))}


class NeuralCollapse(ScalarMetric):
    """NC1 = tr(Σ_within Σ_between⁺) / C (Papyan et al., 2020): falls toward 0 as each class collapses
    to its mean. Meant for the layer feeding the classifier (needs class labels)."""

    name = "neural_collapse"

    def compute(self, data):
        if not data.classes:
            return {}
        x = data.rows().astype(np.float64)
        classes, codes = np.unique(data.labels, return_inverse=True)
        if len(classes) < 2:
            return {}
        means = np.stack([x[codes == c].mean(axis=0) for c in range(len(classes))])
        between = (means - x.mean(axis=0)) / np.sqrt(len(classes))
        within = x - means[codes]
        return {"nc1": float(((within @ np.linalg.pinv(between)) ** 2).sum() / len(x) / len(classes))}


class WeightStats(_Tracked):
    """The layer's own parameters: total norm, mean stable rank of its matrices (‖W‖²_F / ‖W‖²_2),
    and relative change ‖W - W_ref‖ / ‖W_ref‖ since the last (or first) epoch run."""

    name = "weight_stats"

    def compute(self, data):
        params = [] if data.module is None else [p.detach().float().cpu().numpy() for p in data.module.parameters()]
        if not params:
            return {}
        flat = np.concatenate([p.ravel() for p in params])
        norm = float(np.linalg.norm(flat))
        ranks = []
        for p in params:
            if p.ndim >= 2:
                s = np.linalg.svd(p.reshape(p.shape[0], -1), compute_uv=False)
                if s[0] > 0:
                    ranks.append(float((s**2).sum() / s[0] ** 2))
        result = {"weight_norm": norm}
        if ranks:
            result["weight_stable_rank"] = float(np.mean(ranks))
        ref = self._swap(data, flat)
        if ref is not None and ref.shape == flat.shape and np.linalg.norm(ref) > 0:
            key = "weight_update" if self.reference == "last" else "weight_change_from_first"
            result[key] = float(np.linalg.norm(flat - ref) / np.linalg.norm(ref))
        return result


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
