from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from layertrace.metrics import Metric, ProbeData, effective_rank, linear_cka


class FigureMetric(Metric):
    """A plot every few epochs. `compute` returns the plot's data as a dict, saved to
    <probe>/<name>/<step>.json and drawn to a PNG next to it and in the viewer.

    Kinds: {"kind": "scatter", "groups": [{"name", "x", "y"}]}, {"kind": "line", "x", "y"},
    {"kind": "bars", "edges", "counts"}, {"kind": "heatmap", "z", "rows"}, {"kind": "message", "text"}.
    Optional keys: "xlabel", "ylabel", "logy", "note"; a scatter group's "c" colors its points on a
    ramp; a heatmap's "cols" names its columns and "vmin"/"vmax" fix its color range.
    """

    default_every = 5

    def compute(self, data: ProbeData) -> dict:
        raise NotImplementedError


class PCA(FigureMetric):
    """First two principal components of the rows, colored by label."""

    name = "pca"

    def compute(self, data):
        rows = data.rows() - data.rows().mean(axis=0)
        if not rows.any():
            return message("collapsed output: all rows identical")
        u, s, _ = np.linalg.svd(rows, full_matrices=False)
        points = np.zeros((len(rows), 2))
        points[:, : min(2, len(s))] = u[:, :2] * s[:2]
        var = np.append(s**2 / (s**2).sum(), 0.0)
        return scatter(points, data, f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")


class TSNE(FigureMetric):
    """t-SNE of the rows, colored by label (needs scikit-learn)."""

    name = "tsne"

    def __init__(self, perplexity: float = 30.0, every: int | None = None):
        super().__init__(every)
        self.perplexity = perplexity

    def compute(self, data):
        from sklearn.manifold import TSNE as SkTSNE

        unique, inverse = np.unique(data.rows(), axis=0, return_inverse=True)
        if len(unique) < 5:
            return message(f"collapsed output: {len(unique)} distinct row(s)")
        perplexity = min(self.perplexity, (len(unique) - 1) / 3)
        points = SkTSNE(perplexity=perplexity, init="pca", random_state=0).fit_transform(unique)
        return scatter(points[inverse.ravel()], data)


class SVDSpectrum(FigureMetric):
    """Normalized singular values of the centered rows, with the effective rank."""

    name = "svd_spectrum"

    def compute(self, data):
        rows = data.rows()
        s = np.linalg.svd(rows - rows.mean(axis=0), compute_uv=False)
        s = s / s.max() if s.max() > 0 else s
        return {
            "kind": "line",
            "x": list(range(1, len(s) + 1)),
            "y": _numbers(np.maximum(s, 1e-8)),
            "logy": True,
            "xlabel": "component",
            "ylabel": "singular value / max",
            "note": f"effective rank {effective_rank(rows):.2f} / {rows.shape[1]}",
        }


class Histogram(FigureMetric):
    """Distribution of all values."""

    name = "histogram"

    def compute(self, data):
        counts, edges = np.histogram(data.values.ravel(), bins=100)
        return {"kind": "bars", "edges": _numbers(edges), "counts": counts.tolist(), "logy": True, "xlabel": "value"}


class LayerCKA(Metric):
    """Linear CKA between every pair of probes carrying this metric, as one heatmap per run, saved
    under the "layers" probe. Shows which layers represent the same thing, and how that settles
    during training. Only probes that saw the same samples (same rows) are compared."""

    name = "layer_cka"
    default_every = 5


def cka_matrix(names: list[str], rows: list[np.ndarray], sources: list[int]) -> dict:
    n = len(names)
    z = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            if sources[i] == sources[j] and len(rows[i]) == len(rows[j]):
                z[i, j] = z[j, i] = linear_cka(rows[i], rows[j])
    return {
        "kind": "heatmap",
        "z": _numbers(z),
        "rows": names,
        "cols": names,
        "vmin": 0,
        "vmax": 1,
        "note": "linear CKA",
    }


class AttentionHeatmap(FigureMetric):
    """Mean attention weight per step (x) for each label (y)."""

    name = "attention_heatmap"

    def compute(self, data):
        w = data.values.reshape(len(data.values), -1, data.values.shape[-1]).mean(axis=1)
        labels = data.labels if data.classes else np.zeros(len(w), dtype=int)
        classes = np.unique(labels)
        heat = np.stack([w[labels == c].mean(axis=0) for c in classes])
        names = [data.class_name(c) for c in classes] if data.classes else ["all"]
        return {"kind": "heatmap", "z": _numbers(heat), "rows": names, "xlabel": "step (oldest → newest)"}


def message(text: str) -> dict:
    return {"kind": "message", "text": text}


def scatter(points: np.ndarray, data: ProbeData, xlabel: str | None = None, ylabel: str | None = None) -> dict:
    if data.labels is None:
        groups = [{"name": "all", "x": _numbers(points[:, 0]), "y": _numbers(points[:, 1])}]
    elif data.continuous:
        groups = [
            {"name": "label", "x": _numbers(points[:, 0]), "y": _numbers(points[:, 1]), "c": _numbers(data.labels)}
        ]
    else:
        groups = [
            {"name": data.class_name(c), "x": _numbers(points[m, 0]), "y": _numbers(points[m, 1])}
            for c in np.unique(data.labels)
            for m in [data.labels == c]
        ]
    return {"kind": "scatter", "groups": groups, "xlabel": xlabel, "ylabel": ylabel}


def draw(spec: dict, title: str = "") -> Figure:
    """Draw a figure's saved data with matplotlib, e.g. draw(json.load(open("encoder/pca/epoch005.json")))."""
    kind = spec["kind"]
    if spec.get("note"):
        title = f"{title} | {spec['note']}" if title else spec["note"]
    if kind == "message":
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot()
        ax.text(0.5, 0.5, spec["text"], ha="center", va="center")
        ax.set_axis_off()
    elif kind == "scatter":
        fig = Figure(figsize=(9, 8), layout="constrained")
        ax = fig.add_subplot()
        cmap = colormaps["tab20"]
        for i, g in enumerate(spec["groups"]):
            if "c" in g:
                c = np.array(g["c"], dtype=float)
                fig.colorbar(ax.scatter(g["x"], g["y"], s=6, alpha=0.8, c=c, cmap="viridis"), ax=ax, label=g["name"])
            else:
                ax.scatter(g["x"], g["y"], s=6, alpha=0.7, color=cmap(i % cmap.N), label=g["name"])
        if len(spec["groups"]) > 1:
            ax.legend(markerscale=3, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        if not spec.get("xlabel"):
            ax.set_xticks([])
            ax.set_yticks([])
    elif kind == "line":
        fig = Figure(figsize=(7, 4), layout="constrained")
        ax = fig.add_subplot()
        ax.plot(spec["x"], spec["y"], marker="o", ms=3)
        ax.grid(alpha=0.3)
    elif kind == "bars":
        fig = Figure(figsize=(7, 4), layout="constrained")
        ax = fig.add_subplot()
        ax.stairs(spec["counts"], spec["edges"], fill=True)
        ax.grid(alpha=0.3)
    elif kind == "heatmap":
        rows, cols = spec["rows"], spec.get("cols")
        fig = Figure(figsize=(10, max(2.5, 0.25 * len(rows) + 1.5)), layout="constrained")
        ax = fig.add_subplot()
        z = np.array(spec["z"], dtype=float)
        image = ax.imshow(
            z, aspect="auto", cmap="viridis", interpolation="nearest", vmin=spec.get("vmin"), vmax=spec.get("vmax")
        )
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        if cols:
            ax.set_xticks(range(len(cols)), cols, fontsize=7, rotation=45, ha="right")
        fig.colorbar(image, ax=ax)
    else:
        raise ValueError(f"unknown figure kind {kind!r}")
    if kind != "message":
        if spec.get("logy"):
            ax.set_yscale("log")
        ax.set_xlabel(spec.get("xlabel") or "")
        ax.set_ylabel(spec.get("ylabel") or "")
    ax.set_title(title)
    return fig


def _numbers(a) -> list:
    a = np.asarray(a, dtype=float)
    if a.ndim > 1:
        return [_numbers(row) for row in a]
    return [float(f"{v:.4g}") if np.isfinite(v) else None for v in a]
