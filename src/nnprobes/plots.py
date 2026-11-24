from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

from nnprobes.metrics import Metric, ProbeData, effective_rank


class FigureMetric(Metric):
    """One plot per run, saved as <probe>/<name>/<step>.png. Build a matplotlib Figure, not pyplot."""

    default_every = 5

    def render(self, data: ProbeData, title: str) -> Figure:
        raise NotImplementedError


class PCA(FigureMetric):
    """First two principal components of the rows, colored by label."""

    name = "pca"

    def render(self, data, title):
        rows = data.rows() - data.rows().mean(axis=0)
        if not rows.any():
            return _message(title, "collapsed output: all rows identical")
        u, s, _ = np.linalg.svd(rows, full_matrices=False)
        points = np.zeros((len(rows), 2))
        points[:, : min(2, len(s))] = u[:, :2] * s[:2]
        var = np.append(s**2 / (s**2).sum(), 0.0)
        return _scatter(points, data, title, f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")


class TSNE(FigureMetric):
    """t-SNE of the rows, colored by label (needs scikit-learn)."""

    name = "tsne"

    def __init__(self, perplexity: float = 30.0, every: int | None = None):
        super().__init__(every)
        self.perplexity = perplexity

    def render(self, data, title):
        from sklearn.manifold import TSNE as SkTSNE

        unique, inverse = np.unique(data.rows(), axis=0, return_inverse=True)
        if len(unique) < 5:
            return _message(title, f"collapsed output: {len(unique)} distinct row(s)")
        perplexity = min(self.perplexity, (len(unique) - 1) / 3)
        points = SkTSNE(perplexity=perplexity, init="pca", random_state=0).fit_transform(unique)
        return _scatter(points[inverse.ravel()], data, title)


class SVDSpectrum(FigureMetric):
    """Normalized singular values of the centered rows, with the effective rank."""

    name = "svd_spectrum"

    def render(self, data, title):
        rows = data.rows()
        s = np.linalg.svd(rows - rows.mean(axis=0), compute_uv=False)
        s = s / s.max() if s.max() > 0 else s
        fig = Figure(figsize=(7, 4), layout="constrained")
        ax = fig.add_subplot()
        ax.semilogy(np.arange(1, len(s) + 1), np.maximum(s, 1e-8), marker="o", ms=3)
        ax.set_xlabel("component")
        ax.set_ylabel("singular value / max")
        ax.set_title(f"{title} | effective rank {effective_rank(rows):.2f} / {rows.shape[1]}")
        ax.grid(alpha=0.3)
        return fig


class Histogram(FigureMetric):
    """Distribution of all values."""

    name = "histogram"

    def render(self, data, title):
        fig = Figure(figsize=(7, 4), layout="constrained")
        ax = fig.add_subplot()
        ax.hist(data.values.ravel(), bins=100, log=True)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return fig


class AttentionHeatmap(FigureMetric):
    """Mean attention weight per step (x) for each label (y)."""

    name = "attention_heatmap"

    def render(self, data, title):
        w = data.values.reshape(len(data.values), -1, data.values.shape[-1]).mean(axis=1)
        labels = data.labels if data.labels is not None else np.zeros(len(w), dtype=int)
        classes = np.unique(labels)
        heat = np.stack([w[labels == c].mean(axis=0) for c in classes])
        names = [data.class_name(c) for c in classes] if data.labels is not None else ["all"]

        fig = Figure(figsize=(10, max(2.5, 0.25 * len(names) + 1.5)), layout="constrained")
        ax = fig.add_subplot()
        image = ax.imshow(heat, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_xlabel("step (oldest -> newest)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        return fig


def _message(title: str, text: str) -> Figure:
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot()
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def _scatter(points, data: ProbeData, title, xlabel=None, ylabel=None) -> Figure:
    fig = Figure(figsize=(9, 8), layout="constrained")
    ax = fig.add_subplot()
    if data.labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.7)
    else:
        cmap = colormaps["tab20"]
        for i, c in enumerate(np.unique(data.labels)):
            m = data.labels == c
            ax.scatter(points[m, 0], points[m, 1], s=6, alpha=0.7, color=cmap(i % cmap.N), label=data.class_name(c))
        ax.legend(markerscale=3, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_title(title)
    if xlabel is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return fig
