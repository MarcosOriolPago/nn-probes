from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from nnprobes.metrics import ProbeData
from nnprobes.plots import draw


def step_label(step: int | str) -> str:
    return step if isinstance(step, str) else f"epoch{step:03d}"


class RunFolder:
    """Everything a Probes object writes:

    probes.csv                        step, probe, metric, value
    <probe>/curves.png                every scalar over epochs, named runs as dashed lines
    <probe>/<figure>/<step>.json      the figure's data, e.g. encoder/pca/epoch005.json
    <probe>/<figure>/<step>.png       the same figure drawn with matplotlib
    <probe>/activations/<step>.npz    values and labels, from SaveActivations
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.csv_path = self.path / "probes.csv"

    def write_scalars(self, step: int | str, probe: str, values: dict[str, float]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["step", "probe", "metric", "value"])
            writer.writerows([step, probe, key, value] for key, value in values.items())

    def save_figure(self, step: int | str, probe: str, name: str, spec: dict) -> None:
        path = self.path / probe / name / step_label(step)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, separators=(",", ":")))
        tmp.replace(path.with_suffix(".json"))
        draw(spec, f"{probe} | {name} | {step_label(step)}").savefig(path.with_suffix(".png"), dpi=150)

    def save_activations(self, step: int | str, data: ProbeData) -> None:
        path = self.path / data.name / "activations" / f"{step_label(step)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = data.labels if data.labels is not None else np.array([])
        np.savez_compressed(path, values=data.values, labels=labels)

    def plot_curves(self, probe: str) -> None:
        series: dict[str, dict[int, float]] = {}
        marks: dict[str, dict[str, float]] = {}
        with self.csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["probe"] != probe:
                    continue
                step, key, value = row["step"], row["metric"], float(row["value"])
                if step.lstrip("-").isdigit():
                    series.setdefault(key, {})[int(step)] = value
                else:
                    marks.setdefault(key, {})[step] = value
        keys = sorted(series.keys() | marks.keys())
        if not keys:
            return

        cols = min(4, len(keys))
        rows = math.ceil(len(keys) / cols)
        fig = Figure(figsize=(4.5 * cols, 3.3 * rows), layout="constrained")
        for i, key in enumerate(keys):
            ax = fig.add_subplot(rows, cols, i + 1)
            if key in series:
                steps = sorted(series[key])
                ax.plot(steps, [series[key][s] for s in steps], marker="o", ms=3)
            for j, (tag, value) in enumerate(marks.get(key, {}).items()):
                ax.axhline(value, ls="--", lw=1, color=f"C{j + 1}", label=f"{tag}: {value:.3g}")
            if key in marks:
                ax.legend(fontsize=7)
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        fig.suptitle(probe)
        (self.path / probe).mkdir(parents=True, exist_ok=True)
        fig.savefig(self.path / probe / "curves.png", dpi=120)
