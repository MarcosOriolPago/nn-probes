from __future__ import annotations

import functools
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nnprobes.metrics import Metric, ProbeData, QuantInfo, SaveActivations, ScalarMetric
from nnprobes.output import RunFolder
from nnprobes.plots import FigureMetric
from nnprobes.viewer import serve as serve_viewer


def probe(*metrics: Metric, name: str | None = None):
    """Decorator probing what an nn.Module method returns.

    class Encoder(nn.Module):
        @probe(EffectiveRank(), PCA(every=5), name="z_enc")
        def forward(self, x): ...

    The name defaults to the method name. When the module appears several times in a
    model, each instance is a separate probe, prefixed with its path ("blocks.0.z_enc").
    """

    def decorate(method):
        wrapper = method if hasattr(method, "_probe_metrics") else _recording(method)
        wrapper._probe_metrics[:0] = metrics
        if name is not None:
            wrapper._probe_name = name
        return wrapper

    return decorate


def _recording(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        out = method(self, *args, **kwargs)
        record = self.__dict__.get("_probe_record")
        if record is not None:
            record(wrapper, out)
        return out

    wrapper._probe_metrics = []
    wrapper._probe_name = method.__name__
    return wrapper


class Probes:
    """Watches layers while a model trains and saves what it sees under `out_dir`.

    probes = Probes("runs/exp1")
    probes.add(model.encoder, EffectiveRank(), PCA(every=5))
    for epoch in range(epochs):
        train(...)
        probes.run(model, val_loader, epoch)

    Methods decorated with @probe are found automatically. Probing costs nothing outside run().
    serve=True also opens a live viewer of the run in the browser, for as long as the script runs.
    """

    def __init__(
        self,
        out_dir: str | Path,
        class_names: Sequence[str] | None = None,
        max_samples: int = 2000,
        serve: bool = False,
    ):
        self.folder = RunFolder(out_dir)
        self.viewer = serve_viewer(out_dir) if serve else None
        self.class_names = class_names
        self.max_samples = max_samples
        self._layers: dict[nn.Module, tuple[str | None, list[Metric]]] = {}
        self._previous: dict[str, np.ndarray] = {}

    def add(self, layer: nn.Module, *metrics: Metric, name: str | None = None) -> Probes:
        """Probe the output of `layer`. The name defaults to the layer's path in the model."""
        old_name, old_metrics = self._layers.get(layer, (None, []))
        self._layers[layer] = (name or old_name, old_metrics + list(metrics))
        return self

    @torch.no_grad()
    def run(self, model: nn.Module, data: Iterable, step: int | str) -> dict[str, dict[str, float]]:
        """Run `data` through `model` and compute every metric due at `step`.

        data: batches of `x` or `(x, y, ...)`; use the same, unshuffled loader each time.
        step: the epoch, or a name such as "best" or "test" to run every metric.
        Returns the scalar results, {probe: {metric: value}}.
        """
        due = {}
        for name, (target, metrics) in self._targets(model).items():
            due_metrics = [m for m in metrics if m.is_due(step)]
            if due_metrics:
                due[name] = (target, due_metrics)
        if not due:
            return {}

        captured = self._capture(model, data, {name: target for name, (target, _) in due.items()})
        results = {}
        for name, (_, metrics) in due.items():
            if name not in captured:
                warnings.warn(f"probe {name!r} recorded nothing: it was not called by the model", stacklevel=2)
                continue
            probe_data = captured[name]
            if isinstance(step, str):
                probe_data.previous = None
            else:
                self._previous[name] = probe_data.values
            results[name] = self._evaluate(probe_data, metrics, step)
        return {name: values for name, values in results.items() if values}

    def _evaluate(self, data: ProbeData, metrics: list[Metric], step: int | str) -> dict[str, float]:
        scalars = {}
        for metric in metrics:
            try:
                if isinstance(metric, ScalarMetric):
                    scalars.update(metric.compute(data))
                elif isinstance(metric, FigureMetric):
                    self.folder.save_figure(step, data.name, metric.name, metric.compute(data))
                elif isinstance(metric, SaveActivations):
                    self.folder.save_activations(step, data)
            except Exception as e:
                warnings.warn(f"{metric.name} failed on probe {data.name!r}: {e!r}", stacklevel=3)
        if scalars:
            self.folder.write_scalars(step, data.name, scalars)
            self.folder.plot_curves(data.name)
        return scalars

    def _targets(self, model: nn.Module) -> dict[str, tuple[object, list[Metric]]]:
        """{probe name: (layer or (module, method), metrics)} for added layers and decorated methods."""
        paths = {module: path for path, module in model.named_modules()}
        targets = {}
        for layer, (name, metrics) in self._layers.items():
            targets[name or paths.get(layer) or type(layer).__name__] = (layer, metrics)

        decorated = [
            (path, module, method)
            for path, module in model.named_modules()
            for method in _decorated_methods(type(module))
        ]
        counts = Counter(method._probe_name for _, _, method in decorated)
        for path, module, method in decorated:
            name = method._probe_name
            if counts[name] > 1 and path:
                name = f"{path}.{name}"
            targets[name] = ((module, method), list(method._probe_metrics))
        return targets

    def _capture(self, model: nn.Module, data: Iterable, targets: dict[str, object]) -> dict[str, ProbeData]:
        values: dict[str, list[np.ndarray]] = {name: [] for name in targets}
        quant: dict[str, QuantInfo] = {}

        def store(name, out):
            q = _quant_info(out)
            if q is not None:
                quant.setdefault(name, q)
                out = out.value
            while isinstance(out, (tuple, list)):
                out = out[0]
            values[name].append(out.detach().float().cpu().numpy())

        hooks, recorders = [], {}
        for name, target in targets.items():
            if isinstance(target, nn.Module):
                hooks.append(target.register_forward_hook(lambda m, a, out, name=name: store(name, out)))
            else:
                module, method = target
                recorders.setdefault(module, {})[method] = name
        for module, names in recorders.items():
            module.__dict__["_probe_record"] = lambda method, out, names=names: (
                store(names[method], out) if method in names else None
            )

        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        was_training = model.training
        model.eval()
        labels = []
        try:
            for batch in data:
                x, y = _split(batch)
                model(x.to(device))
                labels.append(None if y is None else torch.as_tensor(y).cpu().numpy())
        finally:
            model.train(was_training)
            for hook in hooks:
                hook.remove()
            for module in recorders:
                module.__dict__.pop("_probe_record", None)

        labels = None if any(y is None or y.ndim != 1 for y in labels) else np.concatenate(labels)
        rng = np.random.default_rng(0)
        captured = {}
        for name, chunks in values.items():
            if not chunks:
                continue
            v = np.concatenate(chunks)
            row_labels = None
            if labels is not None and len(v) % len(labels) == 0:
                row_labels = np.repeat(labels, len(v) // len(labels))
            if len(v) > self.max_samples:
                keep = np.sort(rng.choice(len(v), self.max_samples, replace=False))
                v = v[keep]
                row_labels = None if row_labels is None else row_labels[keep]
            captured[name] = ProbeData(name, v, row_labels, self._previous.get(name), quant.get(name), self.class_names)
        return captured


@functools.cache
def _decorated_methods(cls: type) -> tuple:
    found, seen = [], set()
    for klass in cls.__mro__:
        for attr, fn in vars(klass).items():
            if attr not in seen:
                seen.add(attr)
                if getattr(fn, "_probe_metrics", None):
                    found.append(fn)
    return tuple(found)


def _split(batch) -> tuple:
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1] if len(batch) > 1 else None
    return batch, None


def _quant_info(out) -> QuantInfo | None:
    if not all(hasattr(out, a) for a in ("value", "scale", "zero_point", "bit_width")):
        return None
    return QuantInfo(
        scale=out.scale.detach().float().cpu().numpy(),
        zero_point=out.zero_point.detach().float().cpu().numpy(),
        bit_width=int(out.bit_width),
        signed=bool(out.signed),
    )
