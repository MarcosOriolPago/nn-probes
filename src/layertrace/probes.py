from __future__ import annotations

import functools
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from layertrace.metrics import Metric, ProbeData, QuantInfo, SaveActivations, ScalarMetric
from layertrace.output import RunFolder
from layertrace.plots import FigureMetric, LayerCKA, cka_matrix
from layertrace.viewer import serve as serve_viewer

Reduce = str | Callable[[torch.Tensor], torch.Tensor] | None
LABEL_KEYS = ("labels", "label", "y", "target", "targets")


def probe(*metrics: Metric, name: str | None = None, reduce: Reduce = None):
    """Decorator probing what an nn.Module method returns.

    class Encoder(nn.Module):
        @probe(EffectiveRank(), PCA(every=5), name="z_enc")
        def forward(self, x): ...

    The name defaults to the method name. When the module appears several times in a
    model, each instance is a separate probe, prefixed with its path ("blocks.0.z_enc").
    `reduce` works as in Probes.add.
    """
    _check_reduce(reduce)

    def decorate(method):
        wrapper = method if hasattr(method, "_probe_metrics") else _recording(method)
        wrapper._probe_metrics[:0] = metrics
        if name is not None:
            wrapper._probe_name = name
        if reduce is not None:
            wrapper._probe_reduce = reduce
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
    wrapper._probe_reduce = None
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
    forward(model, batch) replaces the default call, model(x) or model(**batch) for dict batches.
    """

    def __init__(
        self,
        out_dir: str | Path,
        class_names: Sequence[str] | None = None,
        max_samples: int = 2000,
        serve: bool = False,
        forward: Callable[[nn.Module, object], object] | None = None,
    ):
        self.folder = RunFolder(out_dir)
        self.viewer = serve_viewer(out_dir) if serve else None
        self.class_names = class_names
        self.max_samples = max_samples
        self.forward = forward
        self._layers: dict[nn.Module, tuple[str | None, list[Metric], Reduce]] = {}

    def add(self, layer: nn.Module, *metrics: Metric, name: str | None = None, reduce: Reduce = None) -> Probes:
        """Probe the output of `layer`. The name defaults to the layer's path in the model.

        reduce: how each sample becomes rows of features. "flatten" (default) makes one row per
        sample; "channels" averages the axes after the channel axis ([N, C, H, W] → [N, C]);
        "tokens" makes one row per position ([N, T, D] → [N·T, D]); or a function of the tensor.
        """
        _check_reduce(reduce)
        old_name, old_metrics, old_reduce = self._layers.get(layer, (None, [], None))
        self._layers[layer] = (name or old_name, old_metrics + list(metrics), reduce or old_reduce)
        return self

    @torch.no_grad()
    def run(self, model: nn.Module, data: Iterable, step: int | str) -> dict[str, dict[str, float]]:
        """Run `data` through `model` and compute every metric due at `step`.

        data: batches of `x` or `(x, y, ...)`; use the same, unshuffled loader each time.
        step: the epoch, or a name such as "best" or "test" to run every metric.
        Returns the scalar results, {probe: {metric: value}}.
        """
        due = {}
        for name, (target, metrics, reduce) in self._targets(model).items():
            due_metrics = [m for m in metrics if m.is_due(step)]
            if due_metrics:
                due[name] = (target, due_metrics, reduce)
        if not due:
            return {}

        captured, sources = self._capture(model, data, {name: (t, r) for name, (t, _, r) in due.items()}, step)
        results, compared = {}, []
        for name, (_, metrics, _) in due.items():
            if name not in captured:
                warnings.warn(f"probe {name!r} recorded nothing: it was not called by the model", stacklevel=2)
                continue
            results[name] = self._evaluate(captured[name], metrics, step)
            if any(isinstance(m, LayerCKA) for m in metrics):
                compared.append(name)
        if len(compared) > 1:
            try:
                spec = cka_matrix(compared, [captured[n].rows() for n in compared], [sources[n] for n in compared])
                self.folder.save_figure(step, "layers", LayerCKA.name, spec)
            except Exception as e:
                warnings.warn(f"layer_cka failed: {e!r}", stacklevel=2)
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

    def _targets(self, model: nn.Module) -> dict[str, tuple[object, list[Metric], Reduce]]:
        """{probe name: (layer or (module, method), metrics, reduce)} for added layers and decorated methods."""
        paths = {module: path for path, module in model.named_modules()}
        targets = {}
        for layer, (name, metrics, reduce) in self._layers.items():
            targets[name or paths.get(layer) or type(layer).__name__] = (layer, metrics, reduce)

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
            targets[name] = ((module, method), list(method._probe_metrics), method._probe_reduce)
        return targets

    def _capture(
        self, model: nn.Module, data: Iterable, targets: dict[str, tuple[object, Reduce]], step: int | str
    ) -> tuple[dict[str, ProbeData], dict[str, int]]:
        """The ProbeData of each probe, and how many rows it had before subsampling."""
        values: dict[str, list[np.ndarray]] = {name: [] for name in targets}
        quant: dict[str, QuantInfo] = {}

        def store(name, out):
            q = _quant_info(out)
            if q is not None:
                quant.setdefault(name, q)
                out = out.value
            out = _first_tensor(out)
            if out is None:
                return
            out = _reduce(out.detach(), targets[name][1])
            values[name].append(out.float().cpu().numpy())

        hooks, recorders = [], {}
        for name, (target, _) in targets.items():
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
                if self.forward is not None:
                    self.forward(model, batch)
                elif isinstance(x, Mapping):
                    model(**_to(x, device))
                else:
                    model(_to(x, device))
                labels.append(None if y is None else torch.as_tensor(y).cpu().numpy())
        finally:
            model.train(was_training)
            for hook in hooks:
                hook.remove()
            for module in recorders:
                module.__dict__.pop("_probe_record", None)

        labels = None if not labels or any(y is None or y.ndim != 1 for y in labels) else np.concatenate(labels)
        captured, sources = {}, {}
        for name, chunks in values.items():
            if not chunks:
                continue
            v = np.concatenate(chunks)
            sources[name] = len(v)
            row_labels = None
            if labels is not None and len(labels) and len(v) % len(labels) == 0:
                row_labels = np.repeat(labels, len(v) // len(labels))
            if len(v) > self.max_samples:
                # one generator per probe: probes with as many rows keep the same samples, every run
                keep = np.sort(np.random.default_rng(0).choice(len(v), self.max_samples, replace=False))
                v = v[keep]
                row_labels = None if row_labels is None else row_labels[keep]
            target = targets[name][0]
            module = target if isinstance(target, nn.Module) else target[0]
            captured[name] = ProbeData(name, v, row_labels, step, module, quant.get(name), self.class_names)
        return captured, sources


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
    """(inputs, labels) of a batch: x, (x, y, ...), or a dict with the labels under one of LABEL_KEYS."""
    if isinstance(batch, Mapping):
        key = next((k for k in LABEL_KEYS if k in batch), None)
        return {k: v for k, v in batch.items() if k != key}, None if key is None else batch[key]
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1] if len(batch) > 1 else None
    return batch, None


def _to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: _to(v, device) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, device) for v in x)
    return x


def _first_tensor(out) -> torch.Tensor | None:
    """The tensor a layer returned: itself, the first of a tuple, or the first value of a dict/ModelOutput."""
    while not isinstance(out, torch.Tensor):
        if isinstance(out, Mapping):
            out = next((v for v in out.values() if v is not None), None)
        elif isinstance(out, (tuple, list)) and out:
            out = out[0]
        else:
            return None
    return out


def _check_reduce(reduce: Reduce) -> None:
    if not (reduce is None or callable(reduce) or reduce in ("flatten", "channels", "tokens")):
        raise ValueError(f"reduce must be 'flatten', 'channels', 'tokens' or a function, not {reduce!r}")


def _reduce(out: torch.Tensor, reduce: Reduce) -> torch.Tensor:
    if callable(reduce):
        return reduce(out)
    if reduce == "channels" and out.ndim > 2:
        return out.flatten(2).mean(dim=2)
    if reduce == "tokens" and out.ndim > 2:
        return out.reshape(-1, out.shape[-1])
    return out


def _quant_info(out) -> QuantInfo | None:
    if not all(hasattr(out, a) for a in ("value", "scale", "zero_point", "bit_width")):
        return None
    return QuantInfo(
        scale=out.scale.detach().float().cpu().numpy(),
        zero_point=out.zero_point.detach().float().cpu().numpy(),
        bit_width=int(out.bit_width),
        signed=bool(out.signed),
    )
