"""Metric registry and pyiqa adapter.

IQAEvaluator only ever sees the `Metric` protocol below — it does not know
pyiqa exists. All pyiqa-specific code (imports, create_metric, tensor
shape quirks) lives in `PyIQAMetric`. Swapping the IQA backend means
writing a new adapter class; IQAEvaluator is untouched.

Metrics are held by `MetricRegistry` instances — build one per evaluation
run and pass it to IQAEvaluator/evaluate(). There is no global registry, so
an IQA run and a segmentation run never interfere.

To add a custom metric without touching main.py or pyiqa, call
`MetricRegistry.register_metric()` with an object implementing `Metric`.

Built-in metrics (below) are exposed as `MetricSpec` constants (`PSNR`,
`SSIM`, ...) — nothing is registered until the caller opts in by passing
them to a registry, e.g. `MetricRegistry(PSNR, SSIM)`.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol, Sequence, runtime_checkable

import pyiqa
import torch

import radimagenet_lpips  # noqa: F401 — registers RadImageNetLPIPS in pyiqa
import clip_iqa_medical   # noqa: F401 — registers ClipIQALung / ClipIQABrain in pyiqa

from constants import RESNET50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MetricDirection = Literal["higher_is_better", "lower_is_better"]
MetricChannels  = Literal["gray", "rgb"]


@runtime_checkable
class Metric(Protocol):
    """Adapter boundary: anything callable this way can be registered as a metric.

    input/target: batch tensor (N, C, H, W), float32 in [0,1] — the same
    format ImageLoader.tensor / .rgb_tensor produce. target is None for
    no-reference metrics. Returns one score per slice in the batch.
    """
    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> Sequence[float]: ...


@dataclass(frozen=True)
class MetricSpec:
    """Static description of one IQA metric.

    Attributes:
        name:      metric name (also the ImageEvaluatorRecord field name for builtins).
        direction: whether a higher or lower score indicates better quality.
        reference: True for full-reference metrics (need a target image).
        channels:  "gray" -> use ImageLoader.tensor; "rgb" -> use ImageLoader.rgb_tensor.
        factory:   builds the Metric instance (lazily, cached by MetricRegistry).
        builtin:   True for framework-shipped metrics (dedicated record field);
                   False for user-registered metrics (stored in record.extra).
        description: human-readable explanation of what the metric measures,
                     shown to users choosing a metric.
        domain:      the domain the metric's defaults are calibrated for,
                     e.g. "medical (MONAI)". Empty string means domain-agnostic.
    """
    name:      str
    direction: MetricDirection
    reference: bool
    channels:  MetricChannels
    factory:   Callable[[], Metric]
    builtin:      bool = True
    description:  str  = ""
    domain:       str  = ""


class PyIQAMetric:
    """Adapter that makes a pyiqa metric satisfy the Metric protocol."""

    def __init__(self, name: str, **kwargs):
        self._name   = name
        self._kwargs = kwargs
        self._impl: Optional[torch.nn.Module] = None

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[float]:
        if self._impl is None:
            self._impl = pyiqa.create_metric(self._name, as_loss=False, device=DEVICE, **self._kwargs)
        scores = self._impl(input, target) if target is not None else self._impl(input)
        scores = scores.squeeze(-1) if scores.dim() == 2 else scores
        return [float(s.item()) for s in scores]


class MetricRegistry:
    """Holds registered MetricSpecs and lazily-instantiated Metric objects.

    Build one instance per evaluation run and pass it explicitly — there is
    no global registry. Two instances never share specs or cached metric
    objects, so an IQA run and a segmentation run can proceed side by side:

        iqa = MetricRegistry(*BUILTIN_METRICS)
        seg = MetricRegistry(*SEGMENTATION_METRICS)
    """

    def __init__(self, *specs: MetricSpec):
        self._specs: dict[str, MetricSpec] = {}
        self._cache: dict[str, Metric] = {}
        self.register(*specs)

    def register(self, *specs: MetricSpec) -> None:
        for spec in specs:
            self._specs[spec.name] = spec
            self._cache.pop(spec.name, None)

    def register_metric(
        self,
        name: str,
        metric: Metric,
        *,
        direction: MetricDirection,
        reference: bool,
        channels: MetricChannels = "rgb",
    ) -> None:
        """Hook a custom metric into this registry.

        No pyiqa import and no edit to main.py or ImageEvaluatorRecord
        required. `metric` just needs to implement the Metric protocol. Its
        scores show up as a column named `name` in the report (via
        ImageEvaluatorRecord.extra).
        """
        self.register(MetricSpec(name, direction, reference, channels,
                                 factory=lambda: metric, builtin=False))

    def get_metric(self, name: str) -> Metric:
        if name not in self._cache:
            self._cache[name] = self._specs[name].factory()
        return self._cache[name]

    @property
    def specs(self) -> list[MetricSpec]:
        return list(self._specs.values())

    @property
    def direction(self) -> dict[str, MetricDirection]:
        return {spec.name: spec.direction for spec in self._specs.values()}


# ---------------------------------------------------------------------------
# Built-in metrics (pyiqa-backed)
# ---------------------------------------------------------------------------

def _pyiqa_factory(name: str, **kwargs) -> Callable[[], Metric]:
    return lambda: PyIQAMetric(name, **kwargs)


# Imported here (rather than alongside the other module-level imports above)
# to avoid a circular import: monai_metrics.py does `from metrics import
# MetricSpec`, which requires MetricSpec to already be defined in this module.
from segmentation_metrics.monai_metrics import (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)


# Full-reference metrics (need a target image)
PSNR              = MetricSpec("psnr",              "higher_is_better", True,  "gray", _pyiqa_factory("psnr"))
SSIM              = MetricSpec("ssim",              "higher_is_better", True,  "gray", _pyiqa_factory("ssim"))
LPIPS             = MetricSpec("lpips",             "lower_is_better",  True,  "rgb",  _pyiqa_factory("lpips"))
DISTS             = MetricSpec("dists",             "lower_is_better",  True,  "rgb",  _pyiqa_factory("dists"))
RADIMAGENET_LPIPS = MetricSpec("radimagenet_lpips", "lower_is_better",  True,  "rgb",  _pyiqa_factory("radimagenet_lpips", backbone_path=str(RESNET50)))
# No-reference metrics
CLIPIQA           = MetricSpec("clipiqa",           "higher_is_better", False, "rgb",  _pyiqa_factory("clipiqa"))
CLIP_IQA_LUNG     = MetricSpec("clip_iqa_lung",     "higher_is_better", False, "rgb",  _pyiqa_factory("clip_iqa_lung"))
CLIP_IQA_BRAIN    = MetricSpec("clip_iqa_brain",    "higher_is_better", False, "rgb",  _pyiqa_factory("clip_iqa_brain"))
BRISQUE           = MetricSpec("brisque",           "lower_is_better",  False, "rgb",  _pyiqa_factory("brisque"))
NIQE              = MetricSpec("niqe",              "lower_is_better",  False, "rgb",  _pyiqa_factory("niqe"))

# Convenience bundle for "just register everything" — not registered by default.
BUILTIN_METRICS = (
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE,
)

# MONAI-backed segmentation-quality metrics (evaluate masks, not images) —
# kept separate from BUILTIN_METRICS so main.py's raw-image CLI is unaffected.
SEGMENTATION_METRICS = (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)

