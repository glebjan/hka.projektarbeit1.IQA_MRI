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

MetricDirection = Literal["higher_is_better", "lower_is_better", "not_ranked"]
MetricChannels  = Literal["gray", "rgb"]
ScoringMode     = Literal["slice", "volume"]

# Duplicated from image_loader.Spacing on purpose: metrics.py must not import
# image_loader — this is only a structural type alias, not a dependency.
Spacing = tuple[float, float, float]

REASON_DEEP_2D = (
    "compares images with a neural network trained on flat 2D pictures, so it "
    "has no way to see a stack of slices as one 3D body"
)
REASON_NO_VOLUME_IMPL = "has no volumetric implementation"


@dataclass(frozen=True)
class ModeSupport:
    """The metric can serve this mode; `factory` builds the instance.

    For `slice_mode` the factory takes no argument. For `volume_mode` it takes
    the image's voxel spacing (`Spacing` or None) — surface-distance metrics
    need it, the others ignore it.
    """
    factory: Callable[..., "Metric"]


@dataclass(frozen=True)
class ModeUnsupported:
    """The metric cannot serve this mode; `reason` is shown to the user verbatim."""
    reason: str


ModeCapability = ModeSupport | ModeUnsupported


@dataclass(frozen=True)
class SkippedMetric:
    name:   str
    reason: str


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
        slice_mode:  ModeSupport (builds the per-slice Metric, lazily, cached by
                     MetricRegistry) or ModeUnsupported (with a reason) for
                     per-slice scoring.
        volume_mode: same, for whole-volume scoring. Defaults to
                     ModeUnsupported(REASON_NO_VOLUME_IMPL) — most metrics only
                     implement slice mode.
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
    slice_mode:  ModeCapability
    volume_mode: ModeCapability = ModeUnsupported(REASON_NO_VOLUME_IMPL)
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
        self._cache: dict[tuple[str, str, Optional[Spacing]], Metric] = {}
        self.register(*specs)

    def register(self, *specs: MetricSpec) -> None:
        for spec in specs:
            self._specs[spec.name] = spec
            for key in [k for k in self._cache if k[0] == spec.name]:
                del self._cache[key]

    def register_metric(
        self,
        name: str,
        metric: Metric,
        *,
        direction: MetricDirection,
        reference: bool,
        channels: MetricChannels = "rgb",
        volume_factory: Optional[Callable[[Optional[Spacing]], Metric]] = None,
        volume_reason: str = REASON_NO_VOLUME_IMPL,
    ) -> None:
        """Hook a custom metric into this registry.

        No pyiqa import and no edit to main.py or ImageEvaluatorRecord
        required. `metric` just needs to implement the Metric protocol. Its
        scores show up as a column named `name` in the report (via
        ImageEvaluatorRecord.extra).

        `volume_factory` takes the image's voxel spacing and returns a Metric
        that scores a whole `(1, C, D, H, W)` volume. Omit it for a slice-only
        metric; `volume_reason` is then what the user is told when they ask for
        volume mode.
        """
        self.register(MetricSpec(
            name, direction, reference, channels,
            slice_mode=ModeSupport(lambda: metric),
            volume_mode=(ModeSupport(volume_factory) if volume_factory is not None
                         else ModeUnsupported(volume_reason)),
            builtin=False,
        ))

    def get_metric(
        self,
        name:    str,
        mode:    ScoringMode = "slice",
        spacing: Optional[Spacing] = None,
    ) -> Metric:
        """Build (or reuse) the metric instance for one name, mode and geometry.

        The defaults keep the plain `get_metric(name)` call valid, which is what
        IQAEvaluator uses.
        """
        key = (name, mode, spacing)
        if key not in self._cache:
            capability = self._capability(self._specs[name], mode)
            if isinstance(capability, ModeUnsupported):
                raise ValueError(
                    f"metric '{name}' cannot be scored in {mode} mode: {capability.reason}"
                )
            self._cache[key] = (
                capability.factory() if mode == "slice" else capability.factory(spacing)
            )
        return self._cache[key]

    @staticmethod
    def _capability(spec: MetricSpec, mode: ScoringMode) -> ModeCapability:
        return spec.slice_mode if mode == "slice" else spec.volume_mode

    def select(self, mode: ScoringMode) -> tuple[list[MetricSpec], list[SkippedMetric]]:
        """Split the registered specs into those that can serve `mode` and those that cannot.

        Pure query — it prints nothing and raises nothing. The caller decides how
        to report the skipped metrics.
        """
        applicable: list[MetricSpec] = []
        skipped:    list[SkippedMetric] = []
        for spec in self._specs.values():
            capability = self._capability(spec, mode)
            if isinstance(capability, ModeSupport):
                applicable.append(spec)
            else:
                skipped.append(SkippedMetric(spec.name, capability.reason))
        return applicable, skipped

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


def _volumetric_iqa_factory(kind: str) -> Callable[[Optional[Spacing]], Metric]:
    """Late-import factory for the MONAI-backed volume metrics.

    Imported inside the function so `metrics` stays importable without MONAI's
    IQA module being loaded for a slice-only run.
    """
    def build(spacing: Optional[Spacing]) -> Metric:
        from volumetric_iqa import MonaiPSNRMetric, MonaiSSIMMetric
        return MonaiPSNRMetric() if kind == "psnr" else MonaiSSIMMetric()

    return build


# Imported here (rather than alongside the other module-level imports above)
# to avoid a circular import: monai_metrics.py does `from metrics import
# MetricSpec`, which requires MetricSpec to already be defined in this module.
from segmentation_metrics.monai_metrics import (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)
from segmentation_metrics.boundary_iou import BOUNDARY_IOU
from segmentation_metrics.volume_metrics import (
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)


# Full-reference metrics (need a target image)
PSNR = MetricSpec("psnr", "higher_is_better", True, "gray",
                  ModeSupport(_pyiqa_factory("psnr")),
                  ModeSupport(_volumetric_iqa_factory("psnr")))
SSIM = MetricSpec("ssim", "higher_is_better", True, "gray",
                  ModeSupport(_pyiqa_factory("ssim")),
                  ModeSupport(_volumetric_iqa_factory("ssim")))
LPIPS             = MetricSpec("lpips",             "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("lpips")),  ModeUnsupported(REASON_DEEP_2D))
DISTS             = MetricSpec("dists",             "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("dists")),  ModeUnsupported(REASON_DEEP_2D))
RADIMAGENET_LPIPS = MetricSpec("radimagenet_lpips", "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("radimagenet_lpips", backbone_path=str(RESNET50))), ModeUnsupported(REASON_DEEP_2D))
# No-reference metrics
CLIPIQA           = MetricSpec("clipiqa",           "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clipiqa")),        ModeUnsupported(REASON_DEEP_2D))
CLIP_IQA_LUNG     = MetricSpec("clip_iqa_lung",     "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clip_iqa_lung")),  ModeUnsupported(REASON_DEEP_2D))
CLIP_IQA_BRAIN    = MetricSpec("clip_iqa_brain",    "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clip_iqa_brain")), ModeUnsupported(REASON_DEEP_2D))
BRISQUE           = MetricSpec("brisque",           "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("brisque")),        ModeUnsupported(REASON_DEEP_2D))
NIQE              = MetricSpec("niqe",              "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("niqe")),           ModeUnsupported(REASON_DEEP_2D))

# Convenience bundle for "just register everything" — not registered by default.
BUILTIN_METRICS = (
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE,
)

# MONAI-backed segmentation-quality metrics (evaluate masks, not images) —
# kept separate from BUILTIN_METRICS so main.py's raw-image CLI is unaffected.
SEGMENTATION_METRICS = (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU,
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)

