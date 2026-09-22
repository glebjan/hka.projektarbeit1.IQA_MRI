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

DreamSim is the one built-in that is not pyiqa-backed and not part of
`BUILTIN_METRICS`: build its spec with `dreamsim_spec(...)` (or use the
`DREAMSIM` default) and register it explicitly. The description types
(`MetricSpec`, `ModeSupport`, and friends) live in the dependency-free leaf
module `metric_spec.py`; this module re-exports them for convenience.
"""

from typing import Callable, Literal, Optional

import pyiqa
import torch

from iqaevaluator import radimagenet_lpips  # noqa: F401 — registers RadImageNetLPIPS in pyiqa
from iqaevaluator import clip_iqa_medical   # noqa: F401 — registers ClipIQALung / ClipIQABrain in pyiqa

from iqaevaluator.constants import RESNET50
from iqaevaluator.metric_spec import (  # noqa: F401 — re-exported; the public names live here
    DEVICE, Metric, MetricChannels, MetricDirection, MetricSpec, ModeCapability,
    ModeSupport, ModeUnsupported, REASON_DEEP_2D, REASON_NO_VOLUME_IMPL,
    SkippedMetric, Spacing,
)
from iqaevaluator.segmentation_metrics.monai_metrics import (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)
from iqaevaluator.segmentation_metrics.boundary_iou import BOUNDARY_IOU
from iqaevaluator.segmentation_metrics.volume_metrics import (
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)
from iqaevaluator.dreamsim_metric import DREAMSIM, dreamsim_spec

ScoringMode = Literal["slice", "volume"]


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
        # (N, 1), (N,) or — niqe on a single image — a 0-d scalar: one score per image.
        return [float(s) for s in scores.reshape(-1)]


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
        from iqaevaluator.volumetric_iqa import MonaiPSNRMetric, MonaiSSIMMetric
        return MonaiPSNRMetric() if kind == "psnr" else MonaiSSIMMetric()

    return build


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
FSIM              = MetricSpec("fsim",              "higher_is_better", True,  "rgb",  ModeSupport(_pyiqa_factory("fsim")),   ModeUnsupported(REASON_DEEP_2D))
GMSD              = MetricSpec("gmsd",              "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("gmsd")),   ModeUnsupported(REASON_DEEP_2D))
VSI               = MetricSpec("vsi",               "higher_is_better", True,  "rgb",  ModeSupport(_pyiqa_factory("vsi")),    ModeUnsupported(REASON_DEEP_2D))
# No-reference metrics
CLIPIQA           = MetricSpec("clipiqa",           "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clipiqa")),        ModeUnsupported(REASON_DEEP_2D))
CLIP_IQA_LUNG     = MetricSpec("clip_iqa_lung",     "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clip_iqa_lung")),  ModeUnsupported(REASON_DEEP_2D))
CLIP_IQA_BRAIN    = MetricSpec("clip_iqa_brain",    "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clip_iqa_brain")), ModeUnsupported(REASON_DEEP_2D))
BRISQUE           = MetricSpec("brisque",           "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("brisque")),        ModeUnsupported(REASON_DEEP_2D))
NIQE              = MetricSpec("niqe",              "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("niqe")),           ModeUnsupported(REASON_DEEP_2D))
MUSIQ             = MetricSpec("musiq",             "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("musiq")),          ModeUnsupported(REASON_DEEP_2D))
MANIQA            = MetricSpec("maniqa",            "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("maniqa")),         ModeUnsupported(REASON_DEEP_2D))
PAQ2PIQ           = MetricSpec("paq2piq",           "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("paq2piq")),        ModeUnsupported(REASON_DEEP_2D))
PIQE              = MetricSpec("piqe",              "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("piqe")),           ModeUnsupported(REASON_DEEP_2D))
ILNIQE            = MetricSpec("ilniqe",            "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("ilniqe")),         ModeUnsupported(REASON_DEEP_2D))

# Convenience bundle for "just register everything" — not registered by default.
BUILTIN_METRICS = (
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS, FSIM, GMSD, VSI,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE,
    MUSIQ, MANIQA, PAQ2PIQ, PIQE, ILNIQE,
)

# MONAI-backed segmentation-quality metrics (evaluate masks, not images) —
# kept separate from BUILTIN_METRICS so main.py's raw-image CLI is unaffected.
SEGMENTATION_METRICS = (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU,
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)

