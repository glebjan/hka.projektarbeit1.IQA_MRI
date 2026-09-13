"""Metric description types — a leaf module with no framework imports.

`metrics.py` (registry, pyiqa adapter, built-in specs), every module under
`segmentation_metrics/` and `dreamsim_metric.py` all need `MetricSpec` and
its companions. Keeping them here, importing only torch, means the metric
modules never import `metrics.py` and `metrics.py` can import them at its
top like any other module — no import cycle, no import-order rules.

`metrics.py` re-exports every name below, so `from iqaevaluator.metrics
import MetricSpec` keeps working.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Sequence, runtime_checkable

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MetricDirection = Literal["higher_is_better", "lower_is_better", "not_ranked"]
MetricChannels  = Literal["gray", "rgb"]

Spacing = tuple[float, float, float]
"""Voxel size in millimetres, ordered like the array axes: (depth, height, width)."""

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

    "[0, 1]" is produced by the run's `normalization.Normalizer` (default
    `MinMax`, per volume; full-reference pairs share the target's range —
    see `image_loader.load_pair`). A metric therefore never sees absolute
    intensities; the scale it was given is in the report's `scale_lo/hi`
    columns. Under `Mask()` the batch holds exactly {0, 1}; the segmentation
    adapters require that and reject anything else. Under `Raw()` the batch
    keeps its stored dtype — only the panoptic-quality adapter is meant to
    receive that.
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
