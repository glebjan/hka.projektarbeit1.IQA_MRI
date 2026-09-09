"""Metric registry and pyiqa adapter.

IQAEvaluator only ever sees the `Metric` protocol below — it does not know
pyiqa exists. All pyiqa-specific code (imports, create_metric, tensor
shape quirks) lives in `PyIQAMetric`. Swapping the IQA backend means
writing a new adapter class; IQAEvaluator is untouched.

To add a custom metric without touching main.py or pyiqa, call
`register_metric()` with an object implementing `Metric`.

Dimensionality
--------------
Every metric here is 2D: it scores one slice at a time, and a volume-level
number is an aggregation over per-slice scores (see IQAEvaluator), never a
true 3D measurement. That is a property of the metrics, not a shortcut:

  fsim  phase congruency from a 2D log-Gabor bank (4 scales x 4 orientations
        in the 2D Fourier plane) plus 2D Scharr gradients.
  gmsd  2D Prewitt kernels; the score is the std of the 2D gradient-magnitude
        similarity map. Of the three this is the only one with a natural 3D
        form (3x3x3 kernels, magnitude over x/y/z), which would however no
        longer be comparable to published GMSD values.
  vsi   SDSP saliency needs a 2D FFT, a 2D log-Gabor filter and a 2D centre
        bias prior, and resizes internally to 256x256.

Greyscale caveats (these metrics were designed for RGB photographs):

  fsim  Its chromatic term is exactly neutral on replicated greyscale
        (I = Q = 0 => S_C == 1), so FSIMc and FSIM agree bit for bit here.
        The pyiqa default chromatic=True is kept for comparability.
  gmsd  pyiqa's to_y_channel(x, 255) rounds to 8 bit. For faint distortions
        on low-contrast slices that quantisation is measurable, so treat
        very small gmsd values as resolution-limited.
  vsi   The colour conspicuity term degenerates on greyscale, leaving
        saliency = frequency prior x centre prior. Constant slices score
        exactly 1.0 regardless of content — ImageLoader.empty_slice_mask
        already excludes those from evaluation.
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
    """
    name:      str
    direction: MetricDirection
    reference: bool
    channels:  MetricChannels
    factory:   Callable[[], Metric]
    builtin:   bool = True


class PyIQAMetric:
    """Adapter that makes a pyiqa metric satisfy the Metric protocol."""

    def __init__(self, name: str, **kwargs: object) -> None:
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
    """Holds registered MetricSpecs and lazily-instantiated Metric objects."""

    def __init__(self) -> None:
        self._specs: dict[str, MetricSpec] = {}
        self._cache: dict[str, Metric] = {}

    def register(self, spec: MetricSpec) -> None:
        self._specs[spec.name] = spec
        self._cache.pop(spec.name, None)

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


registry = MetricRegistry()


def register_metric(
    name: str,
    metric: Metric,
    *,
    direction: MetricDirection,
    reference: bool,
    channels: MetricChannels = "rgb",
) -> None:
    """Hook a custom metric into the evaluation pipeline.

    No pyiqa import, no editing main.py or ImageEvaluatorRecord required.
    `metric` just needs to implement the Metric protocol. Its scores show
    up as a column named `name` in the report (via ImageEvaluatorRecord.extra).
    """
    registry.register(MetricSpec(name, direction, reference, channels, factory=lambda: metric, builtin=False))


# ---------------------------------------------------------------------------
# Built-in metrics (pyiqa-backed)
# ---------------------------------------------------------------------------

def _pyiqa_factory(name: str, **kwargs: object) -> Callable[[], Metric]:
    return lambda: PyIQAMetric(name, **kwargs)


_BUILTIN_SPECS: list[MetricSpec] = [
    # Full-reference metrics
    MetricSpec("psnr",              "higher_is_better", True,  "gray", _pyiqa_factory("psnr")),
    MetricSpec("ssim",              "higher_is_better", True,  "gray", _pyiqa_factory("ssim")),
    # fsim / gmsd / vsi are 2D-only by construction (2D log-Gabor banks, 2D
    # gradient kernels, 2D FFT saliency) — see module docstring. They are
    # computed per slice like every other metric; a volume-level number is an
    # aggregation over slices, never a true 3D measurement.
    # channels="rgb" is mandatory, not a preference: fsim asserts 3 channels for
    # its chromatic term and gmsd's to_y_channel asserts (N, 3, H, W).
    MetricSpec("fsim",              "higher_is_better", True,  "rgb",  _pyiqa_factory("fsim")),
    MetricSpec("gmsd",              "lower_is_better",  True,  "rgb",  _pyiqa_factory("gmsd")),
    MetricSpec("vsi",               "higher_is_better", True,  "rgb",  _pyiqa_factory("vsi")),
    MetricSpec("lpips",             "lower_is_better",  True,  "rgb",  _pyiqa_factory("lpips")),
    MetricSpec("dists",             "lower_is_better",  True,  "rgb",  _pyiqa_factory("dists")),
    MetricSpec("radimagenet_lpips", "lower_is_better",  True,  "rgb",  _pyiqa_factory("radimagenet_lpips", backbone_path=str(RESNET50))),
    # No-reference metrics
    MetricSpec("clipiqa",           "higher_is_better", False, "rgb",  _pyiqa_factory("clipiqa")),
    MetricSpec("clip_iqa_lung",     "higher_is_better", False, "rgb",  _pyiqa_factory("clip_iqa_lung")),
    MetricSpec("clip_iqa_brain",    "higher_is_better", False, "rgb",  _pyiqa_factory("clip_iqa_brain")),
    MetricSpec("brisque",           "lower_is_better",  False, "rgb",  _pyiqa_factory("brisque")),
    MetricSpec("niqe",              "lower_is_better",  False, "rgb",  _pyiqa_factory("niqe")),
]

for _spec in _BUILTIN_SPECS:
    registry.register(_spec)
