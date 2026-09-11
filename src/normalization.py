"""Intensity normalization strategies.

Every metric backend this framework uses — pyiqa, DreamSim, MONAI — takes
float32 input in [0, 1] and scales from there internally (x255 for
brisque/niqe/piqe/vsi, [-1, 1] for lpips/musiq, ImageNet/CLIP statistics for
maniqa/clipiqa/dists, max_val=1.0 for MONAI PSNR). None of them normalizes per
image. So the loader has to, and this module decides *how*.

A `Normalizer` picks the (lo, hi) range that is mapped onto [0, 1]. The mapping
itself is fixed and lives in `scale()` — the only place the formula exists.

Strategies:
    MinMax()                      the image's own extremes (default)
    Percentile(lower, upper)      robust extremes; opt-in, changes every score
    FixedRange(range, name)       a range decided elsewhere — how the input of a
                                  full-reference pair is put on the target's scale
    Raw()                         no scaling; for masks and label maps, whose
                                  integer dtype must survive

Conventions followed: fastMRI scales prediction and reference on the
reference's range; MONAI/nnU-Net keep label maps as integers; TorchIO/MONAI
offer percentile scaling for training, not for evaluation.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch


@dataclass(frozen=True)
class IntensityRange:
    """The raw interval that `scale()` maps onto [0, 1]."""
    lo: float
    hi: float


@runtime_checkable
class Normalizer(Protocol):
    """Chooses the range for one image. `name` ends up in the report."""
    name: str

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]: ...


def scale(
    raw: np.ndarray,
    rng: Optional[IntensityRange],
    *,
    label: str = "image",
) -> torch.Tensor:
    """Map `raw` onto [0, 1] over `rng`.

    - `rng is None`: the data is returned untouched, dtype preserved (masks).
    - `rng.hi > rng.lo`: `clip((raw - lo) / (hi - lo), 0, 1)`. No epsilon — the
      guard already rules out division by zero, and an epsilon would make the
      top of the range fall short of 1.0.
    - `rng.hi == rng.lo`: a constant image. Its value is kept (clipped to
      [0, 1]) and a warning names the file, so a fully-filled mask stays filled
      instead of silently becoming empty.

    `label` is only used in the warning.
    """
    if rng is None:
        return torch.from_numpy(np.ascontiguousarray(raw))
    arr = np.ascontiguousarray(raw, dtype=np.float32)
    if rng.hi > rng.lo:
        arr = (arr - np.float32(rng.lo)) / np.float32(rng.hi - rng.lo)
    else:
        kept = min(max(rng.lo, 0.0), 1.0)
        print(
            f"[{label}] constant image: every voxel is {rng.lo:g}. Kept as "
            f"{kept:g} instead of scaled, so a filled mask stays filled."
        )
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False))


@dataclass(frozen=True)
class MinMax:
    """The image's own minimum and maximum. Today's behaviour, the default."""
    name: str = "minmax"

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:
        return IntensityRange(float(raw.min()), float(raw.max()))


@dataclass(frozen=True)
class Percentile:
    """Robust extremes: a single spike voxel no longer compresses the image.

    Opt-in. Every score changes under it, including psnr and ssim, so results
    are only comparable with other runs that used the same percentiles. In a
    full-reference run the percentiles are taken from the target and applied
    to both images (see `image_loader.load_pair`), so the prediction's own
    overshoot is clipped against the reference's scale, not hidden.
    """
    lower: float = 0.5
    upper: float = 99.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower < self.upper <= 100.0):
            raise ValueError(
                f"Percentile bounds must satisfy 0 <= lower < upper <= 100, "
                f"got lower={self.lower}, upper={self.upper}."
            )

    @property
    def name(self) -> str:
        return f"percentile_{self.lower:g}_{self.upper:g}"

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:
        lo, hi = np.percentile(raw, [self.lower, self.upper])
        return IntensityRange(float(lo), float(hi))


@dataclass(frozen=True)
class FixedRange:
    """A range decided elsewhere; the data is ignored.

    `name` should be the name of the strategy that produced `range`, so the
    report says how the scale was chosen rather than "fixed". `range=None`
    behaves like `Raw()`.
    """
    range: Optional[IntensityRange]
    name: str = "fixed"

    def __post_init__(self) -> None:
        if self.range is not None and self.range.hi < self.range.lo:
            raise ValueError(
                f"FixedRange needs lo <= hi, got lo={self.range.lo}, hi={self.range.hi}."
            )

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:
        return self.range


@dataclass(frozen=True)
class Raw:
    """No scaling. Use for masks and label maps.

    The decoded dtype survives, so an integer label map reaches the
    segmentation metrics as integers and `as_mask(label=...)` can select one
    label. Note that a PNG mask stores 0/255, not 0/1 — load those with
    `MinMax()`, which maps them to exactly {0.0, 1.0}.
    """
    name: str = "raw"

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:
        return None


NORMALIZER_NAMES: dict[str, type] = {
    "minmax":     MinMax,
    "percentile": Percentile,
    "raw":        Raw,
}


def normalizer_from_name(name: str) -> Normalizer:
    """Build a strategy with default parameters from its CLI name."""
    try:
        return NORMALIZER_NAMES[name]()
    except KeyError:
        raise ValueError(
            f"'{name}' is not a normalization strategy. Choose one of: "
            f"{', '.join(NORMALIZER_NAMES)}."
        ) from None
