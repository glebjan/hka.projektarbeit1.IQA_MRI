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
    """One strategy for turning a decoded array into the tensor metrics see.

    `range_of` says what [0, 1] stands for (None when nothing is scaled),
    `apply` produces the `(D, H, W)` tensor, `empty_slices` says which slices
    carry nothing worth scoring. `name` ends up in the report. `source` is
    the file name, used only in messages.
    """
    name: str

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]: ...
    def apply(self, raw: np.ndarray, *, source: str) -> torch.Tensor: ...
    def empty_slices(self, raw: np.ndarray) -> torch.Tensor: ...


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
    - `rng.hi < rng.lo`: an inverted range, never produced by a shipped
      strategy (`MinMax` and `Percentile` always have `lo <= hi`; `FixedRange`
      validates it at construction) — raises `ValueError` rather than falling
      into the branch below and being mislabeled.
    - `rng.hi == rng.lo` and `raw.min() == raw.max()`: a genuinely constant
      image. Its value is kept (clipped to [0, 1]) and a warning names the
      file, so a fully-filled mask stays filled instead of silently becoming
      empty.
    - `rng.hi == rng.lo` but `raw.min() != raw.max()`: the *range* collapsed
      even though the *image* did not — e.g. a `Percentile` span whose 0.5th
      and 99.5th percentiles both land on background, or (via `load_pair`) a
      constant target handing its degenerate `FixedRange` to a non-constant
      input. Calling this image "constant" would be false, so the warning
      says the range is degenerate instead; the data is still clipped to
      [0, 1] as-is, same as the constant-image case.

    `label` is only used in the warning.
    """
    if rng is None:
        return torch.from_numpy(np.ascontiguousarray(raw))
    arr = np.ascontiguousarray(raw, dtype=np.float32)
    if rng.hi > rng.lo:
        arr = (arr - np.float32(rng.lo)) / np.float32(rng.hi - rng.lo)
    elif rng.hi < rng.lo:
        raise ValueError(
            f"IntensityRange is inverted for [{label}]: lo={rng.lo:g} > hi={rng.hi:g}."
        )
    else:
        kept = min(max(rng.lo, 0.0), 1.0)
        raw_lo, raw_hi = float(raw.min()), float(raw.max())
        if raw_lo == raw_hi:
            print(
                f"[{label}] constant image: every voxel is {rng.lo:g}. Kept as "
                f"{kept:g} instead of scaled, so a filled mask stays filled."
            )
        else:
            print(
                f"[{label}] degenerate range: the scale collapsed to {rng.lo:g} "
                f"although the image is not constant (raw min {raw_lo:g}, max "
                f"{raw_hi:g}). Values are clipped to [0, 1] as-is instead of "
                f"scaled."
            )
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False))


def sparse_slices(raw: np.ndarray) -> torch.Tensor:
    """True for slices with (almost) no content, judged on the raw intensities.

    A slice is empty when its spread is below 0.1 % of the volume's intensity
    span, or (ordinary intensity images only, see below) its lift above the
    volume floor is below that same 0.1 %. The span is measured between the
    0.5th and 99.5th percentiles, so one spike voxel cannot shrink it and mark
    healthy slices empty. For a volume whose foreground is rarer than 0.5 %
    the percentile span collapses to zero and the extremes are used instead —
    and in that fallback regime the lift test is skipped entirely, because a
    sparse image's slice mean sits only a tiny fraction of the way from floor
    to ceiling by construction. A constant volume has no span at all and
    every slice counts as empty.

    This is the heuristic for intensity images. Masks loaded with `Mask()`
    use an exact voxel count instead (`Mask.empty_slices`).
    """
    arr = raw.astype(np.float32, copy=False)
    lo, hi = (float(v) for v in np.percentile(arr, [0.5, 99.5]))
    used_extremes = hi <= lo
    if used_extremes:
        lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    if span <= 0.0:
        return torch.ones(arr.shape[0], dtype=torch.bool)
    flat = arr.reshape(arr.shape[0], -1)
    std = flat.std(axis=1)
    empty = std < 1e-3 * span
    if not used_extremes:
        lift = flat.mean(axis=1) - lo
        empty = empty | (lift < 1e-3 * span)
    return torch.from_numpy(empty)


class _RangeBased:
    """`apply`/`empty_slices` shared by every strategy that maps a range onto [0, 1]."""

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:  # overridden
        raise NotImplementedError

    def apply(self, raw: np.ndarray, *, source: str) -> torch.Tensor:
        return scale(raw, self.range_of(raw), label=source)

    def empty_slices(self, raw: np.ndarray) -> torch.Tensor:
        return sparse_slices(raw)


@dataclass(frozen=True)
class MinMax(_RangeBased):
    """The image's own minimum and maximum. Today's behaviour, the default."""
    name: str = "minmax"

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:
        return IntensityRange(float(raw.min()), float(raw.max()))


@dataclass(frozen=True)
class Percentile(_RangeBased):
    """Robust extremes: a single spike voxel no longer compresses the image.

    Opt-in. Every score changes under it, including psnr and ssim, so results
    are only comparable with other runs that used the same percentiles. In a
    full-reference run the percentiles are taken from the target and applied
    to both images (see `image_loader.load_pair`), so the prediction's own
    overshoot is clipped against the reference's scale, not hidden.

    When the foreground is rarer than `lower`/`upper` allow for — a small ROI
    or lesion mask in a large field of view, or a heavily zero-padded volume —
    both percentiles land on background and the span collapses to zero even
    though the image itself is far from constant. `range_of` then falls back
    to the image's own extremes, mirroring the same fallback in
    `sparse_slices`; a real but sparse image is scaled, not binarized.
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
        lo, hi = (float(v) for v in np.percentile(raw, [self.lower, self.upper]))
        if hi <= lo:
            lo, hi = float(raw.min()), float(raw.max())
        return IntensityRange(lo, hi)


@dataclass(frozen=True)
class FixedRange(_RangeBased):
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
class Raw(_RangeBased):
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
