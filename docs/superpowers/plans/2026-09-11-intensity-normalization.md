# Intensity Normalization Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move intensity normalization out of the image decoders into a per-run `Normalizer` strategy, scale full-reference pairs on the target's range, load masks raw, and write the scale used into every report row.

**Architecture:** A new `src/normalization.py` owns `IntensityRange`, the `Normalizer` protocol, four strategies (`MinMax`, `Percentile`, `FixedRange`, `Raw`) and the single `scale()` function. `LoadedImage` carries the decoded array (`raw`) instead of a tensor; `ImageLoader` applies the strategy lazily and exposes `raw_range` / `intensity_range`. `load_pair()` implements the full-reference rule once. Evaluators copy the scale into five new record fields; `empty_slice_mask` moves to raw data.

**Tech Stack:** Python 3.14, numpy, torch, nibabel, pydicom, SimpleITK, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-11-intensity-normalization-design.md`

## Global Constraints

- All code lives flat in `src/`; modules import each other by bare name (`from normalization import MinMax`). No package, no relative imports.
- One component per file. `metrics.py` must not import `image_loader.py` (it duplicates `Spacing` on purpose); `normalization.py` must import neither.
- Run everything from the repository root with the venv binaries: `.venv/bin/pytest`, `.venv/bin/python`. There is no bare `python` on PATH.
- Tests import `src/` via `tests/conftest.py`; run `.venv/bin/pytest tests/<file>.py -q`.
- The `Metric` contract stays: batches are `(N, C, H, W)`, float32 in `[0, 1]`. Only the `Raw` strategy may emit another dtype, and only for masks.
- Every `TODO(norm-N)` comment is removed by the end; decisions that need recording become ordinary comments or docstring sentences.
- Commit after each task with a Conventional Commits subject (`feat(loader): …`, `test(...)`, `docs: …`) and the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Never push. The wiki task commits locally only.

---

## File Structure

| File | Responsibility after this plan |
|---|---|
| `src/normalization.py` (new) | `IntensityRange`, `Normalizer` protocol, `MinMax`, `Percentile`, `FixedRange`, `Raw`, `scale()`, `normalizer_from_name()`, `NORMALIZER_NAMES` |
| `src/image_loader.py` | Decoders return `LoadedImage(raw=…)`; `ImageLoader(path, normalizer)` with `.raw`, `.raw_range`, `.intensity_range`, `.tensor`, `.empty_slice_mask`; `load_pair()` |
| `src/records.py` | Five new fields: `normalization`, `scale_lo`, `scale_hi`, `input_min`, `input_max` |
| `src/iqa_evaluator.py` | `_scale_fields()` helper; FR emptiness = input AND target blank |
| `src/volume_evaluator.py` | Uses `_scale_fields()` |
| `src/evaluation_result.py` | `aggregate_volumes()` fills all three counts with 0 on empty rows |
| `src/segmentation_metrics/monai_metrics.py` | Adapter casts to float32; TODO replaced by a note on `Raw` |
| `src/segmentation_metrics/volume.py` | TODO replaced by a note on `Raw` |
| `src/metrics.py` | `Metric` docstring names the per-run normalizer |
| `src/main.py` | `evaluate(..., normalization=)`, `--normalization` CLI flag |
| `tests/test_normalization.py` (new) | Strategy and `scale()` tests |
| `CLAUDE.md`, `README.md`, `src/evaluation.ipynb`, wiki | Documentation |

---

### Task 1: `normalization.py` — strategies and `scale()`

**Files:**
- Create: `src/normalization.py`
- Test: `tests/test_normalization.py`

**Interfaces:**
- Consumes: nothing from the codebase.
- Produces:
  - `IntensityRange(lo: float, hi: float)` — frozen dataclass.
  - `Normalizer` protocol: attribute `name: str`, method `range_of(raw: np.ndarray) -> Optional[IntensityRange]`.
  - `MinMax()`, `Percentile(lower=0.5, upper=99.5)`, `FixedRange(range: Optional[IntensityRange], name: str = "fixed")`, `Raw()`.
  - `scale(raw: np.ndarray, rng: Optional[IntensityRange], *, label: str = "image") -> torch.Tensor` — `(D, H, W)` in, `(D, H, W)` out (no channel axis).
  - `NORMALIZER_NAMES: dict[str, type]` = `{"minmax": MinMax, "percentile": Percentile, "raw": Raw}` and `normalizer_from_name(name: str) -> Normalizer`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalization.py
"""Tests for src/normalization.py — strategies and the scale() mapping."""
import numpy as np
import pytest
import torch

from normalization import (
    NORMALIZER_NAMES, FixedRange, IntensityRange, MinMax, Normalizer,
    Percentile, Raw, normalizer_from_name, scale,
)


class TestScale:
    def test_maps_range_onto_exactly_zero_and_one(self):
        raw = np.array([[[0.0, 1e-8]]], dtype=np.float32)
        t = scale(raw, IntensityRange(0.0, 1e-8))
        assert float(t.min()) == 0.0
        assert float(t.max()) == 1.0

    def test_clips_values_outside_the_range(self):
        raw = np.array([[[-5.0, 5.0, 15.0]]], dtype=np.float32)
        t = scale(raw, IntensityRange(0.0, 10.0))
        assert t.tolist() == [[[0.0, 0.5, 1.0]]]

    def test_output_is_float32_without_channel_axis(self):
        raw = np.random.default_rng(0).integers(0, 4000, (3, 4, 5), dtype=np.uint16)
        t = scale(raw, IntensityRange(0.0, 4000.0))
        assert t.dtype == torch.float32
        assert t.shape == (3, 4, 5)

    def test_constant_image_keeps_its_clipped_value_and_warns(self, capsys):
        raw = np.full((2, 3, 3), 42.0, dtype=np.float32)
        t = scale(raw, IntensityRange(42.0, 42.0), label="flat.png")
        assert torch.all(t == 1.0)
        out = capsys.readouterr().out
        assert "flat.png" in out and "constant" in out

    def test_constant_zero_image_stays_zero(self):
        raw = np.zeros((1, 3, 3), dtype=np.float32)
        assert torch.all(scale(raw, IntensityRange(0.0, 0.0)) == 0.0)

    def test_none_range_preserves_dtype_and_values(self):
        raw = np.array([[[0, 1, 2, 3]]], dtype=np.int16)
        t = scale(raw, None)
        assert t.dtype == torch.int16
        assert t.tolist() == [[[0, 1, 2, 3]]]

    def test_none_range_accepts_non_contiguous_input(self):
        raw = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
        t = scale(raw, None)
        assert t.shape == (4, 2, 3)


class TestMinMax:
    def test_uses_the_extremes(self):
        raw = np.array([[[100.0, 110.0, 4000.0]]])
        assert MinMax().range_of(raw) == IntensityRange(100.0, 4000.0)

    def test_name(self):
        assert MinMax().name == "minmax"


class TestPercentile:
    def test_ignores_a_single_spike(self):
        raw = np.full((1, 100, 100), 120.0)
        raw[0, :50] = 100.0
        raw[0, 0, 0] = 4000.0
        rng = Percentile().range_of(raw)
        assert rng.lo == pytest.approx(100.0)
        assert rng.hi == pytest.approx(120.0)

    def test_scaled_bulk_spans_the_unit_interval_and_spike_clips(self):
        raw = np.linspace(100.0, 140.0, 10_000, dtype=np.float32).reshape(1, 100, 100)
        raw[0, 0, 0] = 4000.0
        t = scale(raw, Percentile().range_of(raw))
        assert float(t[0, 0, 0]) == 1.0
        assert float(t[0, 50, 50]) == pytest.approx(0.5, abs=0.02)

    def test_name_encodes_the_percentiles(self):
        assert Percentile().name == "percentile_0.5_99.5"
        assert Percentile(1, 99).name == "percentile_1_99"

    @pytest.mark.parametrize("lower,upper", [(99.5, 0.5), (50, 50), (-1, 99), (0, 101)])
    def test_invalid_bounds_raise(self, lower, upper):
        with pytest.raises(ValueError):
            Percentile(lower, upper)


class TestFixedRange:
    def test_ignores_the_data(self):
        rng = IntensityRange(0.0, 800.0)
        assert FixedRange(rng, "minmax").range_of(np.array([[[5.0]]])) is rng

    def test_carries_the_name_it_was_given(self):
        assert FixedRange(IntensityRange(0.0, 1.0), "minmax").name == "minmax"

    def test_none_range_means_raw(self):
        assert FixedRange(None, "raw").range_of(np.array([[[5.0]]])) is None

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError):
            FixedRange(IntensityRange(10.0, 0.0), "x")


class TestRaw:
    def test_returns_no_range(self):
        assert Raw().range_of(np.array([[[1, 2]]])) is None

    def test_name(self):
        assert Raw().name == "raw"


class TestProtocol:
    @pytest.mark.parametrize("strategy", [MinMax(), Percentile(), FixedRange(None, "raw"), Raw()])
    def test_every_strategy_satisfies_normalizer(self, strategy):
        assert isinstance(strategy, Normalizer)


class TestFromName:
    def test_known_names(self):
        assert isinstance(normalizer_from_name("minmax"), MinMax)
        assert isinstance(normalizer_from_name("percentile"), Percentile)
        assert isinstance(normalizer_from_name("raw"), Raw)

    def test_names_match_the_registry(self):
        assert set(NORMALIZER_NAMES) == {"minmax", "percentile", "raw"}

    def test_unknown_name_lists_the_choices(self):
        with pytest.raises(ValueError, match="minmax"):
            normalizer_from_name("zscore")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_normalization.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'normalization'`

- [ ] **Step 3: Write `src/normalization.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_normalization.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/normalization.py tests/test_normalization.py
git commit -m "feat(loader): add intensity normalization strategies

MinMax, Percentile, FixedRange and Raw decide which (lo, hi) is mapped onto
[0, 1]; scale() is the single place the mapping lives. Nothing uses them yet.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `LoadedImage.raw` and `ImageLoader(normalizer)`

**Files:**
- Modify: `src/image_loader.py` (whole file: `LoadedImage`, the four decoders, `ImageLoader`)
- Modify: `tests/test_image_loader.py`, `tests/test_image_geometry.py`, `tests/test_iqa_evaluator.py`

**Interfaces:**
- Consumes: Task 1 (`MinMax`, `Normalizer`, `IntensityRange`, `scale`).
- Produces:
  - `LoadedImage(raw: np.ndarray, spacing=None, is_volumetric=False)` — `raw` is `(D, H, W)`, dtype as decoded.
  - `ImageLoader(path: Path, normalizer: Normalizer = MinMax())` with attributes `path`, `suffix`, `normalizer` and properties `raw`, `raw_range: IntensityRange`, `intensity_range: Optional[IntensityRange]`, `tensor` (`(D, 1, H, W)`), `rgb_tensor`, `spacing`, `is_volumetric`, `empty_slice_mask` (unchanged formula until Task 4).

- [ ] **Step 1: Update the tests that construct or read `LoadedImage`**

In `tests/test_image_loader.py`:

1. Delete `_to_normalized_channel_tensor` from the import list (line 19) and delete the whole `TestToNormalizedChannelTensor` class (lines 30–54).
2. Replace every decoder-tensor access — `_load_pil(p).tensor`, `_load_dicom(p).tensor`, `_load_nifti(p).tensor`, `_load_sitk(p).tensor` — with `ImageLoader(p).tensor`. Decoders no longer produce a tensor. (Lines 65, 74, 157, 167, 185, 195, 218, 226 at the time of writing; use `grep -n ")\.tensor" tests/test_image_loader.py`.)
3. Replace `test_empty_slice_mask_flat_image`'s comment `# Constant array → all values < 1e-3 after normalisation → empty` with `# Constant array → zero spread → every slice empty`.
4. Add to `TestImageLoader`:

```python
    def test_flat_image_keeps_its_value_and_warns(self, tmp_path, capsys):
        arr = np.full((64, 64), 42, dtype="uint8")
        p = tmp_path / "flat.png"
        Image.fromarray(arr).save(p)
        t = ImageLoader(p).tensor
        assert torch.all(t == 1.0)
        assert "flat.png" in capsys.readouterr().out

    def test_raw_is_the_decoded_array(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        assert loader.raw.dtype == np.uint8
        assert loader.raw.shape == (1, IMG_SIZE, IMG_SIZE)

    def test_raw_range_and_intensity_range_under_minmax(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        assert loader.raw_range.lo == float(loader.raw.min())
        assert loader.raw_range.hi == float(loader.raw.max())
        assert loader.intensity_range == loader.raw_range

    def test_percentile_strategy_is_applied(self, tmp_path):
        arr = np.linspace(100, 140, 64 * 64, dtype=np.float32).reshape(64, 64)
        arr[0, 0] = 4000.0
        p = tmp_path / "spike.nii"
        nib.save(nib.Nifti1Image(arr[..., np.newaxis], np.eye(4)), str(p))
        t = ImageLoader(p, Percentile()).tensor
        assert float(t.max()) == 1.0
        assert float(t[0, 0, 32, 32]) == pytest.approx(0.5, abs=0.05)

    def test_raw_strategy_preserves_integer_labels(self, tmp_path):
        labels = np.zeros((8, 8, 3), dtype=np.int16)
        labels[:4, :, 0] = 1
        labels[4:, :, 1] = 2
        labels[:, :4, 2] = 3
        p = tmp_path / "labels.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        loader = ImageLoader(p, Raw())
        assert loader.tensor.dtype == torch.int16
        assert loader.intensity_range is None
        assert sorted(torch.unique(loader.tensor).tolist()) == [0, 1, 2, 3]

    def test_default_normalizer_is_minmax(self, synthetic_png):
        assert ImageLoader(synthetic_png).normalizer == MinMax()
```

   and extend the imports at the top of the file with `from normalization import MinMax, Percentile, Raw`; define `IMG_SIZE = 96` at module level (it must match `tests/conftest.py`).

In `tests/test_image_geometry.py`:

1. Replace the `TestLoadedImage` test with:

```python
class TestLoadedImage:
    def test_is_a_frozen_dataclass_with_three_fields(self):
        li = LoadedImage(raw=np.zeros((1, 2, 2), dtype="float32"), spacing=(1.0, 2.0, 3.0), is_volumetric=True)
        assert li.raw.shape == (1, 2, 2)
        assert li.spacing == (1.0, 2.0, 3.0)
        assert li.is_volumetric is True
        with pytest.raises(Exception):
            li.spacing = (1.0, 1.0, 1.0)
```

2. Replace every `loaded.tensor` with `loaded.raw` in the file (`grep -n "\.tensor" tests/test_image_geometry.py`). `raw` is `(D, H, W)`, so `.shape[0]` still is the slice count; delete any assertion of the form `.shape[1] == 1` on a decoder result, because the channel axis no longer exists there.

In `tests/test_iqa_evaluator.py`:

1. Add `import numpy as np` and `from normalization import MinMax` to the imports.
2. Replace `_make_loader` with:

```python
def _make_loader(n_slices: int = 3, h: int = 64, w: int = 64) -> ImageLoader:
    """Build an ImageLoader without touching the filesystem."""
    loader = object.__new__(ImageLoader)
    loader.path = _FakePath()
    loader.suffix = ".png"
    loader.normalizer = MinMax()
    loader._loaded = LoadedImage(np.random.default_rng(0).random((n_slices, h, w)).astype("float32"))
    loader._tensor = None
    loader._intensity_range = None
    loader._intensity_range_known = False
    return loader
```

3. In `test_empty_slices_have_no_metric_values`, replace `loader._loaded.tensor[0] = torch.zeros(1, 64, 64)` with `loader._loaded.raw[0] = 0.0`.
4. Around line 234, replace `assert loader._loaded.tensor.shape[0] == n` with `assert loader._loaded.raw.shape[0] == n`.

- [ ] **Step 2: Run the three test files to verify they fail**

Run: `.venv/bin/pytest tests/test_image_loader.py tests/test_image_geometry.py tests/test_iqa_evaluator.py -q`
Expected: failures such as `TypeError: LoadedImage.__init__() got an unexpected keyword argument 'raw'` and `ImportError: cannot import name 'MinMax'` is *not* expected (Task 1 provides it); `AttributeError: 'ImageLoader' object has no attribute 'raw'`.

- [ ] **Step 3: Rewrite `LoadedImage`, the decoders and `ImageLoader`**

Replace the top of `src/image_loader.py` down to the end of `_load_sitk` with:

```python
"""Image loading: format-specific decoders, ImageLoader, filename matching.

Decoders return the raw intensities the file holds (after the steps that are
part of decoding: DICOM rescale slope/intercept and MONOCHROME1 inversion,
NIfTI canonical reorientation, axis transposition). How those intensities
reach the [0, 1] every metric expects is decided per run by a
`normalization.Normalizer`, applied lazily by `ImageLoader`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from PIL import Image

from normalization import FixedRange, IntensityRange, MinMax, Normalizer, scale

Spacing = tuple[float, float, float]


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image plus the geometry the decoder knew about.

    Attributes:
        raw:           (D, H, W) array in the dtype the decoder produced —
                       uint8 for PNG/JPEG, float32 for DICOM (slope/intercept
                       applied), the file's own dtype for NIfTI and
                       SimpleITK formats, so an integer label map stays integer.
        spacing:       physical voxel size in millimetres, ordered to match the
                       array's axes: (d_depth, d_height, d_width) for (D, H, W).
                       The names are array axes, not anatomical ones — for a
                       NIfTI transposed to (Z, X, Y), for instance, the tuple
                       is anatomically (dz, dx, dy). None when the format
                       carries no geometry (PNG/JPEG) or when the depth axis
                       is not spatial.
        is_volumetric: True only when the depth axis is a real spatial axis with
                       more than one slice. False for 2D formats, for a single
                       slice, and for 4D NIfTI (whose depth axis mixes time and
                       space, see `_load_nifti`).
    """
    raw:           np.ndarray
    spacing:       Optional[Spacing] = None
    is_volumetric: bool = False


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_pil(path: Path) -> LoadedImage:
    grayscale = np.asarray(Image.open(path).convert("L"))
    return LoadedImage(grayscale[np.newaxis])


def _dicom_array_to_depth_first(pixel_array: np.ndarray, photometric: str) -> np.ndarray:
    pixel_array = np.squeeze(pixel_array)
    if pixel_array.ndim == 2:
        return pixel_array[np.newaxis]
    if pixel_array.ndim == 3:
        if pixel_array.shape[-1] in (3, 4) and photometric.startswith("RGB"):
            luminance = (
                0.2989 * pixel_array[..., 0].astype(np.float32)
                + 0.5870 * pixel_array[..., 1].astype(np.float32)
                + 0.1140 * pixel_array[..., 2].astype(np.float32)
            )
            return luminance[np.newaxis]
        return pixel_array
    raise ValueError(f"Unsupported DICOM pixel_array shape {pixel_array.shape}")


def _load_dicom(path: Path) -> LoadedImage:
    dicom_dataset = pydicom.dcmread(str(path))
    photometric = str(getattr(dicom_dataset, "PhotometricInterpretation", "MONOCHROME2"))
    pixel_array = _dicom_array_to_depth_first(
        dicom_dataset.pixel_array, photometric
    ).astype(np.float32)
    # RescaleSlope/Intercept turn stored values into the modality's units and
    # are part of decoding. WindowCenter/WindowWidth are deliberately not read:
    # they describe how a viewer should display the image, not what it holds.
    slope = float(getattr(dicom_dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dicom_dataset, "RescaleIntercept", 0.0) or 0.0)
    pixel_array = pixel_array * slope + intercept
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # PixelSpacing is [row spacing, column spacing] = (dy, dx).
    pixel_spacing = getattr(dicom_dataset, "PixelSpacing", None)
    thickness = getattr(dicom_dataset, "SliceThickness", None)
    spacing: Optional[Spacing] = None
    if pixel_spacing is not None and thickness:
        spacing = (float(thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))

    # A cine series stacks frames over time, not over space. FrameTime / CineRate
    # are the usual markers; without them a multi-frame series is taken as spatial.
    is_cine = hasattr(dicom_dataset, "FrameTime") or hasattr(dicom_dataset, "CineRate")
    depth = int(pixel_array.shape[0])
    return LoadedImage(
        pixel_array,
        spacing,
        is_volumetric=(depth > 1 and spacing is not None and not is_cine),
    )


def _load_nifti(path: Path) -> LoadedImage:
    image = nib.as_closest_canonical(nib.load(str(path)))
    # asanyarray keeps the stored dtype (an int16 label map stays int16) and
    # applies scl_slope/scl_inter only when the header declares them;
    # get_fdata() would force float64 on everything.
    data = np.asanyarray(image.dataobj)
    zooms = image.header.get_zooms()
    if data.ndim == 3:
        depth_first = np.transpose(data, (2, 0, 1))
        # zooms are (dx, dy, dz) for array axes (X, Y, Z); after the transpose
        # the array axes are (Z, X, Y), so the spacing follows as (dz, dx, dy).
        spacing: Optional[Spacing] = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
        volumetric = depth_first.shape[0] > 1
    elif data.ndim == 4:
        # Time and depth are flattened into a single axis here, so that axis is
        # not spatial and no honest 3-tuple of voxel sizes describes it.
        depth_first = np.transpose(data, (3, 2, 0, 1)).reshape(-1, data.shape[0], data.shape[1])
        spacing = None
        volumetric = False
    else:
        raise ValueError(f"Unsupported NIfTI ndim {data.ndim} for {path}")
    return LoadedImage(depth_first, spacing, volumetric)


def _load_sitk(path: Path) -> LoadedImage:
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image)
    raw_spacing = image.GetSpacing()  # (x, y, z) — the reverse of the array's axes
    spacing: Optional[Spacing] = None
    if volume.ndim == 2:
        volume = volume[np.newaxis]
    elif volume.ndim == 3:
        if len(raw_spacing) == 3:
            spacing = (float(raw_spacing[2]), float(raw_spacing[1]), float(raw_spacing[0]))
    else:
        raise ValueError(f"Unsupported SimpleITK array shape {volume.shape} for {path}")
    return LoadedImage(
        volume,
        spacing,
        is_volumetric=(spacing is not None and volume.shape[0] > 1),
    )
```

Leave `_LOADERS`, `canonical_suffix`, `is_supported` and the filename-matching section as they are. Replace the `ImageLoader` class with:

```python
class ImageLoader:
    """Lazy decoder plus per-run normalization.

    `normalizer` decides what `[0, 1]` stands for in `.tensor`. The default,
    `MinMax()`, scales the image's own extremes. For a full-reference pair use
    `load_pair()`, which puts the input on the target's scale.
    """

    def __init__(self, path: Path, normalizer: Normalizer = MinMax()):
        self.path = path
        self.suffix = canonical_suffix(path)
        if self.suffix not in _LOADERS:
            raise ValueError(f"Unsupported format: {path}")
        self.normalizer = normalizer
        self._loaded: Optional[LoadedImage] = None
        self._tensor: Optional[torch.Tensor] = None
        self._intensity_range: Optional[IntensityRange] = None
        self._intensity_range_known = False

    @property
    def _image(self) -> LoadedImage:
        if self._loaded is None:
            self._loaded = _LOADERS[self.suffix](self.path)
        return self._loaded

    @property
    def raw(self) -> np.ndarray:
        """The decoded (D, H, W) array, dtype as the file stored it."""
        return self._image.raw

    @property
    def raw_range(self) -> IntensityRange:
        """The image's own extremes, whatever the strategy."""
        return IntensityRange(float(self.raw.min()), float(self.raw.max()))

    @property
    def intensity_range(self) -> Optional[IntensityRange]:
        """What `[0, 1]` in `.tensor` stands for; None under `Raw`."""
        if not self._intensity_range_known:
            self._intensity_range = self.normalizer.range_of(self.raw)
            self._intensity_range_known = True
        return self._intensity_range

    @property
    def tensor(self) -> torch.Tensor:
        """(D, 1, H, W); float32 in [0, 1] for every strategy but `Raw`."""
        if self._tensor is None:
            self._tensor = scale(self.raw, self.intensity_range, label=self.path.name).unsqueeze(1)
        return self._tensor

    @property
    def spacing(self) -> Optional[Spacing]:
        """Physical voxel size (dz, dy, dx) in mm, or None if the format has none."""
        return self._image.spacing

    @property
    def is_volumetric(self) -> bool:
        """True when the depth axis is a real spatial axis with more than one slice."""
        return self._image.is_volumetric

    @property
    def rgb_tensor(self) -> torch.Tensor:
        return self.tensor.expand(-1, 3, -1, -1)

    @property
    def empty_slice_mask(self) -> torch.Tensor:
        volume = self.tensor.squeeze(1)
        return (volume.mean(dim=(1, 2)) < 1e-3) | (volume.std(dim=(1, 2)) < 1e-3)

    def log_tensor_shape(self) -> torch.Size:
        shape = self.tensor.shape
        print(f"[{self.path.name}] tensor size: {tuple(shape)}")
        return shape
```

(`empty_slice_mask` is rewritten in Task 4; `FixedRange` is imported now for Task 3's `load_pair`.)

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py`
Expected: all pass. `empty_slice_mask` under `Raw` is not exercised yet.

- [ ] **Step 5: Commit**

```bash
git add src/image_loader.py tests/test_image_loader.py tests/test_image_geometry.py tests/test_iqa_evaluator.py
git commit -m "feat(loader): decoders return raw data; ImageLoader applies a Normalizer

LoadedImage carries the decoded array instead of a tensor. Scaling to [0, 1]
happens lazily in ImageLoader through the strategy it was given (MinMax by
default). NIfTI keeps the stored dtype so label maps survive Raw().

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `load_pair()` — full-reference pairs share the target's scale

**Files:**
- Modify: `src/image_loader.py` (add `load_pair` after the `ImageLoader` class)
- Test: `tests/test_image_loader.py`

**Interfaces:**
- Consumes: Task 2 `ImageLoader`, Task 1 `FixedRange`.
- Produces: `load_pair(input_path: Path, target_path: Path, normalizer: Normalizer = MinMax()) -> tuple[ImageLoader, ImageLoader]` returning `(input, target)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_loader.py`:

```python
# ---------------------------------------------------------------------------
# load_pair
# ---------------------------------------------------------------------------

def _save_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(path))
    return path


class TestLoadPair:
    def test_input_is_scaled_on_the_targets_range(self, tmp_path):
        target = np.random.default_rng(0).random((16, 16, 4)) * 800
        inp_p = _save_nifti(tmp_path / "inp.nii", 2 * target + 500)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        inp, tgt = load_pair(inp_p, tgt_p)
        assert inp.intensity_range == tgt.intensity_range
        assert inp.normalizer.name == "minmax"

    def test_affine_bias_is_no_longer_invisible(self, tmp_path):
        # The regression this design fixes: under independent MinMax the two
        # tensors were identical, so every full-reference metric saw a perfect
        # match. On the target's scale the bias shows up as clipping.
        target = np.random.default_rng(0).random((16, 16, 4)) * 800
        inp_p = _save_nifti(tmp_path / "inp.nii", 2 * target + 500)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        independent = torch.equal(ImageLoader(inp_p).tensor, ImageLoader(tgt_p).tensor)
        inp, tgt = load_pair(inp_p, tgt_p)
        assert independent
        assert not torch.equal(inp.tensor, tgt.tensor)
        assert float(inp.tensor.max()) == 1.0

    def test_strategy_is_taken_from_the_target(self, tmp_path):
        target = np.random.default_rng(1).random((16, 16, 4)) * 100
        target[0, 0, 0] = 5000.0
        inp_p = _save_nifti(tmp_path / "inp.nii", target)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        inp, tgt = load_pair(inp_p, tgt_p, Percentile())
        assert inp.normalizer.name == "percentile_0.5_99.5"
        assert inp.intensity_range.hi < 5000.0

    def test_raw_pair_stays_raw(self, tmp_path):
        labels = np.zeros((8, 8, 2), dtype=np.int16); labels[:4] = 1
        p1 = tmp_path / "a.nii"; p2 = tmp_path / "b.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p1))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p2))
        inp, tgt = load_pair(p1, p2, Raw())
        assert inp.intensity_range is None and tgt.intensity_range is None
        assert inp.tensor.dtype == torch.int16
        assert inp.normalizer.name == "raw"
```

Add `load_pair` to the `from image_loader import (...)` list at the top of the file.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_image_loader.py -q -k LoadPair`
Expected: `ImportError: cannot import name 'load_pair'`

- [ ] **Step 3: Implement `load_pair`**

Append to `src/image_loader.py` after the `ImageLoader` class:

```python
def load_pair(
    input_path: Path,
    target_path: Path,
    normalizer: Normalizer = MinMax(),
) -> tuple[ImageLoader, ImageLoader]:
    """Load a full-reference pair on ONE scale — the target's.

    The target is scaled by `normalizer`; the input is then scaled by the
    range the target produced, so a prediction that is uniformly too bright
    or too flat is measured as such instead of being normalized away
    (fastMRI's convention: `data_range` comes from the reference). Anything
    outside the target's range is clipped to [0, 1] in the input; the raw
    extremes remain visible as `ImageLoader.raw_range`.

    Under `Raw()` the target yields no range and the input stays raw too.

    Returns `(input, target)`.
    """
    target = ImageLoader(target_path, normalizer)
    inp = ImageLoader(input_path, FixedRange(target.intensity_range, name=normalizer.name))
    return inp, target
```

- [ ] **Step 4: Run the file**

Run: `.venv/bin/pytest tests/test_image_loader.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/image_loader.py tests/test_image_loader.py
git commit -m "feat(loader): load_pair scales a full-reference pair on the target

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `empty_slice_mask` on raw data

**Files:**
- Modify: `src/image_loader.py` (`ImageLoader.empty_slice_mask`)
- Test: `tests/test_image_loader.py`

**Interfaces:**
- Consumes: Task 2.
- Produces: `ImageLoader.empty_slice_mask -> torch.BoolTensor` of shape `(D,)`, computed from `raw`, independent of `normalizer`.

- [ ] **Step 1: Write the failing tests**

Append to the `TestImageLoader` class in `tests/test_image_loader.py`:

```python
    def test_empty_slice_mask_survives_a_spike_voxel(self, tmp_path):
        # One corrupt voxel used to compress the whole volume into a sliver of
        # [0, 1], pushing every slice's std under the threshold.
        vol = np.random.default_rng(0).random((32, 32, 6)).astype(np.float32) * 100
        vol[0, 0, 0] = 1e6
        p = tmp_path / "spike.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        mask = ImageLoader(p).empty_slice_mask
        assert not mask.any()

    def test_empty_slice_mask_flags_the_blank_slice_only(self, tmp_path):
        vol = np.random.default_rng(0).random((32, 32, 4)).astype(np.float32) * 100
        vol[:, :, 2] = 0.0
        p = tmp_path / "hole.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        assert ImageLoader(p).empty_slice_mask.tolist() == [False, False, True, False]

    def test_empty_slice_mask_is_independent_of_the_strategy(self, tmp_path):
        vol = np.random.default_rng(0).random((32, 32, 4)).astype(np.float32) * 100
        vol[:, :, 2] = 0.0
        p = tmp_path / "hole.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        assert torch.equal(ImageLoader(p, Raw()).empty_slice_mask, ImageLoader(p).empty_slice_mask)

    def test_empty_slice_mask_keeps_a_sparse_mask_slice(self, tmp_path):
        # A label map whose foreground is below 0.5 % of the volume: the
        # percentile span collapses to 0 and must fall back to the extremes.
        labels = np.zeros((64, 64, 4), dtype=np.int16)
        labels[10:13, 10, 1] = 1
        p = tmp_path / "sparse.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        assert ImageLoader(p, Raw()).empty_slice_mask.tolist() == [True, False, True, True]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_image_loader.py -q -k empty_slice_mask`
Expected: `test_empty_slice_mask_survives_a_spike_voxel` fails (all slices flagged), `test_empty_slice_mask_keeps_a_sparse_mask_slice` fails under `Raw` (unscaled values, threshold meaningless), `..._independent_of_the_strategy` fails for the same reason.

- [ ] **Step 3: Rewrite `empty_slice_mask`**

Replace the property in `src/image_loader.py`:

```python
    @property
    def empty_slice_mask(self) -> torch.Tensor:
        """True for slices with (almost) no content; computed on the raw data.

        A slice is empty when its spread or its lift above the volume floor is
        below 0.1 % of the volume's intensity span. The span is measured
        between the 0.5th and 99.5th percentiles, so one spike voxel cannot
        shrink it and mark healthy slices empty. For a volume whose foreground
        is rarer than 0.5 % (a small lesion mask) the percentile span collapses
        to zero and the extremes are used instead. A constant volume has no
        span at all and every slice counts as empty.
        """
        raw = self.raw.astype(np.float32, copy=False)
        lo, hi = (float(v) for v in np.percentile(raw, [0.5, 99.5]))
        if hi <= lo:
            lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo
        if span <= 0.0:
            return torch.ones(raw.shape[0], dtype=torch.bool)
        flat = raw.reshape(raw.shape[0], -1)
        std  = flat.std(axis=1)
        lift = flat.mean(axis=1) - lo
        return torch.from_numpy((std < 1e-3 * span) | (lift < 1e-3 * span))
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py`
Expected: all pass, including the existing `test_empty_slice_mask_flat_image`, `test_empty_slice_mask_structured_image` and `test_empty_slices_have_no_metric_values`.

- [ ] **Step 5: Commit**

```bash
git add src/image_loader.py tests/test_image_loader.py
git commit -m "fix(loader): test slice emptiness on raw data with a spike-proof span

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Scale columns in every report row

**Files:**
- Modify: `src/records.py:21` (after `is_empty`)
- Modify: `src/iqa_evaluator.py` (`run_evaluation`, new `_scale_fields`)
- Modify: `src/volume_evaluator.py` (`run_evaluation`)
- Test: `tests/test_records.py`, `tests/test_iqa_evaluator.py`, `tests/test_volume_evaluator.py`

**Interfaces:**
- Consumes: Task 2 (`ImageLoader.normalizer`, `.intensity_range`, `.raw_range`).
- Produces:
  - `ImageEvaluatorRecord` fields `normalization: Optional[str]`, `scale_lo: Optional[float]`, `scale_hi: Optional[float]`, `input_min: Optional[float]`, `input_max: Optional[float]`, all default `None`, declared right after `is_empty`.
  - `IQAEvaluator._scale_fields(self) -> dict[str, object]` used by both evaluators.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_records.py`:

```python
class TestScaleFields:
    NAMES = ("normalization", "scale_lo", "scale_hi", "input_min", "input_max")

    @pytest.mark.parametrize("name", NAMES)
    def test_field_defaults_to_none(self, name):
        assert getattr(ImageEvaluatorRecord(image_id="x"), name) is None

    def test_declared_right_after_is_empty(self):
        keys = list(ImageEvaluatorRecord.__annotations__)
        i = keys.index("is_empty")
        assert keys[i + 1 : i + 6] == list(self.NAMES)

    def test_values_survive_to_dict(self):
        r = ImageEvaluatorRecord(image_id="x", normalization="minmax", scale_lo=0.0,
                                 scale_hi=800.0, input_min=3.0, input_max=900.0)
        d = r.to_dict()
        assert d["normalization"] == "minmax" and d["scale_hi"] == 800.0 and d["input_max"] == 900.0
```

(`pytest` and `ImageEvaluatorRecord` are already imported in that file; add `import pytest` if not.)

Append to `tests/test_iqa_evaluator.py`:

```python
class TestScaleFieldsOnRecords:
    def _registry(self):
        return MetricRegistry(PSNR)

    def test_minmax_run_reports_its_own_range(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        rec = IQAEvaluator(loader, None, self._registry()).run_evaluation()[0]
        assert rec.normalization == "minmax"
        assert (rec.scale_lo, rec.scale_hi) == (loader.raw_range.lo, loader.raw_range.hi)
        assert (rec.input_min, rec.input_max) == (loader.raw_range.lo, loader.raw_range.hi)

    def test_full_reference_run_reports_the_targets_scale(self, tmp_path):
        import nibabel as nib
        from image_loader import load_pair
        target = np.random.default_rng(0).random((16, 16, 3)) * 800
        for name, arr in (("inp.nii", 2 * target + 500), ("tgt.nii", target)):
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(tmp_path / name))
        inp, tgt = load_pair(tmp_path / "inp.nii", tmp_path / "tgt.nii")
        rec = IQAEvaluator(inp, tgt, self._registry()).run_evaluation()[0]
        assert (rec.scale_lo, rec.scale_hi) == (tgt.raw_range.lo, tgt.raw_range.hi)
        assert rec.input_max > rec.scale_hi          # the overshoot is visible
        assert rec.input_min > rec.scale_lo

    def test_raw_run_has_no_scale(self, tmp_path):
        import nibabel as nib
        from normalization import Raw
        labels = np.zeros((8, 8, 2), dtype=np.int16); labels[:4] = 1
        p = tmp_path / "m.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        rec = IQAEvaluator(ImageLoader(p, Raw()), None, MetricRegistry()).run_evaluation()[0]
        assert rec.normalization == "raw"
        assert rec.scale_lo is None and rec.scale_hi is None
        assert (rec.input_min, rec.input_max) == (0.0, 1.0)
```

Append to `tests/test_volume_evaluator.py`:

```python
class TestScaleFieldsInVolumeMode:
    def test_volume_row_carries_the_scale(self, nifti_volume):
        from metrics import PSNR
        loader = ImageLoader(nifti_volume)
        rec = VolumeEvaluator(loader, None, MetricRegistry(PSNR)).run_evaluation()[0]
        assert rec.normalization == "minmax"
        assert (rec.scale_lo, rec.scale_hi) == (loader.raw_range.lo, loader.raw_range.hi)
        assert (rec.input_min, rec.input_max) == (loader.raw_range.lo, loader.raw_range.hi)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_records.py tests/test_iqa_evaluator.py tests/test_volume_evaluator.py -q -k "Scale"`
Expected: `AttributeError: 'ImageEvaluatorRecord' object has no attribute 'normalization'` and the `to_dict`/`__init__` keyword errors.

- [ ] **Step 3: Add the fields and fill them**

In `src/records.py`, directly after `is_empty:            bool            = False`:

```python
    # What [0, 1] stood for in this run (see normalization.py). In a
    # full-reference run scale_lo/hi is the TARGET's range; input_min/max are
    # the input's raw extremes, so input_max > scale_hi means the prediction
    # overshot the reference and was clipped. None under the Raw strategy.
    normalization:       Optional[str]   = None
    scale_lo:            Optional[float] = None
    scale_hi:            Optional[float] = None
    input_min:           Optional[float] = None
    input_max:           Optional[float] = None
```

In `src/iqa_evaluator.py`, add to the "Internal helpers" section:

```python
    def _scale_fields(self) -> dict[str, object]:
        """The normalization columns of a record, read from the input loader."""
        rng = self.input.intensity_range
        raw = self.input.raw_range
        return {
            "normalization": self.input.normalizer.name,
            "scale_lo":      None if rng is None else rng.lo,
            "scale_hi":      None if rng is None else rng.hi,
            "input_min":     raw.lo,
            "input_max":     raw.hi,
        }
```

and in `run_evaluation`, build the records with the extra fields:

```python
        scale_fields = self._scale_fields()
        records = [
            ImageEvaluatorRecord(
                image_id=self._format_slice_id(i),
                source_model=self.source_model,
                mode=mode,
                slice_index=i,
                is_empty=bool(empty_mask[i].item()),
                **scale_fields,
            )
            for i in range(D)
        ]
```

In `src/volume_evaluator.py`, `run_evaluation`:

```python
        record = ImageEvaluatorRecord(
            image_id=self._format_volume_id(),
            source_model=self.source_model,
            mode="full_reference" if has_target else "no_reference",
            scoring="volume",
            slice_index=None,
            is_empty=bool(self.input.empty_slice_mask.all().item()),
            **self._scale_fields(),
        )
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py`
Expected: all pass. `to_frame()` picks the new columns up from `__annotations__`; `test_evaluation_result.py` needs no change.

- [ ] **Step 5: Commit**

```bash
git add src/records.py src/iqa_evaluator.py src/volume_evaluator.py tests/test_records.py tests/test_iqa_evaluator.py tests/test_volume_evaluator.py
git commit -m "feat(report): write the intensity scale into every row

normalization, scale_lo/hi and input_min/max make the scale behind psnr's
data_range explicit and show a full-reference bias without a new metric.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Full-reference emptiness and `aggregate_volumes()`

**Files:**
- Modify: `src/iqa_evaluator.py:72-76` (`run_evaluation`, empty mask)
- Modify: `src/evaluation_result.py:77-190` (`aggregate_volumes` docstring and body)
- Test: `tests/test_iqa_evaluator.py`, `tests/test_evaluation_result.py`

**Interfaces:**
- Consumes: Task 4 `empty_slice_mask`.
- Produces: in slice mode with a target, `record.is_empty == input_empty and target_empty`; `aggregate_volumes()` fills `v_pred`, `v_gt`, `tp` with `0.0` on `is_empty` rows.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_iqa_evaluator.py`:

```python
class TestFullReferenceEmptiness:
    def test_blank_input_over_occupied_target_is_scored(self):
        inp = _make_loader(3); tgt = _make_loader(3)
        inp._loaded.raw[0] = 0.0
        records = IQAEvaluator(inp, tgt, MetricRegistry(PSNR)).run_evaluation()
        assert records[0].is_empty is False
        assert records[0].psnr is not None

    def test_blank_on_both_sides_is_empty(self):
        inp = _make_loader(3); tgt = _make_loader(3)
        inp._loaded.raw[0] = 0.0
        tgt._loaded.raw[0] = 0.0
        records = IQAEvaluator(inp, tgt, MetricRegistry(PSNR)).run_evaluation()
        assert records[0].is_empty is True
        assert records[0].psnr is None

    def test_no_reference_still_uses_the_input_alone(self):
        inp = _make_loader(3)
        inp._loaded.raw[0] = 0.0
        records = IQAEvaluator(inp, None, MetricRegistry()).run_evaluation()
        assert records[0].is_empty is True
```

In `tests/test_evaluation_result.py`, replace `test_blank_prediction_slice_zeroes_only_the_prediction_counts` and `test_blank_prediction_slice_is_warned_about_by_name` with:

```python
    def test_blank_slice_contributes_zeros_to_all_counts(self):
        # `is_empty` on a full-reference row means BOTH masks are blank there,
        # so every count on that slice is a real zero.
        records = [
            _empty_slice_record("vol", 0),
            _slice_record("vol", 1, v_pred=8, v_gt=4, tp=4),
        ]
        row = _result(records).aggregate_volumes().loc["vol"]
        assert row["v_pred"] == 8.0
        assert row["v_gt"] == 4.0
        assert row["tp"] == 4.0
        assert row["dice"] == pytest.approx(2 * 4 / 12)

    def test_blank_slice_is_not_warned_about(self, capsys):
        records = [
            _empty_slice_record("vol", 0),
            _slice_record("vol", 1, v_pred=8, v_gt=4, tp=4),
        ]
        _result(records).aggregate_volumes()
        assert capsys.readouterr().out == ""
```

Also update the docstring of `_empty_slice_record` to `"""A slice blank on both sides: is_empty True, no counts recorded at all."""`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_iqa_evaluator.py tests/test_evaluation_result.py -q -k "Emptiness or blank_slice"`
Expected: `test_blank_input_over_occupied_target_is_scored` fails (`is_empty` True), `test_blank_slice_contributes_zeros_to_all_counts` fails (`v_gt` NaN), `test_blank_slice_is_not_warned_about` fails (warning printed).

- [ ] **Step 3: Implement**

In `src/iqa_evaluator.py`, `run_evaluation`, replace `empty_mask = self.input.empty_slice_mask` with:

```python
        # A slice is skipped only when there is nothing to measure on either
        # side. A blank prediction over an occupied reference is a failure and
        # gets scored; in a no-reference run the input alone decides.
        empty_mask = self.input.empty_slice_mask
        if has_target:
            empty_mask = empty_mask & self.target.empty_slice_mask
```

(`has_target` is assigned two lines below today; move its assignment above this block.)

In `src/evaluation_result.py`, replace the docstring paragraph of `aggregate_volumes` that begins `A count can be missing for two different reasons` through `so this never raises for that reason.` with:

```
        A count is missing only when the metric raised on that slice (see
        `IQAEvaluator`). Nothing about the slice is known then, so all three
        counts stay NaN and the volume's totals come out NaN; a warning names
        the affected volumes. Slices flagged `is_empty` are blank on both
        sides — `IQAEvaluator` only skips a slice when input and target are
        both empty — so their counts are real zeros and are filled in as such.
        One incomplete volume does not invalidate the rest of the report, so
        this never raises for that reason.
```

and replace the body from `counts = df[[...]].copy()` through the two `if blank:` / `if failed:` blocks with:

```python
        counts = df[["image_id", "is_empty", "v_pred", "v_gt", "tp"]].copy()
        counts["image_id"] = counts["image_id"].str.replace(_SLICE_SUFFIX, "", regex=True)
        # An empty slice is blank on both sides, so every count there is 0.
        for col in ("v_pred", "v_gt", "tp"):
            fillable = counts["is_empty"] & counts[col].isna()
            counts.loc[fillable, col] = 0.0
        grouped = counts.groupby("image_id")[["v_pred", "v_gt", "tp"]].sum(skipna=False)

        v_pred_sum = grouped["v_pred"].astype(float)
        v_gt_sum   = grouped["v_gt"].astype(float)
        tp_sum     = grouped["tp"].astype(float)
        denom      = v_pred_sum + v_gt_sum

        grouped["dice"]      = np.where(denom == 0, np.nan, 2.0 * tp_sum / denom)
        grouped["vs"]        = np.where(denom == 0, np.nan,
                                        1.0 - (v_pred_sum - v_gt_sum).abs() / denom)
        grouped["vs_signed"] = np.where(denom == 0, np.nan,
                                        2.0 * (v_pred_sum - v_gt_sum) / denom)

        failed = list(grouped.index[grouped[["v_pred", "v_gt", "tp"]].isna().any(axis=1)])
        if failed:
            print(
                "Volume-level numbers are incomplete for "
                f"{', '.join(failed)}: some of their slices have voxel "
                "counts that were never computed, most likely because a "
                "metric failed on them during the run. Their volume-level "
                "numbers are left empty rather than computed from an "
                "incomplete sum — check the run's output for the metric "
                "failure."
            )

        return grouped
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py`
Expected: all pass. If a remaining test in `test_evaluation_result.py` asserts the old NaN-on-blank behaviour (search for `blank`), it belongs to the two tests replaced above; remove it.

- [ ] **Step 5: Commit**

```bash
git add src/iqa_evaluator.py src/evaluation_result.py tests/test_iqa_evaluator.py tests/test_evaluation_result.py
git commit -m "fix(evaluator): a slice is empty only when input and target are blank

A blank prediction over an occupied reference is now scored as the failure
it is. aggregate_volumes() therefore fills every count on an empty slice
with 0 and loses its unknown-reference branch.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Segmentation path — raw masks reach the metrics

**Files:**
- Modify: `src/segmentation_metrics/monai_metrics.py:10-14` (module docstring) and `MonaiSegmentationMetric._binarize`
- Modify: `src/segmentation_metrics/volume.py:32-38` (comment in `as_mask`)
- Test: `tests/test_segmentation_metrics.py`, `tests/test_volume.py`

**Interfaces:**
- Consumes: Task 2 `Raw`.
- Produces: `MonaiSegmentationMetric.__call__` accepts integer tensors; `as_mask` unchanged in behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_segmentation_metrics.py` (uses the existing `MonaiSegmentationMetric` and `compute_dice` imports of that file):

```python
class TestIntegerMasks:
    def test_identical_integer_masks_score_perfect_dice(self):
        g = torch.Generator().manual_seed(0)
        mask = torch.randint(0, 2, (2, 1, 16, 16), generator=g)          # int64
        metric = MonaiSegmentationMetric(compute_dice, include_background=True)
        assert metric(mask, mask.clone()) == pytest.approx([1.0, 1.0])

    def test_threshold_zero_makes_every_label_foreground(self):
        labels = torch.zeros((1, 1, 8, 8), dtype=torch.int16)
        labels[0, 0, :4] = 1
        labels[0, 0, 4:] = 3
        metric = MonaiSegmentationMetric(compute_dice, include_background=True, threshold=0.0)
        assert metric(labels, torch.ones_like(labels)) == pytest.approx([1.0])
```

Append to `tests/test_volume.py`:

```python
class TestRawLabelMapEndToEnd:
    def test_one_vs_rest_on_a_raw_loaded_label_map(self, tmp_path):
        import nibabel as nib
        from image_loader import ImageLoader
        from normalization import Raw
        labels = np.zeros((8, 8, 2), dtype=np.int16)
        labels[:4, :, 0] = 1
        labels[4:, :, 0] = 2
        labels[:, :3, 1] = 3
        p = tmp_path / "labels.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        pred = ImageLoader(p, Raw()).tensor[:, 0].numpy()          # (D, H, W) int16
        assert v_pred(pred, pred, label=2) == 32.0
        assert v_pred(pred, pred, label=3) == 24.0
        assert vs(pred, pred, label=2) == 1.0
```

(Add `import numpy as np` and `from segmentation_metrics.volume import v_pred, vs` if that file does not import them already.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_segmentation_metrics.py tests/test_volume.py -q -k "Integer or RawLabel"`
Expected: `test_identical_integer_masks_score_perfect_dice` fails inside MONAI with a dtype error (or returns a wrong value); the label-map test passes already if Task 2 is in place — that is fine, it pins the behaviour.

- [ ] **Step 3: Implement**

In `src/segmentation_metrics/monai_metrics.py`, replace `_binarize`:

```python
    def _binarize(self, t: torch.Tensor) -> torch.Tensor:
        # Masks loaded with normalization.Raw() arrive in their stored integer
        # dtype; MONAI's functionals want floats either way.
        t = t.float()
        return (t > self._threshold).float() if self._threshold is not None else t
```

Replace the module-docstring paragraph that starts `TODO(norm-6): that route is currently lossy.` with:

```
Load mask files with `normalization.Raw()` (`ImageLoader(path, Raw())` or
`load_pair(..., Raw())`) so they arrive unscaled in their stored dtype. A
binary 0/1 mask then needs no `threshold`. For a multi-label map, pass
`threshold=0.0` to score every non-zero label as foreground; these adapters do
not select a single label — the numpy-backed metrics in `volume.py` do, via
`label=`. PNG masks store 0/255 and should be loaded with `MinMax()` instead,
which maps them to exactly {0.0, 1.0}.
```

In `src/segmentation_metrics/volume.py`, replace the `# TODO(norm-6): ...` comment block inside `as_mask` (the seven lines before `if np.issubdtype(x.dtype, np.integer):`) with:

```python
    # Integer input is a label map; it reaches this branch when the file was
    # loaded with normalization.Raw(). Any other strategy scales it to floats
    # and the threshold branch above applies.
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/monai_metrics.py src/segmentation_metrics/volume.py tests/test_segmentation_metrics.py tests/test_volume.py
git commit -m "fix(segmentation): accept raw integer masks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `evaluate(normalization=)` and the `--normalization` CLI flag

**Files:**
- Modify: `src/main.py` (`evaluate` signature, docstring, `_run_one`; `main()` argparse)
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: Task 3 `load_pair`, Task 1 `normalizer_from_name`, `NORMALIZER_NAMES`.
- Produces: `evaluate(input_path, target_path=None, *, registry, mode="slice", normalization: Normalizer = MinMax())`; CLI `--normalization {minmax,percentile,raw}` (default `minmax`). `main.py` re-exports `MinMax`, `Percentile`, `Raw`, `Normalizer`, `load_pair`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_main.py` (the file already imports `main_module`, `evaluate`, `MetricRegistry`, `PSNR`, `SSIM`, `sys`, and has `_make_png`):

```python
class TestNormalizationOption:
    def test_default_is_minmax(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png")
        df = evaluate(inp, registry=MetricRegistry(PSNR)).to_frame()
        assert df["normalization"].iloc[0] == "minmax"

    def test_strategy_is_passed_to_the_loader(self, tmp_path):
        from normalization import Percentile
        inp = _make_png(tmp_path / "inp.png")
        df = evaluate(inp, registry=MetricRegistry(PSNR), normalization=Percentile()).to_frame()
        assert df["normalization"].iloc[0] == "percentile_0.5_99.5"

    def test_full_reference_run_uses_the_targets_range(self, tmp_path):
        import nibabel as nib
        import numpy as np
        from image_loader import ImageLoader
        target = np.random.default_rng(0).random((16, 16, 3)) * 800
        inp_p = tmp_path / "inp.nii"; tgt_p = tmp_path / "tgt.nii"
        nib.save(nib.Nifti1Image((2 * target + 500).astype(np.float32), np.eye(4)), str(inp_p))
        nib.save(nib.Nifti1Image(target.astype(np.float32), np.eye(4)), str(tgt_p))
        df = evaluate(inp_p, tgt_p, registry=MetricRegistry(PSNR)).to_frame()
        tgt_range = ImageLoader(tgt_p).raw_range
        assert df["scale_lo"].iloc[0] == tgt_range.lo
        assert df["scale_hi"].iloc[0] == tgt_range.hi
        assert df["input_max"].iloc[0] > tgt_range.hi

    def test_main_reexports_the_strategies(self):
        from normalization import MinMax, Percentile, Raw
        assert main_module.MinMax is MinMax
        assert main_module.Percentile is Percentile
        assert main_module.Raw is Raw


class TestNormalizationCLI:
    def test_flag_selects_the_strategy(self, tmp_path, monkeypatch, capsys):
        import pandas as pd
        monkeypatch.setattr("main.BUILTIN_METRICS", (PSNR,))
        inp = _make_png(tmp_path / "inp.png")
        report = tmp_path / "report.csv"
        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp), "--normalization", "percentile"])
        main_module.main()
        assert pd.read_csv(report)["normalization"].iloc[0] == "percentile_0.5_99.5"

    def test_unknown_flag_value_is_rejected(self, tmp_path, monkeypatch):
        inp = _make_png(tmp_path / "inp.png")
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp), "--normalization", "zscore"])
        with pytest.raises(SystemExit):
            main_module.main()
```

(Add `import pytest` at the top of the file if it is missing.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_main.py -q -k Normalization`
Expected: `TypeError: evaluate() got an unexpected keyword argument 'normalization'`, `AttributeError: module 'main' has no attribute 'MinMax'`, and the CLI test fails on the unrecognised argument.

- [ ] **Step 3: Implement**

In `src/main.py`:

1. Extend the imports:

```python
from image_loader import ImageLoader, find_matching_target, list_images, load_pair  # noqa: F401
from normalization import (  # noqa: F401 — re-exported for users
    NORMALIZER_NAMES, MinMax, Normalizer, Percentile, Raw, normalizer_from_name,
)
```

2. Change the signature and docstring of `evaluate`:

```python
def evaluate(
    input_path: Path,
    target_path: Optional[Path] = None,
    *,
    registry: MetricRegistry,
    mode: ScoringMode = "slice",
    normalization: Normalizer = MinMax(),
) -> EvaluationResult:
```

   Add this `Args` entry after `mode:` and delete the `TODO(norm-4/norm-5)` paragraph:

```
        normalization: how raw intensities reach the [0, 1] every metric
                     expects. `MinMax()` (default) scales each image's own
                     extremes; `Percentile()` scales robust extremes and
                     changes every score, so compare only against runs with
                     the same setting; `Raw()` skips scaling and is the right
                     choice for mask files. With a target, both images are
                     put on the TARGET's range (fastMRI's convention), so a
                     prediction that is uniformly too bright is measured as
                     such. The scale used is written into every row as
                     `normalization`, `scale_lo`, `scale_hi`; the input's raw
                     extremes as `input_min`, `input_max`.

                     Scaling is per volume, never per slice: a slice's tensor
                     therefore depends on the rest of its stack, and slice
                     rows of one volume are not independent samples.
```

3. In `_run_one`, replace the loader construction:

```python
            if tgt is not None:
                input_loader, target_loader = load_pair(inp, tgt, normalization)
                target_loader.log_tensor_shape()
            else:
                input_loader, target_loader = ImageLoader(inp, normalization), None
            input_loader.log_tensor_shape()
```

   (Keep the `target_loader: Optional[ImageLoader]` type by declaring it before the `if`.)

4. In `main()`, add after the `--mode` argument:

```python
    parser.add_argument(
        "--normalization", choices=list(NORMALIZER_NAMES), default="minmax",
        help="How intensities reach [0, 1]: each image's extremes (minmax), "
             "robust 0.5/99.5 percentiles (percentile), or no scaling (raw, for masks).",
    )
```

   and pass it on: `evaluate(args.input, args.target, registry=registry, mode=args.mode, normalization=normalizer_from_name(args.normalization))`.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat(main): choose the normalization strategy per run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: TODO sweep and in-repo documentation

**Files:**
- Modify: `src/metrics.py:80-92` (`Metric` docstring)
- Modify: `CLAUDE.md`, `README.md:93`, `src/evaluation.ipynb`
- Verify: no `TODO(norm` remains anywhere in `src/` or `tests/`

**Interfaces:**
- Consumes: everything above.
- Produces: documentation only.

- [ ] **Step 1: Sweep**

Run: `grep -rn "TODO(norm" src tests CLAUDE.md README.md`
Expected after this task: no output. At this point the only hits should be `src/metrics.py` (norm-10 in the `Metric` docstring). If others remain, they were missed in Tasks 2–8 — replace each with a plain sentence stating the decision, as the tasks above did.

- [ ] **Step 2: `Metric` docstring**

In `src/metrics.py`, replace the `TODO(norm-10)` paragraph inside `Metric`'s docstring with:

```
    "[0, 1]" is produced by the run's `normalization.Normalizer` (default
    `MinMax`, per volume; full-reference pairs share the target's range —
    see `image_loader.load_pair`). A metric therefore never sees absolute
    intensities; the scale it was given is in the report's `scale_lo/hi`
    columns. Under the `Raw` strategy the batch keeps its stored dtype —
    only segmentation metrics are meant to receive that.
```

- [ ] **Step 3: `CLAUDE.md`**

Replace the `image_loader.py` paragraph with:

```
**`image_loader.py`** — format-specific decoders (PIL/DICOM/NIfTI/SimpleITK) that return `LoadedImage(raw, spacing, is_volumetric)` with the file's own dtype; `ImageLoader(path, normalizer=MinMax())` (lazy; `.raw`, `.raw_range`, `.intensity_range`, `.tensor` = `(D, 1, H, W)` float32 in [0,1], `.rgb_tensor`, `.empty_slice_mask` on raw data); `load_pair(inp, tgt, normalizer)` puts a full-reference pair on the target's scale; input/target filename matching (`list_images`, `find_matching_target`).

**`normalization.py`** — how raw intensities reach [0,1]: `IntensityRange`, the `Normalizer` protocol, `MinMax` (default), `Percentile(0.5, 99.5)` (opt-in, changes every score), `FixedRange` (used by `load_pair`), `Raw` (masks; dtype preserved), and the single mapping `scale()`. `normalizer_from_name()` backs the CLI flag. Every row records `normalization`, `scale_lo/hi`, `input_min/max`.
```

Add to the `main.py` paragraph: `evaluate(..., normalization=MinMax())`; CLI `--normalization {minmax,percentile,raw}`. Add below the metric summary table:

```
Full-reference runs scale input and target on the **target's** range (fastMRI convention); no-reference runs scale each image's own. Load masks with `Raw()`; PNG 0/255 masks with `MinMax()`.
```

- [ ] **Step 4: `README.md`**

Replace line 93 (`Every format is normalized to ...`) with:

```
Every format decodes to a `(D, H, W)` array in its stored dtype, where `D` is the number of slices (2D inputs become `D=1`). `ImageLoader` then scales it to a `(D, 1, H, W)` float32 tensor in `[0, 1]` using the run's normalization strategy: `MinMax()` (default, the image's own extremes), `Percentile()` (robust extremes, opt-in) or `Raw()` (no scaling, for masks). A full-reference pair is scaled on the **target's** range, so a systematically too-bright prediction is measured as such. The report records the scale in `normalization`, `scale_lo`, `scale_hi`, `input_min`, `input_max`.
```

- [ ] **Step 5: Notebook**

Run this script from the repository root:

```python
# .venv/bin/python - <<'EOF'
import json
p = "src/evaluation.ipynb"
nb = json.load(open(p))
found_paths = found_call = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "INPUT  = Path(" in src and "NORMALIZATION" not in src:
        src = src.replace(
            'TARGET = Path("data/IXI_test_resampled/")',
            'TARGET = Path("data/IXI_test_resampled/")\n\n'
            '# How intensities reach [0, 1]: MinMax() (default), Percentile(), Raw() for masks.\n'
            '# With a target, both images are scaled on the TARGET\'s range.\n'
            'NORMALIZATION = MinMax()',
        )
        cell["source"] = src.splitlines(keepends=True)
        found_paths = True
    if "result = evaluate(INPUT, TARGET, registry=registry)" in src:
        src = src.replace(
            "result = evaluate(INPUT, TARGET, registry=registry)",
            "result = evaluate(INPUT, TARGET, registry=registry, normalization=NORMALIZATION)",
        )
        cell["source"] = src.splitlines(keepends=True)
        found_call = True
assert found_paths and found_call, (found_paths, found_call)
json.dump(nb, open(p, "w"), indent=1, ensure_ascii=False)
open(p, "a").write("\n")
# EOF
```

Then open the notebook's import cell (the one that has `from main import ...`) and add `MinMax` to that import list — `main.py` re-exports it (Task 8). Confirm with:

```bash
grep -c "normalization=NORMALIZATION" src/evaluation.ipynb
```

Expected: `1`.

Add one markdown cell after the `df.head()` cell (edit the JSON in the same way: insert `{"cell_type": "markdown", "metadata": {}, "source": [...]}`) with:

```
`scale_lo` / `scale_hi` is the intensity range mapped to `[0, 1]` — the target's range in a full-reference run. `input_min` / `input_max` are the prediction's raw extremes; `input_max > scale_hi` means the prediction overshot the reference and was clipped. `psnr` uses `data_range = 1.0` on this scale.
```

- [ ] **Step 6: Run the suite and the sweep**

Run: `.venv/bin/pytest tests -q --deselect tests/test_dreamsim_integration.py && grep -rn "TODO(norm" src tests CLAUDE.md README.md; echo "exit=$?"`
Expected: tests pass; grep prints nothing and `exit=1`.

- [ ] **Step 7: Commit**

```bash
git add src/metrics.py CLAUDE.md README.md src/evaluation.ipynb
git commit -m "docs: describe the normalization strategy and retire the TODO(norm) markers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Wiki pages

**Files:**
- Wiki repository (separate): `git@github.com:glebjan/hka.projektarbeit1.IQA_MRI.wiki.git`, cloned into the session scratchpad directory. Pages: `Loading-Images.md`, `Results-and-Reports.md`, `Interpreting-Scores.md`, `Segmentation-Metrics.md`.

**Interfaces:**
- Consumes: the final API from Tasks 1–8.
- Produces: a local wiki commit. **Do not push**; report the clone path and ask the user.

- [ ] **Step 1: Clone**

```bash
git clone git@github.com:glebjan/hka.projektarbeit1.IQA_MRI.wiki.git "$SCRATCHPAD/wiki"
```

(`$SCRATCHPAD` is the session's scratchpad directory from the system prompt.) If the clone fails for lack of SSH access, stop this task and report it; everything else in the plan is independent of it.

- [ ] **Step 2: `Loading-Images.md`**

Add a section `## Normalization` after the section that describes `.tensor`:

````markdown
## Normalization

Decoders return the file's raw intensities (`ImageLoader.raw`, `(D, H, W)`, stored dtype).
How they reach the `[0, 1]` every metric expects is a per-run choice:

| Strategy | Range mapped to `[0, 1]` | When |
|---|---|---|
| `MinMax()` | the image's own min/max | default |
| `Percentile(lower=0.5, upper=99.5)` | robust extremes | spike voxels distort the range; changes **every** score |
| `Raw()` | none — dtype preserved | mask and label-map files |

```python
from image_loader import ImageLoader, load_pair
from normalization import MinMax, Percentile, Raw

img        = ImageLoader(path, Percentile())
inp, tgt   = load_pair(inp_path, tgt_path)          # full-reference: ONE scale, the target's
mask       = ImageLoader(mask_path, Raw())
```

`load_pair` scales the target with the strategy and the input with the range the target produced.
A prediction that is uniformly too bright is clipped against the reference, not normalized away.
`ImageLoader.intensity_range` is that range; `ImageLoader.raw_range` the image's own extremes.

PNG masks store 0/255 — load them with `MinMax()`, which maps them to exactly `{0.0, 1.0}`.
A constant image keeps its (clipped) value and prints a warning instead of turning into zeros.
````

- [ ] **Step 3: `Results-and-Reports.md`**

Add after the column list:

```markdown
### Scale columns

| Column | Meaning |
|---|---|
| `normalization` | strategy name: `minmax`, `percentile_0.5_99.5`, `raw` |
| `scale_lo`, `scale_hi` | the raw range mapped to `[0, 1]`; the **target's** in a full-reference run; empty under `raw` |
| `input_min`, `input_max` | the input's raw extremes |

`input_max > scale_hi` means the prediction overshot the reference and was clipped.
`psnr` is computed with `data_range = 1.0` on this scale, so it is comparable across images
only when their `scale_lo/hi` are.

`aggregate_volumes()`: a slice is `is_empty` only when input **and** target are blank there,
so its voxel counts are real zeros and are filled in. Volumes come out NaN only when a
metric failed on one of their slices.
```

- [ ] **Step 4: `Interpreting-Scores.md`**

Add a section:

```markdown
## Intensity scale

- **Full-reference runs share the target's range** (fastMRI convention). Under independent
  min-max scaling, `norm(x) == norm(2x + 500)` and no full-reference metric could see a
  brightness or contrast bias. Now the bias shows as clipping and lower scores, and the
  report's `input_min/max` against `scale_lo/hi` say how large it was.
- **Percentile scaling changes every score**, including `psnr` and `ssim`. Compare only
  against runs with the same `normalization` value.
- **Scaling is per volume, never per slice.** A slice's tensor depends on the rest of its
  stack; edge slices with little anatomy occupy a narrow part of `[0, 1]` and read as poor
  to no-reference metrics. Slice rows of one volume are not independent samples — treat
  `describe()` and box plots over them as descriptive, not inferential.
```

- [ ] **Step 5: `Segmentation-Metrics.md`**

Add near the top, after the sentence on what the metrics take as input:

```markdown
Load mask files with `Raw()` so they arrive unscaled in their stored dtype:

```python
inp, tgt = load_pair(pred_mask, gt_mask, Raw())
```

Binary 0/1 masks need no `threshold`. For a multi-label map, the MONAI metrics take
`threshold=0.0` (every non-zero label is foreground); the voxel-count metrics
(`vs`, `v_pred`, `v_gt`, `tp`) select one label via `label=` in `segmentation_metrics.volume`.
PNG masks store 0/255 and should be loaded with `MinMax()` instead.
```

- [ ] **Step 6: Commit locally, do not push**

```bash
cd "$SCRATCHPAD/wiki" && git add -A && git commit -m "Document the normalization strategy, load_pair and the scale columns"
```

Report the clone path and that the push is pending the user's decision.

---

## Self-Review

**Spec coverage.** §1 raw `LoadedImage` → Task 2. §2 strategies and `scale()` (no epsilon, constant image, `FixedRange(None)`) → Task 1. §3 `ImageLoader` properties, `load_pair`, `evaluate(normalization=)`, CLI → Tasks 2, 3, 8. §4 five report columns → Task 5. §5 raw empty mask with percentile span, FR emptiness, `aggregate_volumes` → Tasks 4, 6. §6 segmentation path, float cast, TODO replacement → Task 7. §7 every TODO resolved → Tasks 2, 4, 7, 8, 9 (sweep in 9). Error handling: `Percentile`/`FixedRange` validation → Task 1; constant warning → Task 1; `Raw` pair → Task 3. Testing list: every bullet has a test in Tasks 1–8 (`MinMax` bit-for-bit equality is covered by the unchanged existing loader tests plus `test_raw_range_and_intensity_range_under_minmax`). Documentation → Tasks 9, 10.

**Placeholder scan.** No TBD/TODO steps; every code step shows the code. Task 2 step 1 items 2 and Task 9 step 1 name greps to locate lines because the exact numbers shift — the replacement itself is stated.

**Type consistency.** `IntensityRange(lo, hi)`, `Normalizer.name`/`range_of`, `scale(raw, rng, *, label)`, `FixedRange(range, name)`, `load_pair(input_path, target_path, normalizer) -> (input, target)`, `ImageLoader.raw/raw_range/intensity_range/normalizer`, `_scale_fields()`, record fields `normalization/scale_lo/scale_hi/input_min/input_max`, `normalizer_from_name`, `NORMALIZER_NAMES` are spelled identically in every task.

One deliberate deviation from the spec's §5 formula: when the 0.5–99.5 percentile span is zero (foreground rarer than 0.5 %, e.g. a small lesion mask) the mask falls back to the raw extremes instead of declaring every slice empty. Without it, `Raw` mask volumes would be skipped wholesale. The spec's intent (spike-proof, otherwise today's semantics) is preserved.
