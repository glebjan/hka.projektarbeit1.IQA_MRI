# Scoring Mode: 2D Slices and 3D Volumes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every metric be scored either per 2D slice (as today) or once over a whole 3D volume, chosen per evaluation run.

**Architecture:** A volume score is the *same* metric called once with a `(1, C, D, H, W)` sample instead of `D` separate `(1, C, H, W)` samples. Each `MetricSpec` declares, per mode, either a factory (it can do this) or a plain-language reason (it cannot). Image decoders start returning voxel spacing and whether the depth axis is spatial at all. `IQAEvaluator` is frozen and gets a `VolumeEvaluator` subclass, hidden behind a `build_evaluator()` factory so callers keep selecting behaviour by mode.

**Tech Stack:** Python 3.14, PyTorch, MONAI 1.6.0, pyiqa, nibabel, pydicom, SimpleITK, SciPy, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-scoring-mode-2d-3d-design.md`

## Global Constraints

- Run everything from the repo root with the venv active: `source .venv/bin/activate`. Tests: `pytest tests/ -q`. The `tests/conftest.py` puts `src/` on `sys.path`, so test files import bare module names (`from metrics import ...`).
- **`src/iqa_evaluator.py` must not be modified. Not one line.** It may be subclassed. Every other file is open. Any change that would force an edit there is wrong — find another way.
- The per-slice code path must stay behaviourally identical. The 274 existing tests are the evidence; they must all still pass at the end of every task.
- Modes are exactly two: `"slice"` and `"volume"`. There is no `"both"`.
- Metrics that cannot serve the chosen mode are **skipped**, never fatal — except when *no* metric can serve it, which is a hard error.
- Error and skip messages follow: what happened → why it is a property of the thing, not the user's mistake → what to type next. No shape tuples, no class names, no the word "unsupported". Messages are English, matching the codebase.
- Spacing is always ordered to match the tensor's axes: `(dz, dy, dx)` in millimetres.
- Code, comments, docstrings and commit messages in English. The existing files mix a few German inline comments; do not add more.
- `src/data.py` needs **no** change (it uses `ImageLoader`, not the decoder functions). The spec's module table lists it in error.

---

### Task 1: Image geometry — `LoadedImage`, spacing, `is_volumetric`

Decoders currently return a bare tensor and throw the header away. They must return spacing and whether the depth axis is spatial, because everything downstream depends on it.

**Files:**
- Modify: `src/image_loader.py` (all four decoders, `_LOADERS`, `ImageLoader`)
- Modify: `tests/conftest.py` (add volumetric fixtures)
- Modify: `tests/test_image_loader.py` (decoder call sites now return `LoadedImage`)
- Modify: `tests/test_iqa_evaluator.py:233` (cache attribute renamed)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LoadedImage(tensor: torch.Tensor, spacing: Optional[tuple[float, float, float]], is_volumetric: bool)` — frozen dataclass, exported from `image_loader`.
  - `Spacing = tuple[float, float, float]` type alias, exported from `image_loader`.
  - `ImageLoader.spacing -> Optional[Spacing]`, `ImageLoader.is_volumetric -> bool`.
  - `ImageLoader` caches on `self._loaded: Optional[LoadedImage]` (was `self._tensor`).
  - conftest fixtures `nifti_volume`, `nifti_4d`, `sitk_volume`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/conftest.py`:

```python
@pytest.fixture()
def nifti_volume(tmp_path: Path) -> Path:
    """A 3D NIfTI with anisotropic voxels: 1.0 x 1.0 x 1.2 mm, shape (X=8, Y=10, Z=6)."""
    import nibabel as nib

    data = np.random.default_rng(3).random((8, 10, 6)).astype("float32")
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    p = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), p)
    return p


@pytest.fixture()
def nifti_4d(tmp_path: Path) -> Path:
    """A 4D NIfTI (X=8, Y=10, Z=6, T=3) — depth and time get flattened on load."""
    import nibabel as nib

    data = np.random.default_rng(4).random((8, 10, 6, 3)).astype("float32")
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    p = tmp_path / "vol4d.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), p)
    return p


@pytest.fixture()
def sitk_volume(tmp_path: Path) -> Path:
    """A 3D volume written with SimpleITK, spacing (x=0.5, y=0.5, z=2.0) mm."""
    import SimpleITK as sitk

    arr = np.random.default_rng(5).random((6, 10, 8)).astype("float32")  # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 2.0))
    p = tmp_path / "vol.nrrd"
    sitk.WriteImage(img, str(p))
    return p
```

Create `tests/test_image_geometry.py`:

```python
"""Tests for voxel geometry: LoadedImage, spacing, is_volumetric."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from image_loader import (
    ImageLoader,
    LoadedImage,
    _load_nifti,
    _load_pil,
    _load_sitk,
)


class TestLoadedImage:
    def test_is_a_frozen_dataclass_with_three_fields(self):
        li = LoadedImage(tensor=torch.zeros(1, 1, 2, 2), spacing=(1.0, 2.0, 3.0), is_volumetric=True)
        assert li.tensor.shape == (1, 1, 2, 2)
        assert li.spacing == (1.0, 2.0, 3.0)
        assert li.is_volumetric is True
        with pytest.raises(Exception):
            li.spacing = (1.0, 1.0, 1.0)


class TestPilGeometry:
    def test_png_has_no_spacing_and_is_not_volumetric(self, synthetic_png: Path):
        loaded = _load_pil(synthetic_png)
        assert isinstance(loaded, LoadedImage)
        assert loaded.tensor.shape[0] == 1
        assert loaded.spacing is None
        assert loaded.is_volumetric is False


class TestNiftiGeometry:
    def test_3d_spacing_is_reordered_to_dz_dy_dx(self, nifti_volume: Path):
        loaded = _load_nifti(nifti_volume)
        # header zooms are (dx, dy, dz) = (1.0, 1.0, 1.2); tensor axes are (Z, X, Y)
        assert loaded.tensor.shape == (6, 1, 8, 10)
        assert loaded.spacing == pytest.approx((1.2, 1.0, 1.0))

    def test_3d_is_volumetric(self, nifti_volume: Path):
        assert _load_nifti(nifti_volume).is_volumetric is True

    def test_4d_is_not_volumetric_and_has_no_spacing(self, nifti_4d: Path):
        loaded = _load_nifti(nifti_4d)
        assert loaded.tensor.shape[0] == 6 * 3  # depth and time flattened together
        assert loaded.is_volumetric is False
        assert loaded.spacing is None


class TestSitkGeometry:
    def test_spacing_axis_order_is_reversed(self, sitk_volume: Path):
        loaded = _load_sitk(sitk_volume)
        # GetSpacing() is (x, y, z) = (0.5, 0.5, 2.0); the array is (z, y, x)
        assert loaded.tensor.shape == (6, 1, 10, 8)
        assert loaded.spacing == pytest.approx((2.0, 0.5, 0.5))

    def test_is_volumetric(self, sitk_volume: Path):
        assert _load_sitk(sitk_volume).is_volumetric is True


class TestSingleSliceIsNotVolumetric:
    def test_depth_one_nifti(self, tmp_path: Path):
        import nibabel as nib

        data = np.random.default_rng(6).random((8, 10, 1)).astype("float32")
        p = tmp_path / "flat.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), p)
        assert _load_nifti(p).is_volumetric is False


class TestImageLoaderProperties:
    def test_exposes_spacing_and_is_volumetric(self, nifti_volume: Path):
        loader = ImageLoader(nifti_volume)
        assert loader.spacing == pytest.approx((1.2, 1.0, 1.0))
        assert loader.is_volumetric is True
        assert loader.tensor.shape == (6, 1, 8, 10)

    def test_png_through_loader(self, synthetic_png: Path):
        loader = ImageLoader(synthetic_png)
        assert loader.spacing is None
        assert loader.is_volumetric is False

    def test_loads_only_once(self, nifti_volume: Path):
        loader = ImageLoader(nifti_volume)
        first = loader.tensor
        assert loader.spacing is not None
        assert loader.tensor is first
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_geometry.py -q`
Expected: FAIL with `ImportError: cannot import name 'LoadedImage' from 'image_loader'`.

- [ ] **Step 3: Implement `LoadedImage` and the decoders**

In `src/image_loader.py`, add after the imports:

```python
from dataclasses import dataclass

Spacing = tuple[float, float, float]


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image plus the geometry the decoder knew about.

    Attributes:
        tensor:        (D, 1, H, W) float32 in [0, 1].
        spacing:       physical voxel size as (dz, dy, dx) in millimetres,
                       ordered to match the tensor's axes. None when the format
                       carries no geometry (PNG/JPEG) or when the depth axis is
                       not spatial.
        is_volumetric: True only when the depth axis is a real spatial axis with
                       more than one slice. False for 2D formats, for a single
                       slice, and for 4D NIfTI (whose depth axis mixes time and
                       space, see `_load_nifti`).
    """
    tensor:        torch.Tensor
    spacing:       Optional[Spacing] = None
    is_volumetric: bool = False
```

Rewrite the four decoders (only the return statements and geometry extraction change):

```python
def _load_pil(path: Path) -> LoadedImage:
    grayscale = np.asarray(Image.open(path).convert("L"))
    return LoadedImage(_to_normalized_channel_tensor(grayscale[np.newaxis]))


def _load_dicom(path: Path) -> LoadedImage:
    dicom_dataset = pydicom.dcmread(str(path))
    photometric = str(getattr(dicom_dataset, "PhotometricInterpretation", "MONOCHROME2"))
    pixel_array = _dicom_array_to_depth_first(
        dicom_dataset.pixel_array, photometric
    ).astype(np.float32)
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
        _to_normalized_channel_tensor(pixel_array),
        spacing,
        is_volumetric=(depth > 1 and spacing is not None and not is_cine),
    )


def _load_nifti(path: Path) -> LoadedImage:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = image.get_fdata()
    zooms = image.header.get_zooms()
    if data.ndim == 3:
        depth_first = np.transpose(data, (2, 0, 1))
        # zooms are (dx, dy, dz) for array axes (X, Y, Z); after the transpose
        # the tensor axes are (Z, X, Y), so the spacing follows as (dz, dx, dy).
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
    return LoadedImage(_to_normalized_channel_tensor(depth_first), spacing, volumetric)


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
        _to_normalized_channel_tensor(volume),
        spacing,
        is_volumetric=(spacing is not None and volume.shape[0] > 1),
    )
```

Update the registry type and `ImageLoader`:

```python
_LOADERS: dict[str, Callable[[Path], LoadedImage]] = {
    ...unchanged entries...
}
```

```python
class ImageLoader:
    def __init__(self, path: Path):
        self.path = path
        self.suffix = canonical_suffix(path)
        if self.suffix not in _LOADERS:
            raise ValueError(f"Unsupported format: {path}")
        self._loaded: Optional[LoadedImage] = None

    @property
    def _image(self) -> LoadedImage:
        if self._loaded is None:
            self._loaded = _LOADERS[self.suffix](self.path)
        return self._loaded

    @property
    def tensor(self) -> torch.Tensor:
        return self._image.tensor

    @property
    def spacing(self) -> Optional[Spacing]:
        """Physical voxel size (dz, dy, dx) in mm, or None if the format has none."""
        return self._image.spacing

    @property
    def is_volumetric(self) -> bool:
        """True when the depth axis is a real spatial axis with more than one slice."""
        return self._image.is_volumetric
```

Leave `rgb_tensor`, `empty_slice_mask` and `log_tensor_shape` exactly as they are — they go through `self.tensor`.

- [ ] **Step 4: Fix the existing call sites**

In `tests/test_image_loader.py`, every direct decoder call now yields a `LoadedImage`. Append `.tensor` at each of these lines: 65, 74, 157, 167, 185, 195, 218, 226. Example:

```python
        t = _load_pil(p).tensor
```

Line 205 asserts that a bad NIfTI raises; leave it unchanged.

In `tests/test_iqa_evaluator.py:233`, the cache attribute changed:

```python
        loader._loaded = _load_nifti(p)
```

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS — the new geometry tests plus all 274 existing ones.

- [ ] **Step 6: Commit**

```bash
git add src/image_loader.py tests/conftest.py tests/test_image_geometry.py tests/test_image_loader.py tests/test_iqa_evaluator.py
git commit -m "feat(loader): decoders report voxel spacing and whether depth is spatial"
```

---

### Task 2: Mode capability model on `MetricSpec` and `MetricRegistry`

Each spec declares, per mode, either a factory or a reason it cannot serve that mode. The registry answers which metrics apply and builds them for the right mode.

**Files:**
- Modify: `src/metrics.py`
- Modify: `src/segmentation_metrics/monai_metrics.py` (five `MetricSpec(...)` constructions)
- Modify: `src/segmentation_metrics/boundary_iou.py` (one `MetricSpec(...)` construction)
- Modify: `tests/test_metrics.py`
- Test: `tests/test_metrics.py` (new classes appended)

**Interfaces:**
- Consumes: `Spacing` from `image_loader` (Task 1).
- Produces:
  - `ScoringMode = Literal["slice", "volume"]`
  - `MetricDirection = Literal["higher_is_better", "lower_is_better", "not_ranked"]`
  - `ModeSupport(factory: Callable[..., Metric])`, `ModeUnsupported(reason: str)`
  - `MetricSpec.slice_mode`, `MetricSpec.volume_mode` — the `factory` field is **gone**
  - `SkippedMetric(name: str, reason: str)`
  - `MetricRegistry.get_metric(name, mode="slice", spacing=None) -> Metric`
  - `MetricRegistry.select(mode) -> tuple[list[MetricSpec], list[SkippedMetric]]`
  - `MetricRegistry.register_metric(..., volume_factory=None, volume_reason=...)`
  - `REASON_DEEP_2D` — the shared skip reason for the eight CNN/CLIP metrics

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
# ---------------------------------------------------------------------------
# Mode capability model
# ---------------------------------------------------------------------------

from metrics import (
    ModeSupport,
    ModeUnsupported,
    SkippedMetric,
    REASON_DEEP_2D,
)


def _spec(name, *, volume=False, reason="no volumetric implementation"):
    """Build a throwaway spec whose metric returns one score per sample."""
    def make(*_args):
        def metric(inp, tgt=None):
            return [float(inp[i].mean()) for i in range(inp.shape[0])]
        return metric

    return MetricSpec(
        name=name,
        direction="higher_is_better",
        reference=False,
        channels="gray",
        slice_mode=ModeSupport(make),
        volume_mode=ModeSupport(make) if volume else ModeUnsupported(reason),
        builtin=False,
    )


class TestModeCapability:
    def test_supported_mode_carries_a_factory(self):
        spec = _spec("m", volume=True)
        assert isinstance(spec.volume_mode, ModeSupport)

    def test_unsupported_mode_carries_a_reason(self):
        spec = _spec("m", reason="only reads flat pictures")
        assert isinstance(spec.volume_mode, ModeUnsupported)
        assert spec.volume_mode.reason == "only reads flat pictures"


class TestRegistrySelect:
    def test_partitions_specs_by_mode(self):
        registry = MetricRegistry(_spec("can", volume=True), _spec("cannot"))
        applicable, skipped = registry.select("volume")
        assert [s.name for s in applicable] == ["can"]
        assert skipped == [SkippedMetric("cannot", "no volumetric implementation")]

    def test_slice_mode_keeps_everything(self):
        registry = MetricRegistry(_spec("can", volume=True), _spec("cannot"))
        applicable, skipped = registry.select("slice")
        assert [s.name for s in applicable] == ["can", "cannot"]
        assert skipped == []

    def test_select_prints_nothing(self, capsys):
        MetricRegistry(_spec("cannot")).select("volume")
        assert capsys.readouterr().out == ""


class TestGetMetricByMode:
    def test_default_mode_is_slice(self):
        registry = MetricRegistry(_spec("m", volume=True))
        assert registry.get_metric("m") is registry.get_metric("m", "slice")

    def test_volume_and_slice_instances_are_cached_separately(self):
        registry = MetricRegistry(_spec("m", volume=True))
        assert registry.get_metric("m", "volume") is not registry.get_metric("m", "slice")
        assert registry.get_metric("m", "volume") is registry.get_metric("m", "volume")

    def test_spacing_is_part_of_the_cache_key(self):
        registry = MetricRegistry(_spec("m", volume=True))
        a = registry.get_metric("m", "volume", (1.0, 1.0, 1.0))
        b = registry.get_metric("m", "volume", (2.0, 1.0, 1.0))
        assert a is not b
        assert registry.get_metric("m", "volume", (1.0, 1.0, 1.0)) is a

    def test_unsupported_mode_raises(self):
        registry = MetricRegistry(_spec("m"))
        with pytest.raises(ValueError, match="volume"):
            registry.get_metric("m", "volume")


class TestRegisterMetricVolume:
    def test_defaults_to_slice_only(self, fake_metric):
        registry = MetricRegistry()
        registry.register_metric("f", fake_metric, direction="higher_is_better",
                                 reference=False, channels="gray")
        applicable, skipped = registry.select("volume")
        assert applicable == []
        assert [s.name for s in skipped] == ["f"]

    def test_accepts_a_volume_factory(self, fake_metric):
        registry = MetricRegistry()
        registry.register_metric("f", fake_metric, direction="higher_is_better",
                                 reference=False, channels="gray",
                                 volume_factory=lambda spacing: fake_metric)
        applicable, _ = registry.select("volume")
        assert [s.name for s in applicable] == ["f"]


class TestBuiltinCapabilities:
    def test_eight_builtins_cannot_do_volume(self):
        registry = MetricRegistry(*BUILTIN_METRICS)
        _, skipped = registry.select("volume")
        assert sorted(s.name for s in skipped) == sorted([
            "lpips", "dists", "radimagenet_lpips", "clipiqa",
            "clip_iqa_lung", "clip_iqa_brain", "brisque", "niqe",
        ])

    def test_all_share_the_same_reason(self):
        registry = MetricRegistry(*BUILTIN_METRICS)
        _, skipped = registry.select("volume")
        assert {s.reason for s in skipped} == {REASON_DEEP_2D}

    def test_no_builtin_is_skipped_in_slice_mode(self):
        registry = MetricRegistry(*BUILTIN_METRICS)
        _, skipped = registry.select("slice")
        assert skipped == []


class TestNotRankedDirection:
    def test_direction_accepts_not_ranked(self):
        spec = MetricSpec(
            name="count", direction="not_ranked", reference=True, channels="gray",
            slice_mode=ModeSupport(lambda *_: (lambda inp, tgt=None: [0.0])),
            builtin=False,
        )
        assert MetricRegistry(spec).direction == {"count": "not_ranked"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -q`
Expected: FAIL with `ImportError: cannot import name 'ModeSupport' from 'metrics'`.

- [ ] **Step 3: Implement the capability model in `src/metrics.py`**

Replace the `MetricDirection` alias and add the mode types near the top:

```python
MetricDirection = Literal["higher_is_better", "lower_is_better", "not_ranked"]
MetricChannels  = Literal["gray", "rgb"]
ScoringMode     = Literal["slice", "volume"]

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
```

Change `MetricSpec`: drop `factory`, add the two mode fields (update the docstring's `factory:` line to describe them):

```python
    name:      str
    direction: MetricDirection
    reference: bool
    channels:  MetricChannels
    slice_mode:  ModeCapability
    volume_mode: ModeCapability = ModeUnsupported(REASON_NO_VOLUME_IMPL)
    builtin:      bool = True
    description:  str  = ""
    domain:       str  = ""
```

Add to `MetricRegistry` (replacing `get_metric`):

```python
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
```

The cache is now keyed by tuple, so `register()` must clear every entry for a name:

```python
    def register(self, *specs: MetricSpec) -> None:
        for spec in specs:
            self._specs[spec.name] = spec
            for key in [k for k in self._cache if k[0] == spec.name]:
                del self._cache[key]
```

Update the cache annotation in `__init__`:

```python
        self._cache: dict[tuple[str, str, Optional[Spacing]], Metric] = {}
```

Extend `register_metric`:

```python
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
```

- [ ] **Step 4: Migrate the sixteen spec constructions**

In `src/metrics.py`, wrap every built-in factory and give the eight deep-2D metrics their shared reason. `PSNR` and `SSIM` get their volume factories in Task 5; for now they carry a placeholder-free reason that Task 5 replaces:

```python
# Full-reference metrics (need a target image)
PSNR              = MetricSpec("psnr",              "higher_is_better", True,  "gray", ModeSupport(_pyiqa_factory("psnr")))
SSIM              = MetricSpec("ssim",              "higher_is_better", True,  "gray", ModeSupport(_pyiqa_factory("ssim")))
LPIPS             = MetricSpec("lpips",             "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("lpips")),  ModeUnsupported(REASON_DEEP_2D))
DISTS             = MetricSpec("dists",             "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("dists")),  ModeUnsupported(REASON_DEEP_2D))
RADIMAGENET_LPIPS = MetricSpec("radimagenet_lpips", "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("radimagenet_lpips", backbone_path=str(RESNET50))), ModeUnsupported(REASON_DEEP_2D))
# No-reference metrics
CLIPIQA           = MetricSpec("clipiqa",           "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clipiqa")),        ModeUnsupported(REASON_DEEP_2D))
CLIP_IQA_LUNG     = MetricSpec("clip_iqa_lung",     "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clip_iqa_lung")),  ModeUnsupported(REASON_DEEP_2D))
CLIP_IQA_BRAIN    = MetricSpec("clip_iqa_brain",    "higher_is_better", False, "rgb",  ModeSupport(_pyiqa_factory("clip_iqa_brain")), ModeUnsupported(REASON_DEEP_2D))
BRISQUE           = MetricSpec("brisque",           "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("brisque")),        ModeUnsupported(REASON_DEEP_2D))
NIQE              = MetricSpec("niqe",              "lower_is_better",  False, "rgb",  ModeSupport(_pyiqa_factory("niqe")),           ModeUnsupported(REASON_DEEP_2D))
```

The `TestBuiltinCapabilities` tests above expect exactly these eight names with `REASON_DEEP_2D`, and `psnr`/`ssim` to be absent from the skip list — so `PSNR`/`SSIM` must **not** be given a reason here. Give them a temporary volume factory that reuses the pyiqa one; Task 5 swaps it for MONAI:

```python
PSNR = MetricSpec("psnr", "higher_is_better", True, "gray",
                  ModeSupport(_pyiqa_factory("psnr")),
                  ModeSupport(lambda spacing: PyIQAMetric("psnr")))
SSIM = MetricSpec("ssim", "higher_is_better", True, "gray",
                  ModeSupport(_pyiqa_factory("ssim")),
                  ModeSupport(lambda spacing: PyIQAMetric("ssim")))
```

In `src/segmentation_metrics/monai_metrics.py`, each of the five specs currently has `factory=lambda: metric`. Replace with `slice_mode=ModeSupport(lambda: metric)` and import `ModeSupport` from `metrics`. Volume factories come in Task 4. Same edit in `src/segmentation_metrics/boundary_iou.py` for the one spec there; Task 6 adds its volume factory.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS. If `tests/test_segmentation_metrics.py` or `tests/test_boundary_iou.py` assert on `spec.factory`, change those assertions to `spec.slice_mode.factory`.

- [ ] **Step 6: Commit**

```bash
git add src/metrics.py src/segmentation_metrics/monai_metrics.py src/segmentation_metrics/boundary_iou.py tests/test_metrics.py tests/test_segmentation_metrics.py tests/test_boundary_iou.py
git commit -m "feat(metrics): declare per-mode capability on MetricSpec and select in the registry"
```

---

### Task 3: `VolumeEvaluator` and `build_evaluator()`

**Files:**
- Create: `src/volume_evaluator.py`
- Create: `src/evaluator_factory.py`
- Modify: `src/records.py`
- Create: `tests/test_volume_evaluator.py`
- Modify: `tests/test_records.py`, `tests/test_evaluation_result.py` (column set gains `scoring`)

**Interfaces:**
- Consumes: `ImageLoader.is_volumetric` / `.spacing` (Task 1); `MetricRegistry.get_metric(name, mode, spacing)`, `ModeUnsupported` (Task 2).
- Produces:
  - `ImageEvaluatorRecord.scoring: str = "slice"`, `ImageEvaluatorRecord.slice_index: Optional[int] = 0`
  - `VolumeEvaluator(input_image, target_image, registry, source_model=None)`
  - `build_evaluator(input_image, target_image, registry, mode="slice", source_model=None) -> IQAEvaluator`

**Note for the implementer:** `IQAEvaluator._compute_batch()` calls `self.registry.get_metric(spec.name)`, which resolves to slice mode. Because `IQAEvaluator` is frozen, `VolumeEvaluator` overrides that method rather than inheriting it. What the subclass does inherit is `__init__`'s shape check, `_format_slice_id()`, and being an `IQAEvaluator` for callers.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_volume_evaluator.py`:

```python
"""Tests for VolumeEvaluator and build_evaluator."""
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluator_factory import build_evaluator
from image_loader import ImageLoader
from iqa_evaluator import IQAEvaluator
from metrics import MetricRegistry, MetricSpec, ModeSupport, ModeUnsupported
from volume_evaluator import VolumeEvaluator


class _ShapeSpy:
    """A Metric that records the shape it was handed and returns one score per sample."""

    def __init__(self, spacing=None):
        self.seen_shapes = []
        self.spacing = spacing

    def __call__(self, input, target=None):
        self.seen_shapes.append(tuple(input.shape))
        return [float(input[i].mean()) for i in range(input.shape[0])]


@pytest.fixture()
def spy_registry():
    """A registry with one metric that works in both modes, plus the spy instances."""
    slice_spy = _ShapeSpy()
    volume_spies = {}

    def volume_factory(spacing):
        volume_spies[spacing] = _ShapeSpy(spacing)
        return volume_spies[spacing]

    spec = MetricSpec(
        name="spy", direction="higher_is_better", reference=False, channels="gray",
        slice_mode=ModeSupport(lambda: slice_spy),
        volume_mode=ModeSupport(volume_factory),
        builtin=False,
    )
    return MetricRegistry(spec), slice_spy, volume_spies


class TestVolumeSample:
    def test_metric_receives_one_five_dimensional_sample(self, nifti_volume: Path, spy_registry):
        registry, _, volume_spies = spy_registry
        loader = ImageLoader(nifti_volume)          # (6, 1, 8, 10)
        VolumeEvaluator(loader, None, registry).run_evaluation()
        spy = next(iter(volume_spies.values()))
        assert spy.seen_shapes == [(1, 1, 6, 8, 10)]

    def test_spacing_reaches_the_volume_factory(self, nifti_volume: Path, spy_registry):
        registry, _, volume_spies = spy_registry
        VolumeEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()
        assert list(volume_spies) == [(1.2, 1.0, 1.0)]


class TestVolumeRecords:
    def test_one_record_per_volume(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        records = VolumeEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()
        assert len(records) == 1

    def test_record_fields(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        record = VolumeEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()[0]
        assert record.scoring == "volume"
        assert record.slice_index is None
        assert record.image_id == "vol"          # no _sNNN suffix
        assert record.mode == "no_reference"
        assert record.extra["spy"] is not None

    def test_source_model_prefixes_the_id(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        record = VolumeEvaluator(
            ImageLoader(nifti_volume), None, registry, source_model="smore"
        ).run_evaluation()[0]
        assert record.image_id == "smore/vol"


class TestVolumeGuards:
    def test_rejects_a_non_volumetric_image(self, synthetic_png: Path, spy_registry):
        registry, _, _ = spy_registry
        with pytest.raises(ValueError, match="3D"):
            VolumeEvaluator(ImageLoader(synthetic_png), None, registry)

    def test_rejects_a_spacing_mismatch(self, nifti_volume: Path, tmp_path: Path, spy_registry):
        import nibabel as nib

        registry, _, _ = spy_registry
        data = np.asarray(nib.load(str(nifti_volume)).get_fdata())
        other = tmp_path / "other.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), other)
        with pytest.raises(ValueError, match="voxel"):
            VolumeEvaluator(ImageLoader(nifti_volume), ImageLoader(other), registry)

    def test_skips_metrics_that_cannot_do_volume(self, nifti_volume: Path):
        spec = MetricSpec(
            name="flat", direction="higher_is_better", reference=False, channels="gray",
            slice_mode=ModeSupport(lambda: (lambda inp, tgt=None: [0.0])),
            volume_mode=ModeUnsupported("only reads flat pictures"),
            builtin=False,
        )
        record = VolumeEvaluator(ImageLoader(nifti_volume), None, MetricRegistry(spec)).run_evaluation()[0]
        assert "flat" not in record.extra

    def test_empty_slices_are_not_dropped(self, tmp_path: Path, spy_registry):
        import nibabel as nib

        registry, _, volume_spies = spy_registry
        data = np.random.default_rng(7).random((8, 10, 6)).astype("float32")
        data[:, :, :2] = 0.0                     # two blank slices
        p = tmp_path / "gappy.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.2, 1.0])), p)
        VolumeEvaluator(ImageLoader(p), None, registry).run_evaluation()
        assert next(iter(volume_spies.values())).seen_shapes == [(1, 1, 6, 8, 10)]


class TestBuildEvaluator:
    def test_slice_mode_returns_the_frozen_evaluator(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        ev = build_evaluator(ImageLoader(nifti_volume), None, registry, mode="slice")
        assert type(ev) is IQAEvaluator

    def test_volume_mode_returns_the_subclass(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        ev = build_evaluator(ImageLoader(nifti_volume), None, registry, mode="volume")
        assert isinstance(ev, VolumeEvaluator)

    def test_default_is_slice(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        assert type(build_evaluator(ImageLoader(nifti_volume), None, registry)) is IQAEvaluator

    def test_rejects_an_unknown_mode(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        with pytest.raises(ValueError, match="slice"):
            build_evaluator(ImageLoader(nifti_volume), None, registry, mode="3d")


class TestSliceRecordsUnchanged:
    def test_slice_records_say_slice(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        records = IQAEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()
        assert len(records) == 6
        assert all(r.scoring == "slice" for r in records)
        assert records[0].slice_index == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_volume_evaluator.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'volume_evaluator'`.

- [ ] **Step 3: Add the record fields**

In `src/records.py`, inside `ImageEvaluatorRecord`, change `slice_index` and insert `scoring` right after `mode`:

```python
    mode:                str             = "no_reference"
    scoring:             str             = "slice"
    slice_index:         Optional[int]   = 0
```

Extend the class docstring:

```python
"""ImageEvaluatorRecord — one row of results.

`scoring` says what the row covers: "slice" (one 2D slice, `slice_index` set)
or "volume" (one whole 3D volume, `slice_index` None). A run has exactly one
scoring mode, so a report holds exactly one kind of row.
"""
```

- [ ] **Step 4: Implement `VolumeEvaluator`**

Create `src/volume_evaluator.py`:

```python
"""VolumeEvaluator — scores a whole 3D volume as a single sample.

Extends IQAEvaluator without modifying it. Where IQAEvaluator hands each metric
a `(D, C, H, W)` batch of slices, this hands it one `(1, C, D, H, W)` sample, so
metrics that understand 3D geometry (surface distance, panoptic quality,
boundary bands) measure across slices instead of within them.

Two deliberate differences from the slice path:

- Empty slices are NOT filtered. Removing slices from a volume is not filtering,
  it is a change of geometry: the body gains holes and surface distances start
  measuring edges that do not exist.
- Metrics are built per image geometry, because HD95, ASSD, NSD and Boundary IoU
  need the voxel spacing to return physical distances.

`_compute_batch` is overridden rather than inherited: the inherited version calls
`registry.get_metric(name)`, which resolves to slice mode, and IQAEvaluator is
frozen by requirement.
"""

from typing import Optional

import torch

from image_loader import ImageLoader, strip_all_extensions
from iqa_evaluator import IQAEvaluator
from metrics import DEVICE, MetricChannels, MetricRegistry, MetricSpec, ModeUnsupported
from records import ImageEvaluatorRecord


class VolumeEvaluator(IQAEvaluator):
    """Computes one registry's metrics over one input/target volume pair."""

    def __init__(
        self,
        input_image:  ImageLoader,
        target_image: Optional[ImageLoader],
        registry:     MetricRegistry,
        source_model: Optional[str] = None,
    ):
        super().__init__(input_image, target_image, registry, source_model)

        if not input_image.is_volumetric:
            raise ValueError(
                f"'{input_image.path.name}' is not a 3D volume, so it cannot be "
                "scored in volume mode. Its slices are not stacked along a "
                "spatial axis — this is the case for PNG and JPEG images, for a "
                "single slice, and for 4D scans whose frames are time steps. "
                "Score it with mode='slice' instead."
            )
        if target_image is not None and target_image.spacing != input_image.spacing:
            raise ValueError(
                f"'{input_image.path.name}' and '{target_image.path.name}' are "
                f"recorded at different voxel sizes ({input_image.spacing} vs "
                f"{target_image.spacing}, in millimetres as depth/height/width). "
                "Comparing them directly would measure that size difference "
                "rather than image quality. Resample one onto the other's grid "
                "first — and note that which one you resample changes the scores."
            )
        self.spacing = input_image.spacing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_volume(self, img: ImageLoader, channels: MetricChannels) -> torch.Tensor:
        """(D, C, H, W) -> (1, C, D, H, W): the whole volume as a single sample."""
        base = img.tensor if channels == "gray" else img.rgb_tensor
        return base.permute(1, 0, 2, 3).unsqueeze(0)

    def _compute_volume(self, spec: MetricSpec) -> Optional[float]:
        metric = self.registry.get_metric(spec.name, "volume", self.spacing)
        inp = self._pick_volume(self.input, spec.channels).to(DEVICE)
        try:
            if spec.reference:
                ref = self._pick_volume(self.target, spec.channels).to(DEVICE)
                scores = list(metric(inp, ref))
            else:
                scores = list(metric(inp))
        except Exception as exc:
            print(f"[{self.input.path}] metric '{spec.name}' failed: {exc}")
            return None
        return scores[0] if scores else None

    def _format_volume_id(self) -> str:
        base = strip_all_extensions(self.input.path)
        return f"{self.source_model}/{base}" if self.source_model else base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation(self) -> list[ImageEvaluatorRecord]:
        """Evaluate every applicable metric once over the whole volume."""
        has_target = self.target is not None
        record = ImageEvaluatorRecord(
            image_id=self._format_volume_id(),
            source_model=self.source_model,
            mode="full_reference" if has_target else "no_reference",
            scoring="volume",
            slice_index=None,
            is_empty=bool(self.input.empty_slice_mask.all().item()),
        )

        for spec in self.registry.specs:
            if spec.reference and not has_target:
                continue
            if isinstance(spec.volume_mode, ModeUnsupported):
                continue
            value = self._compute_volume(spec)
            if spec.builtin:
                setattr(record, spec.name, value)
            else:
                record.extra[spec.name] = value

        return [record]
```

- [ ] **Step 5: Implement the factory**

Create `src/evaluator_factory.py`:

```python
"""build_evaluator — pick the evaluator for a scoring mode.

Callers select behaviour by mode everywhere else in the framework, so the split
between IQAEvaluator (slices) and VolumeEvaluator (volumes) is kept out of their
way. Both return `list[ImageEvaluatorRecord]` from `run_evaluation()`, so calling
code is unaffected by which one it gets.

This lives in its own module because iqa_evaluator.py cannot import
VolumeEvaluator (which imports IQAEvaluator) without a circular import, and
main.py is scheduled for replacement.
"""

from typing import Optional

from image_loader import ImageLoader
from iqa_evaluator import IQAEvaluator
from metrics import MetricRegistry, ScoringMode
from volume_evaluator import VolumeEvaluator


def build_evaluator(
    input_image:  ImageLoader,
    target_image: Optional[ImageLoader],
    registry:     MetricRegistry,
    mode:         ScoringMode = "slice",
    source_model: Optional[str] = None,
) -> IQAEvaluator:
    """Return the evaluator for `mode`.

    Args:
        mode: "slice" scores every 2D slice separately; "volume" scores the
            whole 3D stack once.

    Raises:
        ValueError: for any other mode.
    """
    if mode == "slice":
        return IQAEvaluator(input_image, target_image, registry, source_model)
    if mode == "volume":
        return VolumeEvaluator(input_image, target_image, registry, source_model)
    raise ValueError(
        f"'{mode}' is not a scoring mode. Use 'slice' to score every 2D slice "
        "separately, or 'volume' to score the whole 3D stack at once."
    )
```

- [ ] **Step 6: Run the full suite and fix column assertions**

Run: `pytest tests/ -q`
Expected: PASS. `EvaluationResult.to_frame()` derives columns from
`ImageEvaluatorRecord.__annotations__`, so `scoring` now appears between `mode`
and `slice_index`. Update any assertion in `tests/test_records.py` or
`tests/test_evaluation_result.py` that lists the exact column set.

- [ ] **Step 7: Commit**

```bash
git add src/volume_evaluator.py src/evaluator_factory.py src/records.py tests/test_volume_evaluator.py tests/test_records.py tests/test_evaluation_result.py
git commit -m "feat(evaluator): add VolumeEvaluator subclass and build_evaluator factory"
```

---

### Task 4: Volume factories for the five MONAI segmentation metrics

**Files:**
- Modify: `src/segmentation_metrics/monai_metrics.py`
- Modify: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Consumes: `ModeSupport` (Task 2); `Spacing` (Task 1).
- Produces: `DICE`, `HAUSDORFF95`, `NSD`, `ASSD`, `PANOPTIC_QUALITY` all carry a `volume_mode` of `ModeSupport`. `hausdorff95_metric`, `surface_dice_metric` and `average_surface_distance_metric` thread the runtime spacing into MONAI.

**Background:** MONAI's `compute_dice`, `compute_hausdorff_distance`, `compute_surface_dice` and `compute_average_surface_distance` accept `(N, C, H, W)` and `(N, C, D, H, W)` alike. `compute_panoptic_quality` takes one spatial array per sample; the existing adapter indexes `y_pred[i, 0]`, which yields `(D, H, W)` from a 5D tensor — verified to work unchanged. Only the distance metrics need `spacing`; passing a runtime spacing must not silently override one a user set explicitly on the builder.

**Deviation from the spec.** The spec's error table says a metric that needs
spacing is skipped when spacing is missing. This plan instead falls back to voxel
units, matching what the spec already sanctions for `boundary_iou`. Reason: two
different behaviours for the same missing input would be arbitrary, and a
distance in voxels is still useful for comparing two runs on the same grid — it
is only meaningless *across* grids. The fallback must be announced, so
`_volume_factory` prints a one-line warning the first time it builds a distance
metric without spacing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_segmentation_metrics.py`:

```python
# ---------------------------------------------------------------------------
# Volume mode
# ---------------------------------------------------------------------------

import torch
from metrics import MetricRegistry, ModeSupport
from segmentation_metrics.monai_metrics import (
    ASSD, DICE, HAUSDORFF95, NSD, PANOPTIC_QUALITY, hausdorff95_metric,
)


def _volume_pair(shape=(1, 1, 6, 12, 12)):
    """(pred, gt) 5D binary volumes; pred is gt shifted by one voxel along x."""
    gt = torch.zeros(shape)
    gt[..., 2:4, 3:9, 3:9] = 1.0
    pred = torch.zeros(shape)
    pred[..., 2:4, 3:9, 4:10] = 1.0
    return pred, gt


class TestSegmentationVolumeMode:
    def test_all_five_support_volume(self):
        for spec in (DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY):
            assert isinstance(spec.volume_mode, ModeSupport), spec.name

    def test_dice_scores_a_five_dimensional_sample(self):
        metric = MetricRegistry(DICE).get_metric("dice", "volume", (1.0, 1.0, 1.0))
        pred, gt = _volume_pair()
        scores = metric(pred, gt)
        assert len(scores) == 1
        assert 0.0 < scores[0] < 1.0

    def test_panoptic_quality_scores_a_five_dimensional_sample(self):
        metric = MetricRegistry(PANOPTIC_QUALITY).get_metric("panoptic_quality", "volume", None)
        pred, gt = _volume_pair()
        assert len(metric(pred, gt)) == 1

    def test_spacing_changes_hausdorff_distance(self):
        pred, gt = _volume_pair()
        isotropic = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", (1.0, 1.0, 1.0))
        coarse    = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", (1.0, 1.0, 3.0))
        assert coarse(pred, gt)[0] > isotropic(pred, gt)[0]

    def test_missing_spacing_falls_back_to_voxel_units(self):
        pred, gt = _volume_pair()
        metric = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", None)
        assert metric(pred, gt)[0] > 0.0

    def test_explicit_builder_spacing_wins_over_runtime_spacing(self):
        pred, gt = _volume_pair()
        spec = hausdorff95_metric(spacing=(1.0, 1.0, 1.0))
        pinned = MetricRegistry(spec).get_metric("hausdorff95", "volume", (1.0, 1.0, 5.0))
        free   = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", (1.0, 1.0, 1.0))
        assert pinned(pred, gt)[0] == pytest.approx(free(pred, gt)[0])

    def test_slice_mode_still_works(self):
        metric = MetricRegistry(DICE).get_metric("dice")
        pred, gt = _volume_pair((4, 1, 12, 12))
        assert len(metric(pred, gt)) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_segmentation_metrics.py -q -k Volume`
Expected: FAIL — `DICE.volume_mode` is a `ModeUnsupported`.

- [ ] **Step 3: Implement the volume factories**

In `src/segmentation_metrics/monai_metrics.py`, import `ModeSupport` and `Spacing` from `metrics`, then add one shared helper:

```python
def _volume_factory(
    compute_fn: Callable[..., torch.Tensor],
    *,
    threshold: Optional[float],
    uses_spacing: bool,
    **monai_kwargs,
) -> Callable[[Optional[Spacing]], MonaiSegmentationMetric]:
    """Build the volume-mode factory for one MONAI functional metric.

    `uses_spacing` marks the distance metrics (HD95, NSD, ASSD), which convert
    voxel counts to physical units. An explicit `spacing` passed to the builder
    always wins: the user pinned it deliberately, and silently replacing it with
    whatever the current file happens to say would be worse than ignoring the
    file.
    """
    def build(spacing: Optional[Spacing]) -> MonaiSegmentationMetric:
        kwargs = dict(monai_kwargs)
        if uses_spacing:
            if spacing is not None:
                kwargs.setdefault("spacing", list(spacing))
            elif "spacing" not in kwargs:
                print(
                    "[warning] no voxel size available, so this distance is "
                    "counted in voxels rather than millimetres. Values are "
                    "comparable between images on the same grid, but not "
                    "between images recorded at different resolutions."
                )
        return MonaiSegmentationMetric(compute_fn, threshold=threshold, **kwargs)

    return build
```

`kwargs.setdefault` is what makes the builder's own `spacing=` win, because it is already present in `monai_kwargs`.

For each of the four `MonaiSegmentationMetric`-backed builders, add the `volume_mode` argument to its `MetricSpec(...)`. `dice_metric` (`uses_spacing=False`):

```python
    return MetricSpec(
        name="dice",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(_volume_factory(
            compute_dice, threshold=threshold, uses_spacing=False, **monai_kwargs)),
        builtin=False,
        ...
    )
```

`hausdorff95_metric` → `compute_hausdorff_distance`, `uses_spacing=True`.
`surface_dice_metric` → `compute_surface_dice`, `uses_spacing=True`.
`average_surface_distance_metric` → `compute_average_surface_distance`, `uses_spacing=True`.

For `panoptic_quality_metric`, the adapter needs no spacing and no change:

```python
        volume_mode=ModeSupport(
            lambda spacing: MonaiPanopticQualityMetric(threshold=threshold, **monai_kwargs)),
```

Extend the `MonaiSegmentationMetric` class docstring with one paragraph:

```
    In volume mode the adapter receives a single `(1, C, D, H, W)` sample and
    returns a single score. MONAI's functional metrics accept 4D and 5D input
    alike, so the adapter itself is unchanged; the distance metrics additionally
    receive `spacing` so their result is in millimetres rather than voxels.
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_segmentation_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/segmentation_metrics/monai_metrics.py tests/test_segmentation_metrics.py
git commit -m "feat(segmentation): score MONAI metrics over whole volumes with physical spacing"
```

---

### Task 5: Volumetric PSNR and SSIM

**Files:**
- Create: `src/volumetric_iqa.py`
- Modify: `src/metrics.py` (`PSNR`, `SSIM` volume factories)
- Create: `tests/test_volumetric_iqa.py`

**Interfaces:**
- Consumes: `Metric` protocol, `ModeSupport` (Task 2).
- Produces: `MonaiPSNRMetric(max_val=1.0)`, `MonaiSSIMMetric(spatial_dims=3, data_range=1.0)` — both satisfy `Metric`.

**Background:** pyiqa takes 4D tensors only, so it cannot score a volume. MONAI 1.6.0 ships `PSNRMetric(max_val)` and `SSIMMetric(spatial_dims, data_range=1.0, win_size=11, ...)`, both of which handle 5D input. `ImageLoader` normalises to [0, 1], so `max_val`/`data_range` are 1.0. These are different estimators from pyiqa's, so a `psnr` column from a slice run and from a volume run are not comparable — state this in the module docstring.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_volumetric_iqa.py`:

```python
"""Tests for MONAI-backed volumetric PSNR and SSIM."""
import pytest
import torch

from metrics import Metric, MetricRegistry, ModeSupport, PSNR, SSIM
from volumetric_iqa import MonaiPSNRMetric, MonaiSSIMMetric


def _volume_pair(noise: float = 0.05):
    torch.manual_seed(0)
    gt = torch.rand(1, 1, 6, 16, 16)
    pred = (gt + noise * torch.randn_like(gt)).clamp(0.0, 1.0)
    return pred, gt


class TestMonaiPSNRMetric:
    def test_satisfies_the_metric_protocol(self):
        assert isinstance(MonaiPSNRMetric(), Metric)

    def test_returns_one_score_per_sample(self):
        pred, gt = _volume_pair()
        assert len(MonaiPSNRMetric()(pred, gt)) == 1

    def test_identical_volumes_score_very_high(self):
        gt = torch.rand(1, 1, 4, 8, 8)
        assert MonaiPSNRMetric()(gt, gt)[0] > 60.0

    def test_more_noise_scores_lower(self):
        low, gt = _volume_pair(noise=0.02)
        high, _ = _volume_pair(noise=0.20)
        metric = MonaiPSNRMetric()
        assert metric(high, gt)[0] < metric(low, gt)[0]

    def test_requires_a_target(self):
        pred, _ = _volume_pair()
        with pytest.raises(ValueError, match="target"):
            MonaiPSNRMetric()(pred)


class TestMonaiSSIMMetric:
    def test_returns_one_score_per_sample(self):
        pred, gt = _volume_pair()
        assert len(MonaiSSIMMetric()(pred, gt)) == 1

    def test_identical_volumes_score_one(self):
        gt = torch.rand(1, 1, 6, 16, 16)
        assert MonaiSSIMMetric()(gt, gt)[0] == pytest.approx(1.0, abs=1e-4)

    def test_more_noise_scores_lower(self):
        low, gt = _volume_pair(noise=0.02)
        high, _ = _volume_pair(noise=0.20)
        metric = MonaiSSIMMetric()
        assert metric(high, gt)[0] < metric(low, gt)[0]

    def test_window_shrinks_to_fit_a_thin_volume(self):
        """A 6-slice stack is thinner than MONAI's default 11-voxel window."""
        thin = torch.rand(1, 1, 6, 16, 16)
        assert MonaiSSIMMetric()._window_for(thin.shape) == 5
        assert MonaiSSIMMetric()(thin, thin)[0] == pytest.approx(1.0, abs=1e-4)

    def test_window_is_never_grown_beyond_the_preferred_size(self):
        thick = torch.rand(1, 1, 40, 40, 40)
        assert MonaiSSIMMetric()._window_for(thick.shape) == 11


class TestSpecsWired:
    def test_psnr_and_ssim_support_volume(self):
        assert isinstance(PSNR.volume_mode, ModeSupport)
        assert isinstance(SSIM.volume_mode, ModeSupport)

    def test_registry_builds_the_monai_backends(self):
        registry = MetricRegistry(PSNR, SSIM)
        assert isinstance(registry.get_metric("psnr", "volume", None), MonaiPSNRMetric)
        assert isinstance(registry.get_metric("ssim", "volume", None), MonaiSSIMMetric)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_volumetric_iqa.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'volumetric_iqa'`.

- [ ] **Step 3: Implement the adapters**

Create `src/volumetric_iqa.py`:

```python
"""MONAI-backed PSNR and SSIM for whole volumes.

pyiqa's metrics take 4D tensors only, so they cannot score a `(1, C, D, H, W)`
sample. MONAI provides both measures with 3D support, and MONAI is already a
dependency of this project.

`SSIMMetric(spatial_dims=3)` uses a 3D gaussian window, so unlike a mean over
per-slice SSIM it actually sees structure across slices.

Note: these are different estimators from the pyiqa ones used in slice mode
(different window and data-range conventions). A `psnr` or `ssim` column from a
slice run and from a volume run are therefore not directly comparable.
"""

from typing import Optional

import torch
from monai.metrics import PSNRMetric, SSIMMetric


class MonaiPSNRMetric:
    """Peak signal-to-noise ratio over a whole volume.

    `max_val` is the maximum possible intensity: ImageLoader normalises every
    image to [0, 1], so the default of 1.0 is correct for this framework.
    """

    def __init__(self, *, max_val: float = 1.0):
        self._impl = PSNRMetric(max_val=max_val)

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[float]:
        if target is None:
            raise ValueError("psnr is a full-reference metric and requires a target image")
        scores = self._impl(y_pred=input, y=target)
        return [float(s) for s in scores.flatten()]


class MonaiSSIMMetric:
    """Structural similarity with a 3D window over a whole volume.

    The window is built at call time, because it cannot be larger than the volume
    it slides through: MRI stacks are routinely thinner than 11 slices, and
    MONAI's default window of 11 would fail on them. The window is shrunk to the
    largest odd size that fits every spatial axis, never grown beyond `win_size`.

    Args:
        spatial_dims: 3 for volumes. Kept configurable because MONAI supports 2
            as well, which is useful when comparing against the slice backend.
        data_range: maximum intensity range; 1.0 for this framework's [0, 1] tensors.
        win_size: the preferred gaussian window side length, in voxels.
    """

    def __init__(self, *, spatial_dims: int = 3, data_range: float = 1.0, win_size: int = 11):
        self._spatial_dims = spatial_dims
        self._data_range   = data_range
        self._win_size     = win_size

    def _window_for(self, shape: torch.Size) -> int:
        smallest = min(int(s) for s in shape[-self._spatial_dims:])
        fitted = min(self._win_size, smallest)
        if fitted % 2 == 0:
            fitted -= 1
        return max(3, fitted)

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[float]:
        if target is None:
            raise ValueError("ssim is a full-reference metric and requires a target image")
        impl = SSIMMetric(
            spatial_dims=self._spatial_dims,
            data_range=self._data_range,
            win_size=self._window_for(input.shape),
        )
        scores = impl(y_pred=input, y=target)
        return [float(s) for s in scores.flatten()]
```

- [ ] **Step 4: Wire the specs**

In `src/metrics.py`, replace the temporary pyiqa volume factories from Task 2:

```python
def _volumetric_iqa_factory(kind: str) -> Callable[[Optional[Spacing]], Metric]:
    """Late-import factory for the MONAI-backed volume metrics.

    Imported inside the function so `metrics` stays importable without MONAI's
    IQA module being loaded for a slice-only run.
    """
    def build(spacing: Optional[Spacing]) -> Metric:
        from volumetric_iqa import MonaiPSNRMetric, MonaiSSIMMetric
        return MonaiPSNRMetric() if kind == "psnr" else MonaiSSIMMetric()

    return build


PSNR = MetricSpec("psnr", "higher_is_better", True, "gray",
                  ModeSupport(_pyiqa_factory("psnr")),
                  ModeSupport(_volumetric_iqa_factory("psnr")))
SSIM = MetricSpec("ssim", "higher_is_better", True, "gray",
                  ModeSupport(_pyiqa_factory("ssim")),
                  ModeSupport(_volumetric_iqa_factory("ssim")))
```

- [ ] **Step 5: Run the tests, then the full suite**

Run: `pytest tests/test_volumetric_iqa.py -q`
Expected: PASS.

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/volumetric_iqa.py src/metrics.py tests/test_volumetric_iqa.py
git commit -m "feat(iqa): MONAI-backed volumetric psnr and ssim"
```

---

### Task 6: Boundary IoU in 3D with a physical band

**Files:**
- Modify: `src/segmentation_metrics/boundary_iou.py`
- Modify: `tests/test_boundary_iou.py`

**Interfaces:**
- Consumes: `ModeSupport`, `Spacing` (Task 2).
- Produces: `boundary_region(mask, dilation, sampling=None)` accepting nD; `band_width(shape, dilation_ratio, spacing=None)`; `boundary_iou(..., spacing=None)`; `BoundaryIoUMetric(dilation_ratio, threshold, spacing=None)`; `BOUNDARY_IOU.volume_mode` as `ModeSupport`.

**Background:** `dilation_pixels` measures the band in pixels along the image diagonal and `boundary_region` uses a chessboard distance transform — that is exactly Cheng et al. and the 2D path must keep producing bit-identical numbers. In 3D on anisotropic voxels a band of *d voxels* is physically thicker along the coarse axis: at 1×1×1.2 mm it is 8 mm in-plane but 9.6 mm through-plane, making the metric more forgiving in precisely the direction where through-plane errors live. The volume path therefore measures in millimetres via `distance_transform_edt(sampling=...)`, falling back to the voxel/chessboard definition when spacing is unknown.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boundary_iou.py`:

```python
# ---------------------------------------------------------------------------
# nD boundary bands and volume mode
# ---------------------------------------------------------------------------

import numpy as np
import pytest
import torch

from metrics import MetricRegistry, ModeSupport
from segmentation_metrics.boundary_iou import (
    BOUNDARY_IOU, BoundaryIoUMetric, band_width, boundary_iou, boundary_region,
)


def _cube(shape=(10, 20, 20), lo=(2, 5, 5), hi=(8, 15, 15)):
    m = np.zeros(shape, dtype=bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


class TestBoundaryRegion3D:
    def test_accepts_a_3d_mask(self):
        band = boundary_region(_cube(), 1)
        assert band.shape == (10, 20, 20)
        assert band.dtype == bool

    def test_band_is_a_subset_of_the_mask(self):
        mask = _cube()
        assert np.all(boundary_region(mask, 2) <= mask)

    def test_band_includes_the_caps(self):
        """The first and last occupied slice are surface in 3D, unlike in 2D."""
        mask = _cube()
        band = boundary_region(mask, 1)
        assert band[2].any() and band[7].any()

    def test_a_solid_interior_is_excluded(self):
        mask = np.ones((9, 9, 9), dtype=bool)
        band = boundary_region(mask, 1)
        assert not band[4, 4, 4]

    def test_anisotropic_sampling_thins_the_band_along_the_coarse_axis(self):
        mask = _cube()
        isotropic = boundary_region(mask, 2, sampling=(1.0, 1.0, 1.0))
        coarse = boundary_region(mask, 2, sampling=(3.0, 1.0, 1.0))
        assert coarse.sum() < isotropic.sum()


class TestBandWidth:
    def test_without_spacing_it_matches_the_pixel_diagonal(self):
        assert band_width((256, 256), 0.02) == pytest.approx(0.02 * np.hypot(256, 256))

    def test_with_spacing_it_uses_physical_extent(self):
        # extent = (130*1.2, 256*1.0, 256*1.0) mm
        expected = 0.02 * float(np.linalg.norm([130 * 1.2, 256.0, 256.0]))
        assert band_width((130, 256, 256), 0.02, (1.2, 1.0, 1.0)) == pytest.approx(expected)

    def test_never_below_one(self):
        assert band_width((2, 2), 0.001) >= 1.0


class TestBoundaryIoU3D:
    def test_identical_volumes_score_one(self):
        mask = _cube()
        assert boundary_iou(mask, mask) == pytest.approx(1.0)

    def test_shifted_volume_scores_below_one(self):
        gt = _cube()
        pred = _cube(lo=(2, 6, 5), hi=(8, 16, 15))
        assert boundary_iou(pred, gt) < 1.0

    def test_both_empty_is_nan(self):
        empty = np.zeros((6, 8, 8), dtype=bool)
        assert np.isnan(boundary_iou(empty, empty))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes"):
            boundary_iou(_cube(), _cube(shape=(10, 20, 21)))


class TestTwoDimensionalPathUnchanged:
    def test_chessboard_band_is_used_without_spacing(self):
        from scipy.ndimage import distance_transform_cdt

        mask = np.zeros((40, 40), dtype=bool)
        mask[8:32, 8:32] = True
        padded = np.pad(mask, 1).astype(np.uint8)
        expected = mask & (distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1] <= 3)
        assert np.array_equal(boundary_region(mask, 3), expected)


class TestBoundaryIoUVolumeMode:
    def test_spec_supports_volume(self):
        assert isinstance(BOUNDARY_IOU.volume_mode, ModeSupport)

    def test_registry_passes_spacing_into_the_adapter(self):
        metric = MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", (1.2, 1.0, 1.0))
        assert isinstance(metric, BoundaryIoUMetric)
        assert metric._spacing == (1.2, 1.0, 1.0)

    def test_scores_a_five_dimensional_sample(self):
        gt = torch.from_numpy(_cube().astype("float32")).unsqueeze(0).unsqueeze(0)
        pred = torch.from_numpy(_cube(lo=(2, 6, 5), hi=(8, 16, 15)).astype("float32")).unsqueeze(0).unsqueeze(0)
        metric = MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", (1.0, 1.0, 1.0))
        scores = metric(pred, gt)
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_falls_back_without_spacing(self):
        gt = torch.from_numpy(_cube().astype("float32")).unsqueeze(0).unsqueeze(0)
        metric = MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", None)
        assert metric(gt, gt)[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_boundary_iou.py -q -k "3D or BandWidth or VolumeMode"`
Expected: FAIL with `ImportError: cannot import name 'band_width'`.

- [ ] **Step 3: Generalise the geometry**

In `src/segmentation_metrics/boundary_iou.py`, import both transforms and replace `dilation_pixels` with `band_width`, keeping `dilation_pixels` as a thin wrapper so existing callers and tests keep working:

```python
from scipy.ndimage import distance_transform_cdt, distance_transform_edt


def band_width(
    shape: tuple[int, ...],
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """Boundary band width for a mask of `shape`, floored at 1.

    Without `spacing` the width is a fraction of the diagonal in voxels — the
    paper's definition. With `spacing` it is a fraction of the diagonal in
    millimetres, which keeps the band equally thick along every axis on
    anisotropic data instead of widening it along the coarse one.

    Args:
        shape: the mask's shape, 2D or 3D.
        dilation_ratio: fraction of the diagonal (paper default 0.02).
        spacing: physical size per axis, same length as `shape`.
    """
    extent = np.asarray(shape, dtype=float)
    if spacing is not None:
        extent = extent * np.asarray(spacing, dtype=float)
    return max(1.0, dilation_ratio * float(np.linalg.norm(extent)))


def dilation_pixels(
    shape: tuple[int, int], dilation_ratio: float = DEFAULT_DILATION_RATIO
) -> int:
    """Boundary band width in whole pixels — the paper's 2D definition.

    Kept for the 2D path and for callers that want an integer pixel count;
    `band_width` is the general form.
    """
    h, w = shape
    return max(1, int(round(dilation_ratio * float(np.hypot(h, w)))))
```

Generalise `boundary_region`:

```python
def boundary_region(
    mask: np.ndarray,
    dilation: float,
    sampling: Optional[tuple[float, ...]] = None,
) -> np.ndarray:
    """The band of width `dilation` lying just inside `mask`'s contour.

    Without `sampling` this is `mask & ~erode(mask, dilation)` with a
    (2*dilation+1) cubic structuring element, via one chessboard distance
    transform — Cheng et al.'s definition, and bit-identical to the previous 2D
    implementation.

    With `sampling` the distance is euclidean and measured in physical units, so
    the band is a ball of radius `dilation` millimetres rather than a cube of
    `dilation` voxels. On anisotropic data that is the difference between a band
    that is equally thick everywhere and one that is thicker along the coarse
    axis.

    The one-voxel zero pad supplies the background ring that makes an object
    clipped by the array border count that clipped edge as boundary.

    Args:
        mask: 2D or 3D array, coerced to bool.
        dilation: band width, in voxels without `sampling`, in physical units with it.
        sampling: physical size per axis, same length as `mask.ndim`.

    Returns:
        Bool array of `mask`'s shape, always a subset of `mask`.
    """
    m = np.asarray(mask, dtype=bool)
    padded = np.pad(m, 1).astype(np.uint8)
    interior = (slice(1, -1),) * m.ndim
    if sampling is None:
        distance = distance_transform_cdt(padded, metric="chessboard")[interior]
    else:
        distance = distance_transform_edt(padded, sampling=sampling)[interior]
    return m & (distance <= dilation)
```

Note the removed `ndim != 2` guard and the `interior` slice replacing the hard-coded `[1:-1, 1:-1]`.

- [ ] **Step 4: Generalise `boundary_iou` and the adapter**

Replace the 2D guard in `boundary_iou` and thread spacing through:

```python
def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    label: int = 1,
    threshold: float = 0.5,
    spacing: Optional[tuple[float, ...]] = None,
) -> float:
```

Body changes only in these lines:

```python
    if pred.ndim not in (2, 3):
        raise ValueError(f"boundary_iou expects 2D or 3D masks, got shape {pred.shape}")

    pred_mask = as_mask(pred, label, threshold)
    gt_mask = as_mask(gt, label, threshold)

    width = band_width(pred_mask.shape, dilation_ratio, spacing)
    pred_band = boundary_region(pred_mask, width, spacing)
    gt_band = boundary_region(gt_mask, width, spacing)
```

Add to the docstring's Args: `spacing: physical size per axis. Without it the band is measured in voxels (the paper's definition); with it, in physical units.`

Give `BoundaryIoUMetric` a spacing and let it handle both ranks:

```python
    def __init__(
        self,
        *,
        dilation_ratio: float = DEFAULT_DILATION_RATIO,
        threshold: float = 0.5,
        spacing: Optional[tuple[float, ...]] = None,
    ):
        self._dilation_ratio = dilation_ratio
        self._threshold      = threshold
        self._spacing        = spacing
```

and in `__call__`, replace `pred[i, 0]` scoring with:

```python
        for i in range(pred.shape[0]):
            score = boundary_iou(
                pred[i, 0],
                gt[i, 0],
                dilation_ratio=self._dilation_ratio,
                threshold=self._threshold,
                spacing=self._spacing,
            )
```

`pred[i, 0]` yields `(H, W)` from a 4D tensor and `(D, H, W)` from a 5D one, so the loop needs no other change.

Add the volume factory to the spec:

```python
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(lambda spacing: BoundaryIoUMetric(
            dilation_ratio=dilation_ratio, threshold=threshold, spacing=spacing)),
```

Add to the module docstring:

```
In volume mode the band is measured in physical units via a euclidean distance
transform with `sampling=spacing`, because a band of `d` voxels is physically
thicker along a coarse axis — on 1x1x1.2 mm data, 8 mm in-plane against 9.6 mm
through-plane — which makes the metric more forgiving in exactly the direction
where through-plane errors occur. Without spacing it falls back to the paper's
voxel/chessboard band. The 2D path is unchanged and stays bit-identical to
Cheng et al.; a cubic band and a ball-shaped band are not directly comparable,
so slice-mode and volume-mode scores should not be compared with each other.
```

- [ ] **Step 5: Run the tests, then the full suite**

Run: `pytest tests/test_boundary_iou.py -q`
Expected: PASS, including every pre-existing 2D test.

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/segmentation_metrics/boundary_iou.py tests/test_boundary_iou.py
git commit -m "feat(segmentation): 3D boundary bands measured in physical units"
```

---

### Task 7: Volumetric similarity and voxel counts as metrics

**Files:**
- Create: `src/segmentation_metrics/volume_metrics.py`
- Modify: `src/segmentation_metrics/volume.py` (docstring only)
- Modify: `src/metrics.py` (`SEGMENTATION_METRICS` bundle, re-exports)
- Modify: `src/main.py` (re-exports)
- Create: `tests/test_volume_metrics.py`

**Interfaces:**
- Consumes: `as_mask`, `vs`, `vs_signed`, `v_pred`, `v_gt`, `tp` from `segmentation_metrics.volume`; `ModeSupport` (Task 2).
- Produces: `VolumeFunctionMetric` adapter; specs `VS`, `VS_SIGNED`, `V_PRED`, `V_GT`, `TP`; `VOLUME_METRICS` tuple; `SEGMENTATION_METRICS` extended with all five.

**Background:** `volume.py` is deliberately pure NumPy with no `MetricSpec` import; the adapters live in a separate module so it stays that way. VS is registered in **both** modes: per-slice VS is a complementary diagnostic, not a broken one — a model can move mass from slice A to slice B, leaving the totals untouched, so volume VS reads 1.0 while per-slice VS shows the smearing. The counts `v_pred`, `v_gt` and `tp` are raw numbers rather than quality scores, hence `direction="not_ranked"`; they exist so `aggregate_volumes()` (Task 8) can rebuild volume-level dice and VS exactly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_volume_metrics.py`:

```python
"""Tests for the VS / count metric specs."""
import numpy as np
import pytest
import torch

from metrics import MetricRegistry, ModeSupport, SEGMENTATION_METRICS
from segmentation_metrics.volume_metrics import (
    TP, V_GT, V_PRED, VS, VS_SIGNED, VOLUME_METRICS,
)


def _pair_4d():
    """Two slices: slice 0 identical, slice 1 prediction twice the size."""
    gt = torch.zeros(2, 1, 8, 8)
    pred = torch.zeros(2, 1, 8, 8)
    gt[0, 0, 1:3, 1:3] = 1.0
    pred[0, 0, 1:3, 1:3] = 1.0
    gt[1, 0, 1:3, 1:3] = 1.0
    pred[1, 0, 1:5, 1:3] = 1.0
    return pred, gt


def _pair_5d():
    pred, gt = _pair_4d()
    return pred.permute(1, 0, 2, 3).unsqueeze(0), gt.permute(1, 0, 2, 3).unsqueeze(0)


class TestSpecShape:
    def test_all_five_support_both_modes(self):
        for spec in VOLUME_METRICS:
            assert isinstance(spec.slice_mode, ModeSupport), spec.name
            assert isinstance(spec.volume_mode, ModeSupport), spec.name

    def test_counts_are_not_ranked(self):
        assert {s.direction for s in (V_PRED, V_GT, TP)} == {"not_ranked"}

    def test_vs_is_higher_is_better(self):
        assert VS.direction == "higher_is_better"

    def test_included_in_the_segmentation_bundle(self):
        names = {s.name for s in SEGMENTATION_METRICS}
        assert {"vs", "vs_signed", "v_pred", "v_gt", "tp"} <= names


class TestSliceMode:
    def test_vs_per_slice(self):
        metric = MetricRegistry(VS).get_metric("vs")
        pred, gt = _pair_4d()
        scores = metric(pred, gt)
        assert scores[0] == pytest.approx(1.0)          # identical
        assert scores[1] == pytest.approx(1.0 - 4 / 12)  # 8 vs 4 voxels

    def test_counts_per_slice(self):
        pred, gt = _pair_4d()
        assert MetricRegistry(V_PRED).get_metric("v_pred")(pred, gt) == [4.0, 8.0]
        assert MetricRegistry(V_GT).get_metric("v_gt")(pred, gt) == [4.0, 4.0]
        assert MetricRegistry(TP).get_metric("tp")(pred, gt) == [4.0, 4.0]


class TestVolumeMode:
    def test_vs_over_the_whole_volume(self):
        metric = MetricRegistry(VS).get_metric("vs", "volume", (1.0, 1.0, 1.0))
        pred, gt = _pair_5d()
        # totals: pred 12, gt 8  ->  1 - 4/20
        assert metric(pred, gt)[0] == pytest.approx(1.0 - 4 / 20)

    def test_vs_is_spacing_invariant(self):
        pred, gt = _pair_5d()
        fine = MetricRegistry(VS).get_metric("vs", "volume", (1.0, 1.0, 1.0))
        coarse = MetricRegistry(VS).get_metric("vs", "volume", (3.0, 1.0, 1.0))
        assert fine(pred, gt)[0] == pytest.approx(coarse(pred, gt)[0])

    def test_vs_signed_is_positive_when_oversegmenting(self):
        metric = MetricRegistry(VS_SIGNED).get_metric("vs_signed", "volume", None)
        pred, gt = _pair_5d()
        assert metric(pred, gt)[0] > 0.0

    def test_counts_sum_over_the_volume(self):
        pred, gt = _pair_5d()
        assert MetricRegistry(V_PRED).get_metric("v_pred", "volume", None)(pred, gt) == [12.0]
        assert MetricRegistry(TP).get_metric("tp", "volume", None)(pred, gt) == [8.0]


class TestUndefined:
    def test_two_empty_masks_give_none(self):
        empty = torch.zeros(1, 1, 4, 4, 4)
        assert MetricRegistry(VS).get_metric("vs", "volume", None)(empty, empty) == [None]

    def test_requires_a_target(self):
        pred, _ = _pair_5d()
        with pytest.raises(ValueError, match="target"):
            MetricRegistry(VS).get_metric("vs", "volume", None)(pred)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_volume_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'segmentation_metrics.volume_metrics'`.

- [ ] **Step 3: Implement the adapter and specs**

Create `src/segmentation_metrics/volume_metrics.py`:

```python
"""Volumetric Similarity and voxel counts as registrable metrics.

`volume.py` stays pure NumPy with no framework imports; the MetricSpec wrappers
live here, which also keeps it out of the circular-import dance the other
segmentation modules need.

VS is registered for BOTH scoring modes on purpose. Per-slice VS is not a broken
measurement, it is a complementary one: if every slice has the right area the
total volume is right, but a right total volume does not imply right per-slice
areas. A model that moves mass from one slice to the next leaves the totals
untouched, so volume VS reads 1.0 while the structure is smeared along the
depth axis — and per-slice VS shows it.

Note that VS compares sizes, not overlap: two masks of equal size that do not
touch anywhere still score 1.0. It is only meaningful beside dice.

`v_pred`, `v_gt` and `tp` are raw voxel counts, not quality scores, so their
direction is "not_ranked". They exist so that a per-slice run can be aggregated
to correct volume-level numbers by `EvaluationResult.aggregate_volumes()` —
summing counts and then taking the ratio, never averaging ratios.

Usage: import `metrics` before this module, same as the other segmentation
metric modules.
"""

from typing import Callable, Optional

import numpy as np
import torch

from metrics import MetricSpec, ModeSupport
from segmentation_metrics.volume import tp as _tp
from segmentation_metrics.volume import v_gt as _v_gt
from segmentation_metrics.volume import v_pred as _v_pred
from segmentation_metrics.volume import vs as _vs
from segmentation_metrics.volume import vs_signed as _vs_signed


class VolumeFunctionMetric:
    """Adapter turning a `volume.py` function into a Metric.

    The wrapped functions take two NumPy arrays of any matching shape, so the
    same adapter serves both scoring modes: in slice mode it sees `(N, C, H, W)`
    and scores each sample's channel 0; in volume mode it sees `(1, C, D, H, W)`
    and scores the whole `(D, H, W)` body at once.

    Args:
        fn: one of `vs`, `vs_signed`, `v_pred`, `v_gt`, `tp`.
        threshold: binarization cutoff for float masks (`value >= threshold`).
            ImageLoader tensors are floats in [0, 1], so a cutoff always applies.
    """

    def __init__(self, fn: Callable[..., float], *, threshold: float = 0.5):
        self._fn        = fn
        self._threshold = threshold

    def __call__(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> list[Optional[float]]:
        if target is None:
            raise ValueError(
                f"'{self._fn.__name__}' compares two masks and requires a target mask"
            )
        pred = input.detach().cpu().numpy()
        gt   = target.detach().cpu().numpy()
        scores: list[Optional[float]] = []
        for i in range(pred.shape[0]):
            value = self._fn(pred[i, 0], gt[i, 0], threshold=self._threshold)
            scores.append(None if np.isnan(value) else float(value))
        return scores


def _both_modes(fn: Callable[..., float], threshold: float) -> tuple[ModeSupport, ModeSupport]:
    """Slice and volume capability for a function that is already shape-agnostic."""
    metric = VolumeFunctionMetric(fn, threshold=threshold)
    return ModeSupport(lambda: metric), ModeSupport(lambda spacing: metric)


def vs_metric(*, threshold: float = 0.5) -> MetricSpec:
    """Volumetric Similarity (Taha & Hanbury): 1 - |Vp - Vg| / (Vp + Vg).

    Range [0, 1]; 1.0 means the two masks have the same size. Spacing-invariant:
    the voxel volume cancels in numerator and denominator.
    """
    slice_mode, volume_mode = _both_modes(_vs, threshold)
    return MetricSpec(
        name="vs", direction="higher_is_better", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            "Volumetric Similarity: how closely the predicted and reference "
            "masks agree in size (1.0 = same size). It is not an overlap "
            "measure — two masks of equal size that never touch also score 1.0 "
            "— so read it beside dice. Domain-agnostic and independent of voxel "
            "size."
        ),
        domain="",
    )


def vs_signed_metric(*, threshold: float = 0.5) -> MetricSpec:
    """Signed Volumetric Similarity (SimpleITK convention), range [-2, 2].

    Negative means the prediction is smaller than the reference
    (undersegmentation), positive means larger (oversegmentation).
    """
    slice_mode, volume_mode = _both_modes(_vs_signed, threshold)
    return MetricSpec(
        name="vs_signed", direction="not_ranked", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            "Signed Volumetric Similarity: the direction of the size error "
            "(negative = the prediction is too small, positive = too large, "
            "0.0 = the sizes match). Domain-agnostic and independent of voxel size."
        ),
        domain="",
    )


def _count_metric(name: str, fn: Callable[..., float], what: str, threshold: float) -> MetricSpec:
    slice_mode, volume_mode = _both_modes(fn, threshold)
    return MetricSpec(
        name=name, direction="not_ranked", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            f"Raw voxel count: {what}. Not a quality score — it is reported so "
            "that per-slice runs can be summed into correct volume-level dice "
            "and volumetric similarity."
        ),
        domain="",
    )


def v_pred_metric(*, threshold: float = 0.5) -> MetricSpec:
    return _count_metric("v_pred", _v_pred, "voxels marked in the prediction", threshold)


def v_gt_metric(*, threshold: float = 0.5) -> MetricSpec:
    return _count_metric("v_gt", _v_gt, "voxels marked in the reference", threshold)


def tp_metric(*, threshold: float = 0.5) -> MetricSpec:
    return _count_metric("tp", _tp, "voxels marked in both masks", threshold)


VS        = vs_metric()
VS_SIGNED = vs_signed_metric()
V_PRED    = v_pred_metric()
V_GT      = v_gt_metric()
TP        = tp_metric()

VOLUME_METRICS = (VS, VS_SIGNED, V_PRED, V_GT, TP)
```

The wrapped functions take `label` and `threshold` as keyword arguments and
`_check_shapes` runs inside each of them, so the adapter passes only `threshold`.

- [ ] **Step 4: Register them in the bundle**

In `src/metrics.py`, next to the existing late imports:

```python
from segmentation_metrics.volume_metrics import (
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)
```

and extend the bundle:

```python
SEGMENTATION_METRICS = (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU,
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)
```

Add `VS, VS_SIGNED, V_PRED, V_GT, TP` to the `from metrics import (...)` re-export block in `src/main.py`.

In `src/segmentation_metrics/volume.py`, extend `aggregate_patient`'s docstring with:

```
    This is the correct way to get volume-level numbers out of a per-slice run:
    it sums the counts before dividing, so it never averages ratios. A run with
    `mode="volume"` computes the same quantities directly;
    `EvaluationResult.aggregate_volumes()` wraps this function for the per-slice
    case.
```

- [ ] **Step 5: Run the tests, then the full suite**

Run: `pytest tests/test_volume_metrics.py -q`
Expected: PASS.

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/segmentation_metrics/volume_metrics.py src/segmentation_metrics/volume.py src/metrics.py src/main.py tests/test_volume_metrics.py
git commit -m "feat(segmentation): register volumetric similarity and voxel counts as metrics"
```

---

### Task 8: `EvaluationResult.aggregate_volumes()`

**Files:**
- Modify: `src/evaluation_result.py`
- Modify: `tests/test_evaluation_result.py`

**Interfaces:**
- Consumes: the `v_pred` / `v_gt` / `tp` columns produced by Task 7; `ImageEvaluatorRecord.scoring` (Task 3).
- Produces: `EvaluationResult.aggregate_volumes() -> pd.DataFrame`, indexed by `image_id` prefix, with columns `v_pred`, `v_gt`, `tp`, `dice`, `vs`, `vs_signed`.

**Background:** a per-slice run's `.describe()` averages ratios, which is wrong for dice and VS and meaningless for the distance metrics. This method sums the counts per volume and then divides — the same arithmetic as `volume.aggregate_patient()`. Metrics that cannot be reconstructed from per-slice values (hausdorff95, assd, nsd, panoptic_quality, boundary_iou and every deep-2D metric) are deliberately **absent** from the result rather than averaged.

Rows are grouped by the volume an `image_id` belongs to. Slice ids are formatted by `IQAEvaluator._format_slice_id` as `[<model>/]<stem>_sNNN`, so stripping a trailing `_sNNN` yields the volume key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation_result.py`:

```python
# ---------------------------------------------------------------------------
# aggregate_volumes
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from evaluation_result import EvaluationResult, _EvaluatedImage
from metrics import MetricRegistry
from records import ImageEvaluatorRecord
from segmentation_metrics.volume_metrics import TP, V_GT, V_PRED, VS


def _slice_record(image_id, idx, *, v_pred, v_gt, tp, hd95=None):
    r = ImageEvaluatorRecord(image_id=f"{image_id}_s{idx:03d}", scoring="slice", slice_index=idx)
    r.extra.update({"v_pred": v_pred, "v_gt": v_gt, "tp": tp})
    if hd95 is not None:
        r.extra["hausdorff95"] = hd95
    return r


def _result(records):
    return EvaluationResult(
        [_EvaluatedImage(input_path=Path("vol.nii.gz"), records=records)],
        MetricRegistry(V_PRED, V_GT, TP, VS),
    )


class TestAggregateVolumes:
    def test_one_row_per_volume(self):
        records = [
            _slice_record("vol", 0, v_pred=4, v_gt=4, tp=4),
            _slice_record("vol", 1, v_pred=8, v_gt=4, tp=4),
        ]
        df = _result(records).aggregate_volumes()
        assert list(df.index) == ["vol"]

    def test_counts_are_summed(self):
        records = [
            _slice_record("vol", 0, v_pred=4, v_gt=4, tp=4),
            _slice_record("vol", 1, v_pred=8, v_gt=4, tp=4),
        ]
        row = _result(records).aggregate_volumes().loc["vol"]
        assert row["v_pred"] == 12.0
        assert row["v_gt"] == 8.0
        assert row["tp"] == 8.0

    def test_dice_is_the_ratio_of_sums_not_the_mean_of_ratios(self):
        records = [
            _slice_record("vol", 0, v_pred=4, v_gt=4, tp=4),     # per-slice dice 1.0
            _slice_record("vol", 1, v_pred=8, v_gt=4, tp=4),     # per-slice dice 2/3
        ]
        row = _result(records).aggregate_volumes().loc["vol"]
        assert row["dice"] == pytest.approx(2 * 8 / 20)          # 0.8, not 0.8333
        assert row["vs"] == pytest.approx(1 - 4 / 20)

    def test_two_volumes_are_grouped_separately(self):
        records = [
            _slice_record("a", 0, v_pred=4, v_gt=4, tp=4),
            _slice_record("b", 0, v_pred=2, v_gt=6, tp=2),
        ]
        df = _result(records).aggregate_volumes()
        assert sorted(df.index) == ["a", "b"]

    def test_model_prefix_is_kept_in_the_key(self):
        records = [_slice_record("smore/vol", 0, v_pred=4, v_gt=4, tp=4)]
        assert list(_result(records).aggregate_volumes().index) == ["smore/vol"]

    def test_non_reconstructible_metrics_are_absent(self):
        records = [
            _slice_record("vol", 0, v_pred=4, v_gt=4, tp=4, hd95=2.0),
            _slice_record("vol", 1, v_pred=8, v_gt=4, tp=4, hd95=9.0),
        ]
        df = _result(records).aggregate_volumes()
        assert "hausdorff95" not in df.columns

    def test_empty_masks_give_nan_ratios(self):
        records = [_slice_record("vol", 0, v_pred=0, v_gt=0, tp=0)]
        row = _result(records).aggregate_volumes().loc["vol"]
        assert np.isnan(row["dice"]) and np.isnan(row["vs"])

    def test_missing_counts_are_reported_clearly(self):
        record = ImageEvaluatorRecord(image_id="vol_s000", scoring="slice", slice_index=0)
        result = EvaluationResult(
            [_EvaluatedImage(input_path=Path("vol.nii.gz"), records=[record])],
            MetricRegistry(),
        )
        with pytest.raises(ValueError, match="v_pred"):
            result.aggregate_volumes()

    def test_refuses_a_volume_mode_result(self):
        record = ImageEvaluatorRecord(image_id="vol", scoring="volume", slice_index=None)
        result = EvaluationResult(
            [_EvaluatedImage(input_path=Path("vol.nii.gz"), records=[record])],
            MetricRegistry(),
        )
        with pytest.raises(ValueError, match="already"):
            result.aggregate_volumes()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation_result.py -q -k AggregateVolumes`
Expected: FAIL with `AttributeError: 'EvaluationResult' object has no attribute 'aggregate_volumes'`.

- [ ] **Step 3: Implement it**

In `src/evaluation_result.py`, add the import and the method:

```python
import re

_SLICE_SUFFIX = re.compile(r"_s\d+$")
```

```python
    def aggregate_volumes(self) -> pd.DataFrame:
        """Volume-level numbers from a per-slice run, computed correctly.

        Averaging per-slice scores is wrong for every ratio metric: the mean of
        per-slice ratios is not the ratio of the summed volume, and a slice
        holding three voxels would count as much as one holding three thousand.
        This method sums the voxel counts per volume first and divides once.

        Only metrics that can be rebuilt from per-slice counts appear: `dice`,
        `vs` and `vs_signed`. Distance and perceptual metrics (hausdorff95,
        assd, nsd, panoptic_quality, boundary_iou, lpips, ...) cannot be
        reconstructed from per-slice values at all and are deliberately left
        out rather than averaged — run again with mode="volume" for those.

        Returns:
            DataFrame indexed by volume id (the slice id without its `_sNNN`
            suffix) with columns v_pred, v_gt, tp, dice, vs, vs_signed. Ratio
            columns are NaN where the denominator is 0.

        Raises:
            ValueError: if the run was already scored per volume, or if the
                voxel-count metrics were not registered.
        """
        df = self.to_frame()

        if not df.empty and (df["scoring"] == "volume").any():
            raise ValueError(
                "This report was already scored one row per volume, so there is "
                "nothing left to aggregate. Use to_frame() to read it."
            )

        missing = [c for c in ("v_pred", "v_gt", "tp") if c not in df.columns]
        if missing:
            raise ValueError(
                "Volume-level numbers are built from the voxel counts v_pred, "
                f"v_gt and tp, and this report has no {', '.join(missing)} "
                "column. Add the counting metrics to the registry and run "
                "again: MetricRegistry(..., V_PRED, V_GT, TP)."
            )

        counts = df[["image_id", "v_pred", "v_gt", "tp"]].copy()
        counts["image_id"] = counts["image_id"].str.replace(_SLICE_SUFFIX, "", regex=True)
        grouped = counts.groupby("image_id")[["v_pred", "v_gt", "tp"]].sum(min_count=1)

        v_pred_sum = grouped["v_pred"].astype(float)
        v_gt_sum   = grouped["v_gt"].astype(float)
        tp_sum     = grouped["tp"].astype(float)
        denom      = v_pred_sum + v_gt_sum

        grouped["dice"]      = np.where(denom == 0, np.nan, 2.0 * tp_sum / denom)
        grouped["vs"]        = np.where(denom == 0, np.nan,
                                        1.0 - (v_pred_sum - v_gt_sum).abs() / denom)
        grouped["vs_signed"] = np.where(denom == 0, np.nan,
                                        2.0 * (v_pred_sum - v_gt_sum) / denom)
        return grouped
```

Add `import numpy as np` at the top of the file, and extend the class docstring's usage block:

```
    Volume-level numbers from a per-slice run:
        result.aggregate_volumes()
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `pytest tests/test_evaluation_result.py -q`
Expected: PASS.

Run: `pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation_result.py tests/test_evaluation_result.py
git commit -m "feat(result): aggregate_volumes rebuilds volume-level dice and vs from counts"
```

---

### Task 9: Mode plumbing through `evaluate()` and the CLI

The last piece: choosing the mode from outside, reporting skipped metrics once per run, and skipping files that are not volumes.

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `build_evaluator` (Task 3); `MetricRegistry.select` (Task 2); `ImageLoader.is_volumetric` (Task 1).
- Produces: `evaluate(input_path, target_path=None, *, registry, mode="slice")`; `report_skipped_metrics(skipped, mode) -> None`; CLI flag `--mode {slice,volume}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_main.py`:

```python
# ---------------------------------------------------------------------------
# Scoring mode
# ---------------------------------------------------------------------------

import pytest

from main import evaluate, report_skipped_metrics
from metrics import MetricRegistry, MetricSpec, ModeSupport, ModeUnsupported, SkippedMetric


def _spec(name, *, volume, reason="only reads flat pictures"):
    def make(*_args):
        return lambda inp, tgt=None: [float(inp[i].mean()) for i in range(inp.shape[0])]

    return MetricSpec(
        name=name, direction="higher_is_better", reference=False, channels="gray",
        slice_mode=ModeSupport(make),
        volume_mode=ModeSupport(make) if volume else ModeUnsupported(reason),
        builtin=False,
    )


class TestSkipMessage:
    def test_names_every_skipped_metric(self, capsys):
        report_skipped_metrics(
            [SkippedMetric("lpips", "reads flat pictures"),
             SkippedMetric("niqe", "reads flat pictures")],
            "volume",
        )
        out = capsys.readouterr().out
        assert "lpips" in out and "niqe" in out

    def test_states_the_reason(self, capsys):
        report_skipped_metrics([SkippedMetric("lpips", "reads flat pictures")], "volume")
        assert "reads flat pictures" in capsys.readouterr().out

    def test_names_the_other_mode_as_the_way_out(self, capsys):
        report_skipped_metrics([SkippedMetric("lpips", "r")], "volume")
        assert 'mode="slice"' in capsys.readouterr().out

    def test_silent_when_nothing_is_skipped(self, capsys):
        report_skipped_metrics([], "volume")
        assert capsys.readouterr().out == ""


class TestEvaluateMode:
    def test_default_is_slice(self, nifti_volume):
        result = evaluate(nifti_volume, None, registry=MetricRegistry(_spec("m", volume=True)))
        assert len(result.to_frame()) == 6

    def test_volume_mode_yields_one_row(self, nifti_volume):
        result = evaluate(nifti_volume, None,
                          registry=MetricRegistry(_spec("m", volume=True)), mode="volume")
        df = result.to_frame()
        assert len(df) == 1
        assert df.iloc[0]["scoring"] == "volume"

    def test_skip_message_appears_once_per_run(self, tmp_path, nifti_volume, capsys):
        import shutil

        directory = tmp_path / "many"
        directory.mkdir()
        for i in range(3):
            shutil.copy(nifti_volume, directory / f"vol{i}.nii.gz")
        registry = MetricRegistry(_spec("ok", volume=True), _spec("flat", volume=False))
        evaluate(directory, None, registry=registry, mode="volume")
        assert capsys.readouterr().out.count("flat") == 1

    def test_hard_error_when_no_metric_can_serve_the_mode(self, nifti_volume):
        registry = MetricRegistry(_spec("flat", volume=False))
        with pytest.raises(ValueError, match="none of"):
            evaluate(nifti_volume, None, registry=registry, mode="volume")

    def test_non_volumetric_files_are_skipped_not_fatal(self, tmp_path, nifti_volume, synthetic_png, capsys):
        import shutil

        directory = tmp_path / "mixed"
        directory.mkdir()
        shutil.copy(nifti_volume, directory / "vol.nii.gz")
        shutil.copy(synthetic_png, directory / "flat.png")
        registry = MetricRegistry(_spec("m", volume=True))
        result = evaluate(directory, None, registry=registry, mode="volume")
        assert len(result.to_frame()) == 1
        assert "flat.png" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -q -k "SkipMessage or EvaluateMode"`
Expected: FAIL with `ImportError: cannot import name 'report_skipped_metrics' from 'main'`.

- [ ] **Step 3: Implement the message and the mode**

In `src/main.py`, add the imports and the reporter:

```python
from evaluator_factory import build_evaluator
from metrics import ScoringMode, SkippedMetric

_OTHER_MODE = {"slice": "volume", "volume": "slice"}


def report_skipped_metrics(skipped: list[SkippedMetric], mode: ScoringMode) -> None:
    """Tell the user, once per run, which metrics will not be computed and why.

    Metrics are grouped by reason so a shared explanation is printed once. Prints
    nothing when nothing is skipped.
    """
    if not skipped:
        return
    by_reason: dict[str, list[str]] = {}
    for item in skipped:
        by_reason.setdefault(item.reason, []).append(item.name)

    print(f"\nSome metrics cannot be scored in {mode} mode and will be skipped.")
    for reason, names in by_reason.items():
        print(f"\n  {', '.join(sorted(names))}")
        print(f"  This metric {reason}.")
    print(f'\nTo score them, run again with mode="{_OTHER_MODE[mode]}".\n')
```

Change `evaluate()`'s signature and body:

```python
def evaluate(
    input_path: Path,
    target_path: Optional[Path] = None,
    *,
    registry: MetricRegistry,
    mode: ScoringMode = "slice",
) -> EvaluationResult:
```

Add to the docstring:

```
        mode:        "slice" scores every 2D slice separately and produces one
                     row per slice; "volume" scores each 3D stack once and
                     produces one row per volume. Metrics that cannot serve the
                     chosen mode are skipped with a message; if none can, this
                     raises.
```

Insert right after the docstring, before `evaluated: list[...] = []`:

```python
    applicable, skipped = registry.select(mode)
    if not applicable:
        raise ValueError(
            f"None of the selected metrics can be scored in {mode} mode, so this "
            f'run would compute nothing. Run again with mode="{_OTHER_MODE[mode]}", '
            "or pick metrics that work on whole volumes (dice, hausdorff95, nsd, "
            "assd, panoptic_quality, boundary_iou, vs, psnr, ssim)."
        )
    report_skipped_metrics(skipped, mode)
```

Replace the evaluator construction inside `_run_one`:

```python
            if mode == "volume" and not input_loader.is_volumetric:
                print(
                    f"[{inp.name}] skipped: this is not a 3D volume. Its slices "
                    "are not stacked along a spatial axis, which is the case for "
                    "PNG and JPEG images, for a single slice, and for 4D scans "
                    "whose frames are time steps."
                )
                return
            records = build_evaluator(
                input_loader, target_loader, registry, mode
            ).run_evaluation()
```

- [ ] **Step 4: Add the CLI flag**

In `main()`:

```python
    parser.add_argument(
        "--mode", choices=["slice", "volume"], default="slice",
        help="Score every 2D slice separately (default) or each 3D volume once.",
    )
```

and pass it through:

```python
    result = evaluate(args.input, args.target, registry=registry, mode=args.mode)
```

- [ ] **Step 5: Run the tests, then the full suite**

Run: `pytest tests/test_main.py -q`
Expected: PASS.

Run: `pytest tests/ -q`
Expected: PASS — every new test plus the 274 originals.

- [ ] **Step 6: Add the cross-check test**

This is the strongest available evidence that both paths are right: two
independent routes to one number. Append to `tests/test_volume_metrics.py`:

```python
class TestCrossCheck:
    def test_volume_dice_equals_aggregated_slice_dice(self, tmp_path):
        """Volume-mode dice and aggregate_volumes()'s dice must agree exactly."""
        import nibabel as nib
        from main import evaluate
        from metrics import MetricRegistry
        from segmentation_metrics.monai_metrics import DICE
        from segmentation_metrics.volume_metrics import TP, V_GT, V_PRED

        gt = np.zeros((12, 12, 6), dtype="float32")
        gt[3:9, 3:9, 1:5] = 1.0
        pred = np.zeros_like(gt)
        pred[4:10, 3:9, 1:5] = 1.0
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        gt_path = tmp_path / "case_gt.nii.gz"
        pred_path = tmp_path / "case_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred, affine), pred_path)
        nib.save(nib.Nifti1Image(gt, affine), gt_path)

        volume_dice = evaluate(
            pred_path, gt_path, registry=MetricRegistry(DICE), mode="volume"
        ).to_frame().iloc[0]["dice"]

        aggregated = evaluate(
            pred_path, gt_path, registry=MetricRegistry(V_PRED, V_GT, TP), mode="slice"
        ).aggregate_volumes().iloc[0]["dice"]

        assert volume_dice == pytest.approx(aggregated, abs=1e-6)
```

Run: `pytest tests/test_volume_metrics.py::TestCrossCheck -q`
Expected: PASS. If it fails, the discrepancy is real — do not adjust the
tolerance; find which path is wrong.

- [ ] **Step 7: Commit**

```bash
git add src/main.py tests/test_main.py tests/test_volume_metrics.py
git commit -m "feat(cli): choose scoring mode per run and report skipped metrics once"
```

---

## After the plan

Not part of any task, recorded so they are not lost:

- **`CLAUDE.md` is stale.** It documents `mask_writer.py`, `MaskWriter` and
  `records.best_slice_per_metric()`, none of which exist. The obsolete section
  should be removed, and the file updated for the new modules — deliberately
  left out of this plan.
- The multi-volume loading class that replaces `main.evaluate()`, which is also
  where resampling belongs.
- Reslicing to sagittal/coronal for evaluation.
