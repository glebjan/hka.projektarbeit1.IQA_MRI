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


_LOADERS: dict[str, Callable[[Path], LoadedImage]] = {
    ".png":  _load_pil,
    ".jpg":  _load_pil,
    ".jpeg": _load_pil,
    ".dcm":  _load_dicom,
    ".nii":  _load_nifti,
    ".nrrd": _load_sitk,
    ".mha":  _load_sitk,
    ".mhd":  _load_sitk,
}


def canonical_suffix(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return ".nii"
    return path.suffix.lower()


def is_supported(path: Path) -> bool:
    return canonical_suffix(path) in _LOADERS


# ---------------------------------------------------------------------------
# Filename matching (input <-> target discovery)
# ---------------------------------------------------------------------------

_MIN_MATCH_PREFIX_LENGTH = 4


def strip_all_extensions(path: Path) -> str:
    return path.name.split(".")[0]


def _shared_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and is_supported(p))


def find_matching_target(input_path: Path, targets: list[Path]) -> Optional[Path]:
    input_stem = strip_all_extensions(input_path)
    best_match: Optional[Path] = None
    longest_prefix = 0
    for candidate in targets:
        length = _shared_prefix_length(input_stem, strip_all_extensions(candidate))
        if length > longest_prefix:
            best_match, longest_prefix = candidate, length
    return best_match if longest_prefix >= _MIN_MATCH_PREFIX_LENGTH else None


# ---------------------------------------------------------------------------
# ImageLoader
# ---------------------------------------------------------------------------

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
        """True for slices with (almost) no content; computed on the raw data.

        A slice is empty when its spread is below 0.1 % of the volume's
        intensity span, or (ordinary intensity images only, see below) its
        lift above the volume floor is below that same 0.1 %. The span is
        measured between the 0.5th and 99.5th percentiles, so one spike voxel
        cannot shrink it and mark healthy slices empty. For a volume whose
        foreground is rarer than 0.5 % (a small lesion mask) the percentile
        span collapses to zero and the extremes are used instead — and in
        that fallback regime the lift test is skipped entirely. A sparse
        binary mask's foreground mean sits, by construction, only a tiny
        fraction of the way from floor to ceiling, so the lift test would
        re-flag the very slice the extremes fallback exists to rescue; the
        spread test alone is meaningful there. A constant volume has no span
        at all and every slice counts as empty.
        """
        raw = self.raw.astype(np.float32, copy=False)
        lo, hi = (float(v) for v in np.percentile(raw, [0.5, 99.5]))
        used_extremes = hi <= lo
        if used_extremes:
            lo, hi = float(raw.min()), float(raw.max())
        span = hi - lo
        if span <= 0.0:
            return torch.ones(raw.shape[0], dtype=torch.bool)
        flat = raw.reshape(raw.shape[0], -1)
        std  = flat.std(axis=1)
        empty = std < 1e-3 * span
        if not used_extremes:
            lift = flat.mean(axis=1) - lo
            empty = empty | (lift < 1e-3 * span)
        return torch.from_numpy(empty)

    def log_tensor_shape(self) -> torch.Size:
        shape = self.tensor.shape
        print(f"[{self.path.name}] tensor size: {tuple(shape)}")
        return shape


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
