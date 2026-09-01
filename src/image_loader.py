"""Image loading: format-specific decoders, ImageLoader, filename matching."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from PIL import Image

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


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _to_normalized_channel_tensor(depth_first_array: np.ndarray) -> torch.Tensor:
    arr = depth_first_array.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(1)


def _load_pil(path: Path) -> LoadedImage:
    grayscale = np.asarray(Image.open(path).convert("L"))
    return LoadedImage(_to_normalized_channel_tensor(grayscale[np.newaxis]))


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
