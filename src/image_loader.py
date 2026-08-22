"""Image loading: format-specific decoders, ImageLoader, filename matching."""

from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _to_normalized_channel_tensor(depth_first_array: np.ndarray) -> torch.Tensor:
    arr = depth_first_array.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(1)


def _load_pil(path: Path) -> torch.Tensor:
    grayscale = np.asarray(Image.open(path).convert("L"))
    return _to_normalized_channel_tensor(grayscale[np.newaxis])


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


def _load_dicom(path: Path) -> torch.Tensor:
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
    return _to_normalized_channel_tensor(pixel_array)


def _load_nifti(path: Path) -> torch.Tensor:
    data = nib.as_closest_canonical(nib.load(str(path))).get_fdata()
    if data.ndim == 3:
        depth_first = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        depth_first = np.transpose(data, (3, 2, 0, 1)).reshape(-1, data.shape[0], data.shape[1])
    else:
        raise ValueError(f"Unsupported NIfTI ndim {data.ndim} for {path}")
    return _to_normalized_channel_tensor(depth_first)


def _load_sitk(path: Path) -> torch.Tensor:
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if volume.ndim == 2:
        volume = volume[np.newaxis]
    elif volume.ndim != 3:
        raise ValueError(f"Unsupported SimpleITK array shape {volume.shape} for {path}")
    return _to_normalized_channel_tensor(volume)


_LOADERS: dict[str, Callable[[Path], torch.Tensor]] = {
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
    def __init__(self, path: Path) -> None:
        self.path = path
        self.suffix = canonical_suffix(path)
        if self.suffix not in _LOADERS:
            raise ValueError(f"Unsupported format: {path}")
        self._tensor: Optional[torch.Tensor] = None

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            self._tensor = _LOADERS[self.suffix](self.path)
        return self._tensor

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
