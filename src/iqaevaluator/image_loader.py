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

from iqaevaluator.metric_spec import Spacing
from iqaevaluator.normalization import (
    FixedRange, IntensityRange, Mask, MinMax, Normalizer, Raw,
)


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image plus the geometry the decoder knew about.

    Attributes:
        raw:           (D, H, W), or (D, H, W, 3) for a colour PNG/JPEG. The
                       dtype is the decoder's — uint8 for PNG/JPEG, float32
                       for DICOM (slope/intercept applied), the file's own
                       dtype for NIfTI and SimpleITK formats, so an integer
                       label map stays integer. Only the PIL decoder ever
                       produces a channel axis; the medical formats stay
                       single-channel.
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

# PIL's own ITU-R 601-2 weights, so the same picture stored as RGB and as
# greyscale yields the same numbers.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# Modes PIL already stores as one channel. "LA"/"La" are greyscale-with-alpha:
# convert("L") drops the alpha channel exactly as it would for any other
# mode here, so these files score the same as the same picture without an
# alpha channel. Everything else is treated as colour and normalized to RGB,
# so only two cases can leave this decoder.
_SINGLE_CHANNEL_MODES = frozenset({"1", "L", "I", "I;16", "I;16B", "F", "LA", "La"})

_PIL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def _load_pil(path: Path) -> LoadedImage:
    image = Image.open(path)
    if image.mode in _SINGLE_CHANNEL_MODES:
        return LoadedImage(np.asarray(image.convert("L"))[np.newaxis])
    return LoadedImage(np.asarray(image.convert("RGB"))[np.newaxis])


def to_luma(raw: np.ndarray) -> np.ndarray:
    """(D, H, W, 3) -> (D, H, W) float32; a (D, H, W) array passes through.

    The weights are PIL's, so a colour file and the same picture stored as
    greyscale produce the same intensities (up to PIL's truncation).
    """
    if raw.ndim == 3:
        return raw
    weights = np.asarray(_LUMA_WEIGHTS, dtype=np.float32)
    return (raw.astype(np.float32) * weights).sum(axis=-1)


def collapse_identical_channels(raw: np.ndarray, *, source: str) -> np.ndarray:
    """(D, H, W, 3) -> (D, H, W) when all channels agree; raise otherwise.

    A greyscale mask stored as RGB is common and harmless. A genuinely
    coloured label map is not: mixing its channels would invent labels that
    were never there, and the segmentation metrics would report plausible
    but wrong numbers.
    """
    if raw.ndim == 3:
        return raw
    first = raw[..., :1]
    if not np.array_equal(raw, np.broadcast_to(first, raw.shape)):
        raise ValueError(
            f"[{source}] this is a colour image, and a mask or instance map must "
            "be unambiguous. Supply it as a single-channel file (a 0/1 or 0/255 "
            "mask, or an integer label map) instead of a coloured rendering."
        )
    return raw[..., 0]


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
    components = image.GetNumberOfComponentsPerPixel()
    if components > 1:
        raise ValueError(
            f"[{path.name}] this file holds {components} components per pixel. "
            "Colour/vector images are supported for PNG and JPEG only; extract "
            "the component you want to score."
        )
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
    `load_pair()`, which puts the input on the target's scale when the
    strategy produces one.
    """

    def __init__(self, path: Path, normalizer: Normalizer = MinMax()):
        self.path = path
        self.suffix = canonical_suffix(path)
        if self.suffix not in _LOADERS:
            raise ValueError(f"Unsupported format: {path}")
        self.normalizer = normalizer
        self._loaded: Optional[LoadedImage] = None
        self._raw: Optional[np.ndarray] = None
        self._tensor: Optional[torch.Tensor] = None
        self._intensity_range: Optional[IntensityRange] = None
        self._intensity_range_known = False
        self._force_gray = False

    @property
    def _image(self) -> LoadedImage:
        if self._loaded is None:
            self._loaded = _LOADERS[self.suffix](self.path)
        return self._loaded

    @property
    def raw(self) -> np.ndarray:
        """The decoded array: (D, H, W), or (D, H, W, 3) for a colour file."""
        if self._raw is None:
            raw = self._image.raw
            # Backstop for decoders: `_load_sitk` already rejects vector
            # images at the source, and DICOM/NIfTI never produce a channel
            # axis, so no real file reaches this branch today. Kept in case a
            # future or misbehaving decoder returns one anyway.
            if raw.ndim == 4 and self.suffix not in _PIL_SUFFIXES:
                raise ValueError(
                    f"[{self.path.name}] this format must decode to a single channel, "
                    f"but the decoder returned {raw.shape[-1]} of them. Colour is "
                    "supported for PNG and JPEG only; convert the file or extract "
                    "the component you want to score."
                )
            # Mask collapse runs before the forced luma so a genuinely
            # coloured mask is always refused, even if the normalizer were
            # changed to Mask()/Raw() after force_gray() was called. Once
            # collapsed, the array is (D, H, W), so the luma step below is a
            # no-op for it (to_luma passes ndim == 3 through unchanged).
            if isinstance(self.normalizer, (Mask, Raw)):
                raw = collapse_identical_channels(raw, source=self.path.name)
            if raw.ndim == 4 and self._force_gray:
                raw = to_luma(raw)
            self._raw = raw
        return self._raw

    @property
    def channels(self) -> int:
        """1 for a greyscale image, 3 for a colour one."""
        return 1 if self.raw.ndim == 3 else 3

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
        """(D, C, H, W) with C in {1, 3}. float32 in [0, 1] under every strategy
        except `Raw()` (dtype preserved, unscaled); exactly {0.0, 1.0} under
        `Mask()`. Colour files keep their three channels — what a metric does
        with them is decided by `MetricSpec.channels` and by the metric itself.
        The returned tensor may share storage with this loader's cache (as do
        `.rgb_tensor` and `.gray_tensor`, which are derived from it) — callers
        must not mutate it in place."""
        if self._tensor is None:
            scaled = self.normalizer.apply(self.raw, source=self.path.name)
            self._tensor = (
                scaled.unsqueeze(1) if scaled.ndim == 3 else scaled.permute(0, 3, 1, 2)
            )
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
        """(D, 3, H, W) — colour as it is, greyscale replicated three times."""
        tensor = self.tensor
        return tensor if tensor.shape[1] == 3 else tensor.expand(-1, 3, -1, -1)

    @property
    def gray_tensor(self) -> torch.Tensor:
        """(D, 1, H, W) — colour reduced to luma, greyscale as it is.

        The reduction happens on `.tensor`, i.e. after scaling, not on `.raw`
        before it. Because the luma weights sum to one, this is equivalent to
        reducing first and then scaling — the intensity range still comes
        from the whole colour image, exactly as `.tensor` documents. One
        exception: `scale()` clips to [0, 1] before this property runs, so a
        channel that was clipped shifts the resulting luma slightly, same as
        it would shift any other per-channel computation on `.tensor`."""
        tensor = self.tensor
        if tensor.shape[1] == 1:
            return tensor
        weights = torch.tensor(
            _LUMA_WEIGHTS, dtype=tensor.dtype, device=tensor.device
        ).view(1, 3, 1, 1)
        return (tensor * weights).sum(dim=1, keepdim=True)

    @property
    def empty_slice_mask(self) -> torch.Tensor:
        """True for slices with nothing to score, as the strategy defines it.

        Intensity strategies use `normalization.sparse_slices` (a spread/lift
        heuristic on the raw data); `Mask()` counts foreground voxels exactly.
        Colour is reduced to luma first, so the heuristic keeps the thresholds
        it was calibrated with.
        """
        return self.normalizer.empty_slices(to_luma(self.raw))

    def log_tensor_shape(self) -> torch.Size:
        shape = self.tensor.shape
        print(f"[{self.path.name}] tensor size: {tuple(shape)}")
        return shape

    def force_gray(self) -> None:
        """Compare this image on luma, whatever the file holds.

        Used by `load_pair` when only one side of a pair carries colour. Must
        run before anything is derived from the raw data, because the range
        and the tensor would otherwise be built from the colour version.
        """
        if isinstance(self.normalizer, (Mask, Raw)):
            raise RuntimeError(
                "force_gray() is not valid for masks or instance maps: mixing "
                "their channels would invent labels. Supply a single-channel file."
            )
        if self._tensor is not None or self._intensity_range_known:
            raise RuntimeError(
                "force_gray() must be called before the tensor or the intensity "
                "range is built"
            )
        self._force_gray = True
        self._raw = None


def load_pair(
    input_path: Path,
    target_path: Path,
    normalizer: Normalizer = MinMax(),
) -> tuple[ImageLoader, ImageLoader]:
    """Load a full-reference pair on ONE scale — the target's — when there is one.

    The target is scaled by `normalizer`; if that produced a range, the input
    is scaled by the same range, so a prediction that is uniformly too bright
    or too flat is measured as such instead of being normalized away
    (fastMRI's convention: `data_range` comes from the reference). Anything
    outside the target's range is clipped to [0, 1] in the input; the raw
    extremes remain visible as `ImageLoader.raw_range`.

    Under `Mask()` and `Raw()` the target yields no range and both sides are
    loaded with `normalizer` itself, independently — a 0/1 prediction against
    a 0/255 reference binarizes to the same tensor.

    When only one side carries colour, both are compared on luma: the usual
    cause is a format difference (the same greys stored as RGB), not a colour
    error, and replicating the greyscale side instead would score JPEG's
    channel drift as a colour deviation. The switch happens before any
    scaling, so the target's range is read from the same numbers that are
    compared later.

    Returns `(input, target)`.
    """
    input_image = ImageLoader(input_path, normalizer)
    target = ImageLoader(target_path, normalizer)

    if input_image.channels != target.channels:
        print(
            f"[{input_path.name} vs {target_path.name}] one image is colour and the "
            "other greyscale — both are compared on luma (one channel)."
        )
        input_image.force_gray()
        target.force_gray()

    rng = target.intensity_range
    if rng is not None:
        input_image.normalizer = FixedRange(rng, name=normalizer.name)
    return input_image, target
