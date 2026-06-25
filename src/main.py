import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Literal, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pyiqa
import SimpleITK as sitk
import torch
from PIL import Image
from skimage.filters import threshold_otsu

import radimagenet_lpips   # noqa: F401 — registers RadImageNetLPIPS in pyiqa
import clip_iqa_medical    # noqa: F401 — registers ClipIQALung / ClipIQABrain in pyiqa

from constants import REPORT, RESNET50

BATCH_SIZE = 32  # Slices pro Batch-Call. Bei OOM reduzieren.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Image loading
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


def _canonical_suffix(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return ".nii"
    return path.suffix.lower()


def _is_supported(path: Path) -> bool:
    return _canonical_suffix(path) in _LOADERS


_MIN_MATCH_PREFIX_LENGTH = 4


def _strip_all_extensions(path: Path) -> str:
    return path.name.split(".")[0]


def _shared_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length


def _list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and _is_supported(p))


def _find_matching_target(input_path: Path, targets: list[Path]) -> Optional[Path]:
    input_stem = _strip_all_extensions(input_path)
    best_match: Optional[Path] = None
    longest_prefix = 0
    for candidate in targets:
        length = _shared_prefix_length(input_stem, _strip_all_extensions(candidate))
        if length > longest_prefix:
            best_match, longest_prefix = candidate, length
    return best_match if longest_prefix >= _MIN_MATCH_PREFIX_LENGTH else None


class ImageLoader:
    def __init__(self, path: Path):
        self.path = path
        self.suffix = _canonical_suffix(path)
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


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
# To add a new metric:
#   1. Add a MetricSpec entry to _METRIC_SPECS
#   2. Add the corresponding Optional[float] field to ImageEvaluatorRecord
# ---------------------------------------------------------------------------

_MetricDirection = Literal["higher_is_better", "lower_is_better"]
_MetricChannels  = Literal["gray", "rgb"]


@dataclass(frozen=True)
class MetricSpec:
    """Static description of one IQA metric.

    Attributes:
        name:       pyiqa metric name (also the ImageEvaluatorRecord field name).
        direction:  whether a higher or lower score indicates better quality.
        reference:  True for full-reference metrics (need a target image).
        channels:   "gray" → use ImageLoader.tensor; "rgb" → use ImageLoader.rgb_tensor.
    """
    name:      str
    direction: _MetricDirection
    reference: bool
    channels:  _MetricChannels


_METRIC_SPECS: list[MetricSpec] = [
    # Full-reference metrics
    MetricSpec("psnr",              "higher_is_better", True,  "gray"),
    MetricSpec("ssim",              "higher_is_better", True,  "gray"),
    MetricSpec("lpips",             "lower_is_better",  True,  "rgb"),
    MetricSpec("dists",             "lower_is_better",  True,  "rgb"),
    MetricSpec("radimagenet_lpips", "lower_is_better",  True,  "rgb"),
    # No-reference metrics
    MetricSpec("clipiqa",           "higher_is_better", False, "rgb"),
    MetricSpec("clip_iqa_lung",     "higher_is_better", False, "rgb"),
    MetricSpec("clip_iqa_brain",    "higher_is_better", False, "rgb"),
    MetricSpec("brisque",           "lower_is_better",  False, "rgb"),
    MetricSpec("niqe",              "lower_is_better",  False, "rgb"),
]

# Derived lookup table — kept for backward compatibility with any external code.
_METRIC_DIRECTION: dict[str, _MetricDirection] = {s.name: s.direction for s in _METRIC_SPECS}


# ---------------------------------------------------------------------------
# Metric model cache
# ---------------------------------------------------------------------------

_METRIC_CACHE: dict[str, torch.nn.Module] = {}


def _get_metric(name: str) -> torch.nn.Module:
    if name not in _METRIC_CACHE:
        kwargs: dict = {}
        if name == "radimagenet_lpips":
            kwargs["backbone_path"] = str(RESNET50)
        _METRIC_CACHE[name] = pyiqa.create_metric(name, as_loss=False, device=DEVICE, **kwargs)
    return _METRIC_CACHE[name]


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------

def _segment_otsu(grayscale_slice: np.ndarray) -> np.ndarray:
    if float(grayscale_slice.max()) <= float(grayscale_slice.min()):
        return np.zeros_like(grayscale_slice, dtype=bool)
    return grayscale_slice > threshold_otsu(grayscale_slice)


_SEGMENTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "otsu": _segment_otsu,
}

SEGMENTER = "otsu"
MASK_DIR  = Path("report") / "masks"


def _active_segmenter() -> Callable[[np.ndarray], np.ndarray]:
    return _SEGMENTERS[SEGMENTER]


def _slice_to_uint8(grayscale_float: np.ndarray) -> np.ndarray:
    return (np.clip(grayscale_float, 0.0, 1.0) * 255).astype(np.uint8)


def _mask_to_uint8(binary_mask: np.ndarray) -> np.ndarray:
    return (binary_mask.astype(bool) * 255).astype(np.uint8)


def _apply_color_overlay(
    grayscale_slice: np.ndarray,
    foreground_mask: np.ndarray,
    tint_color: tuple = (255, 0, 0),
    tint_strength: float = 0.4,
) -> np.ndarray:
    rgb = np.stack([_slice_to_uint8(grayscale_slice)] * 3, axis=-1).astype(np.float32)
    foreground = foreground_mask.astype(bool)
    for channel, color_value in enumerate(tint_color):
        rgb[foreground, channel] = (
            rgb[foreground, channel] * (1.0 - tint_strength) + color_value * tint_strength
        )
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class ImageEvaluatorRecord:
    image_id:            str
    source_model:        Optional[str]   = None
    mode:                str             = "no_reference"
    slice_index:         int             = 0
    is_empty:            bool            = False
    # Full-reference metrics (None when no target is available)
    psnr:                Optional[float] = None
    ssim:                Optional[float] = None
    lpips:               Optional[float] = None
    dists:               Optional[float] = None
    radimagenet_lpips:   Optional[float] = None
    # No-reference metrics
    clipiqa:             Optional[float] = None
    clip_iqa_lung:       Optional[float] = None
    clip_iqa_brain:      Optional[float] = None
    brisque:             Optional[float] = None
    niqe:                Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Best-slice helper (module-level so both IQAEvaluator and MaskWriter can use it)
# ---------------------------------------------------------------------------

def _best_slice_per_metric(records: list[ImageEvaluatorRecord]) -> dict[str, int]:
    """Return the slice index that achieves the best score for each metric."""
    value_index_pairs: dict[str, list[tuple[float, int]]] = {
        metric: [] for metric in _METRIC_DIRECTION
    }
    for record in records:
        if record.is_empty:
            continue
        for metric in _METRIC_DIRECTION:
            value = getattr(record, metric, None)
            if value is not None:
                value_index_pairs[metric].append((value, record.slice_index))

    best: dict[str, int] = {}
    for metric, pairs in value_index_pairs.items():
        if not pairs:
            continue
        if _METRIC_DIRECTION[metric] == "higher_is_better":
            _, idx = max(pairs, key=lambda p: p[0])
        else:
            _, idx = min(pairs, key=lambda p: p[0])
        best[metric] = idx
    return best


# ---------------------------------------------------------------------------
# IQAEvaluator — pure metric computation, no file I/O
# ---------------------------------------------------------------------------

class IQAEvaluator:
    """Computes all registered IQA metrics for one input/target image pair.

    The evaluator is intentionally free of file I/O: it only returns
    ImageEvaluatorRecord objects.  Writing results to disk is handled by
    MaskWriter (segmentation images) and EvaluationResult (CSV).
    """

    def __init__(
        self,
        input_image:  ImageLoader,
        target_image: Optional[ImageLoader],
        source_model: Optional[str] = None,
    ):
        self.input        = input_image
        self.target       = target_image
        self.source_model = source_model

        if self.target is not None and self.input.tensor.shape != self.target.tensor.shape:
            raise ValueError(
                f"shape mismatch: input {tuple(self.input.tensor.shape)} "
                f"vs target {tuple(self.target.tensor.shape)}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_tensor_batch(self, img: ImageLoader, channels: _MetricChannels, indices: list[int]) -> torch.Tensor:
        base = img.tensor if channels == "gray" else img.rgb_tensor
        return base[indices]  # (len(indices), C, H, W)

    def _compute_batch(self, spec: MetricSpec, indices: list[int]) -> list[Optional[float]]:
        metric = _get_metric(spec.name)
        inp = self._pick_tensor_batch(self.input, spec.channels, indices).to(DEVICE)
        try:
            if spec.reference:
                ref = self._pick_tensor_batch(self.target, spec.channels, indices).to(DEVICE)
                scores = metric(inp, ref)
            else:
                scores = metric(inp)
            scores = scores.squeeze(-1) if scores.dim() == 2 else scores
            return [float(s.item()) for s in scores]
        except Exception as exc:
            print(f"[{self.input.path}] metric '{spec.name}' batch failed: {exc}")
            return [None] * len(indices)

    def _format_slice_id(self, slice_index: int) -> str:
        base = f"{_strip_all_extensions(self.input.path)}_s{slice_index:03d}"
        return f"{self.source_model}/{base}" if self.source_model else base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation(self) -> list[ImageEvaluatorRecord]:
        """Evaluate all metrics for every slice.  No files are written."""
        D          = self.input.tensor.shape[0]
        empty_mask = self.input.empty_slice_mask
        has_target = self.target is not None
        mode       = "full_reference" if has_target else "no_reference"

        records = [
            ImageEvaluatorRecord(
                image_id=self._format_slice_id(i),
                source_model=self.source_model,
                mode=mode,
                slice_index=i,
                is_empty=bool(empty_mask[i].item()),
            )
            for i in range(D)
        ]

        active = [i for i in range(D) if not records[i].is_empty]

        for spec in _METRIC_SPECS:
            if spec.reference and not has_target:
                continue
            for chunk_start in range(0, len(active), BATCH_SIZE):
                chunk = active[chunk_start : chunk_start + BATCH_SIZE]
                values = self._compute_batch(spec, chunk)
                for idx, value in zip(chunk, values):
                    setattr(records[idx], spec.name, value)

        return records


# ---------------------------------------------------------------------------
# MaskWriter — segmentation images for the best slice of each metric
# ---------------------------------------------------------------------------

class MaskWriter:
    """Writes slice / mask / overlay PNGs for the best-scoring slice per metric."""

    def __init__(self, output_dir: Path = MASK_DIR):
        self.output_dir = output_dir

    def write(
        self,
        input_loader: ImageLoader,
        records: list[ImageEvaluatorRecord],
    ) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem      = _strip_all_extensions(input_loader.path)
        segmenter = _active_segmenter()
        volume    = input_loader.tensor[:, 0].numpy()
        saved: list[Path] = []

        for metric, slice_index in _best_slice_per_metric(records).items():
            gray  = volume[slice_index]
            mask  = segmenter(gray)
            prefix = str(self.output_dir / f"{stem}_{metric}_s{slice_index:03d}")

            slice_path   = Path(prefix + "_slice.png")
            mask_path    = Path(prefix + "_mask.png")
            overlay_path = Path(prefix + "_overlay.png")

            Image.fromarray(_slice_to_uint8(gray),                     mode="L").save(slice_path)
            Image.fromarray(_mask_to_uint8(mask),                      mode="L").save(mask_path)
            Image.fromarray(_apply_color_overlay(gray, mask),          mode="RGB").save(overlay_path)

            saved.extend([slice_path, mask_path, overlay_path])
            print(
                f"[{input_loader.path.name}] {metric} best slice={slice_index}"
                f" -> {slice_path.name}, {mask_path.name}, {overlay_path.name}"
            )

        return saved


# ---------------------------------------------------------------------------
# EvaluationResult — holds computed records, owns all output operations
# ---------------------------------------------------------------------------

@dataclass
class _EvaluatedImage:
    input_path: Path
    records:    list[ImageEvaluatorRecord]


class EvaluationResult:
    """Container for the results of one evaluation run.

    Normal use (notebook, no file I/O):
        result = evaluate()
        df     = result.to_frame()

    Optional report output:
        result.generate_report(Path("report/my_output.csv"))
    """

    def __init__(self, images: list[_EvaluatedImage]):
        self._images = images

    # ------------------------------------------------------------------
    # Pure data access — no file I/O
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return all records as a DataFrame.  Nothing is written to disk."""
        rows    = [record.to_dict() for img in self._images for record in img.records]
        columns = list(ImageEvaluatorRecord.__annotations__.keys())
        return pd.DataFrame(rows, columns=columns)

    # ------------------------------------------------------------------
    # Optional output
    # ------------------------------------------------------------------

    def generate_report(
        self,
        report_path: Path = REPORT,
        mask_dir:    Path = MASK_DIR,
    ) -> pd.DataFrame:
        """Write the CSV report and segmentation mask images, then return the DataFrame.

        Args:
            report_path: Destination for the CSV file.
            mask_dir:    Directory that receives the slice/mask/overlay PNGs.

        Returns:
            The same DataFrame that to_frame() would return.
        """
        df = self.to_frame()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
        print(f"CSV written: {report_path}")

        writer = MaskWriter(mask_dir)
        for img in self._images:
            loader = ImageLoader(img.input_path)
            writer.write(loader, img.records)

        return df


# ---------------------------------------------------------------------------
# Top-level evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    input_path: Path,
    target_path: Optional[Path] = None,
) -> EvaluationResult:
    """Discover input/target images and compute all IQA metrics.

    Args:
        input_path:  Path to an input image file or a directory of images.
        target_path: Optional path to a reference image file or directory.
                     Pass None for a no-reference (NR-only) evaluation.

    No files are written; use EvaluationResult.generate_report() for output.
    """
    evaluated: list[_EvaluatedImage] = []

    def _run_one(inp: Path, tgt: Optional[Path]) -> None:
        try:
            input_loader = ImageLoader(inp)
            input_loader.log_tensor_shape()
            target_loader: Optional[ImageLoader] = None
            if tgt is not None:
                target_loader = ImageLoader(tgt)
                target_loader.log_tensor_shape()
            records = IQAEvaluator(input_loader, target_loader).run_evaluation()
            evaluated.append(_EvaluatedImage(input_path=inp, records=records))
        except Exception as exc:
            print(f"[{inp}] evaluation failed: {exc}")

    if input_path.is_file():
        if target_path is not None and not target_path.is_file():
            print(f"[WARNING] input is a file but target '{target_path}' is not — target ignored.")
        tgt = target_path if (target_path is not None and target_path.is_file()) else None
        _run_one(input_path, tgt)
    elif input_path.is_dir():
        available_targets: list[Path] = []
        if target_path is not None and target_path.is_dir():
            available_targets = _list_images(target_path)
        elif target_path is not None and target_path.is_file():
            available_targets = [target_path]
        elif target_path is not None:
            print(f"[WARNING] target '{target_path}' is neither a file nor a directory — target ignored.")
        input_files = _list_images(input_path)
        if not input_files:
            print(f"[WARNING] No supported image files found in '{input_path}'.")
        matched_targets: set[Path] = set()
        for inp in input_files:
            tgt = _find_matching_target(inp, available_targets)
            if tgt is not None and tgt in matched_targets:
                print(f"[WARNING] '{tgt.name}' matched multiple inputs — last match: '{inp.name}'.")
            if tgt is not None:
                matched_targets.add(tgt)
            _run_one(inp, tgt)
        unmatched = [t for t in available_targets if t not in matched_targets]
        if unmatched:
            print(f"[WARNING] {len(unmatched)} target file(s) had no matching input: "
                  + ", ".join(t.name for t in unmatched))
    else:
        print(f"No input file or directory at {input_path}")

    return EvaluationResult(evaluated)


# ---------------------------------------------------------------------------
# CLI entry point — evaluate and write report (backward-compatible behaviour)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IQA metrics and write a report.")
    parser.add_argument("input", type=Path, help="Input image file or directory.")
    parser.add_argument(
        "target", type=Path, nargs="?", default=None,
        help="Optional reference image file or directory (omit for NR-only evaluation).",
    )
    args = parser.parse_args()
    result = evaluate(args.input, args.target)
    report = result.generate_report(REPORT)
    print(report.describe())
    print(f"Report written: {REPORT}")


if "__main__" == __name__:
    main()
