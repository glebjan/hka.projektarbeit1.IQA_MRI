"""MaskWriter — Otsu segmentation images for the best slice of each metric."""

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

from image_loader import ImageLoader, strip_all_extensions
from records import ImageEvaluatorRecord, best_slice_per_metric

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
    tint_color: tuple[int, int, int] = (255, 0, 0),
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
# MaskWriter
# ---------------------------------------------------------------------------

class MaskWriter:
    """Writes slice / mask / overlay PNGs for the best-scoring slice per metric."""

    def __init__(self, output_dir: Path = MASK_DIR) -> None:
        self.output_dir = output_dir

    def write(
        self,
        input_loader: ImageLoader,
        records: list[ImageEvaluatorRecord],
    ) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem      = strip_all_extensions(input_loader.path)
        segmenter = _active_segmenter()
        volume    = input_loader.tensor[:, 0].numpy()
        saved: list[Path] = []

        for metric, slice_index in best_slice_per_metric(records).items():
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
