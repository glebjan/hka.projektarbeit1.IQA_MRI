from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyiqa
import pydicom
import torch
from PIL import Image
from torchvision import transforms


# ---------- image loading ----------

SUPPORTED_SUFFIXES = {".png", ".dcm"}


def _select_dicom_frame(arr: np.ndarray, photometric: str) -> np.ndarray:
    """Reduce pydicom pixel_array to 2D grayscale or HxWx3 RGB."""
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # RGB plane last: (H, W, 3|4)
        if arr.shape[-1] in (3, 4) and photometric.startswith("RGB"):
            return arr[..., :3]
        # Multi-frame stack: (frames, H, W) -> middle frame
        if arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
            return arr[arr.shape[0] // 2]

    raise ValueError(f"Unsupported DICOM pixel_array shape {arr.shape}")


def _load_dicom(path: Path) -> Image.Image:
    """DICOM -> PIL.Image, windowed via slope/intercept, normalized to uint8."""
    ds = pydicom.dcmread(str(path))
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    arr = _select_dicom_frame(ds.pixel_array, photometric).astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)

    return Image.fromarray(arr.astype(np.uint8))


def load_image(path: Path) -> Image.Image:
    """Dispatch on suffix. Raise on unsupported."""
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        return _load_dicom(path)
    if suffix == ".png":
        return Image.open(path)
    raise ValueError(f"Unsupported image format: {path.suffix}")


# ---------- tensor helpers ----------

_TO_TENSOR = transforms.ToTensor()

_METRIC_CACHE: dict[str, torch.nn.Module] = {}


def _get_metric(name: str) -> torch.nn.Module:
    """Lazy singleton — avoid reloading pyiqa weights per image."""
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = pyiqa.create_metric(name, as_loss=False)
    return _METRIC_CACHE[name]


def pil_to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    """PIL (any mode) -> [1, 3, H, W] float tensor in [0, 1]."""
    return _TO_TENSOR(image.convert("RGB")).unsqueeze(0)


def pil_to_grayscale_tensor(image: Image.Image) -> torch.Tensor:
    """PIL (any mode) -> [1, 1, H, W] float tensor in [0, 1]."""
    return _TO_TENSOR(image.convert("L")).unsqueeze(0)


# ---------- record ----------

@dataclass
class ImageEvaluatorRecord:
    image_id: str
    source_model: Optional[str] = None
    mode: str = "no_reference"  # "full_reference" | "no_reference"

    # --- Full-Reference ---
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    dists: Optional[float] = None

    # --- No-Reference ---
    clipiqa: Optional[float] = None
    brisque: Optional[float] = None
    niqe: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- evaluator ----------

class IQAEvaluator:
    def __init__(
        self,
        input_path: str | Path,
        target_path: Optional[str | Path] = None,
        source_model: Optional[str] = None,
    ):
        self.input_path = Path(input_path)
        self.target_path = Path(target_path) if target_path else None
        self.source_model = source_model

        # Keep source images as PIL — convert per-metric via helpers.
        self.input_img: Image.Image = load_image(self.input_path)
        self.target_img: Optional[Image.Image] = (
            load_image(self.target_path) if self.target_path else None
        )

    # ----- Full-Reference -----

    def __psnr(self) -> float:
        metric = _get_metric("psnr")
        inp = pil_to_grayscale_tensor(self.input_img)
        tgt = pil_to_grayscale_tensor(self.target_img)
        return float(metric(inp, tgt).item())

    def __ssim(self) -> float:
        metric = _get_metric("ssim")
        inp = pil_to_grayscale_tensor(self.input_img)
        tgt = pil_to_grayscale_tensor(self.target_img)
        return float(metric(inp, tgt).item())

    def __lpips(self) -> float:
        metric = _get_metric("lpips")
        inp = pil_to_rgb_tensor(self.input_img)
        tgt = pil_to_rgb_tensor(self.target_img)
        return float(metric(inp, tgt).item())

    def __dists(self) -> float:
        metric = _get_metric("dists")
        inp = pil_to_rgb_tensor(self.input_img)
        tgt = pil_to_rgb_tensor(self.target_img)
        return float(metric(inp, tgt).item())

    # ----- No-Reference -----

    def __clipiqa(self) -> float:
        metric = _get_metric("clipiqa")
        return float(metric(pil_to_rgb_tensor(self.input_img)).item())

    def __brisque(self) -> float:
        metric = _get_metric("brisque")
        return float(metric(pil_to_rgb_tensor(self.input_img)).item())

    def __niqe(self) -> float:
        metric = _get_metric("niqe")
        return float(metric(pil_to_rgb_tensor(self.input_img)).item())

    # ----- orchestration -----

    def _safe_run(self, name: str, fn: Callable[[], float]) -> Optional[float]:
        """Run a metric; on failure log and return None."""
        try:
            return fn()
        except Exception as exc:
            print(f"[{self.input_path}] metric '{name}' failed: {exc}")
            return None

    def run_evaluation(self) -> ImageEvaluatorRecord:
        image_id = (
            f"{self.source_model}/{self.input_path.stem}"
            if self.source_model
            else self.input_path.stem
        )
        record = ImageEvaluatorRecord(
            image_id=image_id,
            source_model=self.source_model,
            mode="full_reference" if self.target_img is not None else "no_reference",
        )

        # NR metrics on every image.
        record.clipiqa = self._safe_run("clipiqa", self.__clipiqa)
        record.brisque = self._safe_run("brisque", self.__brisque)
        record.niqe = self._safe_run("niqe", self.__niqe)

        if self.target_img is not None:
            record.psnr = self._safe_run("psnr", self.__psnr)
            record.ssim = self._safe_run("ssim", self.__ssim)
            record.lpips = self._safe_run("lpips", self.__lpips)
            record.dists = self._safe_run("dists", self.__dists)

        # TODO: Segmentation

        return record


# ---------- dataset driver ----------

def evaluate_dataset(data_root: Path, report_path: Path) -> pd.DataFrame:
    input_root = data_root / "input"
    target_root = data_root / "target"

    columns = list(ImageEvaluatorRecord.__annotations__.keys())
    report = pd.DataFrame(columns=columns)

    if not input_root.is_dir():
        print(f"No input directory at {input_root}")
        report.to_csv(report_path, index=False)
        return report

    for model_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        source_model = model_dir.name
        for input_path in sorted(model_dir.rglob("*")):
            if not input_path.is_file():
                continue
            if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            rel = input_path.relative_to(input_root)
            target_candidate = target_root / rel
            target_path = target_candidate if target_candidate.is_file() else None

            try:
                evaluator = IQAEvaluator(input_path, target_path, source_model)
                record = evaluator.run_evaluation()
            except Exception as exc:
                print(f"[{input_path}] evaluator init/run failed: {exc}")
                continue

            report = pd.concat(
                [report, pd.DataFrame([record.to_dict()])], ignore_index=True
            )
            del record

    report.to_csv(report_path, index=False)
    return report


def main():
    data_root = Path("data")
    report_path = Path("report") / "report.csv"
    report = evaluate_dataset(data_root, report_path)
    print(report)
    print(f"Report written: {report_path}")


if "__main__" == __name__:
    main()
