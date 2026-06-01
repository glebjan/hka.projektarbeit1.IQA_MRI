from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pyiqa
import SimpleITK as sitk
import torch
from PIL import Image

from constants import INPUT, TARGET


# ---------- tensor helper ----------

def _to_tensor(arr_dhw: np.ndarray) -> torch.Tensor:
    """(D, H, W) ndarray -> (D, 1, H, W) float32 tensor min-max normalized to [0, 1]."""
    arr = arr_dhw.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(1)


# ---------- format loaders ----------

def _load_pil(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)[None, ...]  # (1, H, W)
    return _to_tensor(arr)


def _select_dicom_frames(arr: np.ndarray, photometric: str) -> np.ndarray:
    """Reduce pydicom pixel_array to (D, H, W). RGB planes collapsed to luminance."""
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and photometric.startswith("RGB"):
            rgb = arr[..., :3].astype(np.float32)
            lum = (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2])
            return lum[None, ...]
        return arr  # (D, H, W) multi-frame stack

    raise ValueError(f"Unsupported DICOM pixel_array shape {arr.shape}")


def _load_dicom(path: Path) -> torch.Tensor:
    ds = pydicom.dcmread(str(path))
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    arr = _select_dicom_frames(ds.pixel_array, photometric).astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    return _to_tensor(arr)


def _load_nifti(path: Path) -> torch.Tensor:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata()  # (H, W, D) or (H, W, D, T)
    if data.ndim == 3:
        arr = np.transpose(data, (2, 0, 1))  # (D, H, W)
    elif data.ndim == 4:
        # (H, W, D, T) -> (T, D, H, W) -> flatten to (T*D, H, W)
        arr = np.transpose(data, (3, 2, 0, 1)).reshape(-1, data.shape[0], data.shape[1])
    else:
        raise ValueError(f"Unsupported NIfTI ndim {data.ndim} for {path}")
    return _to_tensor(arr)


def _load_sitk(path: Path) -> torch.Tensor:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # already (D, H, W) for 3D
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported SimpleITK array shape {arr.shape} for {path}")
    return _to_tensor(arr)


_LOADERS: dict[str, Callable[[Path], torch.Tensor]] = {
    ".png": _load_pil,
    ".jpg": _load_pil,
    ".jpeg": _load_pil,
    ".dcm": _load_dicom,
    ".nii": _load_nifti,
    ".nrrd": _load_sitk,
    ".mha": _load_sitk,
    ".mhd": _load_sitk,
}


def _normalized_suffix(path: Path) -> str:
    """Return the dispatch suffix, collapsing .nii.gz -> .nii."""
    if path.name.lower().endswith(".nii.gz"):
        return ".nii"
    return path.suffix.lower()


def _is_supported(path: Path) -> bool:
    return _normalized_suffix(path) in _LOADERS


# ---------- ImageHelper ----------

class ImageHelper:
    """Suffix-dispatched lazy loader. Holds (D, 1, H, W) grayscale tensor in [0, 1]."""

    def __init__(self, path: Path):
        self.path = path
        self.suffix = _normalized_suffix(path)
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
    def empty_mask(self) -> torch.Tensor:
        t = self.tensor.squeeze(1)  # (D, H, W)
        return (t.mean(dim=(1, 2)) < 1e-3) | (t.std(dim=(1, 2)) < 1e-3)


# ---------- metric cache ----------

_METRIC_CACHE: dict[str, torch.nn.Module] = {}


def _get_metric(name: str) -> torch.nn.Module:
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = pyiqa.create_metric(name, as_loss=False)
    return _METRIC_CACHE[name]


# ---------- record ----------

@dataclass
class ImageEvaluatorRecord:
    image_id: str
    source_model: Optional[str] = None
    mode: str = "no_reference"  # "full_reference" | "no_reference"
    slice_index: int = 0
    is_empty: bool = False

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
        input_img: ImageHelper,
        target_img: Optional[ImageHelper],
        source_model: Optional[str] = None,
    ):
        self.input = input_img
        self.target = target_img
        self.source_model = source_model

        if self.target is not None and self.input.tensor.shape != self.target.tensor.shape:
            raise ValueError(
                f"shape mismatch: input {tuple(self.input.tensor.shape)} "
                f"vs target {tuple(self.target.tensor.shape)}"
            )

    # ----- metric methods (operate on a single slice index) -----

    # // Full-Reference
    def __psnr(self, i: int) -> float:
        return float(_get_metric("psnr")(self.input.tensor[i:i+1], self.target.tensor[i:i+1]).item())

    def __ssim(self, i: int) -> float:
        return float(_get_metric("ssim")(self.input.tensor[i:i+1], self.target.tensor[i:i+1]).item())

    def __lpips(self, i: int) -> float:
        return float(_get_metric("lpips")(self.input.rgb_tensor[i:i+1], self.target.rgb_tensor[i:i+1]).item())

    def __dists(self, i: int) -> float:
        return float(_get_metric("dists")(self.input.rgb_tensor[i:i+1], self.target.rgb_tensor[i:i+1]).item())

    # // No-Reference

    def __clipiqa(self, i: int) -> float:
        return float(_get_metric("clipiqa")(self.input.rgb_tensor[i:i+1]).item())

    def __brisque(self, i: int) -> float:
        return float(_get_metric("brisque")(self.input.rgb_tensor[i:i+1]).item())

    def __niqe(self, i: int) -> float:
        return float(_get_metric("niqe")(self.input.rgb_tensor[i:i+1]).item())

    # ----- orchestration -----

    def _safe_run(self, name: str, fn: Callable[[], float]) -> Optional[float]:
        try:
            return fn()
        except Exception as exc:
            print(f"[{self.input.path}] metric '{name}' failed: {exc}")
            return None

    def _slice_id(self, i: int) -> str:
        stem = self.input.path.name.split(".")[0]
        base = f"{stem}_s{i:03d}"
        return f"{self.source_model}/{base}" if self.source_model else base

    def run_evaluation(self) -> list[ImageEvaluatorRecord]:
        records: list[ImageEvaluatorRecord] = []
        empty = self.input.empty_mask
        D = self.input.tensor.shape[0]
        has_target = self.target is not None

        for i in range(D):
            rec = ImageEvaluatorRecord(
                image_id=self._slice_id(i),
                source_model=self.source_model,
                mode="full_reference" if has_target else "no_reference",
                slice_index=i,
                is_empty=bool(empty[i].item()),
            )
            if not rec.is_empty:
                rec.clipiqa = self._safe_run("clipiqa", lambda i=i: self.__clipiqa(i))
                rec.brisque = self._safe_run("brisque", lambda i=i: self.__brisque(i))
                rec.niqe = self._safe_run("niqe", lambda i=i: self.__niqe(i))
                if has_target:
                    rec.psnr = self._safe_run("psnr", lambda i=i: self.__psnr(i))
                    rec.ssim = self._safe_run("ssim", lambda i=i: self.__ssim(i))
                    rec.lpips = self._safe_run("lpips", lambda i=i: self.__lpips(i))
                    rec.dists = self._safe_run("dists", lambda i=i: self.__dists(i))
            records.append(rec)

        # TODO: Segmentation

        return records


# ---------- dataset driver ----------

def evaluate_dataset(report_path: Path) -> pd.DataFrame:
    rows: list[dict] = []

    # ---- previous directory-walking implementation (commented out) ----
    # if not INPUT.is_dir():
    #     print(f"No input directory at {INPUT}")
    #     return pd.DataFrame(columns=list(ImageEvaluatorRecord.__annotations__.keys()))
    #
    # def _process_file(input_path: Path, source_model: Optional[str]):
    #     patient_stem = input_path.relative_to(INPUT).parts[0]
    #     target_candidate = TARGET / (patient_stem + ".nii.gz")
    #     target_path = target_candidate if target_candidate.is_file() else None
    #     try:
    #         in_helper = ImageHelper(input_path)
    #         tgt_helper = ImageHelper(target_path) if target_path else None
    #         records = IQAEvaluator(in_helper, tgt_helper, source_model).run_evaluation()
    #     except Exception as exc:
    #         print(f"[{input_path}] evaluator init/run failed: {exc}")
    #         return
    #     rows.extend(r.to_dict() for r in records)
    #
    # subdirs = sorted(p for p in INPUT.iterdir() if p.is_dir())
    # if subdirs:
    #     for model_dir in subdirs:
    #         source_model = model_dir.name
    #         for input_path in sorted(model_dir.rglob("*")):
    #             if input_path.is_file() and _is_supported(input_path):
    #                 _process_file(input_path, source_model)
    # else:
    #     for input_path in sorted(INPUT.rglob("*")):
    #         if input_path.is_file() and _is_supported(input_path):
    #             _process_file(input_path, None)

    # ---- direct single-file input/target ----
    # INPUT is a direct path to a file; TARGET is a direct path to a file (if it exists).
    if not INPUT.is_file():
        print(f"No input file at {INPUT}")
        return pd.DataFrame(columns=list(ImageEvaluatorRecord.__annotations__.keys()))

    target_path = TARGET if TARGET.is_file() else None
    try:
        in_helper = ImageHelper(INPUT)
        tgt_helper = ImageHelper(target_path) if target_path else None
        records = IQAEvaluator(in_helper, tgt_helper).run_evaluation()
        rows.extend(r.to_dict() for r in records)
    except Exception as exc:
        print(f"[{INPUT}] evaluator init/run failed: {exc}")

    columns = list(ImageEvaluatorRecord.__annotations__.keys())
    report = pd.DataFrame(rows, columns=columns)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    return report


def main():
    report_path = Path("report") / "test_report.csv"
    report = evaluate_dataset(report_path)
    print(report)
    print(f"Report written: {report_path}")


if "__main__" == __name__:
    main()
