"""Runs a CalibrationCase through the framework's public path.

The path a user takes is what gets calibrated, not an adapter in isolation:
fixture -> temporary NIfTI (with spacing) -> load_pair(normalizer) ->
MetricRegistry(spec) -> IQAEvaluator / VolumeEvaluator -> record value.
"""
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np

from iqaevaluator.evaluator_factory import build_evaluator
from iqaevaluator.image_loader import load_pair
from iqaevaluator.metric_spec import MetricSpec
from iqaevaluator.metrics import MetricRegistry
from iqaevaluator.records import ImageEvaluatorRecord
from iqaevaluator.segmentation_metrics.boundary_iou import boundary_iou_metric
from iqaevaluator.segmentation_metrics.monai_metrics import (
    average_surface_distance_metric, dice_metric, hausdorff95_metric,
    normalized_surface_dice_metric, panoptic_quality_metric,
)
from iqaevaluator.segmentation_metrics.volume_metrics import (
    tp_metric, v_gt_metric, v_pred_metric, vs_metric,
)

from calibration.cases import ISO, CalibrationCase, Spacing

SPEC_BUILDERS: dict[str, Callable[..., MetricSpec]] = {
    "dice": dice_metric,
    "hausdorff95": hausdorff95_metric,
    "nsd": normalized_surface_dice_metric,
    "assd": average_surface_distance_metric,
    "panoptic_quality": panoptic_quality_metric,
    "boundary_iou": boundary_iou_metric,
    "vs": vs_metric,
    "v_pred": v_pred_metric,
    "v_gt": v_gt_metric,
    "tp": tp_metric,
}


def _storable(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int16)
    return arr.astype(np.float32)


def write_nifti(arr: np.ndarray, path: Path, spacing: Spacing = ISO) -> Path:
    """Write `arr` so that ImageLoader(path).raw == arr and .spacing == spacing.

    The NIfTI decoder transposes (X, Y, Z) -> (Z, X, Y) and reads zooms as
    (dz, dx, dy); so a (D, H, W) array is stored as (H, W, D) with an affine
    of diag(s_h, s_w, s_d). A 2D (H, W) array becomes a single slice.
    """
    if arr.ndim == 2:
        arr = arr[None]
    data = np.ascontiguousarray(np.transpose(_storable(arr), (1, 2, 0)))
    s_d, s_h, s_w = spacing
    nib.save(nib.Nifti1Image(data, np.diag([s_h, s_w, s_d, 1.0])), str(path))
    return path


def framework_records(case: CalibrationCase, tmp_dir: Path) -> list[ImageEvaluatorRecord]:
    pred, gt = case.build()
    spacing = case.spacing or ISO
    pred_path = write_nifti(pred, tmp_dir / f"{case.name}_pred.nii.gz", spacing)
    gt_path = write_nifti(gt, tmp_dir / f"{case.name}_gt.nii.gz", spacing)
    inp, tgt = load_pair(pred_path, gt_path, case.normalizer)
    spec = SPEC_BUILDERS[case.metric](**case.metric_kwargs)
    return build_evaluator(inp, tgt, MetricRegistry(spec), mode=case.mode).run_evaluation()


def framework_value(case: CalibrationCase, tmp_dir: Path) -> Optional[float]:
    """The single record value of a volume-mode or single-slice case."""
    records = framework_records(case, tmp_dir)
    assert len(records) == 1, f"{case.name}: expected one record, got {len(records)}"
    return records[0].extra.get(case.metric)
