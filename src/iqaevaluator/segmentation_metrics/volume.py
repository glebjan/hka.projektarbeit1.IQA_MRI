"""Stateless volume-based segmentation metrics (Dice, VS, AVD).

Operate on binary/label masks per 2D/3D slice; aggregate per-patient via
`aggregate_patient`. No shared state between calls.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch


def as_mask(x: np.ndarray, label: Optional[int] = None, threshold: float = 0.5) -> np.ndarray:
    """Binarize an array to a boolean mask — the framework's single binarization rule.

    `normalization.Mask()` calls this at load time; the numpy metrics below
    call it again on whatever they are handed (a no-op on the {0, 1} floats
    `Mask()` produces).

    - bool array: returned unchanged.
    - float array: values must lie in [0, 1] (probabilities); thresholded
      at `threshold`. Values outside [0, 1] raise ValueError — this usually
      means raw logits were passed without a sigmoid activation.
    - integer array (label map): every non-zero value is foreground when
      `label is None`; otherwise one-vs-rest via `x == label`. A 0/255 PNG
      mask and a 0/1 NIfTI mask therefore binarize identically.
    """
    if x.dtype == bool:
        return x
    if np.issubdtype(x.dtype, np.floating):
        x_min, x_max = float(x.min()), float(x.max())
        if x_min < 0.0 or x_max > 1.0:
            raise ValueError(
                f"as_mask expects float values in [0, 1], got range "
                f"[{x_min}, {x_max}]. Did you forget to apply a sigmoid "
                "before binarizing?"
            )
        return x >= threshold
    if np.issubdtype(x.dtype, np.integer):
        return x != 0 if label is None else x == label
    raise TypeError(f"as_mask does not support dtype {x.dtype}")


# ---------------------------------------------------------------------------
# Shared helpers for the metric adapters (monai_metrics, volume_metrics,
# boundary_iou). Every adapter validates its input and applies the empty-mask
# policy the same way; the numbers below never come from a metric backend.
# ---------------------------------------------------------------------------

NOT_EMPTY = object()
"""`empty_policy`'s answer when both masks have foreground: compute the metric."""


def require_binary(t: torch.Tensor, *, metric: str) -> None:
    """Raise unless every value of `t` is 0 or 1 — the contract `Mask()` fulfils.

    A probability map loaded with `MinMax()`, or a label map loaded with
    `Raw()`, would otherwise be scored on whatever the backend makes of it.
    """
    if not bool(((t == 0) | (t == 1)).all()):
        raise ValueError(
            f"{metric} expects a binary mask with values in {{0, 1}}; "
            "load masks with normalization.Mask()"
        )


def foreground_counts(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample foreground voxel counts of `(N, C, *spatial)` batches."""
    dims = tuple(range(1, y_pred.dim()))
    return (y_pred != 0).sum(dim=dims), (y != 0).sum(dim=dims)


def empty_policy(n_pred: int, n_gt: int, *, one_sided: Optional[float]):
    """The empty-mask policy, one place for every metric.

    Both masks empty: the score is undefined → `None`. Exactly one empty:
    the metric's `one_sided` value — 0.0 for overlap-type scores (Dice, VS,
    NSD, PQ, Boundary IoU), `None` for distances (HD95, ASSD), which have no
    finite value there. Otherwise `NOT_EMPTY`: compute the metric.
    """
    if n_pred == 0 and n_gt == 0:
        return None
    if n_pred == 0 or n_gt == 0:
        return one_sided
    return NOT_EMPTY


def _check_shapes(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt shapes must match, got pred={pred.shape}, "
            f"gt={gt.shape}"
        )


def v_pred(pred: np.ndarray, gt: np.ndarray, *, label: Optional[int] = None, threshold: float = 0.5) -> float:
    """Count of positive voxels in the prediction mask."""
    _check_shapes(pred, gt)
    return float(as_mask(pred, label, threshold).sum())


def v_gt(pred: np.ndarray, gt: np.ndarray, *, label: Optional[int] = None, threshold: float = 0.5) -> float:
    """Count of positive voxels in the reference (ground-truth) mask."""
    _check_shapes(pred, gt)
    return float(as_mask(gt, label, threshold).sum())


def tp(pred: np.ndarray, gt: np.ndarray, *, label: Optional[int] = None, threshold: float = 0.5) -> float:
    """Count of voxels positive in both prediction and reference masks."""
    _check_shapes(pred, gt)
    pred_mask = as_mask(pred, label, threshold)
    gt_mask = as_mask(gt, label, threshold)
    return float((pred_mask & gt_mask).sum())


def vs(pred: np.ndarray, gt: np.ndarray, *, label: Optional[int] = None, threshold: float = 0.5) -> float:
    """Volumetric Similarity, Taha & Hanbury (2015) convention.

    Range [0, 1]; 1 = identical volume. NOT an overlap measure — two
    disjoint masks of equal size score 1.0. NaN when both masks are empty.
    """
    vp = v_pred(pred, gt, label=label, threshold=threshold)
    vg = v_gt(pred, gt, label=label, threshold=threshold)
    denom = vp + vg
    if denom == 0:
        return float("nan")
    return 1.0 - abs(vp - vg) / denom


def vs_signed(pred: np.ndarray, gt: np.ndarray, *, label: Optional[int] = None, threshold: float = 0.5) -> float:
    """Signed Volumetric Similarity, SimpleITK convention.

    Range [-2, 2]; negative = undersegmentation (prediction volume smaller
    than reference), positive = oversegmentation. NaN when both masks are
    empty. Identity: vs == 1 - abs(vs_signed) / 2.
    """
    vp = v_pred(pred, gt, label=label, threshold=threshold)
    vg = v_gt(pred, gt, label=label, threshold=threshold)
    denom = vp + vg
    if denom == 0:
        return float("nan")
    return 2.0 * (vp - vg) / denom


def aggregate_patient(
    df: pd.DataFrame, group_col: str | None = "patient_id"
) -> pd.DataFrame:
    """Aggregate per-slice voxel counts to per-patient volume metrics.

    `df` must have columns `v_pred`, `v_gt`, `tp` — one row per slice, as
    produced by calling `v_pred`/`v_gt`/`tp` per slice. Sums voxel counts
    across all slices of a group BEFORE computing ratio metrics: the mean
    of per-slice ratios is not the same as the ratio of the summed volume,
    and this function always does the latter.

    `group_col=None` treats the entire DataFrame as a single patient.

    Returns a DataFrame indexed by group (or a single row of index 0 when
    `group_col=None`) with columns v_pred, v_gt, tp, vs, vs_signed, dice,
    avd_voxels. Ratio metrics are NaN when their denominator is 0.

    This is the correct way to get volume-level numbers out of a per-slice run:
    it sums the counts before dividing, so it never averages ratios. A run with
    `mode="volume"` computes the same quantities directly;
    `EvaluationResult.aggregate_volumes()` wraps this function for the per-slice
    case.
    """
    if group_col is None:
        grouped = df[["v_pred", "v_gt", "tp"]].sum(skipna=True).to_frame().T
    else:
        grouped = df.groupby(group_col)[["v_pred", "v_gt", "tp"]].sum(skipna=True)

    v_pred_sum = grouped["v_pred"].astype(float)
    v_gt_sum = grouped["v_gt"].astype(float)
    tp_sum = grouped["tp"].astype(float)
    denom = v_pred_sum + v_gt_sum

    grouped["vs"] = np.where(
        denom == 0, np.nan, 1.0 - (v_pred_sum - v_gt_sum).abs() / denom
    )
    grouped["vs_signed"] = np.where(
        denom == 0, np.nan, 2.0 * (v_pred_sum - v_gt_sum) / denom
    )
    grouped["dice"] = np.where(denom == 0, np.nan, 2.0 * tp_sum / denom)
    grouped["avd_voxels"] = v_pred_sum - v_gt_sum

    return grouped
