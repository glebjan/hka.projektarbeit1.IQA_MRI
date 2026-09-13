"""Layer 2: the implementations the community cites, same definitional variant (spec D10).

Every import is local to its function so that cases.py, harness.py and the
hand-value layer work without the [calibration] extras; the tests skip on
ImportError. Variants that differ from MONAI's (spec D10 table) are exposed
only through `divergent_values` for the report, never through an equality
assertion.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from calibration.cases import ISO, CalibrationCase
from iqaevaluator.segmentation_metrics.volume import as_mask

PANOPTICAPI_SHA = "7bb4655548f98f3fedc07bf37e9040a992b054b0"     # master head, 2026-09-13
BOUNDARY_IOU_API_SHA = "37d25586a677b043ed585f10e5c42d4e80176ea9"  # master head, 2026-09-13


def _binary(case: CalibrationCase) -> tuple[np.ndarray, np.ndarray]:
    """The case's masks binarized the way its normalizer would (Mask(label=k) keeps only k)."""
    pred, gt = case.build()
    label = getattr(case.normalizer, "label", None)
    return as_mask(pred, label), as_mask(gt, label)


# --- MedPy: voxel surfaces via cross-structure erosion, same as MONAI ---------

def medpy_dice(pred, gt) -> float:
    from medpy.metric.binary import dc
    return float(dc(pred, gt))


def medpy_assd(pred, gt, spacing) -> float:
    from medpy.metric.binary import assd
    return float(assd(pred, gt, voxelspacing=spacing))


def medpy_asd(pred, gt, spacing) -> float:
    """Directed pred -> gt (MedPy: result -> reference)."""
    from medpy.metric.binary import asd
    return float(asd(pred, gt, voxelspacing=spacing))


def medpy_hd(pred, gt, spacing) -> float:
    from medpy.metric.binary import hd
    return float(hd(pred, gt, voxelspacing=spacing))


def medpy_hd95(pred, gt, spacing) -> float:
    """Divergent variant: P95 over the *concatenated* directed distances."""
    from medpy.metric.binary import hd95
    return float(hd95(pred, gt, voxelspacing=spacing))


# --- surface-distance (DeepMind): area-weighted surface elements — divergent ----

def surface_distance_values(pred, gt, spacing, tolerance_mm: float) -> dict[str, float]:
    import surface_distance as sd
    d = sd.compute_surface_distances(gt, pred, spacing_mm=spacing)
    asd_gt_pred, asd_pred_gt = sd.compute_average_surface_distance(d)
    return {
        "hd95": float(sd.compute_robust_hausdorff(d, 95)),
        "asd_gt_to_pred": float(asd_gt_pred),
        "asd_pred_to_gt": float(asd_pred_gt),
        "nsd": float(sd.compute_surface_dice_at_tolerance(d, tolerance_mm)),
    }


# --- panopticapi: Kirillov et al.'s own PQ ----------------------------------------

def panopticapi_pq(pred_map: np.ndarray, gt_map: np.ndarray, work_dir: Path) -> float:
    """PQ over the "thing" category, background encoded as a stuff segment.

    panopticapi reads RGB PNGs and COCO-panoptic annotations, so both maps are
    written to `work_dir`. Background must be a real segment (its own id,
    category isthing=0) in both maps: left as VOID (0), panopticapi ignores a
    predicted instance lying on VOID instead of counting it as a false
    positive and removes VOID pixels from the matched union — 0.5 instead of
    0.3 on fixture F4.
    """
    from PIL import Image
    from panopticapi.evaluation import pq_compute_single_core
    from panopticapi.utils import id2rgb

    THING, STUFF = 1, 2
    background_id = int(max(pred_map.max(), gt_map.max())) + 1

    def encode(m: np.ndarray) -> np.ndarray:
        out = m.astype(np.uint32).copy()
        out[out == 0] = background_id
        return out

    def segments(m: np.ndarray) -> list[dict]:
        return [
            {"id": int(i), "category_id": STUFF if i == background_id else THING,
             "area": int((m == i).sum()), "iscrowd": 0}
            for i in np.unique(m)
        ]

    gt_dir, pred_dir = work_dir / "pq_gt", work_dir / "pq_pred"
    gt_dir.mkdir(exist_ok=True); pred_dir.mkdir(exist_ok=True)
    g, p = encode(gt_map), encode(pred_map)
    Image.fromarray(id2rgb(g)).save(gt_dir / "a.png")
    Image.fromarray(id2rgb(p)).save(pred_dir / "a.png")
    categories = {
        THING: {"id": THING, "name": "instance", "isthing": 1},
        STUFF: {"id": STUFF, "name": "background", "isthing": 0},
    }
    gt_ann = {"image_id": "a", "file_name": "a.png", "segments_info": segments(g)}
    pred_ann = {"image_id": "a", "file_name": "a.png", "segments_info": segments(p)}
    stat = pq_compute_single_core(0, [(gt_ann, pred_ann)], str(gt_dir), str(pred_dir), categories)
    result, _ = stat.pq_average(categories, isthing=True)
    return float(result["pq"])


# --- boundary-iou-api: vendored (spec D8) ------------------------------------------
#
# Copyright (c) 2021, Bowen Cheng. All rights reserved. BSD-2-Clause.
# Source: https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py
# Copied verbatim (imports moved into the function) because the package's
# setup.py ships only an empty top-level package and cannot be installed as a
# dependency next to a SHA-pinned panopticapi.

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    import cv2
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou_official(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """|G_d ∩ P_d| / |G_d ∪ P_d| with the reference boundary extraction (2D only)."""
    pb = mask_to_boundary(pred.astype(np.uint8), dilation_ratio).astype(bool)
    gb = mask_to_boundary(gt.astype(np.uint8), dilation_ratio).astype(bool)
    union = int((pb | gb).sum())
    return float("nan") if union == 0 else float((pb & gb).sum()) / union


# --- dispatch -----------------------------------------------------------------------

def official_value(case: CalibrationCase, work_dir: Path) -> Optional[float]:
    """Same-variant official value for `case`, or None when none exists (spec D10)."""
    spacing = case.spacing or ISO
    if case.metric == "panoptic_quality":
        pred, gt = case.build()
        return panopticapi_pq(pred, gt, work_dir)
    pred, gt = _binary(case)
    if case.metric == "dice":
        return medpy_dice(pred, gt)
    if case.metric == "assd":
        symmetric = case.metric_kwargs.get("symmetric", True)
        return medpy_assd(pred, gt, spacing) if symmetric else medpy_asd(pred, gt, spacing)
    if case.metric == "hausdorff95":
        if case.metric_kwargs.get("percentile", 95) is None:
            return medpy_hd(pred, gt, spacing)
        return None            # MedPy hd95 and surface-distance use other variants
    if case.metric == "boundary_iou":
        return boundary_iou_official(pred, gt) if pred.ndim == 2 else None   # 3D physical band: framework extension
    return None                # nsd (author implementation is area-weighted), vs, counts


def divergent_values(case: CalibrationCase, work_dir: Path) -> dict[str, float]:
    """Report-only columns: the same metric under a different definitional variant."""
    if case.metric not in ("hausdorff95", "assd", "nsd") or case.kind == "policy":
        return {}
    spacing = case.spacing or ISO
    pred, gt = _binary(case)
    if not pred.any() or not gt.any():
        return {}
    out: dict[str, float] = {}
    tolerance = float(case.metric_kwargs.get("class_thresholds", [1.0])[0])
    try:
        sd = surface_distance_values(pred, gt, spacing, tolerance)
    except ImportError:
        sd = None
    if case.metric == "hausdorff95" and case.metric_kwargs.get("percentile", 95) == 95:
        try:
            out["MedPy hd95 (P95 over concatenated distances)"] = medpy_hd95(pred, gt, spacing)
        except ImportError:
            pass
        if sd:
            out["surface-distance robust_hausdorff(95) (area-weighted)"] = sd["hd95"]
    if case.metric == "assd" and sd:
        out["surface-distance ASD gt→pred (area-weighted)"] = sd["asd_gt_to_pred"]
        out["surface-distance ASD pred→gt (area-weighted)"] = sd["asd_pred_to_gt"]
    if case.metric == "nsd" and sd:
        out[f"surface-distance surface_dice@{tolerance:g} (area-weighted, author impl.)"] = sd["nsd"]
    return out


def reference_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for dist in ("medpy", "surface-distance"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "not installed"
    out["panopticapi"] = f"git {PANOPTICAPI_SHA[:12]}"
    out["boundary-iou-api"] = f"vendored mask_to_boundary, git {BOUNDARY_IOU_API_SHA[:12]}"
    return out
