"""Boundary IoU — boundary-sensitive segmentation overlap (Cheng et al., CVPR 2021).

Mask IoU is biased by object size: the same absolute boundary error scores
better on a large object than on a small one. Boundary IoU removes that bias by
scoring only a thin band along each mask's contour:

    G_d = G \\ erode(G, d)
    P_d = P \\ erode(P, d)
    Boundary IoU = |G_d n P_d| / |G_d u P_d|

`d` is a fraction of the image diagonal (`dilation_ratio`, paper default 0.02),
so the band width is resolution-independent.

Implementation note: erosion by a (2d+1)x(2d+1) square keeps exactly the pixels
whose Chebyshev distance to the nearest background pixel exceeds `d`. So rather
than iterating a 3x3 erosion `d` times (the reference recipe, O(d*H*W)), one
chessboard distance transform gives the band in O(H*W), independent of `d`. The
one-pixel zero pad supplies the background ring that makes an object clipped by
the image border count that clipped edge as boundary, matching the reference.

Reference: Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C. Berg,
Alexander Kirillov. "Boundary IoU: Improving Object-Centric Image Segmentation
Evaluation." CVPR 2021.

In volume mode the band is measured in physical units via a euclidean distance
transform with `sampling=spacing`, because a band of `d` voxels is physically
thicker along a coarse axis — on 1x1x1.2 mm data, 8 mm in-plane against 9.6 mm
through-plane — which makes the metric more forgiving in exactly the direction
where through-plane errors occur. Without spacing it falls back to the paper's
voxel/chessboard band. The 2D path is unchanged and stays bit-identical to
Cheng et al.; a cubic band and a ball-shaped band are not directly comparable,
so slice-mode and volume-mode scores should not be compared with each other.
The physical band is never thinner than the coarsest voxel spacing (see
`band_width`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt, distance_transform_edt

from iqaevaluator.metric_spec import MetricSpec, ModeSupport
from iqaevaluator.segmentation_metrics.volume import as_mask, require_binary

DEFAULT_DILATION_RATIO = 0.02
"""Paper default: boundary band width as a fraction of the image diagonal."""


def band_width(
    shape: tuple[int, ...],
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """Boundary band width for a mask of `shape`.

    Without `spacing` the width is a fraction of the diagonal in voxels,
    rounded to the nearest whole voxel and floored at 1 — the paper's
    definition, and the same value `dilation_pixels` returns; chessboard
    distances are integers, so thresholding at anything else would silently
    shift the band.

    With `spacing` it is a fraction of the diagonal in millimetres, left
    un-rounded, floored at `max(spacing)`: at least one voxel along every
    axis. The nearest background voxel across a face lies exactly one
    spacing away along that axis, so a band thinner than the coarsest
    spacing would contain no face perpendicular to that axis at all — on
    3 mm slices with a 1.2 mm band the whole through-plane boundary
    vanished and a one-slice shift scored better than a one-voxel in-plane
    shift.

    Args:
        shape: the mask's shape, 2D or 3D.
        dilation_ratio: fraction of the diagonal (paper default 0.02).
        spacing: physical size per axis, same length as `shape`.
    """
    extent = np.asarray(shape, dtype=float)
    if spacing is not None:
        spacing_arr = np.asarray(spacing, dtype=float)
        extent = extent * spacing_arr
        return max(float(spacing_arr.max()), dilation_ratio * float(np.linalg.norm(extent)))
    return float(max(1, round(dilation_ratio * float(np.linalg.norm(extent)))))


def dilation_pixels(
    shape: tuple[int, int], dilation_ratio: float = DEFAULT_DILATION_RATIO
) -> int:
    """Boundary band width in whole pixels — the paper's 2D definition.

    `dilation_ratio` of the image diagonal, rounded, with a floor of 1 pixel so
    that small images still get a band.

    Kept for the 2D path and for callers that want an integer pixel count;
    `band_width` is the general form.

    Args:
        shape: (height, width) of the mask.
        dilation_ratio: band width as a fraction of sqrt(H^2 + W^2).
    """
    h, w = shape
    return max(1, int(round(dilation_ratio * float(np.hypot(h, w)))))


def boundary_region(
    mask: np.ndarray,
    dilation: float,
    sampling: Optional[tuple[float, ...]] = None,
) -> np.ndarray:
    """The band of width `dilation` lying just inside `mask`'s contour.

    Without `sampling` this is `mask & ~erode(mask, dilation)` with a
    (2*dilation+1)-cubic structuring element, via one chessboard distance
    transform — Cheng et al.'s definition, and bit-identical to the previous
    2D implementation.

    With `sampling` the distance is euclidean and measured in physical units,
    so the band is a ball of radius `dilation` millimetres rather than a cube
    of `dilation` voxels. On anisotropic data that is the difference between
    a band that is equally thick everywhere and one that is thicker along the
    coarse axis.

    The one-voxel zero pad supplies the background ring that makes an object
    clipped by the array border count that clipped edge as boundary.

    Args:
        mask: 2D or 3D array, coerced to bool.
        dilation: band width, in voxels without `sampling`, in physical units
            with it.
        sampling: physical size per axis, same length as `mask.ndim`.

    Returns:
        Bool array of `mask`'s shape, always a subset of `mask`.
    """
    m = np.asarray(mask, dtype=bool)
    padded = np.pad(m, 1).astype(np.uint8)
    interior = (slice(1, -1),) * m.ndim
    if sampling is None:
        distance = distance_transform_cdt(padded, metric="chessboard")[interior]
    else:
        distance = distance_transform_edt(padded, sampling=sampling)[interior]
    return m & (distance <= dilation)


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    label: Optional[int] = None,
    threshold: float = 0.5,
    spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """Boundary IoU between two masks: IoU restricted to the contour bands.

    Range [0, 1]; 1.0 = the two contours coincide within the band width. Unlike
    mask IoU, the score does not improve just because the object is large.

    Args:
        pred: 2D or 3D predicted mask (bool, float probabilities in [0, 1], or
            an integer label map).
        gt: reference mask, same shape and convention as `pred`.
        dilation_ratio: band width as a fraction of the image diagonal. Larger
            values are more forgiving; at 1.0 the metric equals mask IoU.
        label: for integer label maps, which class to score one-vs-rest; None
            scores every non-zero label.
        threshold: for float masks, the binarization cutoff (`value >= threshold`).
        spacing: physical size per axis. Without it the band is measured in
            voxels (the paper's definition); with it, in physical units.

    Returns:
        The Boundary IoU, or NaN when both contour bands are empty (i.e. both
        masks are empty) and the score is undefined.

    Raises:
        ValueError: if the shapes differ or the inputs are not 2D or 3D.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt shapes must match, got pred={pred.shape}, gt={gt.shape}"
        )
    if pred.ndim not in (2, 3):
        raise ValueError(f"boundary_iou expects 2D or 3D masks, got shape {pred.shape}")

    pred_mask = as_mask(pred, label, threshold)
    gt_mask = as_mask(gt, label, threshold)

    width = band_width(pred_mask.shape, dilation_ratio, spacing)
    pred_band = boundary_region(pred_mask, width, spacing)
    gt_band = boundary_region(gt_mask, width, spacing)

    union = int(np.count_nonzero(pred_band | gt_band))
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(pred_band & gt_band)) / union


class BoundaryIoUMetric:
    """Adapter making `boundary_iou` satisfy the framework's Metric protocol.

    Scores channel 0 of each sample in an `(N, C, H, W)` batch (or the whole
    `(D, H, W)` body of a `(1, C, D, H, W)` volume), looping over the batch,
    and returns one score per sample. Input must be binary — load masks with
    `normalization.Mask()` (`Mask(label=k)` selects one class of a label
    map). Both masks empty → the score is undefined and comes back as `None`;
    one mask empty → 0.0.

    Args:
        dilation_ratio: boundary band width as a fraction of the image diagonal.
        spacing: physical size per axis. Without it (slice mode) the band is
            measured in voxels; with it (volume mode) it is measured in
            physical units, so it stays equally thick along every axis on
            anisotropic data and never thinner than one voxel (`band_width`).
    """

    def __init__(
        self,
        *,
        dilation_ratio: float = DEFAULT_DILATION_RATIO,
        spacing: Optional[tuple[float, ...]] = None,
    ):
        self._dilation_ratio = dilation_ratio
        self._spacing        = spacing

    def __call__(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> list[Optional[float]]:
        if target is None:
            raise ValueError(
                "boundary_iou is a full-reference metric and requires a target mask"
            )
        require_binary(input, metric="boundary_iou")
        require_binary(target, metric="boundary_iou")
        pred = input.detach().cpu().numpy()
        gt   = target.detach().cpu().numpy()
        scores: list[Optional[float]] = []
        for i in range(pred.shape[0]):
            score = boundary_iou(
                pred[i, 0], gt[i, 0],
                dilation_ratio=self._dilation_ratio, spacing=self._spacing,
            )
            scores.append(None if np.isnan(score) else float(score))
        return scores


def boundary_iou_metric(*, dilation_ratio: float = DEFAULT_DILATION_RATIO) -> MetricSpec:
    """Boundary IoU: IoU of the contour bands rather than the whole mask.

    Domain-agnostic: the band width is a fraction of the image diagonal, so
    there is no physical spacing or class count to retune between domains —
    `dilation_ratio` is the single tunable and means the same thing everywhere.
    Input must be binary — load masks with `Mask()`.

    Args:
        dilation_ratio: boundary band width as a fraction of sqrt(H^2 + W^2)
            (default 0.02, the paper's value). Larger is more forgiving; 1.0
            degenerates to plain mask IoU.
    """
    metric = BoundaryIoUMetric(dilation_ratio=dilation_ratio)

    def build_volume_metric(spacing: Optional[tuple[float, ...]]) -> BoundaryIoUMetric:
        if spacing is None:
            print(
                "[WARNING] no voxel size available, so the boundary band is "
                "measured in voxels rather than physical units. A band of the "
                "same voxel width is physically thicker along a coarse axis, "
                "so through-plane boundary errors are judged more leniently "
                "than in-plane ones. Scores stay comparable between images on "
                "the same grid. To get an evenly thick band, use a format that "
                "records the voxel size (NIfTI, NRRD, MHA or DICOM)."
            )
        return BoundaryIoUMetric(dilation_ratio=dilation_ratio, spacing=spacing)

    return MetricSpec(
        name="boundary_iou",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(build_volume_metric),
        builtin=False,
        description=(
            "Boundary IoU: intersection-over-union computed on a thin band "
            "along each mask's contour rather than over the whole mask region "
            "(1.0 = contours coincide). Unlike mask IoU or Dice, the score is "
            "not inflated by object size, so a fixed boundary error costs the "
            "same on a large object as on a small one. Domain-agnostic: the "
            "band width is `dilation_ratio` (default 0.02) of the image "
            "diagonal, so it needs no physical voxel spacing; raise it to "
            "tolerate coarser boundaries. Cheng et al., CVPR 2021."
        ),
        domain="",
    )


BOUNDARY_IOU = boundary_iou_metric()
