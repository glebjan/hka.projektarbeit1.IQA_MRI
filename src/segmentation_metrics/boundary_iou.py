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
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt

from metrics import MetricSpec
from segmentation_metrics.volume import as_mask

DEFAULT_DILATION_RATIO = 0.02
"""Paper default: boundary band width as a fraction of the image diagonal."""


def dilation_pixels(
    shape: tuple[int, int], dilation_ratio: float = DEFAULT_DILATION_RATIO
) -> int:
    """Boundary band width in pixels for an image of `shape`.

    `dilation_ratio` of the image diagonal, rounded, with a floor of 1 pixel so
    that small images still get a band.

    Args:
        shape: (height, width) of the mask.
        dilation_ratio: band width as a fraction of sqrt(H^2 + W^2).
    """
    h, w = shape
    return max(1, int(round(dilation_ratio * float(np.hypot(h, w)))))


def boundary_region(mask: np.ndarray, dilation: int) -> np.ndarray:
    """The `dilation`-pixel-wide band lying just inside `mask`'s contour.

    Equivalent to `mask & ~erode(mask, dilation)` with a (2*dilation+1)-square
    structuring element, computed via one chessboard distance transform.

    Args:
        mask: 2D array, coerced to bool.
        dilation: band width in pixels (see `dilation_pixels`).

    Returns:
        2D bool array, always a subset of `mask`.

    Raises:
        ValueError: if `mask` is not 2D.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"boundary_region expects a 2D mask, got shape {m.shape}")
    padded = np.pad(m, 1).astype(np.uint8)
    distance = distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]
    return m & (distance <= dilation)


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    label: int = 1,
    threshold: float = 0.5,
) -> float:
    """Boundary IoU between two 2D masks: IoU restricted to the contour bands.

    Range [0, 1]; 1.0 = the two contours coincide within the band width. Unlike
    mask IoU, the score does not improve just because the object is large.

    Args:
        pred: 2D predicted mask (bool, float probabilities in [0, 1], or an
            integer label map).
        gt: 2D reference mask, same shape and convention as `pred`.
        dilation_ratio: band width as a fraction of the image diagonal. Larger
            values are more forgiving; at 1.0 the metric equals mask IoU.
        label: for integer label maps, which class to score one-vs-rest.
        threshold: for float masks, the binarization cutoff (`value >= threshold`).

    Returns:
        The Boundary IoU, or NaN when both contour bands are empty (i.e. both
        masks are empty) and the score is undefined.

    Raises:
        ValueError: if the shapes differ or the inputs are not 2D.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt shapes must match, got pred={pred.shape}, gt={gt.shape}"
        )
    if pred.ndim != 2:
        raise ValueError(f"boundary_iou expects 2D masks, got shape {pred.shape}")

    pred_mask = as_mask(pred, label, threshold)
    gt_mask = as_mask(gt, label, threshold)

    dilation = dilation_pixels(pred_mask.shape, dilation_ratio)
    pred_band = boundary_region(pred_mask, dilation)
    gt_band = boundary_region(gt_mask, dilation)

    union = int(np.count_nonzero(pred_band | gt_band))
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(pred_band & gt_band)) / union


class BoundaryIoUMetric:
    """Adapter making `boundary_iou` satisfy the framework's Metric protocol.

    Scores channel 0 of each sample in an `(N, C, H, W)` batch, looping over the
    batch the way `MonaiPanopticQualityMetric` does, and returns one score per
    sample. Undefined scores (both masks empty) come back as `None`.

    Args:
        dilation_ratio: boundary band width as a fraction of the image diagonal.
        threshold: binarization cutoff for float masks (`value >= threshold`).
            Unlike the MONAI adapters there is no "skip binarization" option —
            the band computation needs a boolean array, so a cutoff always
            applies. Masks loaded via `ImageLoader` arrive as exact 0.0/1.0
            floats, for which any cutoff in (0, 1) is equivalent.
        label: for integer label maps, which class to score one-vs-rest.
    """

    def __init__(
        self,
        *,
        dilation_ratio: float = DEFAULT_DILATION_RATIO,
        threshold: float = 0.5,
        label: int = 1,
    ):
        self._dilation_ratio = dilation_ratio
        self._threshold      = threshold
        self._label          = label

    def __call__(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> list[Optional[float]]:
        if target is None:
            raise ValueError(
                "boundary_iou is a full-reference metric and requires a target mask"
            )
        pred = input.detach().cpu().numpy()
        gt   = target.detach().cpu().numpy()
        scores: list[Optional[float]] = []
        for i in range(pred.shape[0]):
            score = boundary_iou(
                pred[i, 0],
                gt[i, 0],
                dilation_ratio=self._dilation_ratio,
                label=self._label,
                threshold=self._threshold,
            )
            scores.append(None if np.isnan(score) else float(score))
        return scores


def boundary_iou_metric(
    *,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    threshold: float = 0.5,
    label: int = 1,
) -> MetricSpec:
    """Boundary IoU: IoU of the contour bands rather than the whole mask.

    Domain-agnostic: the band width is a fraction of the image diagonal, so
    there is no physical spacing or class count to retune between domains —
    `dilation_ratio` is the single tunable and means the same thing everywhere.

    Args:
        dilation_ratio: boundary band width as a fraction of sqrt(H^2 + W^2)
            (default 0.02, the paper's value). Larger is more forgiving; 1.0
            degenerates to plain mask IoU.
        threshold: binarization cutoff for float/probability masks
            (`value >= threshold` -> True).
        label: for integer label maps, which class to score one-vs-rest.
    """
    metric = BoundaryIoUMetric(
        dilation_ratio=dilation_ratio, threshold=threshold, label=label
    )
    return MetricSpec(
        name="boundary_iou",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        factory=lambda: metric,
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
