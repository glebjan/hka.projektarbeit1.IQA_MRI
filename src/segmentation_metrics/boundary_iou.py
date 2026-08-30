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

import numpy as np
from scipy.ndimage import distance_transform_cdt

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
