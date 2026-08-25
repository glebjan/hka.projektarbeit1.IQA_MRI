"""MONAI-backed segmentation-quality metrics for the IQA metric registry.

These metrics evaluate segmentation masks (pred vs. ground-truth label maps),
not image intensity — they answer "how good is this segmentation?" rather
than "how good is this reconstructed image?". Inputs are expected to be
binary/label mask tensors, e.g. images loaded via `ImageLoader` from mask
files the user already produced with their own segmentation pipeline; this
module performs no segmentation itself.

MONAI's defaults are calibrated for the medical-imaging domain (physical
voxel spacing in millimeters, a background-class convention). Each builder
function's MetricSpec.description below states which parameters must be
adjusted to apply the metric in another domain (e.g. materials science:
different physical units via `spacing`, different class counts via
`class_thresholds`/kwargs).
"""

from typing import Callable, Optional

import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_panoptic_quality,
    compute_surface_dice,
)

from metrics import MetricSpec

DOMAIN_MEDICAL = "medical (MONAI)"


class MonaiSegmentationMetric:
    """Adapter for MONAI's one-hot-batch functional metrics (Dice, HD95, NSD, ASSD).

    All four share the call signature `compute_fn(y_pred, y, **kwargs) -> (N, C)`
    tensor (batch x class channels); this adapter averages across the class
    dimension to produce one score per sample, matching the Metric protocol.
    """

    def __init__(self, compute_fn: Callable[..., torch.Tensor], *, threshold: Optional[float] = None, **monai_kwargs):
        self._compute_fn = compute_fn
        self._threshold  = threshold
        self._kwargs     = monai_kwargs

    def _binarize(self, t: torch.Tensor) -> torch.Tensor:
        return (t > self._threshold).float() if self._threshold is not None else t

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]:
        y_pred = self._binarize(input)
        y      = self._binarize(target)
        scores = self._compute_fn(y_pred=y_pred, y=y, **self._kwargs)  # (N, C)
        per_sample = scores.mean(dim=1)
        return [None if torch.isnan(s) else float(s.item()) for s in per_sample]


def dice_metric(*, threshold: Optional[float] = None, **monai_kwargs) -> MetricSpec:
    """Dice similarity coefficient: 2*|pred ∩ gt| / (|pred| + |gt|), 1.0 = perfect overlap.

    Domain: medical (MONAI). Defaults assume a single foreground mask channel
    (include_background=True, since there is nothing else to include). For
    multi-class segmentation, pass a multi-channel one-hot input and set
    include_background=False to exclude a true background channel.
    """
    monai_kwargs.setdefault("include_background", True)
    metric = MonaiSegmentationMetric(compute_dice, threshold=threshold, **monai_kwargs)
    return MetricSpec(
        name="dice",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        factory=lambda: metric,
        builtin=False,
        description=(
            "Dice similarity coefficient: overlap between predicted and "
            "ground-truth segmentation masks (1.0 = perfect overlap, 0.0 = no "
            "overlap). Domain: medical (MONAI). For other domains (e.g. "
            "materials-science multi-phase segmentation), pass a multi-channel "
            "one-hot mask and set include_background=False to exclude a real "
            "background class."
        ),
        domain=DOMAIN_MEDICAL,
    )


DICE = dice_metric()


def hausdorff95_metric(*, threshold: Optional[float] = None, **monai_kwargs) -> MetricSpec:
    """95th-percentile Hausdorff Distance: worst-case boundary error, robust to outlier voxels.

    Domain: medical (MONAI) — the returned distance is in voxel units unless
    `spacing` is supplied (physical units per voxel, e.g. mm for medical scans
    or µm for materials micrographs). Pass `spacing=<value or per-axis list>`
    to get physically meaningful distances in another domain.
    """
    monai_kwargs.setdefault("include_background", True)
    monai_kwargs.setdefault("percentile", 95)
    metric = MonaiSegmentationMetric(compute_hausdorff_distance, threshold=threshold, **monai_kwargs)
    return MetricSpec(
        name="hausdorff95",
        direction="lower_is_better",
        reference=True,
        channels="gray",
        factory=lambda: metric,
        builtin=False,
        description=(
            "95th-percentile Hausdorff Distance: how far the predicted "
            "segmentation boundary is from the ground-truth boundary in the "
            "worst 5% of cases (lower = closer boundaries). Domain: medical "
            "(MONAI). Distance is in voxel units by default; pass "
            "`spacing=<mm-per-voxel or per-axis list>` for physical units, or "
            "the equivalent voxel size for another domain (e.g. µm for "
            "materials micrographs)."
        ),
        domain=DOMAIN_MEDICAL,
    )


HAUSDORFF95 = hausdorff95_metric()


def normalized_surface_dice_metric(*, threshold: Optional[float] = None, **monai_kwargs) -> MetricSpec:
    """Normalized Surface Dice (NSD): fraction of the predicted/gt boundary within a tolerance distance.

    Domain: medical (MONAI). `class_thresholds` is the tolerance distance per
    class (defaults to `[1.0]`, one voxel) — MONAI requires it and treats it
    in the same units as `spacing`. For another domain, set both
    `class_thresholds` (acceptable boundary error) and `spacing` (physical
    voxel size) to that domain's units and tolerance, and extend
    `class_thresholds` to one entry per class if using multi-class masks.
    """
    monai_kwargs.setdefault("include_background", True)
    monai_kwargs.setdefault("class_thresholds", [1.0])
    metric = MonaiSegmentationMetric(compute_surface_dice, threshold=threshold, **monai_kwargs)
    return MetricSpec(
        name="nsd",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        factory=lambda: metric,
        builtin=False,
        description=(
            "Normalized Surface Dice: fraction of the predicted and "
            "ground-truth boundaries that lie within a tolerance distance of "
            "each other (1.0 = all boundary points within tolerance). Domain: "
            "medical (MONAI). Tolerance is set via `class_thresholds` "
            "(default 1 voxel per class) and interpreted in the units of "
            "`spacing`; for another domain (e.g. materials micrographs) set "
            "both to that domain's physical voxel size and acceptable "
            "boundary error, and add one threshold per class for multi-class "
            "masks."
        ),
        domain=DOMAIN_MEDICAL,
    )


NSD = normalized_surface_dice_metric()


def average_surface_distance_metric(*, threshold: Optional[float] = None, **monai_kwargs) -> MetricSpec:
    """Average Symmetric Surface Distance (ASSD): mean boundary distance in both directions.

    Domain: medical (MONAI). Like HD95, the result is in voxel units unless
    `spacing` is supplied. For another domain, set `spacing` to that domain's
    physical voxel size to get a physically meaningful distance.
    """
    monai_kwargs.setdefault("include_background", True)
    monai_kwargs.setdefault("symmetric", True)
    metric = MonaiSegmentationMetric(compute_average_surface_distance, threshold=threshold, **monai_kwargs)
    return MetricSpec(
        name="assd",
        direction="lower_is_better",
        reference=True,
        channels="gray",
        factory=lambda: metric,
        builtin=False,
        description=(
            "Average Symmetric Surface Distance: mean distance between the "
            "predicted and ground-truth boundaries, averaged in both "
            "directions (lower = closer boundaries on average). Domain: "
            "medical (MONAI). Distance is in voxel units by default; pass "
            "`spacing=<mm-per-voxel or per-axis list>` for physical units, or "
            "the equivalent voxel size for another domain (e.g. µm for "
            "materials micrographs)."
        ),
        domain=DOMAIN_MEDICAL,
    )


ASSD = average_surface_distance_metric()
