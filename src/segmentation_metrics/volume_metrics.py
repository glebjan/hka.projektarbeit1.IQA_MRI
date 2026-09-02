"""Volumetric Similarity and voxel counts as registrable metrics.

`volume.py` stays pure NumPy with no framework imports; the MetricSpec wrappers
live here, which also keeps it out of the circular-import dance the other
segmentation modules need.

VS is registered for BOTH scoring modes on purpose. Per-slice VS is not a broken
measurement, it is a complementary one: if every slice has the right area the
total volume is right, but a right total volume does not imply right per-slice
areas. A model that moves mass from one slice to the next leaves the totals
untouched, so volume VS reads 1.0 while the structure is smeared along the
depth axis — and per-slice VS shows it.

Note that VS compares sizes, not overlap: two masks of equal size that do not
touch anywhere still score 1.0. It is only meaningful beside dice.

`v_pred`, `v_gt` and `tp` are raw voxel counts, not quality scores, so their
direction is "not_ranked". They exist so that a per-slice run can be aggregated
to correct volume-level numbers by `EvaluationResult.aggregate_volumes()` —
summing counts and then taking the ratio, never averaging ratios.

Usage: import `metrics` before this module, same as the other segmentation
metric modules.
"""

from typing import Callable, Optional

import numpy as np
import torch

from metrics import MetricSpec, ModeSupport
from segmentation_metrics.volume import tp as _tp
from segmentation_metrics.volume import v_gt as _v_gt
from segmentation_metrics.volume import v_pred as _v_pred
from segmentation_metrics.volume import vs as _vs
from segmentation_metrics.volume import vs_signed as _vs_signed


class VolumeFunctionMetric:
    """Adapter turning a `volume.py` function into a Metric.

    The wrapped functions take two NumPy arrays of any matching shape, so the
    same adapter serves both scoring modes: in slice mode it sees `(N, C, H, W)`
    and scores each sample's channel 0; in volume mode it sees `(1, C, D, H, W)`
    and scores the whole `(D, H, W)` body at once.

    Args:
        fn: one of `vs`, `vs_signed`, `v_pred`, `v_gt`, `tp`.
        threshold: binarization cutoff for float masks (`value >= threshold`).
            ImageLoader tensors are floats in [0, 1], so a cutoff always applies.
    """

    def __init__(self, fn: Callable[..., float], *, threshold: float = 0.5):
        self._fn        = fn
        self._threshold = threshold

    def __call__(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> list[Optional[float]]:
        if target is None:
            raise ValueError(
                f"'{self._fn.__name__}' compares two masks and requires a target mask"
            )
        pred = input.detach().cpu().numpy()
        gt   = target.detach().cpu().numpy()
        scores: list[Optional[float]] = []
        for i in range(pred.shape[0]):
            value = self._fn(pred[i, 0], gt[i, 0], threshold=self._threshold)
            scores.append(None if np.isnan(value) else float(value))
        return scores


def _both_modes(fn: Callable[..., float], threshold: float) -> tuple[ModeSupport, ModeSupport]:
    """Slice and volume capability for a function that is already shape-agnostic."""
    metric = VolumeFunctionMetric(fn, threshold=threshold)
    return ModeSupport(lambda: metric), ModeSupport(lambda spacing: metric)


def vs_metric(*, threshold: float = 0.5) -> MetricSpec:
    """Volumetric Similarity (Taha & Hanbury): 1 - |Vp - Vg| / (Vp + Vg).

    Range [0, 1]; 1.0 means the two masks have the same size. Spacing-invariant:
    the voxel volume cancels in numerator and denominator.
    """
    slice_mode, volume_mode = _both_modes(_vs, threshold)
    return MetricSpec(
        name="vs", direction="higher_is_better", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            "Volumetric Similarity: how closely the predicted and reference "
            "masks agree in size (1.0 = same size). It is not an overlap "
            "measure — two masks of equal size that never touch also score 1.0 "
            "— so read it beside dice. Domain-agnostic and independent of voxel "
            "size."
        ),
        domain="",
    )


def vs_signed_metric(*, threshold: float = 0.5) -> MetricSpec:
    """Signed Volumetric Similarity (SimpleITK convention), range [-2, 2].

    Negative means the prediction is smaller than the reference
    (undersegmentation), positive means larger (oversegmentation).
    """
    slice_mode, volume_mode = _both_modes(_vs_signed, threshold)
    return MetricSpec(
        name="vs_signed", direction="not_ranked", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            "Signed Volumetric Similarity: the direction of the size error "
            "(negative = the prediction is too small, positive = too large, "
            "0.0 = the sizes match). Domain-agnostic and independent of voxel size."
        ),
        domain="",
    )


def _count_metric(name: str, fn: Callable[..., float], what: str, threshold: float) -> MetricSpec:
    slice_mode, volume_mode = _both_modes(fn, threshold)
    return MetricSpec(
        name=name, direction="not_ranked", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            f"Raw voxel count: {what}. Not a quality score — it is reported so "
            "that per-slice runs can be summed into correct volume-level dice "
            "and volumetric similarity."
        ),
        domain="",
    )


def v_pred_metric(*, threshold: float = 0.5) -> MetricSpec:
    return _count_metric("v_pred", _v_pred, "voxels marked in the prediction", threshold)


def v_gt_metric(*, threshold: float = 0.5) -> MetricSpec:
    return _count_metric("v_gt", _v_gt, "voxels marked in the reference", threshold)


def tp_metric(*, threshold: float = 0.5) -> MetricSpec:
    return _count_metric("tp", _tp, "voxels marked in both masks", threshold)


VS        = vs_metric()
VS_SIGNED = vs_signed_metric()
V_PRED    = v_pred_metric()
V_GT      = v_gt_metric()
TP        = tp_metric()

VOLUME_METRICS = (VS, VS_SIGNED, V_PRED, V_GT, TP)
