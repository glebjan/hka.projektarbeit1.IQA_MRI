"""MONAI-backed segmentation-quality metrics for the IQA metric registry.

These metrics evaluate segmentation masks (pred vs. ground-truth label maps),
not image intensity — they answer "how good is this segmentation?" rather
than "how good is this reconstructed image?". This module performs no
segmentation itself.

Input must be binary — load masks with `normalization.Mask()`
(`ImageLoader(path, Mask())` or `load_pair(..., Mask())`). `Mask()` turns
0/255 PNGs, 0/1 NIfTIs and integer label maps (`Mask(label=k)` for one class)
into float32 {0, 1}; every adapter here checks that and rejects anything
else. The one exception is panoptic quality, which also accepts an integer
instance map loaded with `Raw()`.

Empty masks follow one policy (`volume.empty_policy`): with exactly one side
empty, Dice/NSD/PQ score 0.0 and HD95/ASSD are `None` (no finite distance
exists); with both sides empty every score is `None`. MONAI's own answers for
those cases (NaN, inf, or Dice 1.0) never reach a record.

One class per run: MONAI's underlying functionals accept one-hot `(N, C, ...)`
batches, but this adapter enforces `C == 1` and rejects anything wider with a
`ValueError` — there is no one-hot / multi-channel path. A multi-label
dataset is scored one class at a time via `Mask(label=k)`, not by stacking
classes into channels. The one exception is `panoptic_quality`, whose integer
instance maps are a different input shape entirely (see its class docstring).

MONAI's defaults are calibrated for the medical-imaging domain (physical voxel
spacing in millimetres, a background-class convention). Domain parameters are
forwarded unchanged via `**monai_kwargs`; each builder's docstring states
which ones matter in another domain (materials science: different physical
units via `spacing`).

Usage: each builder (`dice_metric()`, `hausdorff95_metric()`, ...) returns a
`MetricSpec` — pass one or more into `MetricRegistry(*specs)` and hand that
registry to `IQAEvaluator`/`VolumeEvaluator`:

    from iqaevaluator.metrics import MetricRegistry
    from iqaevaluator.segmentation_metrics.monai_metrics import DICE, HAUSDORFF95

    registry = MetricRegistry(DICE, HAUSDORFF95)

Use the pre-built constants (`DICE`, `HAUSDORFF95`, `NSD`, `ASSD`,
`PANOPTIC_QUALITY`) for defaults, or call a builder with MONAI keyword
arguments (e.g. `hausdorff95_metric(percentile=None)`) before registering.
"""

import math
from typing import Callable, Optional

import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_panoptic_quality,
    compute_surface_dice,
)

from iqaevaluator.metric_spec import MetricSpec, ModeSupport, Spacing
from iqaevaluator.segmentation_metrics.volume import (
    NOT_EMPTY, empty_policy, foreground_counts, require_binary,
)

DOMAIN_MEDICAL = "medical (MONAI)"

_REMOVED_KNOBS = ("threshold", "label")


def _reject_removed_knobs(name: str, kwargs: dict) -> None:
    """`threshold`/`label` used to be adapter parameters. A stale caller must
    fail here rather than have the keyword forwarded to MONAI (or ignored)."""
    for key in _REMOVED_KNOBS:
        if key in kwargs:
            raise TypeError(
                f"{name} no longer takes '{key}': binarization is decided by the "
                "loader — load masks with normalization.Mask() (Mask(label=k) "
                "selects one class)."
            )


class MonaiSegmentationMetric:
    """Adapter for MONAI's one-hot-batch functional metrics (Dice, HD95, NSD, ASSD).

    All four share the call signature `compute_fn(y_pred, y, **kwargs) -> (N, C)`
    tensor (batch x class channels); MONAI's own signature is one-hot, but
    this adapter requires `C == 1` (a `ValueError` names the metric and
    points to `Mask(label=k)` otherwise) and produces one score per sample,
    matching the Metric protocol.

    Both tensors must be binary (`require_binary`) and single-channel; the
    empty-mask policy is applied per sample before MONAI sees anything, with
    `one_sided` as the score for exactly-one-side-empty (0.0 for Dice/NSD,
    None for HD95/ASSD). A NaN or inf that MONAI still returns for a
    populated pair is recorded as None and printed as a warning — an
    unexpected case must not be silent.

    In volume mode the adapter receives a single `(1, C, D, H, W)` sample and
    returns a single score; the distance metrics additionally receive
    `spacing` so their result is in millimetres rather than voxels.
    """

    def __init__(
        self,
        compute_fn: Callable[..., torch.Tensor],
        *,
        one_sided: Optional[float],
        name: Optional[str] = None,
        **monai_kwargs,
    ):
        self._name = name or compute_fn.__name__
        _reject_removed_knobs(self._name, monai_kwargs)
        self._compute_fn = compute_fn
        self._one_sided  = one_sided
        self._kwargs     = monai_kwargs

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]:
        if target is None:
            raise ValueError(f"'{self._name}' compares two masks and requires a target mask")
        require_binary(input, metric=self._name)
        require_binary(target, metric=self._name)
        if input.shape[1] > 1 or target.shape[1] > 1:
            raise ValueError(
                f"{self._name} scores one class per run: this adapter has no "
                "one-hot / multi-channel path — pass a single-channel mask "
                "and select one class at a time with normalization.Mask(label=k) "
                "instead of stacking classes into channels."
            )
        # MONAI 1.6.0 happens to accept integer input; that is an implementation
        # detail of a third-party library, so the adapter guarantees floats itself.
        y_pred, y = input.float(), target.float()

        n_pred, n_gt = foreground_counts(y_pred, y)
        results: list[Optional[float]] = [None] * y_pred.shape[0]
        active: list[int] = []
        for i in range(y_pred.shape[0]):
            verdict = empty_policy(int(n_pred[i]), int(n_gt[i]), one_sided=self._one_sided)
            if verdict is NOT_EMPTY:
                active.append(i)
            else:
                results[i] = verdict

        if active:
            scores = self._compute_fn(y_pred=y_pred[active], y=y[active], **self._kwargs)  # (n, C)
            per_sample = torch.nanmean(scores.float(), dim=1)
            for i, score in zip(active, per_sample):
                value = float(score.item())
                if not math.isfinite(value):
                    print(
                        f"[WARNING] {self._name}: MONAI returned {value} for sample {i} "
                        "although both masks have foreground; recorded as None."
                    )
                    results[i] = None
                else:
                    results[i] = value
        return results


def _volume_factory(
    compute_fn: Callable[..., torch.Tensor],
    *,
    one_sided: Optional[float],
    name: str,
    uses_spacing: bool,
    **monai_kwargs,
) -> Callable[[Optional[Spacing]], MonaiSegmentationMetric]:
    """Build the volume-mode factory for one MONAI functional metric.

    `uses_spacing` marks the distance metrics (HD95, NSD, ASSD), which convert
    voxel counts to physical units. An explicit `spacing` passed to the builder
    always wins: the user pinned it deliberately, and silently replacing it with
    whatever the current file happens to say would be worse than ignoring the
    file.
    """
    def build(spacing: Optional[Spacing]) -> MonaiSegmentationMetric:
        kwargs = dict(monai_kwargs)
        if uses_spacing:
            if spacing is not None:
                kwargs.setdefault("spacing", list(spacing))
            elif "spacing" not in kwargs:
                print(
                    "[WARNING] no voxel size available, so this distance is "
                    "counted in voxels rather than millimetres. Values are "
                    "comparable between images on the same grid, but not "
                    "between images recorded at different resolutions."
                )
        return MonaiSegmentationMetric(compute_fn, one_sided=one_sided, name=name, **kwargs)

    return build


def _monai_spec(
    name: str,
    compute_fn: Callable[..., torch.Tensor],
    *,
    direction: str,
    one_sided: Optional[float],
    uses_spacing: bool,
    description: str,
    defaults: dict,
    **monai_kwargs,
) -> MetricSpec:
    """One MetricSpec for one MONAI functional; `defaults` are the domain
    defaults a caller's `monai_kwargs` may override."""
    _reject_removed_knobs(f"{name}_metric()", monai_kwargs)
    kwargs = {**defaults, **monai_kwargs}
    metric = MonaiSegmentationMetric(compute_fn, one_sided=one_sided, name=name, **kwargs)
    return MetricSpec(
        name=name,
        direction=direction,
        reference=True,
        channels="gray",
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(_volume_factory(
            compute_fn, one_sided=one_sided, name=name, uses_spacing=uses_spacing, **kwargs)),
        builtin=False,
        description=description,
        domain=DOMAIN_MEDICAL,
    )


def dice_metric(**monai_kwargs) -> MetricSpec:
    """Dice similarity coefficient: 2*|pred ∩ gt| / (|pred| + |gt|), 1.0 = perfect overlap.

    Input must be binary and single-channel — load masks with `Mask()`
    (`Mask(label=k)` for one class of a multi-label dataset). Domain: medical
    (MONAI). Defaults assume that single foreground channel
    (include_background=True, since there is nothing else to include) and
    `ignore_empty=False`, so an empty reference with a non-empty prediction
    scores 0.0 (the empty-mask policy decides these cases anyway).

    Args:
        include_background (kwarg, default True): forwarded to MONAI; with
            the single-channel input this adapter requires there is no
            separate background channel to drop, so leave this at the
            default unless you have a specific reason to change it.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_dice`.
    """
    return _monai_spec(
        "dice", compute_dice, direction="higher_is_better", one_sided=0.0, uses_spacing=False,
        defaults={"include_background": True, "ignore_empty": False},
        description=(
            "Dice similarity coefficient: overlap between predicted and "
            "ground-truth segmentation masks (1.0 = perfect overlap, 0.0 = no "
            "overlap). Domain: medical (MONAI). Scores one class per run — "
            "for a multi-label dataset, load each class in turn with "
            "Mask(label=k)."
        ),
        **monai_kwargs,
    )


DICE = dice_metric()


def hausdorff95_metric(**monai_kwargs) -> MetricSpec:
    """95th-percentile Hausdorff Distance: worst-case boundary error, robust to outlier voxels.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI) —
    the returned distance is in voxel units unless `spacing` is supplied
    (physical units per voxel, e.g. mm for medical scans or µm for materials
    micrographs). Definition: max over both directions of the 95th percentile
    of surface-voxel distances (Taha & Hanbury 2015); with exactly one empty
    mask the score is None.

    Args:
        include_background (kwarg, default True): if False, drops channel 0
            (assumed background class) before scoring.
        percentile (kwarg, default 95): which percentile of boundary-point
            distances to report; None gives the plain (max) Hausdorff distance.
        directed (kwarg, default False): True measures pred→gt only.
        spacing (kwarg, default None): physical size of one voxel (scalar or
            per-axis list); converts the voxel-unit distance to real units.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_hausdorff_distance`.
    """
    return _monai_spec(
        "hausdorff95", compute_hausdorff_distance, direction="lower_is_better", one_sided=None,
        uses_spacing=True, defaults={"include_background": True, "percentile": 95},
        description=(
            "95th-percentile Hausdorff Distance: how far the predicted "
            "segmentation boundary is from the ground-truth boundary in the "
            "worst 5% of cases (lower = closer boundaries). Domain: medical "
            "(MONAI). Distance is in voxel units by default; pass "
            "`spacing=<mm-per-voxel or per-axis list>` for physical units, or "
            "the equivalent voxel size for another domain (e.g. µm for "
            "materials micrographs)."
        ),
        **monai_kwargs,
    )


HAUSDORFF95 = hausdorff95_metric()


def normalized_surface_dice_metric(**monai_kwargs) -> MetricSpec:
    """Normalized Surface Dice (NSD): fraction of the predicted/gt boundary within a tolerance distance.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI).
    `class_thresholds` is the tolerance distance per class (defaults to
    `[1.0]`) — MONAI requires it and treats it in the same units as `spacing`.

    The scoring mode therefore changes what this metric measures, not just its
    scale. Slice mode passes no spacing, so the default tolerance means one
    voxel; volume mode passes the image's voxel size, so the same default means
    one millimetre. On 0.5 mm data volume mode is twice as forgiving as slice
    mode, and on 2 mm data half as forgiving. Pass `class_thresholds`
    explicitly if the two modes have to be read on one scale, and do not
    compare an `nsd` column from a slice run against one from a volume run.

    Args:
        include_background (kwarg, default True): if False, drops channel 0.
        class_thresholds (kwarg, default [1.0]): the tolerance distance for
            the single foreground channel this adapter requires, in the
            units of `spacing`. MONAI takes this as a list; keep it to one
            entry.
        spacing (kwarg, default None): physical size of one voxel.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_surface_dice`.
    """
    return _monai_spec(
        "nsd", compute_surface_dice, direction="higher_is_better", one_sided=0.0,
        uses_spacing=True, defaults={"include_background": True, "class_thresholds": [1.0]},
        description=(
            "Normalized Surface Dice: fraction of the predicted and "
            "ground-truth boundaries that lie within a tolerance distance of "
            "each other (1.0 = all boundary points within tolerance). Domain: "
            "medical (MONAI). Tolerance is set via `class_thresholds` "
            "(default 1 voxel) and interpreted in the units of `spacing`; for "
            "another domain (e.g. materials micrographs) set both to that "
            "domain's physical voxel size and acceptable boundary error."
        ),
        **monai_kwargs,
    )


NSD = normalized_surface_dice_metric()


def average_surface_distance_metric(**monai_kwargs) -> MetricSpec:
    """Average Symmetric Surface Distance (ASSD): mean boundary distance in both directions.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI).
    Like HD95, the result is in voxel units unless `spacing` is supplied, and
    with exactly one empty mask the score is None.

    Args:
        include_background (kwarg, default True): if False, drops channel 0.
        symmetric (kwarg, default True): mean over the pred→gt and gt→pred
            surface distances pooled (True) vs. pred→gt only (False).
        spacing (kwarg, default None): physical size of one voxel.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_average_surface_distance`.
    """
    return _monai_spec(
        "assd", compute_average_surface_distance, direction="lower_is_better", one_sided=None,
        uses_spacing=True, defaults={"include_background": True, "symmetric": True},
        description=(
            "Average Symmetric Surface Distance: mean distance between the "
            "predicted and ground-truth boundaries, averaged in both "
            "directions (lower = closer boundaries on average). Domain: "
            "medical (MONAI). Distance is in voxel units by default; pass "
            "`spacing=<mm-per-voxel or per-axis list>` for physical units, or "
            "the equivalent voxel size for another domain (e.g. µm for "
            "materials micrographs)."
        ),
        **monai_kwargs,
    )


ASSD = average_surface_distance_metric()


class MonaiPanopticQualityMetric:
    """Adapter for MONAI's compute_panoptic_quality, which takes one integer
    instance-label map per image (no batch/channel dim, unlike the other four
    metrics) — this loops over the batch itself.

    Accepts integer-valued tensors only: `{0, 1}` from `Mask()` (a binary mask
    is scored as single-instance PQ, foreground = one instance) or an instance
    map with one integer id per object loaded with `Raw()`. The empty-mask
    policy is Dice's: one side empty → 0.0, both empty → None.
    """

    def __init__(self, **monai_kwargs):
        _reject_removed_knobs("panoptic_quality", monai_kwargs)
        self._kwargs = monai_kwargs

    @staticmethod
    def _require_integer_valued(t: torch.Tensor) -> None:
        f = t.float()
        if not bool((f == f.round()).all()):
            raise ValueError(
                "panoptic_quality expects integer-valued instance maps: load binary "
                "masks with normalization.Mask() and instance maps with normalization.Raw()"
            )

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]:
        if target is None:
            raise ValueError("'panoptic_quality' compares two masks and requires a target mask")
        self._require_integer_valued(input)
        self._require_integer_valued(target)
        scores: list[Optional[float]] = []
        for i in range(input.shape[0]):
            pred_map = input[i, 0].long()
            gt_map   = target[i, 0].long()
            verdict = empty_policy(int((pred_map != 0).sum()), int((gt_map != 0).sum()), one_sided=0.0)
            if verdict is not NOT_EMPTY:
                scores.append(verdict)
                continue
            value = float(compute_panoptic_quality(pred_map, gt_map, **self._kwargs).item())
            if not math.isfinite(value):
                print(
                    f"[WARNING] panoptic_quality: MONAI returned {value} for sample {i} "
                    "although both maps have foreground; recorded as None."
                )
                scores.append(None)
            else:
                scores.append(value)
        return scores


def panoptic_quality_metric(**monai_kwargs) -> MetricSpec:
    """Panoptic Quality (PQ): detection accuracy (instances matched by IoU) times
    segmentation accuracy (mean IoU of matched instances), Kirillov et al. 2019.

    Binary masks from `Mask()` degenerate to a single-instance IoU-based score;
    for true multi-instance PQ load integer instance maps with `Raw()`. Domain:
    medical (MONAI), designed for instance segmentation (cells, lesions). For
    another domain with multiple distinct objects (grains in a micrograph),
    supply one integer id per instance and tune `match_iou_threshold`. Note
    that MONAI adds `smooth_numerator=1e-6` to the denominator.

    Args:
        match_iou_threshold (kwarg, default 0.5): minimum IoU for a predicted
            instance to count as matched; unmatched instances count against
            the score.
        metric_name (kwarg, default "pq"): "pq", "sq" or "rq".
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_panoptic_quality`.
    """
    _reject_removed_knobs("panoptic_quality_metric()", monai_kwargs)
    kwargs = {"match_iou_threshold": 0.5, **monai_kwargs}
    metric = MonaiPanopticQualityMetric(**kwargs)
    return MetricSpec(
        name="panoptic_quality",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(lambda spacing: MonaiPanopticQualityMetric(**kwargs)),
        builtin=False,
        description=(
            "Panoptic Quality: combines instance-detection accuracy (are the "
            "right objects found?) and segmentation accuracy (how well do "
            "matched instances overlap?) into one score (1.0 = perfect). "
            "Domain: medical (MONAI), designed for instance segmentation "
            "(e.g. individual cells or lesions); on a plain binary mask it "
            "reduces to a single-instance IoU score. For another domain with "
            "multiple distinct objects (e.g. grains in a materials "
            "micrograph), supply pred/gt with a unique integer label per "
            "instance and tune `match_iou_threshold` (default 0.5) for that "
            "domain's localization tolerance."
        ),
        domain=DOMAIN_MEDICAL,
    )


PANOPTIC_QUALITY = panoptic_quality_metric()
