# MONAI Segmentation Metrics — Design

Date: 2026-08-25
Branch: `segmentation_metrics`

## Purpose

Add five segmentation-quality metrics backed by MONAI — Dice, Hausdorff Distance 95
(HD95), Normalized Surface Dice (NSD), Average Symmetric Surface Distance (ASSD),
Panoptic Quality (PQ) — to the existing IQA framework's metric registry, following
the same adapter/registry pattern used for pyiqa metrics.

MONAI is medical-imaging-specific (assumes physical voxel spacing in mm, a
background class convention, etc.). Every metric ships a description explaining
what it measures, that its defaults are calibrated for the medical domain, and
which parameters must be adjusted to use it in another domain (e.g. materials
science: different physical units, more material phases/classes).

## Non-goals

- No mask *generation*. Otsu-based generation is being removed (didn't work
  well); the user supplies their own pred/gt label maps (already binary/label
  tensors, e.g. mask images loaded via `ImageLoader` like any other image).
- No merge with the unmerged `volumetric_similarity` branch (custom VS/Dice
  metrics in `segmentation_metrics/volume.py`). That integration happens later,
  separately.
- No change to `main.py`'s CLI pipeline, which runs on raw images — segmentation
  metrics are not added to `BUILTIN_METRICS`.

## Package layout

```
src/segmentation_metrics/
    __init__.py
    monai_metrics.py
```

Mirrors the existing `radimagenet_lpips.py` / `clip_iqa_medical.py` pattern:
a self-contained module that builds `Metric`-protocol-compatible objects.
`metrics.py` imports the constants from it for registration; `IQAEvaluator`
never depends on MONAI directly.

## Metric protocol — unchanged

`Metric.__call__(input, target=None) -> Sequence[float]` on `(N,C,H,W)` float32
batches in `[0,1]`, same contract as pyiqa metrics. Segmentation inputs are
label/binary masks in that same tensor shape (loaded via `ImageLoader` like any
other supported format), not raw intensity images.

## Adapters

Two adapter classes in `monai_metrics.py`:

- `MonaiSegmentationMetric` — wraps Dice, HD95, NSD, ASSD. All four MONAI
  metrics share the call signature `metric(y_pred, y) -> (N, C)` tensor; the
  adapter averages across the class dimension to produce one score per sample,
  matching the `Metric` protocol. Optional `threshold` kwarg binarizes
  non-binary input before calling MONAI (`(x > threshold).float()`); default
  `None` performs no binarization (input assumed already binary/label).
- `MonaiPanopticQualityMetric` — separate adapter for PQ, which takes instance
  label maps and a `num_classes` / matching IoU threshold rather than the
  shared four-metric signature.

Both adapters lazily construct their underlying MONAI metric object on first
call (mirrors `PyIQAMetric`'s lazy `_impl` pattern) and are cached per spec
instance by `MetricRegistry` as usual.

## MetricSpec extension

Two new fields on the existing frozen `MetricSpec` dataclass, both with
defaults so the 10 existing IQA constants are unaffected:

```python
description: str = ""   # what the metric measures, in registry/report context
domain:      str = ""   # e.g. "medical (MONAI)"
```

## User-parameter pass-through

Public builder functions (not the private `_pyiqa_factory` pattern) so users
can construct their own differently-parameterized `MetricSpec` for a given
metric family — this is the concrete answer to "let users pass their own
parameters":

```python
def dice_metric(*, include_background=True, num_classes=None,
                threshold=None, **monai_kwargs) -> MetricSpec: ...

def hausdorff95_metric(*, percentile=95, spacing=None,
                        include_background=True, threshold=None,
                        **monai_kwargs) -> MetricSpec: ...

def normalized_surface_dice_metric(*, class_thresholds, spacing=None,
                                    include_background=True, threshold=None,
                                    **monai_kwargs) -> MetricSpec: ...

def average_surface_distance_metric(*, symmetric=True, spacing=None,
                                     include_background=True, threshold=None,
                                     **monai_kwargs) -> MetricSpec: ...

def panoptic_quality_metric(*, num_classes, match_iou_threshold=0.5,
                             **monai_kwargs) -> MetricSpec: ...
```

Each builder returns a `MetricSpec` with a generated `description` covering:
what the metric measures, that it's a segmentation-quality check (not an IQA
image-quality check), the medical/MONAI domain assumption, and which kwarg(s)
to change for another domain (e.g. `spacing` for physical voxel size in
materials micrographs, `class_thresholds`/`num_classes` for a different class
count).

Default module-level constants (parameter-free instantiation of each builder):

```python
DICE              = dice_metric()
HAUSDORFF95       = hausdorff95_metric()
NSD               = normalized_surface_dice_metric(class_thresholds=[1.0])
ASSD              = average_surface_distance_metric()
PANOPTIC_QUALITY  = panoptic_quality_metric(num_classes=1)
```

All five are full-reference (`reference=True`, need a target mask),
`channels="gray"` (single-channel binary mask; multi-class one-hot input is a
known limitation — `ImageLoader` only produces 1- or 3-channel tensors today).

## Registration

`metrics.py` imports and re-exports the five constants plus a
`SEGMENTATION_METRICS` bundle tuple (mirrors `BUILTIN_METRICS`'s shape,
kept separate from it):

```python
SEGMENTATION_METRICS = (DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY)
```

Nothing auto-registers at import time (same opt-in convention as
`BUILTIN_METRICS`). Kept out of `BUILTIN_METRICS` itself so `main.py`'s
raw-image CLI pipeline is untouched.

## Testing

Extend `tests/test_metrics.py` with a `TestSegmentationMetrics` class mirroring
`TestBuiltinMetrics`: spec-attribute assertions (direction, reference,
channels, builtin) for all five constants, registration via
`isolated_registry.register(*SEGMENTATION_METRICS)`, and adapter-level unit
tests calling each `Metric` on small synthetic binary mask tensors
(`torch.randint(0, 2, (N,1,H,W)).float()`) to check output shape/range without
requiring GPU or real medical data.

## Dependency

Add `monai` to `requirements.txt`.
