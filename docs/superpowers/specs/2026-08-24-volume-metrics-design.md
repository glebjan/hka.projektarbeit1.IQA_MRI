# Volume-Based Segmentation Metrics — Design

## Purpose

Add volume/overlap segmentation metrics (Dice, Volumetric Similarity, Average
Volume Difference) as a standalone module, independent of the existing IQA
(image quality assessment) framework in `src/`. These operate on binary/label
masks, not continuous image-quality scores, and are computed per-slice then
aggregated per-patient/volume — a different data flow from `IQAEvaluator`.

## Location and naming

New package `src/segmentation_metrics/`, containing:
- `src/segmentation_metrics/__init__.py` (empty — marks the package)
- `src/segmentation_metrics/volume.py` — implementation

Rationale: the existing `src/metrics.py` is a *file*, not a package, and
already owns the `metrics` name for pyiqa-backed IQA adapters. A directory
literally named `metrics/` cannot coexist with `metrics.py` in the same
folder. Rather than restructure the existing IQA module (which would force
import changes across `iqa_evaluator.py`, `main.py`, `evaluation_result.py`,
and their tests), the new segmentation-metrics code gets its own package
name. This is the one deviation from the "no package structure, flat files"
rule in `CLAUDE.md` — justified because these metrics are a distinct concern
(segmentation overlap vs. image quality) with no shared code path, and a
single-file package is the smallest structural unit that avoids the name
collision while keeping the two concerns separable.

Tests: `tests/test_volume.py` (flat, matching existing `tests/` layout).

## API

### `as_mask(x, label: int = 1, threshold: float = 0.5) -> np.ndarray`

Binarizes an array to a boolean mask.

- `x.dtype == bool` → returned unchanged.
- Float dtype → values must lie in `[0, 1]` (inclusive); otherwise raises
  `ValueError` naming both the offending min/max and suggesting a missing
  sigmoid activation upstream. Valid arrays are thresholded: `x >= threshold`.
- Integer dtype (label map) → `x == label` (one-vs-rest).
- Any other dtype → `TypeError` (not explicitly in the original spec, but
  needed so silent misbehavior on e.g. complex/string arrays cannot occur).

### Single-slice metrics

All have signature `(pred, gt, *, label: int = 1, threshold: float = 0.5) -> float`.
`label`/`threshold` are keyword-only pass-throughs to `as_mask` for both
`pred` and `gt`, so callers with bool/int arrays never need them, and callers
with float logits can still configure them.

Common preamble for all five: if `pred.shape != gt.shape`, raise `ValueError`
including both shapes in the message. Then `as_mask` both arrays.

- `v_pred(pred, gt, ...)` → `pred_mask.sum()` (count of positive voxels in
  the prediction; `gt` is accepted for signature symmetry but only its shape
  is checked against `pred`).
- `v_gt(pred, gt, ...)` → `gt_mask.sum()`.
- `tp(pred, gt, ...)` → `(pred_mask & gt_mask).sum()`.
- `vs(pred, gt, ...)` → `1 - abs(v_pred - v_gt) / (v_pred + v_gt)`.
  Range `[0, 1]`, `1` = identical volume. Denominator `0` (both masks
  empty) → `nan`. Reference: Taha & Hanbury 2015.
- `vs_signed(pred, gt, ...)` → `2 * (v_pred - v_gt) / (v_pred + v_gt)`.
  Range `[-2, 2]`; negative = undersegmentation (prediction smaller than
  reference), positive = oversegmentation. SimpleITK convention.
  Denominator `0` → `nan`.
  Identity: `vs == 1 - abs(vs_signed) / 2` (documented and tested).

All five call `as_mask` independently rather than sharing cached masks —
keeps each function stateless and independently callable, at the cost of
redundant binarization when several are called on the same pair. Acceptable:
these run per-slice on 2D arrays, not a hot path.

### `aggregate_patient(df: pd.DataFrame, group_col: str | None = "patient_id") -> pd.DataFrame`

Input: a DataFrame with (at least) columns `v_pred`, `v_gt`, `tp` — one row
per slice, already computed by the caller via the single-slice functions
above (this function does not call `as_mask` or take image arrays).

Behavior:
- `group_col=None` → treat the entire DataFrame as a single group (one
  output row).
- Otherwise → `groupby(group_col)`.
- Per group: sum `v_pred`, `v_gt`, `tp` (pandas `.sum()`, which skips `NaN`
  by default — this is the "nanmean"-safety requirement: an empty/degenerate
  slice contributing `NaN` to a column must not zero out or corrupt the
  patient-level sum).
- From the summed totals, compute:
  - `vs = 1 - abs(v_pred_sum - v_gt_sum) / (v_pred_sum + v_gt_sum)`
  - `vs_signed = 2 * (v_pred_sum - v_gt_sum) / (v_pred_sum + v_gt_sum)`
  - `dice = 2 * tp_sum / (v_pred_sum + v_gt_sum)`
  - `avd_voxels = v_pred_sum - v_gt_sum`
  - All four denominators `0` → `nan` (never `0.0`, never a raised
    `ZeroDivisionError` — division is done via `numpy` float ops which
    already produce `nan` for `0/0`, or explicit `np.where` guards).
- Returns a DataFrame indexed/keyed by group, with columns
  `v_pred, v_gt, tp, vs, vs_signed, dice, avd_voxels`.

Critical point (explicit in the spec, tested): the mean of per-slice `vs`/
`dice`/etc. values is **not** the same as the value computed on the summed
volume. `aggregate_patient` always sums first, then derives ratios — never
averages ratios.

## Constraints

- Dependencies: `numpy` and `pandas` only — no new packages.
- No classes, no module-level state — every function is independently
  callable and swappable, matching the adapter-style, stateless philosophy
  already used in `src/metrics.py`'s `Metric` protocol (this module is
  intentionally *not* wired into that protocol — segmentation masks are not
  `(N,C,H,W)` image-quality inputs).
- Type hints and short docstrings (value range + reference) on every
  function.

## Testing plan (`tests/test_volume.py`)

1. Identical masks → `vs == 1.0`, `vs_signed == 0.0`, `dice == 1.0`.
2. Two equal-size, spatially disjoint masks → `vs == 1.0` but `dice == 0.0`
   (documents that VS is not an overlap measure).
3. Undersegmentation → `vs_signed < 0`; oversegmentation → `vs_signed > 0`.
4. Both masks empty → all ratio metrics `nan`.
5. Empty reference, non-empty prediction → `vs == 0.0`.
6. Aggregation: 3 synthetic slices where the per-slice mean of a ratio
   metric is provably different from the volume-level (summed) value;
   assert both numbers explicitly (mean ≠ aggregate).
7. Property test: `vs == 1 - abs(vs_signed) / 2` over multiple random mask
   pairs.
8. `as_mask`: float array with a value outside `[0, 1]` → `ValueError`.

Plus, not explicitly listed in the original request but required by the
design decisions above (added for completeness):
9. `as_mask`: bool array passthrough, int label-map `==` semantics.
10. Shape mismatch between `pred` and `gt` → `ValueError` naming both shapes.
11. `aggregate_patient` with `group_col=None` treats the whole frame as one
    group.
12. `as_mask` on an unsupported dtype (e.g. complex) → `TypeError`.
