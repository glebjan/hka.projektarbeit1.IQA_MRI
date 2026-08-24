# Volume-Based Segmentation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stateless volume/overlap segmentation metrics (Dice, Volumetric
Similarity, Average Volume Difference) and a per-patient aggregation
function, as a new standalone module independent of the existing IQA
framework.

**Architecture:** New package `src/segmentation_metrics/volume.py`. One
binarization helper (`as_mask`), five stateless single-slice metric
functions that each call `as_mask` internally, and one pandas-based
aggregation function (`aggregate_patient`) that sums per-slice voxel counts
before deriving ratio metrics — never averages the per-slice ratios
directly. No classes, no module state, no new dependencies.

**Tech Stack:** Python 3.14, numpy, pandas, pytest (all already present in
`.venv`).

## Global Constraints

- Only `numpy` and `pandas` — no new dependencies.
- Shape mismatch between `pred` and `gt` → `ValueError` naming both shapes.
- Any ratio metric with denominator `0` → `np.nan`, never `0.0`, never
  `ZeroDivisionError`.
- `aggregate_patient` sums `v_pred`/`v_gt`/`tp` per group first, then
  computes ratios from the sums — per-slice ratio means are never used.
- No classes, no module-level/global state — every function independently
  callable and swappable.
- Type hints and short docstrings (value range + reference) on every public
  function.
- `vs` range `[0, 1]` (Taha & Hanbury 2015); `vs_signed` range `[-2, 2]`
  (SimpleITK convention, negative = undersegmentation); identity
  `vs == 1 - abs(vs_signed) / 2` holds and is documented in both docstrings.

---

### Task 1: `as_mask` binarization helper

**Files:**
- Create: `src/segmentation_metrics/__init__.py` (empty file)
- Create: `src/segmentation_metrics/volume.py`
- Test: `tests/test_volume.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `as_mask(x: np.ndarray, label: int = 1, threshold: float = 0.5) -> np.ndarray`
  — used by every metric function in Task 2.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_volume.py
import numpy as np
import pytest

from segmentation_metrics.volume import as_mask


def test_as_mask_bool_passthrough():
    x = np.array([True, False, True])
    result = as_mask(x)
    assert result is x or np.array_equal(result, x)
    assert result.dtype == bool


def test_as_mask_float_thresholds():
    x = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    result = as_mask(x, threshold=0.5)
    np.testing.assert_array_equal(result, [False, False, True, True, True])


def test_as_mask_float_out_of_range_raises():
    x = np.array([0.2, 1.5, -0.1])
    with pytest.raises(ValueError):
        as_mask(x)


def test_as_mask_int_label_map():
    x = np.array([0, 1, 2, 1, 0])
    result = as_mask(x, label=1)
    np.testing.assert_array_equal(result, [False, True, False, True, False])


def test_as_mask_unsupported_dtype_raises():
    x = np.array(["a", "b"])
    with pytest.raises(TypeError):
        as_mask(x)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: `ModuleNotFoundError: No module named 'segmentation_metrics'`
(package does not exist yet).

- [ ] **Step 3: Write the implementation**

`src/segmentation_metrics/__init__.py` is an empty file (no content).

```python
# src/segmentation_metrics/volume.py
"""Stateless volume-based segmentation metrics (Dice, VS, AVD).

Operate on binary/label masks per 2D/3D slice; aggregate per-patient via
`aggregate_patient`. No shared state between calls.
"""
from __future__ import annotations

import numpy as np


def as_mask(x: np.ndarray, label: int = 1, threshold: float = 0.5) -> np.ndarray:
    """Binarize an array to a boolean mask.

    - bool array: returned unchanged.
    - float array: values must lie in [0, 1] (probabilities); thresholded
      at `threshold`. Values outside [0, 1] raise ValueError — this usually
      means raw logits were passed without a sigmoid activation.
    - integer array (label map): one-vs-rest via `x == label`.
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
        return x == label
    raise TypeError(f"as_mask does not support dtype {x.dtype}")


def _check_shapes(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt shapes must match, got pred={pred.shape}, "
            f"gt={gt.shape}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/__init__.py src/segmentation_metrics/volume.py tests/test_volume.py
git commit -m "feat: add as_mask binarization helper for segmentation metrics"
```

---

### Task 2: Single-slice voxel-count and ratio metrics

**Files:**
- Modify: `src/segmentation_metrics/volume.py`
- Test: `tests/test_volume.py`

**Interfaces:**
- Consumes: `as_mask(x, label=1, threshold=0.5) -> np.ndarray` and
  `_check_shapes(pred, gt) -> None` from Task 1.
- Produces:
  - `v_pred(pred, gt, *, label=1, threshold=0.5) -> float`
  - `v_gt(pred, gt, *, label=1, threshold=0.5) -> float`
  - `tp(pred, gt, *, label=1, threshold=0.5) -> float`
  - `vs(pred, gt, *, label=1, threshold=0.5) -> float`
  - `vs_signed(pred, gt, *, label=1, threshold=0.5) -> float`
  All five consumed by Task 3's `aggregate_patient` docstring/tests (as the
  functions callers use to build the per-slice DataFrame) and by Task 4's
  property test.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_volume.py
from segmentation_metrics.volume import v_pred, v_gt, tp, vs, vs_signed


def _disk_mask(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius**2


def test_identical_masks_perfect_scores():
    mask = _disk_mask((20, 20), (10, 10), 5)
    assert vs(mask, mask) == 1.0
    assert vs_signed(mask, mask) == 0.0
    assert 2 * tp(mask, mask) / (v_pred(mask, mask) + v_gt(mask, mask)) == 1.0


def test_disjoint_equal_size_masks_vs_one_dice_zero():
    pred = np.zeros((20, 20), dtype=bool)
    gt = np.zeros((20, 20), dtype=bool)
    pred[0:5, 0:5] = True   # 25 voxels
    gt[15:20, 15:20] = True  # 25 voxels, disjoint
    assert vs(pred, gt) == 1.0
    dice = 2 * tp(pred, gt) / (v_pred(pred, gt) + v_gt(pred, gt))
    assert dice == 0.0


def test_undersegmentation_negative_signed_vs():
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True  # 100 voxels
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:5, 0:5] = True  # 25 voxels, subset -> smaller than gt
    assert vs_signed(pred, gt) < 0


def test_oversegmentation_positive_signed_vs():
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:10] = True  # 100 voxels
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:5, 0:5] = True  # 25 voxels, subset -> pred bigger than gt
    assert vs_signed(pred, gt) > 0


def test_both_empty_masks_yield_nan():
    pred = np.zeros((20, 20), dtype=bool)
    gt = np.zeros((20, 20), dtype=bool)
    assert np.isnan(vs(pred, gt))
    assert np.isnan(vs_signed(pred, gt))


def test_empty_reference_nonempty_prediction_vs_zero():
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:5, 0:5] = True
    gt = np.zeros((20, 20), dtype=bool)
    assert vs(pred, gt) == 0.0


def test_shape_mismatch_raises_with_both_shapes():
    pred = np.zeros((20, 20), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match=r"\(20, 20\).*\(10, 10\)"):
        vs(pred, gt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: `ImportError: cannot import name 'v_pred'` (functions not defined
yet).

- [ ] **Step 3: Write the implementation**

Append to `src/segmentation_metrics/volume.py`:

```python
def v_pred(pred: np.ndarray, gt: np.ndarray, *, label: int = 1, threshold: float = 0.5) -> float:
    """Count of positive voxels in the prediction mask."""
    _check_shapes(pred, gt)
    return float(as_mask(pred, label, threshold).sum())


def v_gt(pred: np.ndarray, gt: np.ndarray, *, label: int = 1, threshold: float = 0.5) -> float:
    """Count of positive voxels in the reference (ground-truth) mask."""
    _check_shapes(pred, gt)
    return float(as_mask(gt, label, threshold).sum())


def tp(pred: np.ndarray, gt: np.ndarray, *, label: int = 1, threshold: float = 0.5) -> float:
    """Count of voxels positive in both prediction and reference masks."""
    _check_shapes(pred, gt)
    pred_mask = as_mask(pred, label, threshold)
    gt_mask = as_mask(gt, label, threshold)
    return float((pred_mask & gt_mask).sum())


def vs(pred: np.ndarray, gt: np.ndarray, *, label: int = 1, threshold: float = 0.5) -> float:
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


def vs_signed(pred: np.ndarray, gt: np.ndarray, *, label: int = 1, threshold: float = 0.5) -> float:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: 12 passed (5 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/volume.py tests/test_volume.py
git commit -m "feat: add single-slice volume and VS segmentation metrics"
```

---

### Task 3: `aggregate_patient` and VS/signed-VS identity property test

**Files:**
- Modify: `src/segmentation_metrics/volume.py`
- Test: `tests/test_volume.py`

**Interfaces:**
- Consumes: `v_pred`, `v_gt`, `tp`, `vs`, `vs_signed` from Task 2 (used only
  in tests, to build the per-slice DataFrame fed into `aggregate_patient`;
  `aggregate_patient` itself takes a DataFrame, not image arrays).
- Produces: `aggregate_patient(df: pd.DataFrame, group_col: str | None = "patient_id") -> pd.DataFrame`
  with output columns `v_pred, v_gt, tp, vs, vs_signed, dice, avd_voxels`.
  This is the final public surface of the module — no later task consumes
  it.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_volume.py
import pandas as pd
from segmentation_metrics.volume import aggregate_patient


def test_aggregate_mean_of_ratios_differs_from_volume_ratio():
    # 3 slices, same patient. Per-slice dice varies a lot; the volume-level
    # dice (computed from summed tp/v_pred/v_gt) must differ from the naive
    # mean of per-slice dice values.
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1"],
            "v_pred": [10, 0, 10],
            "v_gt": [10, 10, 10],
            "tp": [10, 0, 0],
        }
    )
    # per-slice dice: slice0 = 2*10/20=1.0, slice1 = 2*0/10=0.0 (nan-safe: 0/10=0.0),
    # slice2 = 2*0/20=0.0 -> naive mean = 1/3 = 0.3333...
    naive_mean_dice = (1.0 + 0.0 + 0.0) / 3
    result = aggregate_patient(df)
    volume_dice = result.loc["p1", "dice"]
    # volume-level: tp_sum=10, v_pred_sum=20, v_gt_sum=30 -> dice = 2*10/50 = 0.4
    assert volume_dice == pytest.approx(0.4)
    assert volume_dice != pytest.approx(naive_mean_dice)


def test_aggregate_group_col_none_treats_whole_frame_as_one_patient():
    df = pd.DataFrame(
        {
            "v_pred": [10, 20],
            "v_gt": [10, 20],
            "tp": [10, 20],
        }
    )
    result = aggregate_patient(df, group_col=None)
    assert len(result) == 1
    assert result.iloc[0]["dice"] == pytest.approx(1.0)


def test_aggregate_empty_denominator_yields_nan():
    df = pd.DataFrame(
        {"patient_id": ["p1"], "v_pred": [0], "v_gt": [0], "tp": [0]}
    )
    result = aggregate_patient(df)
    assert np.isnan(result.loc["p1", "vs"])
    assert np.isnan(result.loc["p1", "vs_signed"])
    assert np.isnan(result.loc["p1", "dice"])


@pytest.mark.parametrize("seed", range(5))
def test_vs_vs_signed_identity_random_masks(seed):
    rng = np.random.default_rng(seed)
    shape = (30, 30)
    pred = rng.random(shape) > 0.5
    gt = rng.random(shape) > 0.5
    v = vs(pred, gt)
    vsig = vs_signed(pred, gt)
    assert v == pytest.approx(1 - abs(vsig) / 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: `ImportError: cannot import name 'aggregate_patient'`.

- [ ] **Step 3: Write the implementation**

Append to `src/segmentation_metrics/volume.py`:

```python
import pandas as pd


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: all tests passed (12 + 8 new = 20).

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/volume.py tests/test_volume.py
git commit -m "feat: add aggregate_patient for volume-level segmentation metrics"
```

---

### Task 4: Full-suite verification

**Files:** none created/modified — verification only.

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: nothing (terminal task).

- [ ] **Step 1: Run the full new test file with verbose output**

Run: `.venv/bin/python -m pytest tests/test_volume.py -v`
Expected: 20 passed, 0 failed.

- [ ] **Step 2: Run the entire existing project test suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 185 (pre-existing) + 20 (new) = 205 passed, 0 failed.

- [ ] **Step 3: Confirm no new dependencies were introduced**

Run: `git diff main -- requirements.txt`
Expected: empty output (no changes to `requirements.txt`).

No commit needed for this task — it only verifies work already committed in
Tasks 1–3.
