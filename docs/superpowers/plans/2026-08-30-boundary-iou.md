# Boundary IoU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Boundary IoU (Cheng et al., CVPR 2021) to `src/segmentation_metrics/` as a registrable `MetricSpec`, implemented in O(H·W) via a chessboard distance transform instead of iterated erosion.

**Architecture:** One new self-contained module `src/segmentation_metrics/boundary_iou.py` with three layers, built bottom-up: band geometry (`dilation_pixels`, `boundary_region`) → stateless numpy scalar (`boundary_iou`) → `Metric`-protocol adapter and `MetricSpec` builder (`BoundaryIoUMetric`, `boundary_iou_metric`, `BOUNDARY_IOU`). Wire-up appends the constant to `metrics.SEGMENTATION_METRICS` and `main.py`'s re-export list. Mirrors the existing `monai_metrics.py` builder-and-constant shape.

**Tech Stack:** Python 3.14, numpy, `scipy.ndimage` (already pinned at `scipy==1.13.1`), torch, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-boundary-iou-design.md`

## Global Constraints

- **No new dependencies.** `scipy` and `numpy` are already in `requirements.txt`. Do **not** add OpenCV / `cv2`.
- Run everything from the repo root with the venv active: `source .venv/bin/activate`. `tests/conftest.py` puts `src/` on `sys.path`, so tests import bare module names (`from segmentation_metrics.boundary_iou import ...`).
- Tests live in the flat `tests/` directory — no subpackages. New file: `tests/test_boundary_iou.py`.
- Metric name string is exactly `"boundary_iou"`.
- `builtin=False` on the `MetricSpec` (routes the score into `record.extra`, no `ImageEvaluatorRecord` change).
- Paper default `dilation_ratio = 0.02`; band width floors at 1 pixel.
- The one-pixel zero pad before the distance transform is **required** — it makes an object clipped by the image border count that clipped edge as boundary, matching the reference implementation. Do not remove it as an optimisation.
- Every numeric expectation in this plan was verified against a working prototype before the plan was written. If a test fails with a different number, the implementation is wrong, not the expectation.

---

### Task 1: Band geometry

Builds the two primitives the rest of the module rests on: how wide the boundary band is, and which pixels it covers.

**Files:**
- Create: `src/segmentation_metrics/boundary_iou.py`
- Test: `tests/test_boundary_iou.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `DEFAULT_DILATION_RATIO: float = 0.02`
  - `dilation_pixels(shape: tuple[int, int], dilation_ratio: float = DEFAULT_DILATION_RATIO) -> int`
  - `boundary_region(mask: np.ndarray, dilation: int) -> np.ndarray` (2D bool in, 2D bool out)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_boundary_iou.py`:

```python
"""Tests for src/segmentation_metrics/boundary_iou.py — Boundary IoU (Cheng et al., CVPR 2021)."""
import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from segmentation_metrics.boundary_iou import (
    DEFAULT_DILATION_RATIO,
    boundary_region,
    dilation_pixels,
)


def _reference_boundary(mask: np.ndarray, dilation: int) -> np.ndarray:
    """The paper's reference recipe: zero-pad by 1, erode with a 3x3 kernel
    `dilation` times, crop back, subtract. Used to pin our fast distance-
    transform implementation to the published definition."""
    h, w = mask.shape
    padded = np.pad(mask, 1)
    eroded = binary_erosion(
        padded, structure=np.ones((3, 3), bool), iterations=dilation, border_value=0
    )
    return mask & ~eroded[1 : h + 1, 1 : w + 1]


class TestDilationPixels:
    def test_default_ratio_is_paper_value(self):
        assert DEFAULT_DILATION_RATIO == 0.02

    @pytest.mark.parametrize("shape,expected", [
        ((512, 512), 14),   # diag 724.08 * 0.02 = 14.48 -> 14
        ((400, 400), 11),   # diag 565.69 * 0.02 = 11.31 -> 11
        ((200, 100), 4),    # diag 223.61 * 0.02 =  4.47 ->  4
    ])
    def test_scales_with_image_diagonal(self, shape, expected):
        assert dilation_pixels(shape) == expected

    def test_floors_at_one_pixel(self):
        # 10x10 diagonal is 14.14; 0.02 * 14.14 rounds to 0, must clamp to 1.
        assert dilation_pixels((10, 10)) == 1

    def test_explicit_ratio_overrides_default(self):
        assert dilation_pixels((400, 400), 0.1) == 57  # 565.69 * 0.1 = 56.57


class TestBoundaryRegion:
    def test_band_of_a_square_is_the_shell(self):
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True          # 20x20 square
        # d=1 leaves an 18x18 core; d=2 leaves 16x16.
        assert boundary_region(mask, 1).sum() == 400 - 324
        assert boundary_region(mask, 2).sum() == 400 - 256

    def test_band_saturates_at_the_whole_mask(self):
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        assert boundary_region(mask, 10).sum() == 400

    def test_border_clipped_object_counts_its_clipped_edge(self):
        """An object flush against the image border has no pixels outside it,
        but the reference implementation's zero pad still treats that edge as
        boundary — so a corner square bands identically to a free-floating one."""
        corner = np.zeros((40, 40), bool)
        corner[0:20, 0:20] = True
        assert boundary_region(corner, 2).sum() == 144

    def test_full_mask_bands_from_the_image_border_inward(self):
        full = np.ones((10, 10), bool)
        assert boundary_region(full, 1).sum() == 36   # 100 - 8*8
        assert boundary_region(full, 2).sum() == 64   # 100 - 6*6

    def test_empty_mask_has_empty_band(self):
        assert boundary_region(np.zeros((10, 10), bool), 3).sum() == 0

    def test_result_is_a_subset_of_the_mask(self):
        rng = np.random.default_rng(3)
        mask = rng.random((30, 30)) > 0.4
        band = boundary_region(mask, 2)
        assert np.array_equal(band & mask, band)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            boundary_region(np.ones((2, 8, 8), bool), 1)

    @pytest.mark.parametrize("dilation", [1, 2, 3, 5, 9])
    def test_matches_iterated_erosion_reference(self, dilation):
        """Bit-exact agreement with the published cv2-based recipe, including
        masks that touch the image border."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            h, w = rng.integers(6, 40, 2)
            mask = rng.random((h, w)) > rng.uniform(0.2, 0.8)
            mask[:2, :] |= rng.random((2, w)) > 0.5   # force border contact
            np.testing.assert_array_equal(
                boundary_region(mask, dilation), _reference_boundary(mask, dilation)
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'segmentation_metrics.boundary_iou'`

- [ ] **Step 3: Write the implementation**

Create `src/segmentation_metrics/boundary_iou.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py -v`
Expected: PASS — 18 tests (parametrization expands `test_scales_with_image_diagonal` to 3 and `test_matches_iterated_erosion_reference` to 5).

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/boundary_iou.py tests/test_boundary_iou.py
git commit -m "feat(segmentation): boundary band geometry for Boundary IoU"
```

---

### Task 2: The `boundary_iou` scalar

Turns the band primitives into the published metric over a pair of 2D masks, reusing `volume.as_mask` for binarization so label maps and probability maps behave the same as in `volume.py`.

**Files:**
- Modify: `src/segmentation_metrics/boundary_iou.py` (append)
- Test: `tests/test_boundary_iou.py` (append)

**Interfaces:**
- Consumes: `dilation_pixels`, `boundary_region`, `DEFAULT_DILATION_RATIO` from Task 1; `as_mask(x, label, threshold) -> np.ndarray` from `segmentation_metrics.volume`.
- Produces: `boundary_iou(pred, gt, *, dilation_ratio=DEFAULT_DILATION_RATIO, label=1, threshold=0.5) -> float` — returns `float("nan")` when both bands are empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boundary_iou.py` (and extend the existing import from `segmentation_metrics.boundary_iou` to also bring in `boundary_iou`):

```python
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float("nan") if union == 0 else np.count_nonzero(a & b) / union


def _square(size: int, offset: int, image: int = 400) -> np.ndarray:
    mask = np.zeros((image, image), bool)
    mask[offset : offset + size, offset : offset + size] = True
    return mask


class TestBoundaryIoU:
    def test_identical_masks_score_one(self):
        mask = _square(160, 120)
        assert boundary_iou(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks_score_zero(self):
        a = np.zeros((64, 64), bool); a[2:10, 2:10] = True
        b = np.zeros((64, 64), bool); b[50:60, 50:60] = True
        assert boundary_iou(a, b) == 0.0

    def test_both_empty_is_nan(self):
        empty = np.zeros((64, 64), bool)
        assert np.isnan(boundary_iou(empty, empty))

    def test_one_empty_scores_zero(self):
        mask = _square(160, 120)
        assert boundary_iou(mask, np.zeros_like(mask)) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            boundary_iou(np.zeros((8, 8), bool), np.zeros((8, 9), bool))

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError):
            boundary_iou(np.zeros((2, 8, 8), bool), np.zeros((2, 8, 8), bool))

    def test_full_ratio_degenerates_to_mask_iou(self):
        """dilation_ratio=1.0 makes the band the entire mask, so the metric
        must reduce exactly to plain mask IoU."""
        a, b = _square(160, 20), _square(160, 25)
        assert boundary_iou(a, b, dilation_ratio=1.0) == pytest.approx(_mask_iou(a, b))

    def test_penalises_a_shift_more_than_mask_iou(self):
        a, b = _square(160, 20), _square(160, 25)   # 5-pixel diagonal shift
        assert _mask_iou(a, b) == pytest.approx(0.8841, abs=1e-4)
        assert boundary_iou(a, b) == pytest.approx(0.3822, abs=1e-4)

    def test_is_symmetric(self):
        a, b = _square(160, 20), _square(160, 25)
        assert boundary_iou(a, b) == pytest.approx(boundary_iou(b, a))

    def test_is_insensitive_to_object_scale(self):
        """The paper's central claim. For a fixed 5-pixel shift, mask IoU
        improves steeply as the object grows, while Boundary IoU stays flat."""
        sizes = (40, 80, 160, 320)
        mask_scores, boundary_scores = [], []
        for size in sizes:
            offset = (400 - size) // 2
            a, b = _square(size, offset), _square(size, offset + 5)
            mask_scores.append(_mask_iou(a, b))
            boundary_scores.append(boundary_iou(a, b))

        assert mask_scores == sorted(mask_scores)          # rises with size
        assert mask_scores[0] < 0.65 and mask_scores[-1] > 0.93
        assert max(boundary_scores) - min(boundary_scores) < 0.05
        assert all(0.37 < s < 0.42 for s in boundary_scores)

    def test_thresholds_float_probability_masks(self):
        soft_pred = np.where(_square(160, 120), 0.9, 0.1)
        soft_gt = np.where(_square(160, 120), 0.7, 0.2)
        assert boundary_iou(soft_pred, soft_gt, threshold=0.5) == pytest.approx(1.0)

    def test_selects_one_class_from_an_integer_label_map(self):
        labels = np.zeros((64, 64), np.int32)
        labels[10:30, 10:30] = 1
        labels[40:60, 40:60] = 2
        assert boundary_iou(labels, labels, label=2) == pytest.approx(1.0)
        assert boundary_iou(labels, np.zeros_like(labels), label=2) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py::TestBoundaryIoU -v`
Expected: collection error — `ImportError: cannot import name 'boundary_iou'`

- [ ] **Step 3: Write the implementation**

Add the import at the top of `src/segmentation_metrics/boundary_iou.py`, below the `scipy` import:

```python
from segmentation_metrics.volume import as_mask
```

Append to the same file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py -v`
Expected: PASS — all tests, including the Task 1 set.

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/boundary_iou.py tests/test_boundary_iou.py
git commit -m "feat(segmentation): boundary_iou scalar over 2D mask pairs"
```

---

### Task 3: Metric adapter and MetricSpec builder

Wraps the scalar in the framework's `Metric` protocol — one score per sample of an `(N, C, H, W)` batch — and exposes it as a `MetricSpec`.

**Files:**
- Modify: `src/segmentation_metrics/boundary_iou.py` (append)
- Test: `tests/test_boundary_iou.py` (append)

**Interfaces:**
- Consumes: `boundary_iou`, `DEFAULT_DILATION_RATIO` from Task 2; `MetricSpec` from `metrics`.
- Produces:
  - `BoundaryIoUMetric(*, dilation_ratio=DEFAULT_DILATION_RATIO, threshold=0.5, label=1)` with `__call__(input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]`
  - `boundary_iou_metric(*, dilation_ratio=DEFAULT_DILATION_RATIO, threshold=0.5, label=1) -> MetricSpec`
  - `BOUNDARY_IOU: MetricSpec` — `boundary_iou_metric()` with defaults

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boundary_iou.py` (extend the module imports to add `BOUNDARY_IOU`, `BoundaryIoUMetric`, `boundary_iou_metric`, and add `import torch` plus `from metrics import MetricSpec` at the top):

```python
def _batch(masks: list[np.ndarray]) -> torch.Tensor:
    """Stack 2D masks into the framework's (N, 1, H, W) float32 tensor."""
    return torch.from_numpy(np.stack(masks)[:, None].astype(np.float32))


class TestBoundaryIoUMetricAdapter:
    def test_scores_each_sample_independently(self):
        """Three samples with three distinct expected scores — a per-sample bug
        (e.g. scoring the whole batch at once) cannot pass this."""
        metric = BoundaryIoUMetric()
        a, b = _square(160, 20), _square(160, 25)
        empty = np.zeros((400, 400), bool)
        scores = metric(_batch([a, a, a]), _batch([a, b, empty]))
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.3822, abs=1e-4)
        assert scores[2] == 0.0

    def test_reports_none_for_undefined_scores(self):
        metric = BoundaryIoUMetric()
        empty = np.zeros((400, 400), bool)
        solid = _square(160, 120)
        scores = metric(_batch([empty, solid]), _batch([empty, solid]))
        assert scores[0] is None
        assert scores[1] == pytest.approx(1.0)

    def test_missing_target_raises(self):
        metric = BoundaryIoUMetric()
        with pytest.raises(ValueError):
            metric(_batch([_square(160, 120)]))

    def test_dilation_ratio_is_forwarded(self):
        a, b = _square(160, 20), _square(160, 25)
        wide = BoundaryIoUMetric(dilation_ratio=1.0)(_batch([a]), _batch([b]))
        assert wide[0] == pytest.approx(_mask_iou(a, b))

    def test_threshold_binarizes_soft_masks(self):
        soft_pred = np.where(_square(160, 120), 0.9, 0.1)
        soft_gt = np.where(_square(160, 120), 0.7, 0.2)
        scores = BoundaryIoUMetric(threshold=0.5)(_batch([soft_pred]), _batch([soft_gt]))
        assert scores[0] == pytest.approx(1.0)


class TestBoundaryIoUMetricBuilder:
    def test_returns_metric_spec(self):
        spec = boundary_iou_metric()
        assert isinstance(spec, MetricSpec)
        assert spec.name == "boundary_iou"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == ""
        assert "Boundary IoU" in spec.description

    def test_default_constant_matches_builder_defaults(self):
        assert BOUNDARY_IOU.name == "boundary_iou"
        assert BOUNDARY_IOU.builtin is False

    def test_factory_produces_a_working_metric(self):
        metric = BOUNDARY_IOU.factory()
        mask = _square(160, 120)
        assert metric(_batch([mask]), _batch([mask]))[0] == pytest.approx(1.0)

    def test_builder_overrides_reach_the_metric(self):
        a, b = _square(160, 20), _square(160, 25)
        metric = boundary_iou_metric(dilation_ratio=1.0).factory()
        assert metric(_batch([a]), _batch([b]))[0] == pytest.approx(_mask_iou(a, b))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py -v`
Expected: collection error — `ImportError: cannot import name 'BoundaryIoUMetric'`

- [ ] **Step 3: Write the implementation**

Extend the imports at the top of `src/segmentation_metrics/boundary_iou.py`:

```python
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt

from metrics import MetricSpec
from segmentation_metrics.volume import as_mask
```

Append to the same file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py -v`
Expected: PASS — all tests.

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/boundary_iou.py tests/test_boundary_iou.py
git commit -m "feat(segmentation): BoundaryIoUMetric adapter and MetricSpec builder"
```

---

### Task 4: Register with the framework

Makes `BOUNDARY_IOU` reachable the same way `DICE` and friends are: exported from `metrics.py`, bundled in `SEGMENTATION_METRICS`, re-exported from `main.py`.

**Files:**
- Modify: `src/metrics.py:156-158` (the late segmentation import) and the `SEGMENTATION_METRICS` tuple at the end of the file
- Modify: `src/main.py:9-13` (the re-export list)
- Test: `tests/test_boundary_iou.py` (append)

**Interfaces:**
- Consumes: `BOUNDARY_IOU` from Task 3.
- Produces: `metrics.BOUNDARY_IOU`, `main.BOUNDARY_IOU`, and `BOUNDARY_IOU` as the sixth entry of `metrics.SEGMENTATION_METRICS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boundary_iou.py`:

```python
class TestFrameworkRegistration:
    def test_exported_from_metrics(self):
        import metrics
        assert metrics.BOUNDARY_IOU is BOUNDARY_IOU

    def test_included_in_segmentation_bundle(self):
        from metrics import SEGMENTATION_METRICS
        assert BOUNDARY_IOU in SEGMENTATION_METRICS

    def test_not_in_builtin_bundle(self):
        """main.py's raw-image CLI must stay unaffected by segmentation metrics."""
        from metrics import BUILTIN_METRICS
        assert BOUNDARY_IOU not in BUILTIN_METRICS

    def test_reexported_from_main(self):
        import main
        assert main.BOUNDARY_IOU is BOUNDARY_IOU

    def test_registry_round_trip(self):
        from metrics import MetricRegistry
        registry = MetricRegistry(BOUNDARY_IOU)
        assert "boundary_iou" in registry.direction
        assert registry.direction["boundary_iou"] == "higher_is_better"
        metric = registry.get_metric("boundary_iou")
        mask = _square(160, 120)
        assert metric(_batch([mask]), _batch([mask]))[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_boundary_iou.py::TestFrameworkRegistration -v`
Expected: FAIL — `AttributeError: module 'metrics' has no attribute 'BOUNDARY_IOU'`

- [ ] **Step 3: Wire it up**

In `src/metrics.py`, extend the existing late import block (currently lines 153-158) so it reads:

```python
# Imported here (rather than alongside the other module-level imports above)
# to avoid a circular import: monai_metrics.py does `from metrics import
# MetricSpec`, which requires MetricSpec to already be defined in this module.
from segmentation_metrics.monai_metrics import (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)
from segmentation_metrics.boundary_iou import BOUNDARY_IOU
```

At the end of `src/metrics.py`, extend the bundle:

```python
# MONAI-backed segmentation-quality metrics (evaluate masks, not images) —
# kept separate from BUILTIN_METRICS so main.py's raw-image CLI is unaffected.
SEGMENTATION_METRICS = (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU,
)
```

In `src/main.py`, add `BOUNDARY_IOU` to the segmentation line of the re-export block (currently line 13):

```python
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU, SEGMENTATION_METRICS,
```

- [ ] **Step 4: Run the full suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: PASS — the whole suite, no regressions. Boundary IoU adds no new dependency, so nothing else should move.

- [ ] **Step 5: Verify no new dependency crept in**

Run: `source .venv/bin/activate && python -c "import cv2" 2>&1 | head -1`
Expected: `ModuleNotFoundError: No module named 'cv2'` — confirming the implementation does not rely on OpenCV. Also confirm `git diff --stat requirements.txt` is empty.

- [ ] **Step 6: Commit**

```bash
git add src/metrics.py src/main.py tests/test_boundary_iou.py
git commit -m "feat(segmentation): register boundary_iou in the metric bundle"
```

---

## Notes for the executor

- **`SEGMENTATION_METRICS` ordering.** Append `BOUNDARY_IOU` at the end; the MONAI five stay in their current order so the diff stays readable.
- **The comment above the late import** in `metrics.py` explains the circular-import reason and still applies to the new line — leave it in place rather than duplicating it.
- **Do not touch `CLAUDE.md`.** It is already out of date on this package (it documents a removed `mask_writer.py` and does not mention `segmentation_metrics/` at all). Fixing it is a separate job; expanding this task into a documentation sweep is scope creep.
- **Do not add Boundary IoU to `BUILTIN_METRICS`.** It scores masks, not images, and `main.py`'s CLI runs on raw images.
