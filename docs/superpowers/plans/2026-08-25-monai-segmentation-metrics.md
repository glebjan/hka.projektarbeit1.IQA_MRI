# MONAI Segmentation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five MONAI-backed segmentation-quality metrics (Dice, HD95, NSD, ASSD, Panoptic Quality) to the existing metric registry, following the same adapter/registry pattern as the pyiqa metrics, with per-metric descriptions and a pass-through mechanism for user parameters.

**Architecture:** A new `src/segmentation_metrics/` package holds two adapter classes (`MonaiSegmentationMetric` for the four one-hot-batch metrics, `MonaiPanopticQualityMetric` for PQ, which has a different per-image signature) plus five public builder functions that construct parameterized `MetricSpec`s. `metrics.py` gets two new optional dataclass fields (`description`, `domain`) and re-exports the five default constants plus a `SEGMENTATION_METRICS` bundle, kept separate from `BUILTIN_METRICS` so `main.py`'s raw-image CLI path is untouched.

**Tech Stack:** Python 3.14, PyTorch, MONAI 1.6.0 (functional metrics API: `monai.metrics.compute_dice`, `compute_hausdorff_distance`, `compute_average_surface_distance`, `compute_surface_dice`, `compute_panoptic_quality`), pytest.

## Global Constraints

- Metric protocol stays exactly as-is: `Metric.__call__(input, target=None) -> Sequence[float]` on `(N,C,H,W)` float32 batches in `[0,1]`.
- Segmentation metrics receive pred/gt as already-binarized label mask tensors supplied by the user (no in-framework mask generation, no Otsu).
- `channels="gray"` (single-channel mask), `reference=True` (all five need a target mask) for all five specs.
- `builtin=False` for all five specs — `ImageEvaluatorRecord` is not modified; scores land in `record.extra[name]`, exactly like the existing `register_metric()` path for custom metrics.
- Nothing auto-registers at import time — same opt-in convention as `BUILTIN_METRICS`.
- Segmentation metrics are NOT added to `BUILTIN_METRICS` — `main.py`'s CLI is untouched.
- No changes to `records.py`, `iqa_evaluator.py`, or `main.py`'s registration call.
- Add `monai` to `requirements.txt` (verified working version: 1.6.0).
- Run from `src/` (bare imports, no package prefix), matching the rest of the codebase.

---

## File Structure

```
src/segmentation_metrics/
    __init__.py            # empty, marks package
    monai_metrics.py        # adapters + builder functions + default constants
src/metrics.py               # +2 dataclass fields, +5 re-exported constants, +SEGMENTATION_METRICS bundle
requirements.txt             # +monai
tests/test_segmentation_metrics.py   # adapter-level unit tests (synthetic tensors)
tests/test_metrics.py        # +TestSegmentationMetrics class (spec-attribute + registration tests)
```

---

### Task 1: Add `monai` dependency and verify import

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing
- Produces: `monai` importable in `.venv`

- [ ] **Step 1: Install monai into the project venv**

Run:
```bash
.venv/bin/python -m pip install monai
```
Expected: `Successfully installed monai-1.6.0` (or similar 1.6.x).

- [ ] **Step 2: Verify import works**

Run:
```bash
.venv/bin/python -c "import monai; from monai.metrics import compute_dice, compute_hausdorff_distance, compute_average_surface_distance, compute_surface_dice, compute_panoptic_quality; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Add `monai` to `requirements.txt`**

Open `requirements.txt`, insert alphabetically (between `mpmath` and `multidict`... actually the file isn't alphabetized after edits accumulate — just insert near the other ML deps, e.g. right after `matplotlib==3.9.4`):

```
monai==1.6.0
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add monai dependency for segmentation metrics"
```

---

### Task 2: Extend `MetricSpec` with `description` and `domain` fields

**Files:**
- Modify: `src/metrics.py` (the `MetricSpec` dataclass, currently ending `builtin: bool = True`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `MetricSpec(..., description: str = "", domain: str = "")` — later tasks construct specs passing these two kwargs.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`, inside `TestBuiltinMetrics` or as a small standalone test near it:

```python
class TestMetricSpecDescriptionFields:
    def test_defaults_are_empty_strings(self):
        spec = MetricSpec("dummy", "higher_is_better", False, "gray", lambda: None)
        assert spec.description == ""
        assert spec.domain == ""

    def test_accepts_explicit_values(self):
        spec = MetricSpec(
            "dummy", "higher_is_better", False, "gray", lambda: None,
            description="measures X", domain="medical (MONAI)",
        )
        assert spec.description == "measures X"
        assert spec.domain == "medical (MONAI)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py::TestMetricSpecDescriptionFields -v`
Expected: FAIL — `TypeError: MetricSpec.__init__() got an unexpected keyword argument 'description'`

- [ ] **Step 3: Add the two fields**

In `src/metrics.py`, find:

```python
    name:      str
    direction: MetricDirection
    reference: bool
    channels:  MetricChannels
    factory:   Callable[[], Metric]
    builtin:   bool = True
```

Replace with:

```python
    name:      str
    direction: MetricDirection
    reference: bool
    channels:  MetricChannels
    factory:   Callable[[], Metric]
    builtin:      bool = True
    description:  str  = ""
    domain:       str  = ""
```

Also update the class docstring's `Attributes:` block (directly above) to document the two new fields — append:

```
        description: human-readable explanation of what the metric measures,
                     shown to users choosing a metric.
        domain:      the domain the metric's defaults are calibrated for,
                     e.g. "medical (MONAI)". Empty string means domain-agnostic.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py::TestMetricSpecDescriptionFields tests/test_metrics.py::TestBuiltinMetrics -v`
Expected: all PASS (existing `TestBuiltinMetrics` must still pass unchanged — the two new fields have defaults).

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add description and domain fields to MetricSpec"
```

---

### Task 3: `segmentation_metrics` package skeleton + `MonaiSegmentationMetric` adapter + Dice

**Files:**
- Create: `src/segmentation_metrics/__init__.py`
- Create: `src/segmentation_metrics/monai_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Consumes: `metrics.Metric` protocol (structural, no import needed — duck-typed), `metrics.MetricSpec`, `metrics.MetricDirection`, `metrics.MetricChannels` from `metrics.py`.
- Produces:
  - `class MonaiSegmentationMetric` with `__init__(self, compute_fn, *, threshold=None, **monai_kwargs)` and `__call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]`.
  - `def dice_metric(*, threshold=None, **monai_kwargs) -> MetricSpec`
  - `DICE = dice_metric()` module constant.

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p src/segmentation_metrics
touch src/segmentation_metrics/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_segmentation_metrics.py`:

```python
"""Tests for src/segmentation_metrics/monai_metrics.py — MONAI-backed segmentation metrics."""
import torch
import pytest

from metrics import MetricSpec
from segmentation_metrics.monai_metrics import (
    MonaiSegmentationMetric,
    dice_metric,
    DICE,
)


def _binary_batch(n=2, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randint(0, 2, (n, 1, h, w), generator=g).float()
    gt = torch.randint(0, 2, (n, 1, h, w), generator=g).float()
    return pred, gt


class TestMonaiSegmentationMetricAdapter:
    def test_call_returns_one_score_per_sample(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, include_background=True)
        pred, gt = _binary_batch(n=3)
        scores = metric(pred, gt)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_identical_masks_score_perfect_dice(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, include_background=True)
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)

    def test_threshold_binarizes_before_computing(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, include_background=True, threshold=0.5)
        pred = torch.full((1, 1, 8, 8), 0.7)
        gt = torch.full((1, 1, 8, 8), 0.9)
        scores = metric(pred, gt)
        assert scores[0] == pytest.approx(1.0)  # both binarize to all-ones


class TestDiceMetricBuilder:
    def test_returns_metric_spec(self):
        spec = dice_metric()
        assert isinstance(spec, MetricSpec)
        assert spec.name == "dice"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"
        assert "Dice" in spec.description or "dice" in spec.description

    def test_default_constant_matches_builder_defaults(self):
        assert DICE.name == "dice"
        assert DICE.builtin is False

    def test_factory_produces_working_metric(self):
        spec = dice_metric()
        metric = spec.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2

    def test_user_kwargs_pass_through(self):
        spec = dice_metric(include_background=False)
        metric = spec.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd tests && .venv/../.venv/bin/python -m pytest test_segmentation_metrics.py -v` (or from repo root: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py -v`)
Expected: FAIL — `ModuleNotFoundError: No module named 'segmentation_metrics'`

- [ ] **Step 4: Write `monai_metrics.py`**

Create `src/segmentation_metrics/monai_metrics.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/segmentation_metrics/ tests/test_segmentation_metrics.py
git commit -m "feat: add MonaiSegmentationMetric adapter and dice_metric builder"
```

---

### Task 4: Hausdorff Distance 95 (`hausdorff95_metric`)

**Files:**
- Modify: `src/segmentation_metrics/monai_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Consumes: `MonaiSegmentationMetric` (Task 3), `compute_hausdorff_distance` (already imported in Task 3).
- Produces: `def hausdorff95_metric(*, threshold=None, **monai_kwargs) -> MetricSpec`, `HAUSDORFF95 = hausdorff95_metric()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_segmentation_metrics.py`:

```python
from segmentation_metrics.monai_metrics import hausdorff95_metric, HAUSDORFF95


class TestHausdorff95MetricBuilder:
    def test_returns_metric_spec(self):
        spec = hausdorff95_metric()
        assert spec.name == "hausdorff95"
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_identical_masks_score_zero_distance(self):
        spec = hausdorff95_metric()
        metric = spec.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_default_constant(self):
        assert HAUSDORFF95.name == "hausdorff95"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py::TestHausdorff95MetricBuilder -v`
Expected: FAIL — `ImportError: cannot import name 'hausdorff95_metric'`

- [ ] **Step 3: Add the builder**

In `src/segmentation_metrics/monai_metrics.py`, after `DICE = dice_metric()`, add:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/monai_metrics.py tests/test_segmentation_metrics.py
git commit -m "feat: add hausdorff95_metric builder"
```

---

### Task 5: Normalized Surface Dice (`normalized_surface_dice_metric`)

**Files:**
- Modify: `src/segmentation_metrics/monai_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Consumes: `MonaiSegmentationMetric` (Task 3), `compute_surface_dice` (already imported).
- Produces: `def normalized_surface_dice_metric(*, threshold=None, **monai_kwargs) -> MetricSpec`, `NSD = normalized_surface_dice_metric()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_segmentation_metrics.py`:

```python
from segmentation_metrics.monai_metrics import normalized_surface_dice_metric, NSD


class TestNormalizedSurfaceDiceMetricBuilder:
    def test_returns_metric_spec(self):
        spec = normalized_surface_dice_metric()
        assert spec.name == "nsd"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_identical_masks_score_perfect(self):
        spec = normalized_surface_dice_metric()
        metric = spec.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)

    def test_custom_class_thresholds_override(self):
        spec = normalized_surface_dice_metric(class_thresholds=[2.0])
        metric = spec.factory()
        pred, gt = _binary_batch(n=1)
        scores = metric(pred, gt)
        assert len(scores) == 1

    def test_default_constant(self):
        assert NSD.name == "nsd"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py::TestNormalizedSurfaceDiceMetricBuilder -v`
Expected: FAIL — `ImportError: cannot import name 'normalized_surface_dice_metric'`

- [ ] **Step 3: Add the builder**

In `src/segmentation_metrics/monai_metrics.py`, after `HAUSDORFF95 = hausdorff95_metric()`, add:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/monai_metrics.py tests/test_segmentation_metrics.py
git commit -m "feat: add normalized_surface_dice_metric builder"
```

---

### Task 6: Average Symmetric Surface Distance (`average_surface_distance_metric`)

**Files:**
- Modify: `src/segmentation_metrics/monai_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Consumes: `MonaiSegmentationMetric` (Task 3), `compute_average_surface_distance` (already imported).
- Produces: `def average_surface_distance_metric(*, threshold=None, **monai_kwargs) -> MetricSpec`, `ASSD = average_surface_distance_metric()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_segmentation_metrics.py`:

```python
from segmentation_metrics.monai_metrics import average_surface_distance_metric, ASSD


class TestAverageSurfaceDistanceMetricBuilder:
    def test_returns_metric_spec(self):
        spec = average_surface_distance_metric()
        assert spec.name == "assd"
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_identical_masks_score_zero_distance(self):
        spec = average_surface_distance_metric()
        metric = spec.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_default_constant(self):
        assert ASSD.name == "assd"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py::TestAverageSurfaceDistanceMetricBuilder -v`
Expected: FAIL — `ImportError: cannot import name 'average_surface_distance_metric'`

- [ ] **Step 3: Add the builder**

In `src/segmentation_metrics/monai_metrics.py`, after `NSD = normalized_surface_dice_metric()`, add:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/monai_metrics.py tests/test_segmentation_metrics.py
git commit -m "feat: add average_surface_distance_metric builder"
```

---

### Task 7: Panoptic Quality (`MonaiPanopticQualityMetric` + `panoptic_quality_metric`)

PQ has a different MONAI signature than the other four: `compute_panoptic_quality(pred, gt, ...)` takes a **single** `(H, W)` (or `(H, W, D)`) integer instance-label map per image — no batch or channel dimension, and no `include_background`/`spacing` concept. It needs its own adapter that loops over the batch.

**Files:**
- Modify: `src/segmentation_metrics/monai_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Consumes: `compute_panoptic_quality` (already imported in Task 3).
- Produces: `class MonaiPanopticQualityMetric` with `__init__(self, *, threshold=None, **monai_kwargs)` and `__call__`; `def panoptic_quality_metric(*, threshold=None, **monai_kwargs) -> MetricSpec`; `PANOPTIC_QUALITY = panoptic_quality_metric()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_segmentation_metrics.py`:

```python
from segmentation_metrics.monai_metrics import (
    MonaiPanopticQualityMetric,
    panoptic_quality_metric,
    PANOPTIC_QUALITY,
)


class TestMonaiPanopticQualityMetric:
    def test_call_returns_one_score_per_sample(self):
        metric = MonaiPanopticQualityMetric()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_identical_masks_score_perfect_pq(self):
        metric = MonaiPanopticQualityMetric()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)


class TestPanopticQualityMetricBuilder:
    def test_returns_metric_spec(self):
        spec = panoptic_quality_metric()
        assert spec.name == "panoptic_quality"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_default_constant(self):
        assert PANOPTIC_QUALITY.name == "panoptic_quality"

    def test_factory_produces_working_metric(self):
        spec = panoptic_quality_metric()
        metric = spec.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py::TestMonaiPanopticQualityMetric -v`
Expected: FAIL — `ImportError: cannot import name 'MonaiPanopticQualityMetric'`

- [ ] **Step 3: Add the adapter and builder**

In `src/segmentation_metrics/monai_metrics.py`, after `ASSD = average_surface_distance_metric()`, add:

```python
class MonaiPanopticQualityMetric:
    """Adapter for MONAI's compute_panoptic_quality, which takes one (H, W)
    integer instance-label map per image (no batch/channel dim, unlike the
    other four metrics) — this loops over the batch itself.

    Binary 0/1 masks are treated as a single-instance PQ (foreground = one
    instance). For true multi-instance panoptic quality, supply pred/gt with
    distinct integer instance IDs per object instead of a plain 0/1 mask.
    """

    def __init__(self, *, threshold: Optional[float] = None, **monai_kwargs):
        self._threshold = threshold
        self._kwargs    = monai_kwargs

    def _binarize(self, t: torch.Tensor) -> torch.Tensor:
        return (t > self._threshold).float() if self._threshold is not None else t

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]:
        y_pred = self._binarize(input)
        y      = self._binarize(target)
        scores: list[Optional[float]] = []
        for i in range(y_pred.shape[0]):
            pred_map = y_pred[i, 0].long()
            gt_map   = y[i, 0].long()
            score = compute_panoptic_quality(pred_map, gt_map, **self._kwargs)
            scores.append(None if torch.isnan(score) else float(score.item()))
        return scores


def panoptic_quality_metric(*, threshold: Optional[float] = None, **monai_kwargs) -> MetricSpec:
    """Panoptic Quality (PQ): combines detection accuracy (matching instances
    by IoU) and segmentation accuracy (mean IoU of matched instances) into one score.

    Domain: medical (MONAI) — designed for instance segmentation (e.g.
    individual cells, lesions). For a plain binary mask, PQ degenerates to a
    single-instance IoU-based score. For another domain with multiple
    distinct objects (e.g. grains/particles in a materials micrograph),
    supply pred/gt with a unique integer label per instance instead of a
    binary mask, and tune `match_iou_threshold` (default 0.5) for that
    domain's acceptable localization tolerance.
    """
    monai_kwargs.setdefault("match_iou_threshold", 0.5)
    metric = MonaiPanopticQualityMetric(threshold=threshold, **monai_kwargs)
    return MetricSpec(
        name="panoptic_quality",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        factory=lambda: metric,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_segmentation_metrics.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/segmentation_metrics/monai_metrics.py tests/test_segmentation_metrics.py
git commit -m "feat: add MonaiPanopticQualityMetric adapter and panoptic_quality_metric builder"
```

---

### Task 8: Register constants in `metrics.py` + `SEGMENTATION_METRICS` bundle + registry tests

**Files:**
- Modify: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `DICE`, `HAUSDORFF95`, `NSD`, `ASSD`, `PANOPTIC_QUALITY` from `segmentation_metrics.monai_metrics` (Tasks 3–7).
- Produces: `metrics.SEGMENTATION_METRICS: tuple[MetricSpec, ...]` (5 elements) and the same 5 constants re-exported from `metrics.py`, importable as `from metrics import DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, SEGMENTATION_METRICS`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_metrics.py`, near `TestBuiltinMetrics`:

```python
from segmentation_metrics.monai_metrics import DICE as _DICE  # sanity: same objects re-exported


class TestSegmentationMetrics:
    EXPECTED = {
        "dice":              ("higher_is_better", True, "gray"),
        "hausdorff95":       ("lower_is_better",  True, "gray"),
        "nsd":               ("higher_is_better", True, "gray"),
        "assd":              ("lower_is_better",  True, "gray"),
        "panoptic_quality":  ("higher_is_better", True, "gray"),
    }

    def test_not_registered_by_default(self):
        names = {s.name for s in registry.specs}
        assert not (set(self.EXPECTED) & names)

    def test_all_names_present_in_bundle(self):
        names = {s.name for s in SEGMENTATION_METRICS}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing from SEGMENTATION_METRICS"

    def test_kept_out_of_builtin_metrics(self):
        names = {s.name for s in BUILTIN_METRICS}
        assert not (set(self.EXPECTED) & names)

    @pytest.mark.parametrize("name,attrs", EXPECTED.items())
    def test_spec_attributes(self, name, attrs):
        direction, reference, channels = attrs
        spec = next(s for s in SEGMENTATION_METRICS if s.name == name)
        assert spec.direction == direction, f"{name}: direction mismatch"
        assert spec.reference == reference, f"{name}: reference mismatch"
        assert spec.channels  == channels,  f"{name}: channels mismatch"
        assert spec.builtin   is False,     f"{name}: should not be builtin"
        assert spec.domain    == "medical (MONAI)"
        assert spec.description  # non-empty

    def test_register_opts_in(self, isolated_registry):
        isolated_registry.register(*SEGMENTATION_METRICS)
        names = {s.name for s in isolated_registry.specs}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing after explicit registration"

    def test_metrics_module_reexports_same_objects(self):
        from metrics import DICE
        assert DICE is _DICE
```

Also update the `from metrics import (...)` block at the top of `tests/test_metrics.py` to add:

```python
    SEGMENTATION_METRICS,
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py::TestSegmentationMetrics -v`
Expected: FAIL — `ImportError: cannot import name 'SEGMENTATION_METRICS' from 'metrics'`

- [ ] **Step 3: Wire the constants into `metrics.py`**

In `src/metrics.py`, find the existing import block:

```python
import radimagenet_lpips  # noqa: F401 — registers RadImageNetLPIPS in pyiqa
import clip_iqa_medical   # noqa: F401 — registers ClipIQALung / ClipIQABrain in pyiqa
```

Add directly below it:

```python
from segmentation_metrics.monai_metrics import (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)
```

Then find the end of the file:

```python
# Convenience bundle for "just register everything" — not registered by default.
BUILTIN_METRICS = (
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE,
)
```

Add directly below it:

```python
# MONAI-backed segmentation-quality metrics (evaluate masks, not images) —
# kept separate from BUILTIN_METRICS so main.py's raw-image CLI is unaffected.
SEGMENTATION_METRICS = (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: all PASS (existing tests unaffected, new `TestSegmentationMetrics` passes)

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: register MONAI segmentation metrics in metrics.py"
```

---

### Task 9: Full regression run

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests pass (pre-existing tests + new segmentation-metrics tests), 0 failures.

- [ ] **Step 2: Confirm no accidental auto-registration**

Run:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import metrics
print(len(metrics.registry.specs))
"
```
Expected: `0` — importing `metrics` (and transitively `segmentation_metrics.monai_metrics`) must not register anything.

- [ ] **Step 3: Commit** (only if Steps 1–2 required any fixes; otherwise nothing to commit)

```bash
git add -A
git commit -m "test: verify full regression after MONAI segmentation metrics"
```
