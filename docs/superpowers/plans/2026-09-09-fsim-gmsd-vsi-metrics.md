# FSIM / GMSD / VSI Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three classical full-reference pyiqa metrics FSIM, GMSD and VSI as built-in metrics, following exactly the pattern the ten existing built-ins use.

**Architecture:** Each metric is one `MetricSpec` constant in `src/metrics.py` backed by `PyIQAMetric` via the existing `_pyiqa_factory` helper, appended to the `BUILTIN_METRICS` tuple. Because they are `builtin=True`, both evaluators write their scores with `setattr(record, spec.name, value)`, so `ImageEvaluatorRecord` needs one dedicated `Optional[float]` field per metric. All three are 2D-only, so their `volume_mode` is `ModeUnsupported`. No new files, no new adapter class, no change to `IQAEvaluator` / `VolumeEvaluator` / `EvaluationResult`.

**Tech Stack:** Python 3.14, pyiqa, torch, pytest. venv at `.venv/`.

**Spec:** No separate spec file — this is a bounded change whose design was agreed in chat on 2026-09-09. The design is restated in full under Global Constraints below; the plan is self-contained.

## Global Constraints

- Python must be run from the venv: `source .venv/bin/activate`, or invoke `.venv/bin/python` / `.venv/bin/pytest` directly.
- Tests import src modules with bare names (`from metrics import ...`); `tests/conftest.py` puts `src/` on `sys.path`. Run pytest from the repository root, never from `src/`.
- Metric names, verbatim, lowercase, as pyiqa registers them: `fsim`, `gmsd`, `vsi`.
- Directions, verified by smoke test on 2026-09-09 (identical pair vs. random pair, 2×3×96×96): `fsim` 1.000 → 0.744 (`higher_is_better`), `gmsd` 0.000 → 0.189 (`lower_is_better`), `vsi` 1.000 → 0.857 (`higher_is_better`).
- All three are `reference=True` (full-reference) and `channels="rgb"`. RGB is required, not cosmetic: FSIM's chromatic path and VSI's saliency both need 3 channels, and pyiqa's GMSD defaults to `test_y_channel=True`. `ImageLoader.rgb_tensor` is the channel-replicated grayscale tensor, which satisfies all three.
- `volume_mode` for all three is `ModeUnsupported(REASON_DEEP_2D)`. This is the user's explicit decision, made after being told the reason text ("compares images with a neural network trained on flat 2D pictures…") is factually wrong for these three classical metrics. Do not substitute `REASON_NO_VOLUME_IMPL` and do not add a new reason constant.
- `builtin=True` (the `MetricSpec` default) for all three. Do not pass `description=` or `domain=` — no existing built-in sets them.
- Do not touch `src/iqa_evaluator.py`, `src/volume_evaluator.py`, `src/evaluation_result.py`, or `src/evaluator_factory.py`. They are generic over the registry and need no change.
- Do not add notebook `THRESHOLDS` entries for the three metrics. `THRESHOLDS.get(col, [])` already degrades gracefully to "histogram, no threshold lines", and no MRI-specific literature values are available.
- Commit style: Conventional Commits, and every commit message ends with the trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/metrics.py` | Modify | Three `MetricSpec` constants + three names in `BUILTIN_METRICS`. |
| `src/records.py` | Modify | Three `Optional[float]` fields, so `setattr` in the evaluators lands on a real column. |
| `src/main.py` | Modify | Three names added to the convenience re-export list. |
| `tests/test_metrics.py` | Modify | Spec attributes, bundle membership, volume-mode capability, real pyiqa score behaviour. |
| `tests/test_records.py` | Modify | The three fields exist, default to `None`, and survive `to_dict()`. |
| `tests/test_iqa_evaluator.py` | Modify | End-to-end: a registered built-in lands in its dedicated record field, not in `extra`. |
| `CLAUDE.md` | Modify | Metric summary table + built-in count/constant list. |
| `src/evaluation.ipynb` | Modify | Markdown reference table, metric count, FR-metric list. |

---

## Task 1: Register the three specs in metrics.py

**Files:**
- Modify: `src/metrics.py:270-300` (the built-in spec block and `BUILTIN_METRICS`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `MetricSpec`, `ModeSupport`, `ModeUnsupported`, `REASON_DEEP_2D`, `_pyiqa_factory`, `PyIQAMetric` — all already defined in `src/metrics.py`.
- Produces: module-level constants `FSIM`, `GMSD`, `VSI` of type `MetricSpec`, with `.name` `"fsim"` / `"gmsd"` / `"vsi"`. `BUILTIN_METRICS` becomes a 13-tuple containing them. Task 2 imports these three names.

- [ ] **Step 1: Write the failing spec-attribute and capability tests**

In `tests/test_metrics.py`, add three rows to the `EXPECTED` dict of `class TestBuiltinMetrics` (around line 247), placed after the `radimagenet_lpips` row so the full-reference metrics stay grouped:

```python
        "fsim":               ("higher_is_better", True,  "rgb"),
        "gmsd":               ("lower_is_better",  True,  "rgb"),
        "vsi":                ("higher_is_better", True,  "rgb"),
```

Then, in `class TestBuiltinCapabilities` (around line 454), rename `test_eight_builtins_cannot_do_volume` to `test_eleven_builtins_cannot_do_volume` and add the three names to its expected list. The whole method becomes:

```python
    def test_eleven_builtins_cannot_do_volume(self):
        registry = MetricRegistry(*BUILTIN_METRICS)
        _, skipped = registry.select("volume")
        assert sorted(s.name for s in skipped) == sorted([
            "lpips", "dists", "radimagenet_lpips", "clipiqa",
            "clip_iqa_lung", "clip_iqa_brain", "brisque", "niqe",
            "fsim", "gmsd", "vsi",
        ])
```

Leave `test_all_share_the_same_reason` and `test_no_builtin_is_skipped_in_slice_mode` in that class untouched — the first still passes because all three carry `REASON_DEEP_2D`, and the second still passes because all three support slice mode.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/test_metrics.py -k "TestBuiltinMetrics or TestBuiltinCapabilities" -v
```

Expected: FAIL. `test_all_names_present_in_bundle` fails with `AssertionError: 'fsim' missing from BUILTIN_METRICS`; the three parametrized `test_spec_attributes[fsim-…]` / `[gmsd-…]` / `[vsi-…]` cases fail with `StopIteration` from the `next(...)` lookup; `test_eleven_builtins_cannot_do_volume` fails on the list comparison.

- [ ] **Step 3: Add the three specs**

In `src/metrics.py`, in the full-reference block, immediately after the `RADIMAGENET_LPIPS` line (currently line 288) and before the `# No-reference metrics` comment, insert:

```python
FSIM              = MetricSpec("fsim",              "higher_is_better", True,  "rgb",  ModeSupport(_pyiqa_factory("fsim")),   ModeUnsupported(REASON_DEEP_2D))
GMSD              = MetricSpec("gmsd",              "lower_is_better",  True,  "rgb",  ModeSupport(_pyiqa_factory("gmsd")),   ModeUnsupported(REASON_DEEP_2D))
VSI               = MetricSpec("vsi",               "higher_is_better", True,  "rgb",  ModeSupport(_pyiqa_factory("vsi")),    ModeUnsupported(REASON_DEEP_2D))
```

Then extend the bundle at the bottom of the file so it reads:

```python
# Convenience bundle for "just register everything" — not registered by default.
BUILTIN_METRICS = (
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS, FSIM, GMSD, VSI,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_metrics.py -k "TestBuiltinMetrics or TestBuiltinCapabilities" -v
```

Expected: PASS, all cases.

- [ ] **Step 5: Write the failing score-behaviour test**

The existing `TestPyIQAMetricPSNR` and `TestPyIQAMetricLPIPS` classes prove the adapter actually computes something sane for each family of metric. Add the equivalent for the three new ones. Insert this class in `tests/test_metrics.py` immediately after `class TestPyIQAMetricLPIPS` ends (around line 242), before the `# Built-in metric specs` banner comment:

```python
class TestPyIQAMetricStructuralFR:
    """FSIM / GMSD / VSI: classical full-reference, RGB channel, no downloaded weights.

    `perfect` is the score for two identical images; the metric must move away
    from it once the pair differs. Direction comes from the MetricSpec.
    """

    # name, score for an identical pair, direction
    CASES = [
        ("fsim", 1.0, "higher_is_better"),
        ("gmsd", 0.0, "lower_is_better"),
        ("vsi",  1.0, "higher_is_better"),
    ]

    @pytest.mark.parametrize("name,perfect,_direction", CASES)
    def test_identical_images_hit_the_perfect_score(self, name, perfect, _direction):
        m = PyIQAMetric(name)
        inp = torch.rand(1, 3, 96, 96)
        scores = m(inp, inp)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(perfect, abs=1e-3)

    @pytest.mark.parametrize("name,perfect,direction", CASES)
    def test_different_images_score_worse(self, name, perfect, direction):
        m = PyIQAMetric(name)
        a = torch.rand(1, 3, 96, 96)
        b = torch.rand(1, 3, 96, 96)
        score = m(a, b)[0]
        if direction == "higher_is_better":
            assert score < perfect
        else:
            assert score > perfect

    @pytest.mark.parametrize("name,_perfect,_direction", CASES)
    def test_batch_output_length(self, name, _perfect, _direction):
        m = PyIQAMetric(name)
        inp = torch.rand(3, 3, 96, 96)
        assert len(m(inp, inp)) == 3

    @pytest.mark.parametrize("name,_perfect,_direction", CASES)
    def test_returns_list_of_floats(self, name, _perfect, _direction):
        m = PyIQAMetric(name)
        inp = torch.rand(2, 3, 96, 96)
        assert all(isinstance(s, float) for s in m(inp, inp))
```

- [ ] **Step 6: Run the score-behaviour test**

```bash
.venv/bin/pytest tests/test_metrics.py::TestPyIQAMetricStructuralFR -v
```

Expected: PASS, 12 cases. These tests exercise `PyIQAMetric`, which already exists, so they pass on the first run — that is intentional. They are a characterization test for the three metrics' contract, not a driver for new adapter code. If any case fails, the cause is a real disagreement between the smoke-tested behaviour and the spec: stop and report it rather than loosening the tolerance.

- [ ] **Step 7: Run the whole metrics suite for regressions**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: PASS, no failures.

- [ ] **Step 8: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(metrics): add fsim, gmsd and vsi as built-in full-reference metrics

All three are classical 2D full-reference metrics from pyiqa, registered
through the existing PyIQAMetric adapter. Volume mode is unsupported.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Give the three metrics dedicated record fields

**Files:**
- Modify: `src/records.py:24-29` (the full-reference field block)
- Modify: `src/main.py:9-15` (the re-export import)
- Test: `tests/test_records.py`, `tests/test_iqa_evaluator.py`

**Interfaces:**
- Consumes: `FSIM`, `GMSD`, `VSI` from Task 1.
- Produces: `ImageEvaluatorRecord.fsim`, `.gmsd`, `.vsi`, each `Optional[float] = None`, appearing as report columns in field-declaration order. `main.FSIM` / `main.GMSD` / `main.VSI` re-exported.

Why this task exists: `IQAEvaluator` (`src/iqa_evaluator.py:98-101`) and `VolumeEvaluator` (`src/volume_evaluator.py:117-120`) branch on `spec.builtin` and do `setattr(record, spec.name, value)` for built-ins. Without the field, `setattr` on a non-slotted dataclass silently creates an instance attribute that `dataclasses.asdict()` ignores — the score is computed and then thrown away, with no error. The test in Step 3 is what catches that.

- [ ] **Step 1: Write the failing record-field test**

Append to `tests/test_records.py`:

```python
class TestStructuralFRFields:
    """fsim / gmsd / vsi are builtin metrics, so they need dedicated fields.

    Builtin scores are written with setattr(record, spec.name, value); on a
    dataclass without the field that write is silently dropped by asdict().
    """

    NAMES = ["fsim", "gmsd", "vsi"]

    @pytest.mark.parametrize("name", NAMES)
    def test_field_defaults_to_none(self, name):
        record = ImageEvaluatorRecord(image_id="img")
        assert getattr(record, name) is None

    @pytest.mark.parametrize("name", NAMES)
    def test_field_is_declared_not_ad_hoc(self, name):
        assert name in ImageEvaluatorRecord.__annotations__

    def test_values_survive_to_dict(self):
        record = ImageEvaluatorRecord(image_id="img", fsim=0.9, gmsd=0.05, vsi=0.97)
        d = record.to_dict()
        assert d["fsim"] == 0.9
        assert d["gmsd"] == 0.05
        assert d["vsi"] == 0.97

    def test_not_stored_in_extra(self):
        record = ImageEvaluatorRecord(image_id="img", fsim=0.9)
        assert "fsim" not in record.extra
```

`tests/test_records.py` already has `import pytest` and `from records import ImageEvaluatorRecord` at the top — no import change needed.

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest tests/test_records.py::TestStructuralFRFields -v
```

Expected: FAIL. `test_field_defaults_to_none` fails with `AttributeError: 'ImageEvaluatorRecord' object has no attribute 'fsim'`, `test_field_is_declared_not_ad_hoc` with a plain assertion failure, and the two constructor tests with `TypeError: __init__() got an unexpected keyword argument 'fsim'`.

- [ ] **Step 3: Add the three fields**

In `src/records.py`, in the block under the `# Full-reference metrics` comment, after the `radimagenet_lpips` line:

```python
    fsim:                Optional[float] = None
    gmsd:                Optional[float] = None
    vsi:                 Optional[float] = None
```

- [ ] **Step 4: Run it to verify it passes**

```bash
.venv/bin/pytest tests/test_records.py -v
```

Expected: PASS, no failures.

- [ ] **Step 5: Write the failing end-to-end evaluator test**

This is the test that would have caught a missing field even if Step 1 had been skipped. Append to `tests/test_iqa_evaluator.py`:

```python
class TestStructuralFRMetricsEndToEnd:
    """A registered fsim/gmsd/vsi run must land in the record's own fields."""

    def test_scores_land_in_dedicated_fields(self):
        inp = _make_loader(2, 96, 96)
        tgt = _make_loader(2, 96, 96)
        registry = MetricRegistry(FSIM, GMSD, VSI)
        records = IQAEvaluator(inp, tgt, registry).run_evaluation()

        assert len(records) == 2
        for record in records:
            assert isinstance(record.fsim, float)
            assert isinstance(record.gmsd, float)
            assert isinstance(record.vsi, float)
            assert record.extra == {}
```

Extend that file's existing metrics import to bring the names in:

```python
from metrics import MetricRegistry, MetricSpec, ModeSupport, PSNR, SSIM, FSIM, GMSD, VSI
```

- [ ] **Step 6: Run the end-to-end test**

```bash
.venv/bin/pytest tests/test_iqa_evaluator.py::TestStructuralFRMetricsEndToEnd -v
```

Expected: PASS. `run_evaluation()` is `IQAEvaluator`'s entry point (`src/iqa_evaluator.py:71`) — do not add a wrapper and do not change `iqa_evaluator.py`.

- [ ] **Step 7: Add the re-exports in main.py**

In `src/main.py`, extend the `from metrics import (...)` block so the full-reference line reads:

```python
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS, FSIM, GMSD, VSI,
```

- [ ] **Step 8: Verify the re-export**

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import main; print(main.FSIM.name, main.GMSD.name, main.VSI.name, len(main.BUILTIN_METRICS))"
```

Expected output: `fsim gmsd vsi 13`

- [ ] **Step 9: Run the full suite**

```bash
.venv/bin/pytest -q
```

Expected: PASS, with the pre-existing count (400 as of 2026-09-03) plus the newly added cases, and zero failures. A failure anywhere else means this change had a side effect — stop and report it.

- [ ] **Step 10: Commit**

```bash
git add src/records.py src/main.py tests/test_records.py tests/test_iqa_evaluator.py
git commit -m "$(cat <<'EOF'
feat(records): give fsim, gmsd and vsi their own report columns

Builtin metrics are written with setattr(record, spec.name, value), so
each needs a declared field or asdict() drops the score silently.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update the documentation

**Files:**
- Modify: `CLAUDE.md:43` (built-in count and constant list), `CLAUDE.md:83-86` (metric summary table)
- Modify: `src/evaluation.ipynb` (cell 0 markdown: metric count and reference table; the markdown cell listing FR metrics)
- Test: none — documentation only; verified by reading and by the notebook still parsing as JSON.

**Interfaces:**
- Consumes: the final names and directions from Tasks 1 and 2.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Update the CLAUDE.md metric summary table**

Replace the full-reference rows of the table at `CLAUDE.md:83-86` so the block reads:

```markdown
| psnr, ssim | FR | higher |
| lpips, dists, radimagenet_lpips | FR | lower |
| fsim, vsi | FR | higher |
| gmsd | FR | lower |
| clipiqa, clip_iqa_lung, clip_iqa_brain | NR | higher |
| brisque, niqe | NR | lower |
```

- [ ] **Step 2: Update the built-in count and constant list in CLAUDE.md**

In the `**`metrics.py`**` paragraph at `CLAUDE.md:43`, change `The 10 built-in metrics are exposed as module-level `MetricSpec` constants` to `The 13 built-in metrics are exposed as module-level `MetricSpec` constants`, insert `` `FSIM`, `GMSD`, `VSI`, `` into the constant list after `` `RADIMAGENET_LPIPS`, ``, and change `the `BUILTIN_METRICS` tuple of all ten` to `the `BUILTIN_METRICS` tuple of all thirteen`.

Leave the rest of that paragraph alone. It describes `MetricRegistry` as a singleton named `registry`, which is stale — the code now builds one registry per run — but that staleness predates this change and fixing it is out of scope. Mention it in the handoff instead of silently rewriting it.

- [ ] **Step 3: Update the notebook reference table**

In `src/evaluation.ipynb`, cell 0, add three rows to the markdown threshold table after the `RadImageNet-LPIPS` row. The columns are Metric / Direction / Perfect / Excellent / Good / Average / Bad; these three have no MRI-specific literature thresholds, so mark the graded columns as not established rather than inventing numbers:

```markdown
| FSIM | ↑ higher | 1.0 | — | — | — | — |
| GMSD | ↓ lower | 0 | — | — | — | — |
| VSI | ↑ higher | 1.0 | — | — | — | — |
```

Immediately after the existing BRISQUE/NIQE caveat blockquote in the same cell, add:

```markdown
> **FSIM / GMSD / VSI:** no MRI-specific thresholds are established in the literature, so the graded columns are left blank and these metrics get no threshold lines in the plots below. Use them for relative, within-study comparison.
```

In the same cell, change `using eight IQA metrics implemented in `main.py`` to `using 13 IQA metrics implemented in `main.py``.

- [ ] **Step 4: Update the FR-metric list in the notebook**

In the markdown cell that reads `` `TARGET = None` switches to a no-reference (NR-only) evaluation — full-reference metrics (PSNR, SSIM, LPIPS, DISTS, RadImageNet-LPIPS) will be skipped automatically.``, extend the parenthesized list to `(PSNR, SSIM, LPIPS, DISTS, RadImageNet-LPIPS, FSIM, GMSD, VSI)`.

Do **not** add entries to the `THRESHOLDS` dict in the code cell — `THRESHOLDS.get(col, [])` already handles missing metrics by drawing no threshold lines, and the plot grid sizes itself from `metric_cols`.

- [ ] **Step 5: Verify the notebook is still valid JSON**

```bash
.venv/bin/python -c "import json; nb=json.load(open('src/evaluation.ipynb')); print('cells:', len(nb['cells']))"
```

Expected: prints the cell count without raising.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md src/evaluation.ipynb
git commit -m "$(cat <<'EOF'
docs: document fsim, gmsd and vsi in the metric tables

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Design coverage.** Every point of the agreed design maps to a task: the three specs and `BUILTIN_METRICS` → Task 1 Step 3; `channels="rgb"` and the directions → Task 1 Steps 1 and 5; `ModeUnsupported(REASON_DEEP_2D)` → Task 1 Steps 1 and 3; record fields → Task 2 Step 3; `main.py` re-export → Task 2 Step 7; the two test-file updates in `test_metrics.py` → Task 1 Steps 1 and 5; docs → Task 3; the deliberate omission of notebook thresholds → Global Constraints and Task 3 Step 4.

**Name consistency.** `fsim` / `gmsd` / `vsi` (metric names, record fields, dict keys) and `FSIM` / `GMSD` / `VSI` (spec constants) are used identically in every task. `BUILTIN_METRICS` has 13 entries after Task 1, which is what Task 2 Step 8 asserts.

**Known risk, accepted.** In volume mode the three will be reported as skipped because they "compare images with a neural network trained on flat 2D pictures", which is untrue for phase-congruency, gradient-magnitude and saliency metrics. This is the user's explicit choice, taken to keep `test_all_share_the_same_reason` and the single-group skip message intact.
