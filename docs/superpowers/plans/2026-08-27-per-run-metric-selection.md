# Per-Run Metric Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each evaluation run pick its own metrics by replacing the module-level `registry` singleton with explicitly-passed `MetricRegistry` instances.

**Architecture:** `MetricRegistry` becomes an ordinary instantiable class whose constructor takes specs. The singleton is deleted and the registry travels explicitly: `evaluate(..., registry=)` → `IQAEvaluator(..., registry)` → `EvaluationResult(images, registry)`. `MaskWriter`/Otsu and `best_slice_per_metric()` are deleted outright, removing the last downstream consumer that needed global metric directions.

**Tech Stack:** Python 3.14, PyTorch, pyiqa, MONAI 1.6.0, pandas, pytest.

## Global Constraints

- The `Metric` protocol is unchanged: `__call__(input, target=None) -> Sequence[float]` on `(N,C,H,W)` float32 batches in `[0,1]`.
- `MetricSpec` is unchanged — no new fields, no changed defaults.
- No metric adapter changes (`PyIQAMetric`, `MonaiSegmentationMetric`, `MonaiPanopticQualityMetric`).
- No changes to the metric constants or the `BUILTIN_METRICS` / `SEGMENTATION_METRICS` bundles.
- `registry` is a **required** parameter everywhere it appears — no defaults, no global fallback.
- `IQAEvaluator`'s `registry` parameter is **positional** (third parameter, before `source_model`).
- `evaluate()`'s `registry` parameter is **keyword-only**.
- One `MetricRegistry` instance is shared across all images within a single `evaluate()` run, so network-backed metrics build once per run, not once per image.
- Run everything from the repo root with `.venv/bin/python` (never bare `python`/`pip`); the project's `tests/conftest.py` puts `src/` on `sys.path`, so modules import bare (`from metrics import ...`).
- CLAUDE.md is explicitly NOT updated by this plan (it documents the removed singleton; a separate follow-up).

---

## File Structure

```
src/metrics.py               # MetricRegistry.__init__(*specs); register_metric becomes a method; singleton deleted
src/iqa_evaluator.py         # IQAEvaluator gains required positional `registry`
src/records.py               # best_slice_per_metric + _record_metric_value deleted; no metrics import left
src/evaluation_result.py     # EvaluationResult(images, registry); generate_report loses mask_dir + MaskWriter call
src/main.py                  # evaluate(..., *, registry); main() builds MetricRegistry; re-exports updated
src/mask_writer.py           # DELETED
tests/conftest.py            # isolated_registry fixture DELETED
tests/test_metrics.py        # constructor/method/isolation tests
tests/test_iqa_evaluator.py  # registry argument threaded through
tests/test_records.py        # best_slice_per_metric test class DELETED
tests/test_evaluation_result.py  # EvaluationResult(images, registry); mask PNG test DELETED
tests/test_main.py           # _restrict_to_fast replaced by local MetricRegistry
tests/test_mask_writer.py    # DELETED
src/evaluation.ipynb         # setup cell builds its own MetricRegistry; evaluate() call gains registry=
```

Task order matters: `metrics.py` first (everything depends on it), then the deletions that shrink the surface (`MaskWriter`/`best_slice_per_metric`), then the consumers, then the final integration test.

---

### Task 1: `MetricRegistry` takes specs in its constructor; `register_metric` becomes a method

**Files:**
- Modify: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `MetricRegistry(*specs: MetricSpec)` — constructor now accepts specs.
  - `MetricRegistry.register_metric(self, name: str, metric: Metric, *, direction: MetricDirection, reference: bool, channels: MetricChannels = "rgb") -> None` — moved from module level, same parameters and semantics.
  - The module-level `registry` object and the free `register_metric` function no longer exist.
  - `MetricRegistry.register(*specs)`, `.get_metric(name)`, `.specs`, `.direction` unchanged.

- [ ] **Step 1: Write the failing tests**

In `tests/test_metrics.py`, add this class (put it directly after the existing `TestMetricRegistry` class):

```python
class TestMetricRegistryInstances:
    def test_constructor_accepts_specs(self):
        reg = MetricRegistry(PSNR, SSIM)
        assert {s.name for s in reg.specs} == {"psnr", "ssim"}

    def test_constructor_empty_by_default(self):
        reg = MetricRegistry()
        assert reg.specs == []

    def test_instances_are_independent(self):
        reg_a = MetricRegistry(PSNR)
        reg_b = MetricRegistry(SSIM)
        assert {s.name for s in reg_a.specs} == {"psnr"}
        assert {s.name for s in reg_b.specs} == {"ssim"}

    def test_caches_are_independent(self, fake_metric):
        reg_a = MetricRegistry()
        reg_b = MetricRegistry()
        reg_a.register_metric("shared_name", fake_metric,
                              direction="higher_is_better", reference=False)
        assert reg_a.get_metric("shared_name") is fake_metric
        with pytest.raises(KeyError):
            reg_b.get_metric("shared_name")

    def test_register_metric_is_a_method(self, fake_metric):
        reg = MetricRegistry()
        reg.register_metric("custom", fake_metric,
                            direction="lower_is_better", reference=True, channels="gray")
        spec = next(s for s in reg.specs if s.name == "custom")
        assert spec.builtin is False
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert reg.get_metric("custom") is fake_metric

    def test_no_module_level_singleton(self):
        import metrics
        assert not hasattr(metrics, "registry"), "global registry singleton must be gone"
        assert not callable(getattr(metrics, "register_metric", None)), \
            "free register_metric function must be gone"
```

Then update the import block at the top of `tests/test_metrics.py`: remove `register_metric` and `registry` from the `from metrics import (...)` list, and make sure `MetricRegistry`, `MetricSpec`, `PSNR`, `SSIM`, `BUILTIN_METRICS`, `SEGMENTATION_METRICS`, `DEVICE`, `PyIQAMetric`, `_pyiqa_factory` remain.

Existing tests in that file that reference the global `registry` must be updated in this same step:
- `TestBuiltinMetrics.test_not_registered_by_default` and `TestSegmentationMetrics.test_not_registered_by_default` both assert against the global. Replace each body with a check against a fresh instance:

```python
    def test_not_registered_by_default(self):
        # A fresh registry starts empty — nothing self-registers at import time.
        assert MetricRegistry().specs == []
```

- Any test using the `isolated_registry` fixture in this file: replace the fixture parameter with a locally constructed `MetricRegistry(...)`. For example `TestBuiltinMetrics.test_register_opts_in` becomes:

```python
    def test_register_opts_in(self):
        reg = MetricRegistry()
        reg.register(*BUILTIN_METRICS)
        names = {s.name for s in reg.specs}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing after explicit registration"
```

and `TestSegmentationMetrics.test_register_opts_in` becomes the same with `SEGMENTATION_METRICS`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `TypeError: MetricRegistry() takes no arguments` / `AttributeError: 'MetricRegistry' object has no attribute 'register_metric'`, and `test_no_module_level_singleton` fails because `metrics.registry` still exists.

- [ ] **Step 3: Rewrite the registry section of `src/metrics.py`**

Replace the whole block from `class MetricRegistry:` down to and including the free `register_metric` function (i.e. everything between `class MetricRegistry:` and the `# ---` divider that introduces "Built-in metrics (pyiqa-backed)") with:

```python
class MetricRegistry:
    """Holds registered MetricSpecs and lazily-instantiated Metric objects.

    Build one instance per evaluation run and pass it explicitly — there is
    no global registry. Two instances never share specs or cached metric
    objects, so an IQA run and a segmentation run can proceed side by side:

        iqa = MetricRegistry(*BUILTIN_METRICS)
        seg = MetricRegistry(*SEGMENTATION_METRICS)
    """

    def __init__(self, *specs: MetricSpec):
        self._specs: dict[str, MetricSpec] = {}
        self._cache: dict[str, Metric] = {}
        self.register(*specs)

    def register(self, *specs: MetricSpec) -> None:
        for spec in specs:
            self._specs[spec.name] = spec
            self._cache.pop(spec.name, None)

    def register_metric(
        self,
        name: str,
        metric: Metric,
        *,
        direction: MetricDirection,
        reference: bool,
        channels: MetricChannels = "rgb",
    ) -> None:
        """Hook a custom metric into this registry.

        No pyiqa import and no edit to main.py or ImageEvaluatorRecord
        required. `metric` just needs to implement the Metric protocol. Its
        scores show up as a column named `name` in the report (via
        ImageEvaluatorRecord.extra).
        """
        self.register(MetricSpec(name, direction, reference, channels,
                                 factory=lambda: metric, builtin=False))

    def get_metric(self, name: str) -> Metric:
        if name not in self._cache:
            self._cache[name] = self._specs[name].factory()
        return self._cache[name]

    @property
    def specs(self) -> list[MetricSpec]:
        return list(self._specs.values())

    @property
    def direction(self) -> dict[str, MetricDirection]:
        return {spec.name: spec.direction for spec in self._specs.values()}
```

Note what is gone: the `registry = MetricRegistry()` line and the module-level `def register_metric(...)` function.

Also update the module docstring at the top of `src/metrics.py` — replace the two paragraphs that read:

```
To add a custom metric without touching main.py or pyiqa, call
`register_metric()` with an object implementing `Metric`.

Built-in metrics (below) are exposed as `MetricSpec` constants (`PSNR`,
`SSIM`, ...) — nothing is registered until the caller opts in by calling
`registry.register(...)` with the ones they want.
```

with:

```
Metrics are held by `MetricRegistry` instances — build one per evaluation
run and pass it to IQAEvaluator/evaluate(). There is no global registry, so
an IQA run and a segmentation run never interfere.

To add a custom metric without touching main.py or pyiqa, call
`MetricRegistry.register_metric()` with an object implementing `Metric`.

Built-in metrics (below) are exposed as `MetricSpec` constants (`PSNR`,
`SSIM`, ...) — nothing is registered until the caller opts in by passing
them to a registry, e.g. `MetricRegistry(PSNR, SSIM)`.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: all PASS.

Other test files still import the deleted `registry`/`register_metric` and will fail at collection — that is expected and gets fixed in Tasks 3–6. Do not fix them here.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "refactor: make MetricRegistry instantiable, drop global singleton"
```

---

### Task 2: Delete `MaskWriter`/Otsu and `best_slice_per_metric`

**Files:**
- Delete: `src/mask_writer.py`
- Delete: `tests/test_mask_writer.py`
- Modify: `src/records.py`
- Test: `tests/test_records.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (independent deletion).
- Produces: `records.py` exports only `ImageEvaluatorRecord` and imports nothing from `metrics.py`. `MaskWriter`, `MASK_DIR`, `best_slice_per_metric`, and `_record_metric_value` no longer exist anywhere.

- [ ] **Step 1: Write the failing test**

In `tests/test_records.py`, delete the entire `best_slice_per_metric` test class (the class starting after the `# best_slice_per_metric` comment banner at roughly line 67, containing `test_higher_is_better_selects_max`, `test_lower_is_better_selects_min`, `test_empty_slices_skipped`, `test_none_values_ignored`) along with that comment banner. Update the module docstring's first line from:

```python
"""Tests for src/records.py — ImageEvaluatorRecord, best_slice_per_metric."""
```

to:

```python
"""Tests for src/records.py — ImageEvaluatorRecord."""
```

Update the import line — it currently reads:

```python
from records import ImageEvaluatorRecord, _record_metric_value, best_slice_per_metric
```

Change it to:

```python
from records import ImageEvaluatorRecord
```

Then remove any now-unused imports at the top of that file (e.g. `PSNR`, `LPIPS` imported from `metrics` solely for the deleted class) and delete any test that called `_record_metric_value` directly.

Add this guard test at the end of `tests/test_records.py`:

```python
class TestRecordsHasNoMetricsDependency:
    def test_best_slice_per_metric_is_gone(self):
        import records
        assert not hasattr(records, "best_slice_per_metric")
        assert not hasattr(records, "_record_metric_value")

    def test_mask_writer_module_is_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import mask_writer  # noqa: F401
```

Make sure `import pytest` is present at the top of `tests/test_records.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_records.py::TestRecordsHasNoMetricsDependency -v`
Expected: FAIL — `assert not hasattr(records, "best_slice_per_metric")` fails, and `import mask_writer` succeeds instead of raising.

- [ ] **Step 3: Delete the files and the functions**

```bash
git rm src/mask_writer.py tests/test_mask_writer.py
```

In `src/records.py`, delete everything from the line `def _record_metric_value(` to the end of the file (that is `_record_metric_value` and `best_slice_per_metric` in full), and delete the now-unused import line:

```python
from metrics import registry
```

`src/records.py` ends after `ImageEvaluatorRecord.to_dict()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_records.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/records.py tests/test_records.py
git commit -m "refactor: remove MaskWriter/Otsu and best_slice_per_metric"
```

---

### Task 3: `IQAEvaluator` takes a required registry

**Files:**
- Modify: `src/iqa_evaluator.py`
- Test: `tests/test_iqa_evaluator.py`

**Interfaces:**
- Consumes: `MetricRegistry(*specs)` and `MetricRegistry.register_metric(...)` from Task 1.
- Produces: `IQAEvaluator(input_image: ImageLoader, target_image: Optional[ImageLoader], registry: MetricRegistry, source_model: Optional[str] = None)` — `registry` is the third positional parameter. `self.registry` is the attribute name.

- [ ] **Step 1: Write the failing tests**

In `tests/test_iqa_evaluator.py`, change the import line from:

```python
from metrics import MetricSpec, register_metric, registry, PSNR, SSIM
```

to:

```python
from metrics import MetricRegistry, MetricSpec, PSNR, SSIM
```

Every existing `IQAEvaluator(...)` construction in the file needs the registry argument. For the `TestIQAEvaluatorInit` class, that means:

```python
class TestIQAEvaluatorInit:
    def test_shape_mismatch_raises(self):
        inp = _make_loader(2, 64, 64)
        tgt = _make_loader(2, 32, 32)   # Different spatial size
        with pytest.raises(ValueError, match="shape mismatch"):
            IQAEvaluator(inp, tgt, MetricRegistry())

    def test_matching_shapes_ok(self):
        inp = _make_loader(2, 64, 64)
        tgt = _make_loader(2, 64, 64)
        ev = IQAEvaluator(inp, tgt, MetricRegistry())
        assert ev.input is inp and ev.target is tgt

    def test_no_target_ok(self):
        inp = _make_loader(3)
        ev = IQAEvaluator(inp, None, MetricRegistry())
        assert ev.target is None

    def test_registry_is_stored(self):
        inp = _make_loader(2)
        reg = MetricRegistry(PSNR)
        ev = IQAEvaluator(inp, None, reg)
        assert ev.registry is reg
```

Apply the same mechanical change to every other `IQAEvaluator(...)` call in the file: add a `MetricRegistry(...)` third argument holding whichever metrics that test needs (`MetricRegistry()` where the test does not care about metrics, `MetricRegistry(PSNR, SSIM)` where it previously relied on registered metrics). Any test that used `isolated_registry` or the free `register_metric` uses a local `MetricRegistry` and its `register_metric` method instead — e.g. a test that registered a fake metric becomes:

```python
        reg = MetricRegistry()
        reg.register_metric("fake", fake_metric,
                            direction="higher_is_better", reference=False)
        ev = IQAEvaluator(inp, None, reg)
```

Add this new test class at the end of the file — it is the point of the whole change:

```python
class TestPerRunMetricSelection:
    def test_two_evaluators_compute_different_metrics(self, fake_metric):
        inp = _make_loader(2)

        reg_a = MetricRegistry()
        reg_a.register_metric("metric_a", fake_metric,
                              direction="higher_is_better", reference=False)
        reg_b = MetricRegistry()
        reg_b.register_metric("metric_b", fake_metric,
                              direction="higher_is_better", reference=False)

        recs_a = IQAEvaluator(inp, None, reg_a).run_evaluation()
        recs_b = IQAEvaluator(inp, None, reg_b).run_evaluation()

        assert "metric_a" in recs_a[0].extra
        assert "metric_b" not in recs_a[0].extra
        assert "metric_b" in recs_b[0].extra
        assert "metric_a" not in recs_b[0].extra
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_iqa_evaluator.py -v`
Expected: FAIL — `TypeError: IQAEvaluator.__init__() takes 3 to 4 positional arguments but 4 were given`, and `AttributeError: 'IQAEvaluator' object has no attribute 'registry'`.

- [ ] **Step 3: Thread the registry through `src/iqa_evaluator.py`**

Change the import line from:

```python
from metrics import DEVICE, MetricChannels, MetricSpec, registry
```

to:

```python
from metrics import DEVICE, MetricChannels, MetricRegistry, MetricSpec
```

Replace the `__init__` signature and body's first lines:

```python
    def __init__(
        self,
        input_image:  ImageLoader,
        target_image: Optional[ImageLoader],
        registry:     MetricRegistry,
        source_model: Optional[str] = None,
    ):
        self.input        = input_image
        self.target       = target_image
        self.registry     = registry
        self.source_model = source_model
```

(the shape-mismatch `ValueError` check below it stays exactly as-is)

In `_compute_batch`, change:

```python
        metric = registry.get_metric(spec.name)
```

to:

```python
        metric = self.registry.get_metric(spec.name)
```

In `run_evaluation`, change:

```python
        for spec in registry.specs:
```

to:

```python
        for spec in self.registry.specs:
```

Update the class docstring's first line from:

```python
    """Computes all registered IQA metrics for one input/target image pair.
```

to:

```python
    """Computes one registry's metrics for one input/target image pair.

    The metric set comes from the `registry` passed in, so different
    evaluators in the same process can compute different metrics.
```

(keep the existing "The evaluator is intentionally free of file I/O..." paragraph, but drop its now-stale mention of MaskWriter — it should read "Writing results to disk is handled by EvaluationResult (CSV).")

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_iqa_evaluator.py -v`
Expected: all PASS, including `TestPerRunMetricSelection`.

- [ ] **Step 5: Commit**

```bash
git add src/iqa_evaluator.py tests/test_iqa_evaluator.py
git commit -m "refactor: IQAEvaluator takes an explicit MetricRegistry"
```

---

### Task 4: `EvaluationResult` takes the run's registry; `generate_report` drops masks

**Files:**
- Modify: `src/evaluation_result.py`
- Test: `tests/test_evaluation_result.py`

**Interfaces:**
- Consumes: `MetricRegistry` from Task 1; `IQAEvaluator(inp, tgt, registry)` from Task 3; `MaskWriter` no longer exists (Task 2).
- Produces: `EvaluationResult(images: list[_EvaluatedImage], registry: MetricRegistry)` with `self._registry`; `generate_report(report_path: Path = REPORT) -> pd.DataFrame` (no `mask_dir` parameter, writes CSV only).

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluation_result.py`:

Change the import line `from metrics import register_metric` to `from metrics import MetricRegistry, PSNR, SSIM`.

Update the `_make_result` helper so it builds a registry — change its signature and its final return:

```python
def _make_result(n_images: int = 2, slices_each: int = 2,
                 registry: MetricRegistry | None = None) -> EvaluationResult:
```

and its last line from `return EvaluationResult(images)` to:

```python
    return EvaluationResult(images, registry if registry is not None else MetricRegistry())
```

Replace `test_custom_metric_column` with:

```python
    def test_custom_metric_column(self, fake_metric):
        reg = MetricRegistry()
        reg.register_metric("frame_custom", fake_metric,
                            direction="higher_is_better", reference=False)
        rec = ImageEvaluatorRecord(image_id="t", slice_index=0)
        rec.extra["frame_custom"] = 0.7
        images = [_EvaluatedImage(input_path=Path("/fake/t.png"), records=[rec])]
        res = EvaluationResult(images, reg)
        df = res.to_frame()
        assert "frame_custom" in df.columns
```

Replace the `_build_result_from_real_image` helper with:

```python
    def _build_result_from_real_image(self, tmp_path: Path) -> EvaluationResult:
        """Create a minimal real EvaluationResult from a synthetic PNG."""
        from image_loader import ImageLoader
        from iqa_evaluator import IQAEvaluator

        reg = MetricRegistry(PSNR, SSIM)   # fast metrics only

        arr = (np.random.default_rng(5).random((96, 96)) * 255).astype("uint8")
        inp_p = tmp_path / "inp.png"; tgt_p = tmp_path / "tgt.png"
        Image.fromarray(arr).save(inp_p)
        noise = np.clip(arr.astype(int) + 5, 0, 255).astype("uint8")
        Image.fromarray(noise).save(tgt_p)

        inp = ImageLoader(inp_p); tgt = ImageLoader(tgt_p)
        records = IQAEvaluator(inp, tgt, reg).run_evaluation()
        return EvaluationResult([_EvaluatedImage(input_path=inp_p, records=records)], reg)
```

Delete `test_mask_pngs_written` entirely (MaskWriter is gone). Update the remaining `generate_report` tests to drop the `isolated_registry` fixture parameter and the mask-dir argument:

```python
    def test_csv_written(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        csv_path = tmp_path / "report.csv"
        res.generate_report(csv_path)
        assert csv_path.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        csv_path = tmp_path / "report.csv"
        res.generate_report(csv_path)
        import pandas as pd
        loaded = pd.read_csv(csv_path)
        assert "image_id" in loaded.columns
        assert "psnr" in loaded.columns

    def test_returns_dataframe_equal_to_to_frame(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        df_direct = res.to_frame()
        df_report = res.generate_report(tmp_path / "r.csv")
        assert list(df_direct["image_id"]) == list(df_report["image_id"])

    def test_no_mask_dir_created(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        res.generate_report(tmp_path / "r.csv")
        assert not (tmp_path / "masks").exists()
```

Also drop the `isolated_registry` fixture parameter from any other test in this file that still takes it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evaluation_result.py -v`
Expected: FAIL — `TypeError: EvaluationResult.__init__() takes 2 positional arguments but 3 were given`, plus an `ImportError` for `mask_writer` from `evaluation_result.py`.

- [ ] **Step 3: Update `src/evaluation_result.py`**

Delete these two import lines:

```python
from mask_writer import MaskWriter, MASK_DIR
from metrics import registry
```

and add:

```python
from metrics import MetricRegistry
```

Replace `__init__`:

```python
    def __init__(self, images: list[_EvaluatedImage], registry: MetricRegistry):
        self._images   = images
        self._registry = registry
```

In `to_frame`, change:

```python
        extra_columns = [spec.name for spec in registry.specs if not spec.builtin]
```

to:

```python
        extra_columns = [spec.name for spec in self._registry.specs if not spec.builtin]
```

Replace `generate_report` in full:

```python
    def generate_report(self, report_path: Path = REPORT) -> pd.DataFrame:
        """Write the CSV report, then return the DataFrame.

        Args:
            report_path: Destination for the CSV file.

        Returns:
            The same DataFrame that to_frame() would return.
        """
        df = self.to_frame()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
        print(f"CSV written: {report_path}")
        return df
```

Update the class docstring's example block from:

```
    Optional report output:
        result.generate_report(Path("report/my_output.csv"))
```

— that line stays correct as-is, no change needed there. Do remove the now-stale `ImageLoader` import if nothing else in the file uses it (it was only used to build loaders for `MaskWriter`); check with a quick grep before removing:

```bash
grep -n "ImageLoader" src/evaluation_result.py
```

If the only hit is the import line, delete that import line too.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluation_result.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation_result.py tests/test_evaluation_result.py
git commit -m "refactor: EvaluationResult takes the run's registry, drops mask output"
```

---

### Task 5: `evaluate()` and `main()` thread one registry through the run

**Files:**
- Modify: `src/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `MetricRegistry`, `BUILTIN_METRICS`, `SEGMENTATION_METRICS` from Task 1; `IQAEvaluator(inp, tgt, registry)` from Task 3; `EvaluationResult(images, registry)` and `generate_report(report_path)` from Task 4.
- Produces: `evaluate(input_path: Path, target_path: Optional[Path] = None, *, registry: MetricRegistry) -> EvaluationResult`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_main.py`, change the import line from:

```python
from metrics import registry, PSNR, SSIM
```

to:

```python
from metrics import MetricRegistry, PSNR, SSIM
```

Delete the `_restrict_to_fast` helper entirely and replace it with:

```python
def _fast_registry() -> MetricRegistry:
    """Only psnr+ssim — avoids slow/network metrics during main tests."""
    return MetricRegistry(PSNR, SSIM)
```

Then update every `evaluate()` call in the file to drop the `isolated_registry` fixture and pass a registry instead. Each affected test follows this shape:

```python
    def test_returns_evaluation_result(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png")
        res = main_module.evaluate(inp, registry=_fast_registry())
        assert isinstance(res, EvaluationResult)
```

Apply that mechanically to `test_full_reference_single_file`, `test_no_reference_psnr_none`, `test_target_is_dir_warns_and_ignores`, `test_multiple_images_evaluated`, `test_directory_matching`, `test_empty_directory_warns`, `test_unmatched_targets_warns`, `test_exception_in_one_image_continues`, and `test_nonexistent_path_returns_empty_result` — remove the `isolated_registry` parameter and the `_restrict_to_fast(...)` line, and add `registry=_fast_registry()` to the `evaluate(...)` call, keeping each test's other arguments and assertions unchanged.

Update the two CLI tests to drop `isolated_registry` (the `monkeypatch.setattr("main.BUILTIN_METRICS", (PSNR, SSIM))` line stays — `main()` still builds its registry from that bundle):

```python
    def test_cli_writes_report(self, tmp_path, monkeypatch, capsys):
        # main() builds its registry from main.BUILTIN_METRICS — patch that
        # bundle down to psnr+ssim to keep the CLI test fast.
        monkeypatch.setattr("main.BUILTIN_METRICS", (PSNR, SSIM))
        inp = _make_png(tmp_path / "inp.png")
        report = tmp_path / "report.csv"

        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp)])

        main_module.main()

        assert report.exists()

    def test_cli_with_target(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("main.BUILTIN_METRICS", (PSNR, SSIM))
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)
        report = tmp_path / "report.csv"

        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp), str(tgt)])

        main_module.main()

        assert report.exists()
```

Add this new test class at the end of the file — it proves the shared-cache requirement and the two-run scenario:

```python
class TestEvaluateRegistryThreading:
    def test_registry_is_required(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png")
        with pytest.raises(TypeError):
            main_module.evaluate(inp)

    def test_same_registry_instance_shared_across_images(self, tmp_path, monkeypatch):
        seen = []
        real_init = IQAEvaluator.__init__

        def spy_init(self, input_image, target_image, registry, source_model=None):
            seen.append(registry)
            real_init(self, input_image, target_image, registry, source_model)

        monkeypatch.setattr("iqa_evaluator.IQAEvaluator.__init__", spy_init)

        _make_png(tmp_path / "a.png", seed=0)
        _make_png(tmp_path / "b.png", seed=1)
        reg = _fast_registry()
        main_module.evaluate(tmp_path, registry=reg)

        assert len(seen) == 2, "expected one evaluator per image"
        assert all(r is reg for r in seen), "every image must share one registry instance"

    def test_two_runs_use_different_metrics(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)

        run_psnr = main_module.evaluate(inp, tgt, registry=MetricRegistry(PSNR))
        run_ssim = main_module.evaluate(inp, tgt, registry=MetricRegistry(SSIM))

        df_psnr = run_psnr.to_frame()
        df_ssim = run_ssim.to_frame()
        assert df_psnr["psnr"].notna().any()
        assert df_psnr["ssim"].isna().all()
        assert df_ssim["ssim"].notna().any()
        assert df_ssim["psnr"].isna().all()
```

Make sure `from iqa_evaluator import IQAEvaluator` and `import pytest` are present at the top of `tests/test_main.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_main.py -v`
Expected: FAIL — `TypeError: evaluate() got an unexpected keyword argument 'registry'`.

- [ ] **Step 3: Update `src/main.py`**

Replace the import block:

```python
from metrics import (  # noqa: F401 — re-exported for users
    DEVICE, Metric, MetricSpec, MetricRegistry,
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE, BUILTIN_METRICS,
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, SEGMENTATION_METRICS,
)
```

Change the `evaluate` signature and docstring:

```python
def evaluate(
    input_path: Path,
    target_path: Optional[Path] = None,
    *,
    registry: MetricRegistry,
) -> EvaluationResult:
    """Discover input/target images and compute one registry's metrics.

    Args:
        input_path:  Path to an input image file or a directory of images.
        target_path: Optional path to a reference image file or directory.
                     Pass None for a no-reference (NR-only) evaluation.
        registry:    The metrics to compute. One instance is shared across
                     every image in the run, so network-backed metrics are
                     built once rather than once per image.

    No files are written; use EvaluationResult.generate_report() for output.
    """
```

Inside `_run_one`, change the evaluator construction from:

```python
            records = IQAEvaluator(input_loader, target_loader).run_evaluation()
```

to:

```python
            records = IQAEvaluator(input_loader, target_loader, registry).run_evaluation()
```

Change the final return from `return EvaluationResult(evaluated)` to:

```python
    return EvaluationResult(evaluated, registry)
```

Replace `main()`'s opening lines — the current body starts with a German TODO comment and `registry.register(*BUILTIN_METRICS)`. Replace both with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IQA metrics and write a report.")
    parser.add_argument("input", type=Path, help="Input image file or directory.")
    parser.add_argument(
        "target", type=Path, nargs="?", default=None,
        help="Optional reference image file or directory (omit for NR-only evaluation).",
    )
    args = parser.parse_args()

    # Reference usage: pick the metrics for this run. Swap BUILTIN_METRICS for
    # SEGMENTATION_METRICS (or any subset, e.g. MetricRegistry(PSNR, SSIM)) to
    # evaluate something else — each run owns its own registry.
    registry = MetricRegistry(*BUILTIN_METRICS)

    result = evaluate(args.input, args.target, registry=registry)
    report = result.generate_report(REPORT)
    print(report.describe())
    print(f"Report written: {REPORT}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_main.py -v`
Expected: all PASS, including `TestEvaluateRegistryThreading`.

- [ ] **Step 5: Commit**

```bash
git add src/main.py tests/test_main.py
git commit -m "refactor: evaluate() threads one MetricRegistry through the run"
```

---

### Task 6: Update the exploration notebook to build its own registry

**Files:**
- Modify: `src/evaluation.ipynb`

**Interfaces:**
- Consumes: `MetricRegistry`, `BUILTIN_METRICS`, `SEGMENTATION_METRICS` from Task 1; `evaluate(input, target, *, registry)` from Task 5.
- Produces: nothing later tasks depend on.

The notebook is a documented entry point ("Run notebook interactively" in CLAUDE.md) and currently imports the deleted singleton, so it breaks without this task.

- [ ] **Step 1: Find the two affected cells**

Run:
```bash
grep -n "from metrics import registry\|registry.register\|evaluate(INPUT, TARGET)" src/evaluation.ipynb
```
Expected: one cell importing `registry, BUILTIN_METRICS` and calling `registry.register(*BUILTIN_METRICS)`, and one cell calling `evaluate(INPUT, TARGET)`.

- [ ] **Step 2: Edit the setup cell**

The setup cell's source currently is:

```python
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from main import evaluate
from metrics import registry, BUILTIN_METRICS

# Pick which metrics to compute — swap BUILTIN_METRICS for individual
# constants (e.g. registry.register(PSNR, SSIM)) to run only a subset.
registry.register(*BUILTIN_METRICS)
```

Change it to:

```python
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from main import evaluate
from metrics import MetricRegistry, BUILTIN_METRICS, SEGMENTATION_METRICS

# Pick which metrics this run computes — swap BUILTIN_METRICS for
# SEGMENTATION_METRICS, or list individual constants
# (e.g. MetricRegistry(PSNR, SSIM)), to run a different set.
registry = MetricRegistry(*BUILTIN_METRICS)
```

Use the Jupyter notebook editing tool (NotebookEdit) rather than hand-editing the JSON, so the file stays valid.

- [ ] **Step 3: Edit the evaluation cell**

Change the `evaluate(INPUT, TARGET)` call to:

```python
evaluate(INPUT, TARGET, registry=registry)
```

If the surrounding markdown cell describes the call as `evaluate(INPUT, TARGET)`, update that prose to `evaluate(INPUT, TARGET, registry=registry)` too so the narrative matches the code.

- [ ] **Step 4: Verify the notebook is still valid JSON and its imports resolve**

Run:
```bash
.venv/bin/python -c "
import json, sys
nb = json.load(open('src/evaluation.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
assert 'from metrics import registry' not in src, 'notebook still imports the deleted singleton'
assert 'MetricRegistry(*BUILTIN_METRICS)' in src, 'notebook does not build a registry'
assert 'registry=registry' in src, 'notebook does not pass the registry to evaluate()'
print('notebook ok')
"
```
Expected: `notebook ok`

- [ ] **Step 5: Commit**

```bash
git add src/evaluation.ipynb
git commit -m "docs: update evaluation notebook for per-run registries"
```

---

### Task 7: Remove the obsolete `isolated_registry` fixture and run the full regression

**Files:**
- Modify: `tests/conftest.py`
- Test: whole suite

**Interfaces:**
- Consumes: all prior tasks.
- Produces: no `isolated_registry` fixture anywhere; the suite passes end to end.

- [ ] **Step 1: Confirm nothing still uses the fixture**

Run:
```bash
grep -rn "isolated_registry" tests/
```
Expected: only the fixture definition in `tests/conftest.py` (lines ~57-72) and its comment banner. If any test file still references it, that file was missed in Tasks 1–5 — fix that reference the same way those tasks did (local `MetricRegistry(...)`) before continuing.

- [ ] **Step 2: Delete the fixture**

In `tests/conftest.py`, delete the comment banner and the whole fixture:

```python
# ---------------------------------------------------------------------------
# Registry isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_registry():
    """Yield the global registry with its original state restored afterwards.

    Prevents registry-mutating tests from affecting each other.
    """
    from metrics import registry

    saved_specs = dict(registry._specs)
    saved_cache = dict(registry._cache)
    yield registry
    registry._specs = saved_specs
    registry._cache = saved_cache
```

Leave every other fixture in the file untouched.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all tests pass, 0 failures. The count will be lower than the previous 221 — `tests/test_mask_writer.py` and the `best_slice_per_metric` tests were deleted in Task 2, and one mask-PNG test in Task 4.

- [ ] **Step 4: Confirm the singleton is really gone from the source tree**

Run:
```bash
grep -rn "from metrics import.*\bregistry\b\|metrics\.registry\|^registry = " src/ tests/
```
Expected: no output. A hit means some module still reaches for the deleted global.

Also confirm the two-registry scenario works end to end:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from metrics import MetricRegistry, PSNR, SSIM, DICE
a = MetricRegistry(PSNR, SSIM)
b = MetricRegistry(DICE)
print(sorted(s.name for s in a.specs), sorted(s.name for s in b.specs))
"
```
Expected: `['psnr', 'ssim'] ['dice']`

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test: drop obsolete isolated_registry fixture"
```
