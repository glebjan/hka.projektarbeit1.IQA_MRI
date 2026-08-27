# Per-Run Metric Selection — Design

Date: 2026-08-27
Branch: `segmentation_metrics` (continues after the MONAI segmentation-metrics work)

## Purpose

Let each evaluation run choose its own metrics. Today a single module-level
`registry` singleton in `metrics.py` holds the metric set for the whole
process, so every `IQAEvaluator` in a session computes the same metrics.
The target usage is two independent runs side by side:

```python
iqa = evaluate(images, targets, registry=MetricRegistry(*BUILTIN_METRICS))
seg = evaluate(pred_masks, gt_masks, registry=MetricRegistry(*SEGMENTATION_METRICS))
```

`MetricRegistry` becomes an ordinary instantiable class; the singleton is
removed and the metric set is passed explicitly through the pipeline.

## Non-goals

- No change to the `Metric` protocol, `MetricSpec`, or any metric adapter
  (`PyIQAMetric`, `MonaiSegmentationMetric`, `MonaiPanopticQualityMetric`).
- No change to the metric constants or the `BUILTIN_METRICS` /
  `SEGMENTATION_METRICS` bundles.
- No new metrics.
- CLAUDE.md is not updated as part of this work (it documents the removed
  `registry` singleton and `register_metric()`; a follow-up task).

## Current coupling to the singleton

The global `registry` is read in four places:

| Location | Use |
|---|---|
| `iqa_evaluator.py:47,86` | `registry.get_metric(...)`, `registry.specs` |
| `records.py:41` | `registry.direction` inside `best_slice_per_metric()` |
| `evaluation_result.py:43` | `registry.specs` for the CSV's extra columns |
| `main.py:88` | `registry.register(*BUILTIN_METRICS)` |

Plus `tests/conftest.py:60`, whose `isolated_registry` fixture exists purely
to save/restore the singleton's private `_specs`/`_cache` between tests.

## Design

### 1. `metrics.py` — instantiable registry

`MetricRegistry.__init__` accepts specs directly:

```python
class MetricRegistry:
    def __init__(self, *specs: MetricSpec):
        self._specs: dict[str, MetricSpec] = {}
        self._cache: dict[str, Metric] = {}
        self.register(*specs)
```

`register(*specs)`, `get_metric(name)`, and the `specs` / `direction`
properties keep their current behaviour.

The module-level `registry = MetricRegistry()` singleton is **deleted**.

The free function `register_metric(name, metric, *, direction, reference,
channels="rgb")` becomes a method on `MetricRegistry` with the same name,
parameters, and semantics — it wraps a raw `Metric` object in a
`MetricSpec(builtin=False)` and registers it into that instance. With no
global to register into, a free function no longer makes sense.

### 2. `iqa_evaluator.py` — registry as a required parameter

```python
IQAEvaluator(input_image, target_image, registry, source_model=None)
```

`registry` is positional and required — no default and no fallback, so a
run's metric set is always explicit. `run_evaluation()` iterates
`self.registry.specs` and resolves instances via `self.registry.get_metric`.

### 3. `main.py` — `evaluate()` threads one registry through the run

```python
def evaluate(input_path, target_path=None, *, registry: MetricRegistry) -> EvaluationResult
```

`registry` is a required keyword argument. `evaluate()` passes the same
instance to every per-image `IQAEvaluator`, so lazily-built metric objects
(LPIPS, CLIP-IQA and other network-backed metrics) are constructed once per
run rather than once per image, and forwards it to `EvaluationResult`.

`main()` builds `MetricRegistry(*BUILTIN_METRICS)` and passes it in.
`main.py`'s re-export block drops `registry` and `register_metric` and adds
`MetricRegistry` and `SEGMENTATION_METRICS`.

### 4. `evaluation_result.py` — columns from the run's registry

```python
EvaluationResult(images, registry)
```

`to_frame()` derives its extra columns from `registry.specs` instead of the
global. `generate_report(report_path)` loses its `mask_dir` parameter and
writes only the CSV.

### 5. Deletions

- `src/mask_writer.py` in full (Otsu segmentation and `MaskWriter`) — the
  Otsu approach did not work as intended and is being dropped.
- `best_slice_per_metric()` and its helper `_record_metric_value()` from
  `records.py`; `MaskWriter` was their only consumer. `records.py` then
  imports nothing from `metrics.py`.
- `tests/test_mask_writer.py`.
- The `isolated_registry` fixture in `tests/conftest.py`, obsolete once
  there is no shared global state to isolate.

## Testing

Every test that reached for `isolated_registry` builds a local
`MetricRegistry(PSNR, SSIM)` instead — the fixture's save/restore dance was
only ever compensating for the singleton.

- `tests/test_metrics.py`: constructor-with-specs, `register_metric` as a
  method, and that two `MetricRegistry` instances hold independent specs and
  caches.
- `tests/test_iqa_evaluator.py`: all `IQAEvaluator(...)` constructions gain
  the registry argument.
- `tests/test_records.py`: the `best_slice_per_metric` test class is removed
  along with the function.
- `tests/test_evaluation_result.py`: `EvaluationResult(images, registry)`;
  the `test_mask_pngs_written` test is removed with `MaskWriter`.
- `tests/test_main.py`: `_restrict_to_fast()` is replaced by constructing
  `MetricRegistry(PSNR, SSIM)` and passing it to `evaluate()`.
- One new test covering the point of the change: two `IQAEvaluator`s built
  with different registries in the same process compute different metric
  sets and do not affect each other.

## Breaking changes

- `from metrics import registry` no longer resolves.
- The free `register_metric()` is now `MetricRegistry.register_metric()`.
- `evaluate()` requires `registry=`.
- `EvaluationResult(images)` requires a second argument.
- `generate_report()` no longer accepts `mask_dir` and writes no PNGs.
