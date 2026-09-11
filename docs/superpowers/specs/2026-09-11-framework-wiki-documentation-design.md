# Framework Wiki Documentation — Design

**Date:** 2026-09-11
**Status:** approved
**Target repo:** `https://github.com/glebjan/hka.projektarbeit1.IQA_MRI.wiki.git` (branch `master`)

## Goal

A reader who knows nothing about this repository can, from the wiki alone,
install the framework, choose metrics, register built-in and custom metrics,
configure a metric (DreamSim as the worked example), run an evaluation in
either scoring mode, and turn the resulting records into a report — without
reading the source.

## Decisions

| Decision | Choice |
|---|---|
| Language | English (prose and code) |
| Scope | Everything: IQA metrics, segmentation metrics, slice and volume mode, aggregation |
| Layout | ~12 task-oriented pages plus `_Sidebar.md` |
| Example verification | Every example that needs no weight download is executed; its real output goes into the page. Model-backed examples (dreamsim, musiq, maniqa, ilniqe) are shown as snippets, checked against the test suite, and marked as not executed |
| Documented entry point | Bottom-up: `ImageLoader` → `MetricRegistry` → `build_evaluator` → records → `EvaluationResult` |
| `main.evaluate()` / CLI | Out of the documented API. One paragraph on `Home` states it exists, is a reference implementation, and is scheduled for replacement |
| Diagrams | Inline Mermaid (object flow, spec/capability model, mode dispatch, tensor shapes) |
| Caveats | Own page `Interpreting-Scores` plus short warnings at the points where they bite |
| Metric catalog depth | Overview table plus 2–4 sentences per metric; full parameter documentation only for the configurable ones |
| Install depth | Full install page plus a troubleshooting page carrying the known pitfalls |
| Delivery | Clone, write, commit locally, show the file list and diff, push only after explicit approval |
| Local `README.md` / `CLAUDE.md` | README reduced to install + quickstart + wiki link; CLAUDE.md corrected. Both are git-ignored, so this is a local-only change |

## Prerequisite code change

The documented chain ends at the report, and today `EvaluationResult` can only
be built from `_EvaluatedImage` — a private dataclass whose only field,
`input_path`, is never read (set in `main.py:136` and in tests, never consumed;
`to_frame()` flattens the grouping immediately and the per-image association
already lives in `record.image_id`).

Add one public classmethod to `src/evaluation_result.py`:

```python
@classmethod
def from_records(
    cls,
    records: list[ImageEvaluatorRecord],
    registry: MetricRegistry,
) -> "EvaluationResult"
```

It takes a flat record list — the concatenation of what one or more evaluators
returned — and the registry the run used (needed for the custom-metric
columns). `_EvaluatedImage` stays internal for `main.evaluate()`; nothing else
changes.

Landed test-first on branch `volumetric_similarity`, one commit. New tests:
records from one image, records from several images concatenated, an empty
list, and a custom metric's `extra` column surviving into `to_frame()`.

## Page map

| Page | Content |
|---|---|
| `Home.md` | What the framework does, a 15-line minimal example, the object-flow Mermaid diagram, the page index, the `main.py` status note |
| `Installation.md` | venv, `requirements.txt`, the ResNet50 weight, the DreamSim cache, device selection (cuda/mps/cpu) |
| `Core-Concepts.md` | The five objects and how they hand off. Tensor shapes `(D,1,H,W)`, `(D,3,H,W)`, `(1,C,D,H,W)`. Mermaid object flow |
| `Loading-Images.md` | Supported formats, `.tensor` / `.rgb_tensor`, `spacing`, `is_volumetric`, `empty_slice_mask`, `list_images` / `find_matching_target` for building input/target pairs |
| `Registering-Metrics.md` | `MetricRegistry(PSNR, SSIM)`, `.register()`, no singleton, `BUILTIN_METRICS`, `SEGMENTATION_METRICS`, `.select(mode)` and `SkippedMetric`, `.get_metric()` caching |
| `Metric-Catalog.md` | Table of all metrics (full-reference/no-reference, direction, range, slice/volume support, gray/rgb) plus a short entry per metric |
| `Configuring-Metrics.md` | `dreamsim_spec()` — all six parameters, name derivation, registering several variants at once. MONAI factories (`threshold`, `**monai_kwargs`), `boundary_iou_metric` |
| `Custom-Metrics.md` | The `Metric` protocol, `register_metric(...)`, hand-written `MetricSpec` with `ModeSupport` / `ModeUnsupported`, where the score lands (`record.extra` → CSV column), `volume_factory` |
| `Running-an-Evaluation.md` | `build_evaluator(input, target, registry, mode, source_model).run_evaluation()`, batching, the shape check, what a failing metric does (None, no abort) |
| `Scoring-Modes.md` | slice vs volume, `VolumeEvaluator`, spacing-mismatch abort, why empty slices are kept in volume mode, the skip messages |
| `Results-and-Reports.md` | `EvaluationResult.from_records(...)`, `.to_frame()`, `.aggregate_volumes()` including its NaN rules, `.generate_report(path)`, column order |
| `Segmentation-Metrics.md` | Masks as input, threshold and label, the eleven metrics, voxels vs millimetres |
| `Interpreting-Scores.md` | Volume-wide min-max normalization, slices are not independent samples, the two modes are not one scale, the nsd tolerance changing unit, domain transfer for dreamsim and the CLIP-based metrics |
| `Troubleshooting.md` | setuptools/`pkg_resources`, no CUDA wheels on macOS, Python 3.14 limits, OOM → `BATCH_SIZE`, shape mismatch, "none of the selected metrics can be scored" |
| `_Sidebar.md` | Navigation |

## Verification

A scratchpad script builds a synthetic NIfTI volume (and a mask pair) and runs:
registry construction, `PSNR`/`SSIM` in both modes, a custom metric end to end,
`from_records` → `to_frame` → CSV, `aggregate_volumes`, the `select()` skip
output, and DreamSim name derivation without loading a model. The captured
output is what the pages show.

## Order of work

1. `from_records` — test-first, full suite, one commit
2. Verification script, capture real outputs
3. Write the wiki pages
4. Self-review: every signature, parameter name and default re-checked against `src/`
5. Show file list and diff, stop
6. Push after approval
7. Reduce `README.md`, correct `CLAUDE.md` (local, git-ignored)

## Out of scope

- Documenting `main.evaluate()` as a supported API
- Any other change to framework behaviour
- Translating the wiki into German
