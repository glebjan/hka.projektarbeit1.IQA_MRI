# Metric Calibration, Part A: Segmentation Metrics — Design

Date: 2026-09-13
Branch: `volumetric_similarity`
Source: `docs/reviews/2026-09-13-ultrareview-kalibrierung.md` (review at HEAD
0047d24; its Phase 1 is already done — the package imports, 589 tests collect)

## Purpose

The review's lead question was: *are the metrics tested against a gold
standard, i.e. is the framework calibrated for scientific use?* The answer was
no. This design is the first of three parts (A segmentation, B intensity
metrics, C loader/normalization edge cases) and covers the segmentation
metrics, where the review found the largest semantic gaps: a perfect
prediction can score Dice 0.0 (review 3.1), MONAI silently scores only label 1
(3.3), four binarizers with two semantics (3.6), `inf` in the CSV (3.7), the
boundary band vanishing through-plane (3.9), and real voxels dropped as
"empty" (3.10/3.11).

Calibration here means three things, in this order of authority:

1. **The metric's definition in its paper, applied by hand to a fixture small
   enough that the derivation fits in ten lines.** The derivation is written
   down and reviewed by a human; the number is not trusted, the derivation is
   checked.
2. **The implementation the community cites when publishing the metric**,
   run on the same fixture, with the same definitional variant.
3. **MONAI's own test cases**, run through this framework's loader and
   adapters, proving the adapters do not distort what MONAI computes.

`main.py` is a sandbox and out of scope throughout.

## Decisions

| # | Decision | Alternative rejected |
|---|---|---|
| D1 | One foreground policy, decided at the loader: new `Mask()` normalizer. `Raw()` stays as an escape hatch (dtype preserved) but is no longer valid input for the binary segmentation metrics (PQ keeps accepting `Raw()` instance maps, see Part 2). | Extending `Raw(label=...)` — mixes "unscaled" with "binarized"; 0/255 PNG stays a special case. |
| D2 | Under `Mask()`, a slice is empty iff it has zero foreground voxels (exact count). Skip only when input **and** target are empty. Volume mode does not set `is_empty` at all. | All-false `empty_slices` (every slice scored, `None` noise); a run flag `skip_empty=False` (second knob to forget). |
| D3 | Empty-mask policy: one side empty → Dice/VS/NSD/PQ/Boundary-IoU 0.0, HD95/ASSD `None`; both empty → `None` everywhere. `inf`/`NaN` never reach the CSV. Counts (`v_pred`, `v_gt`, `tp`) and `vs_signed` are not scores and get no policy: one side empty → real counts and ±2.0; both empty → counts 0.0, `vs`/`vs_signed` `None` (NaN → None, as today). | Both empty → 1.0 (Metrics Reloaded, Maier-Hein 2024): inflates means over many empty cases. |
| D4 | Boundary-IoU physical band floor = `max(spacing)` (at least one voxel in every direction), was 1.0 mm. | Voxel-mode only in 3D — loses the anisotropy correction. |
| D5 | `threshold`/`label` removed from every spec builder and adapter. Domain parameters stay (`**monai_kwargs`). Numpy functions keep `label`/`threshold` as standalone utilities; `as_mask` becomes the single binarization rule. | Keep knobs with a no-op default — four paths remain. |
| D6 | One label per run: `Mask(label=k)`. No one-hot / multi-channel path through the loader. | One-hot channels — complicates every adapter for a case nobody asked for. |
| D7 | `Normalizer` protocol gains `apply()` and `empty_slices()`; the loader no longer knows strategy semantics. | `is_mask` flag + `isinstance` branches in the loader. |
| D8 | Official reference packages as optional deps, run in a separate Hatch env `calibration` with its own venv. Git dependencies pinned to a SHA. One forced exception: `boundary-iou-api` cannot be installed as a dependency — its `setup.py` ships only the empty top-level package (no `boundary_iou.utils`) and pins `panopticapi` to an unpinned `archive/master.zip` URL that conflicts with any SHA pin (both verified with uv on Python 3.14) — so its 20-line `mask_to_boundary` is vendored verbatim (BSD-2-Clause, copyright header, source SHA) in `tests/calibration/official.py`. | Vendoring core functions — "copied" is weaker than "imported". |
| D9 | Test suite as a whole moves out of the runtime install: dev tools out of `[project.dependencies]`, `tests/` out of the sdist, single test command `hatch run calibration:test`. Caveat: pyiqa 0.1.15.post2 itself declares pytest, ruff, pre-commit, tensorboard and yapf as runtime dependencies, so they stay installed in `.venv` transitively; what D9 removes is our direct pins and the `tests/` tree. | — (user requirement: users must not pay for tests in disk space) |
| D10 | Framework follows the MONAI variant of each definition (voxel-based surfaces, HD95 = max of directed 95th percentiles — Taha & Hanbury 2015). Divergent official variants appear in the report as extra columns with an explanation. | Switching to `surface-distance` (area-weighted) semantics — MONAI backend would no longer match. |

## Part 0 — Test suite → calibration environment (D8, D9)

Runs first, own commit; everything after is verified with
`hatch run calibration:test`.

`pyproject.toml`:

- `[project.dependencies]`: remove every package that only dev tooling needs
  (`pytest`, `ruff`, `pre_commit`, `virtualenv`, `cfgv`, `identify`,
  `nodeenv`, `yapf`, `distlib`, `iniconfig`, `pluggy`, …). The exact list is
  determined with `uv pip tree`: a package moves only if nothing in the
  runtime closure depends on it. Runtime pins stay exact (deliberate: pyiqa's
  outputs are backend-version-dependent). Review 1.4's "120 → 12 deps" is
  explicitly **not** done.
- `[project.optional-dependencies]`:
  ```toml
  calibration = [
    "surface-distance==0.1",   # DeepMind's own PyPI upload (author DeepMind, homepage deepmind/surface-distance); Nikolov et al. — NSD (author implementation), HD, ASSD (area-weighted variant)
    "medpy==0.5.2",            # Dice, HD, HD95 (concatenated variant), ASSD (voxel variant); sdist only, builds on Python 3.14 / numpy 2.5.3 (verified)
    "panopticapi @ git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0",
  ]
  ```
  with `[tool.hatch.metadata] allow-direct-references = true`. The SHA is the
  master head on 2026-09-13 and is recorded in the calibration report.
  `boundary-iou-api` is deliberately absent (D8): `mask_to_boundary` is
  vendored in `official.py` from SHA `37d25586a677b043ed585f10e5c42d4e80176ea9`.
- `[tool.hatch.envs.calibration]`: `path = ".venv-calibration"`,
  `installer = "uv"`, `features = ["calibration"]`, `dependencies = ["pytest==9.1.1", "ruff==0.16.6", …]`
  (the dev tools removed above, same pins). Scripts:
  `test = "pytest {args:tests}"`, `report = "python tests/calibration/report.py"`.
- `[tool.hatch.envs.default.scripts]`: `test` removed. `evaluate` stays.
- `[tool.hatch.build.targets.sdist] include`: `tests/` removed.

Consequence for the developer: a second venv with its own torch stack
(~1.7 GB). Development happens in `.venv-calibration`; `.venv` exists to prove
the runtime install works without dev tools.

Docs: `CLAUDE.md` (test command, env layout), `README.md`.

Verification: `hatch env create calibration && hatch run calibration:test`
→ all tests pass; `uv pip tree --invert --package pytest` in `.venv` names pyiqa
as the only parent (iqaevaluator no longer appears); `hatch build -t sdist`
contains no `tests/`.

## Part 1 — `Mask()` and the `Normalizer` protocol (D1, D2, D6, D7)

`src/iqaevaluator/normalization.py`:

```python
@runtime_checkable
class Normalizer(Protocol):
    name: str
    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]: ...
    def apply(self, raw: np.ndarray, *, source: str) -> torch.Tensor: ...  # (D, H, W) float32; `source` = file name for messages
    def empty_slices(self, raw: np.ndarray) -> torch.Tensor: ...           # (D,) bool
```

- The kwarg is `source`, not `label`: `Mask(label=k)` already uses `label` for
  the class id. `scale()` keeps its `label=` parameter; `apply` passes `source`
  through to it.
- `MinMax`, `Percentile`, `FixedRange`, `Raw`: `apply` returns
  `scale(raw, self.range_of(raw), label=source)` — unchanged mapping.
  `empty_slices` returns `sparse_slices(raw)`, a module function holding the
  heuristic currently in `ImageLoader.empty_slice_mask` verbatim (std and lift
  tests against a `Percentile().range_of` span, extremes fallback). Behaviour
  for intensity images is bit-identical; existing tests prove it.
- `Mask(label: Optional[int] = None, threshold: float = 0.5)`, frozen
  dataclass. `name` is `"mask"` or `f"mask_label_{label}"`. `range_of` → `None`.
  `apply` → `as_mask(raw, label, threshold)` as float32 `{0.0, 1.0}`.
  `empty_slices` → `apply(raw).sum(dim=(1, 2)) == 0`. A float map outside
  [0, 1] raises `ValueError` naming the file (re-raised from `as_mask` with
  `source` prepended).
- `NORMALIZER_NAMES["mask"] = Mask`; CLI `--normalization mask`.
  `Mask(label=k)` is API-only (no CLI label parser).
- Module docstring: strategy list gains `Mask()`; `Raw()` is described as the
  escape hatch for instance maps (PQ) and anything else that must keep its
  dtype. `Percentile`'s docstring points at `ImageLoader.empty_slice_mask` for
  the shared sparse-foreground fallback — repoint it at `sparse_slices`.

`src/iqaevaluator/segmentation_metrics/volume.py`:

- `as_mask(x, label: Optional[int] = None, threshold: float = 0.5)`:
  bool → unchanged; integer → `x != 0` when `label is None`, else `x == label`;
  float → range check then `x >= threshold`. This is the framework's only
  binarization rule; `Mask.apply` calls it.

`src/iqaevaluator/image_loader.py`:

- `tensor` → `self.normalizer.apply(self.raw, source=self.path.name).unsqueeze(1)`.
- `empty_slice_mask` → `self.normalizer.empty_slices(self.raw)`.
- Docstrings: "float32 in [0, 1] under every strategy except `Raw()`;
  exactly {0, 1} under `Mask()`".
- `load_pair`: couples the input to the target only when the target produced a
  range:
  ```python
  target = ImageLoader(target_path, normalizer)
  rng = target.intensity_range
  inp_norm = normalizer if rng is None else FixedRange(rng, name=normalizer.name)
  inp = ImageLoader(input_path, inp_norm)
  ```
  Under `Mask()` and `Raw()` both sides are therefore loaded independently
  (fixes review 3.1). Under `MinMax`/`Percentile` nothing changes.

`src/iqaevaluator/volume_evaluator.py`: `is_empty` no longer passed; the record
default `False` applies.

`src/iqaevaluator/evaluation_result.py`: `aggregate_volumes` unchanged — with
D2 an `is_empty` slice really has zero voxels on both sides, so filling its
counts with 0 is now correct rather than a heuristic (docstring updated).

Report columns: `normalization = mask | mask_label_k`, `scale_lo/hi` empty (as
under `Raw`), `input_min/max` from `raw_range` as before.

Tests (`tests/test_normalization.py`, `tests/test_image_loader.py`,
`tests/test_iqa_evaluator.py`):

- 0/255 uint8 PNG and 0/1 uint8 NIfTI → identical tensors under `Mask()`.
- `Mask(label=2)` on a `{0,1,2}` int16 map → only label 2 is 1.0.
- float map: `>= 0.5`; values outside [0, 1] → `ValueError` containing the
  file name.
- `empty_slices`: a slice with a single foreground voxel is not empty; an
  all-zero slice is.
- `load_pair` under `Mask()`: 0/1 NIfTI prediction vs 0/255 PNG ground truth,
  identical shapes → `dice == 1.0`, `vs == 1.0` (review 3.1 failure case).
- `load_pair` under `MinMax()`: input still on the target's range (existing
  tests).
- `sparse_slices` equals the previous `empty_slice_mask` on the existing
  intensity fixtures (existing tests keep passing).
- Volume mode: record `is_empty` is `False` for an all-zero input.

## Part 2 — Segmentation adapters (D3, D4, D5)

Shared helpers in `segmentation_metrics/volume.py`:

```python
def require_binary(t: torch.Tensor, *, metric: str) -> None
    # raises ValueError(f"{metric} expects a binary mask with values in {{0, 1}}; "
    #                   "load masks with normalization.Mask()") unless every value is 0 or 1

def foreground_counts(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
    # per-sample voxel counts over (C, *spatial)

NOT_EMPTY = object()
def empty_policy(n_pred: int, n_gt: int, *, one_sided: Optional[float]) -> Optional[float] | NOT_EMPTY
    # both 0 -> None; exactly one 0 -> one_sided; else NOT_EMPTY
```

`one_sided` per metric: Dice 0.0, VS 0.0 (`vs` already does this), NSD 0.0,
Boundary-IoU 0.0, HD95 `None`, ASSD `None`, PQ 0.0.

`segmentation_metrics/monai_metrics.py`:

- `MonaiSegmentationMetric(compute_fn, *, one_sided, **monai_kwargs)`. No
  `threshold`, no `_binarize`. `__call__`: `require_binary` on both tensors →
  `foreground_counts` → policy per sample → `compute_fn` only on the
  remaining samples (as one sub-batch) → `torch.nanmean` over channels →
  scatter back. Any `inf` or `NaN` that still comes out of MONAI becomes
  `None` plus a `print` warning naming the metric and sample — an unexpected
  case must not be silent.
- Dice is built with `ignore_empty=False` (the policy decides anyway; this
  removes MONAI's NaN for empty ground truth).
- The four copy-paste builders collapse into one
  `_monai_spec(name, compute_fn, *, direction, one_sided, uses_spacing, description, defaults: dict, **monai_kwargs)`
  (review 4.1). Public signatures: `dice_metric(**monai_kwargs)`,
  `hausdorff95_metric(**monai_kwargs)`, `normalized_surface_dice_metric(**monai_kwargs)`,
  `average_surface_distance_metric(**monai_kwargs)`. Domain defaults unchanged:
  `percentile=95`, `symmetric=True`, `class_thresholds=[1.0]`,
  `include_background=True`. Docstrings lose the `threshold` paragraph and gain
  one sentence: "Input must be binary — load masks with `Mask()`."
- `MonaiPanopticQualityMetric(**monai_kwargs)`: no `threshold`. Accepts
  integer-valued tensors — `{0, 1}` from `Mask()` (single-instance PQ) or an
  instance map from `Raw()`. Validation: `t == t.round()`, else `ValueError`
  pointing at `Mask()`/`Raw()`. Empty policy as Dice.

`segmentation_metrics/volume_metrics.py`: `VolumeFunctionMetric(fn)` without
`threshold`; `require_binary` first; numpy functions called as before (their
`as_mask` is a no-op on `{0, 1}` floats). `vs_metric()` etc. lose `threshold`.

`segmentation_metrics/boundary_iou.py`:

- `BoundaryIoUMetric(*, dilation_ratio, spacing)` without `threshold`;
  `require_binary`. Docstring paragraph "tensors are always float32 in [0, 1]"
  replaced.
- `band_width(shape, dilation_ratio, spacing)`: with `spacing`,
  `max(max(spacing), dilation_ratio * ‖shape · spacing‖)`. Docstring states
  the floor and why (a band thinner than one voxel along an axis excludes
  every face perpendicular to that axis). Without `spacing` unchanged.
- Module docstring's import-order note is deleted once the import cycle is
  gone (see "Import cycle" below).

Import cycle (review 1.2): `MetricSpec`, `ModeSupport`, `ModeUnsupported`,
`ModeCapability`, `SkippedMetric`, `Spacing`, `REASON_*`, `Metric`,
`MetricChannels`, `MetricDirection` move to a leaf module
`src/iqaevaluator/metric_spec.py`; `metrics.py` re-exports them so every
existing `from iqaevaluator.metrics import MetricSpec` keeps working, and its
bottom imports become normal top imports. This is in scope because Part 2
touches every segmentation module's imports anyway. `image_loader.py` imports
`Spacing` from `metric_spec` as well and drops its duplicate alias (which only
existed because `metrics.py` could not import `image_loader`).

Cross-check (review 2.5): `tests/test_volume_metrics.py::TestCrossCheck` gains
a `{0,1,2}` label-map case under `Mask(label=2)`: volume-mode Dice ==
`aggregate_volumes()` Dice == 0.75 (fixture F6 below).

Unit tests (`tests/test_segmentation_metrics.py`, `test_volume_metrics.py`,
`test_boundary_iou.py`):

- Every adapter raises on a tensor containing 0.5 with a message naming
  `Mask()`.
- Policy table, every cell, every metric (fixture F3).
- No `inf` in any adapter output; `np.isinf(df).any()` is false on a report
  containing one-sided-empty volumes.
- `band_width((8, 40, 40), 0.02, (3, 1, 1)) == 3.0`;
  `band_width((130, 256, 256), 0.02, (1.2, 1, 1))` unchanged (≈ 7.9).
- The removed `threshold=` keyword raises `TypeError` (signature test), so a
  stale caller fails loudly rather than being silently ignored.

## Part 3 — Calibration suite

```
tests/calibration/
  __init__.py
  cases.py                        # CalibrationCase records: fixture, hand value, derivation, source
  official.py                     # adapters: surface-distance, medpy, panopticapi; vendored mask_to_boundary (boundary-iou-api, BSD-2)
  monai_cases.py                  # fixtures + expected values copied from MONAI 1.6.0 tests (Apache-2.0 header)
  test_hand.py                    # framework == hand value            (no extras needed)
  test_official.py                # framework == official, same variant (importorskip)
  test_official_hand.py           # official == hand value              (importorskip) — pins the reference itself
  test_monai_cases.py             # framework == MONAI's own expected values, through Mask()+adapters
  test_random.py                  # framework == official on seeded random blobs, three spacings (importorskip)
  report.py                       # renders docs/calibration/segmentation.md
```

`CalibrationCase`: `name`, `metric`, `build() -> (pred, gt)` (numpy, `(D,H,W)`
or `(H,W)`), `spacing`, `mode` (`slice`/`volume`), `normalizer`
(`Mask()`/`Mask(label=2)`/`Raw()`), `metric_kwargs`, `expected`, `tolerance`,
`derivation` (multi-line string: paper, equation, arithmetic), `source`
(citation).

Framework side always goes through the public path: fixture written to a
temporary NIfTI (with spacing) → `load_pair(..., normalizer)` →
`MetricRegistry(spec)` → `IQAEvaluator`/`VolumeEvaluator` → record value. The
path a user takes is what gets calibrated, not the adapter in isolation.

`report.py` renders one table per metric — case | hand value | derivation |
official (package@version, variant) | framework | Δ — under a header with the
framework git SHA and the MONAI, pyiqa, scipy versions; divergent-variant
columns (D10) carry a footnote. Output `docs/calibration/segmentation.md`,
committed.

### Definitional variants (D10)

| Metric | Framework (MONAI) | Same variant, official | Divergent variant, reported alongside |
|---|---|---|---|
| Dice | 2·TP/(2·TP+FP+FN) on voxels | MedPy `dc` | — |
| HD95 | max(P95(pred→gt), P95(gt→pred)) over surface voxels; surface = mask ∖ erosion (cross structure) — Taha & Hanbury 2015, Eq. for HD_q | (none exactly; MedPy `hd95` takes P95 over the **concatenated** directed distances) | MedPy `hd95`; `surface-distance` `compute_robust_hausdorff(95)` (area-weighted surface elements) |
| ASSD | mean over the concatenated directed surface-voxel distances | MedPy `assd` | `surface-distance` `compute_average_surface_distance` (area-weighted, returns both directions) |
| NSD | (|S_gt within τ| + |S_pred within τ|) / (|S_gt| + |S_pred|) over surface voxels — MONAI | — | `surface-distance` `compute_surface_dice_at_tolerance` (author implementation, area-weighted) |
| PQ | Kirillov 2019 Eq. 1, IoU match > 0.5 | `panopticapi` `pq_compute` | — |
| HD (percentile None) | max over both directed maxima of surface-voxel distances | MedPy `hd` — via `hausdorff95_metric(percentile=None)` | — |
| Boundary IoU | Cheng 2021, band = mask ∖ erode(mask, d) | vendored `mask_to_boundary` from `boundary-iou-api` (2D, cv2 erosion) | — (3D physical mode is this framework's extension; hand value only) |

Verified on F1/F2 with the installed packages: MedPy `dc`/`assd`/`asd`/`hd`
reproduce every MONAI value; `surface-distance` diverges as expected (F1 ASD
0.188 vs 0.1667, F2 NSD 0.9894 vs 0.9912) — those numbers land in the report's
divergent columns, never in an equality assertion.

### Fixtures and hand values

All shapes are `(D, H, W)`; ranges are half-open Python slices. Every value
below has been checked once numerically (MONAI 1.6.0 / scipy) as a sanity
check of the arithmetic — the authority is the derivation, which the reviewer
is asked to verify.

**F1 — Block shift.** Volume `(6, 12, 12)`. GT = `[2:4, 3:9, 3:9]`, Pred =
`[2:4, 3:9, 4:10]`. |GT| = |Pred| = 2·6·6 = 72; overlap W 4:9 → 2·6·5 = 60.

- Dice = 2·60 / (72+72) = 120/144 = **0.833333**. (IoU would be 60/84 = 0.714 —
  the mutation check.)
- Surfaces: the blocks are 2 voxels thick in D, so every voxel has an outside
  neighbour along D; erosion with the cross structure removes all of them →
  S_gt = S_pred = all 72 voxels.
- Directed distances gt→pred: the 60 overlap voxels are also pred-surface
  voxels → 0; the 12 GT voxels at W = 3 have their nearest pred-surface voxel
  at W = 4, same (D, H) → 1. Symmetric for pred→gt (W = 9 → W = 8).
- HD95 iso: P95 of 72 values with 12 ones (top 16.7 %) = **1.0**; also HD (max)
  = 1.0. Spacing (D,H,W) = (1, 1, 3): the W step is 3 mm → **3.0**.
- ASSD iso: (12·1 + 12·1) / (72 + 72) = 24/144 = **0.166667**; directed pred→gt
  = 12/72 = 0.166667 (equal here — F2 separates them). Spacing (1,1,3): **0.5**.
- NSD, τ = 1.0 iso: every surface voxel is within 1 → 144/144 = **1.0**.
  Spacing (1,1,3), τ = 1.0 mm: 12 + 12 voxels at 3 mm fail → 120/144 =
  **0.833333**.

**F2 — Outlier.** Volume `(12, 10, 10)`. GT = cube `[2:6, 2:6, 2:6]` (64
voxels); Pred = GT ∪ {(10, 3, 3)}. Nearest GT voxel to the outlier is
(5, 3, 3) → distance 5.

- |Pred| = 65, overlap 64 → Dice = 128/129 = **0.992248**.
- Surfaces: a 4³ cube loses its 2³ interior under cross erosion → S_gt = 56.
  The isolated voxel is its own surface → S_pred = 57.
- gt→pred: all 56 are pred-surface voxels → all 0. pred→gt: 56 zeros and one 5.
- HD (percentile None, symmetric) = max(0, 5) = **5.0**. HD95: P95 of
  [0]·56 + [5] — the 95th-percentile position 0.95·56 = 53.2 lies among the
  zeros → 0; symmetric max(0, 0) = **0.0**. This is the `percentile=95`
  mutation check (5.0 vs 0.0).
- ASSD symmetric: (0·56 + 0·56 + 5) / (56 + 57) = 5/113 = **0.044248**.
  Directed pred→gt: 5/57 = **0.087719**. The `symmetric` mutation check.
- NSD τ = 1: (56 + 56) / (56 + 57) = 112/113 = **0.991150**.

**F3 — Empty cases.** Volume `(4, 8, 8)`. (i) GT empty, Pred = `[1:3, 2:5, 2:5]`;
(ii) Pred empty, GT = same block; (iii) both empty. Expected values are the
policy table in D3, for Dice, VS, HD95, ASSD, NSD, Boundary-IoU, PQ. In slice
mode (iii) never reaches a metric (both sides empty → skipped) — asserted via
the record's `is_empty` and missing scores; in volume mode it yields `None`.

**F4 — Panoptic quality.** Image `(16, 16)`, integer instance maps loaded with
`Raw()`. GT: instance 1 = `[0:4, 0:4]`, instance 2 = `[6:10, 6:10]` (16 px
each). Pred: instance 1 = `[0:4, 1:5]` (overlap 12, union 20, IoU 0.6 → matched
since > 0.5), no counterpart for GT 2 (FN), instance 3 = `[12:16, 0:4]` (no
overlap, FP). Kirillov et al. 2019, Eq. 1:
PQ = Σ_TP IoU / (|TP| + ½|FP| + ½|FN|) = 0.6 / (1 + 0.5 + 0.5) = **0.3**
(SQ = 0.6, RQ = 0.5). MONAI adds `smooth_numerator=1e-6` to the denominator,
so the framework reports 0.29999986: the case carries `tolerance=1e-6` and
the derivation names the term. The `panopticapi` adapter writes both maps as
`id2rgb` PNGs into a temporary directory and calls `pq_compute_single_core`
with a two-category set; background must be encoded as a **stuff** segment
(own id, category `isthing=0`) in both maps and PQ read from
`PQStat.pq_average(categories, isthing=True)`. Left as VOID (0), panopticapi
ignores the false-positive instance and removes VOID pixels from the matched
union, giving 0.5 instead of 0.3 (verified).

**F5 — Boundary IoU, physical band.** Volume `(8, 40, 40)`, spacing (D,H,W) =
(3, 1, 1). GT = `[2:6, 8:32, 8:32]` (4·24·24 = 2304 voxels). Pred_D =
`[3:7, 8:32, 8:32]` (shift one slice = 3 mm along D); Pred_W = `[2:6, 8:32, 9:33]`
(shift one voxel = 1 mm along W).

- Band width: 0.02·‖(24, 40, 40)‖ = 0.02·√3392 = 1.229 mm. Old floor 1.0 →
  1.229; new floor `max(spacing)` = **3.0**.
- Band (new): a voxel is in the band iff its euclidean distance to the nearest
  background voxel is ≤ 3.0 mm. Inside an axis-aligned box that distance is
  attained perpendicular to the nearest face, so: D = 2 and D = 5 slabs are
  entirely in the band (background 3 mm away along D) → 2·576 = 1152; on the
  D = 3, 4 slabs (6 mm from background along D) only the in-plane ring of
  width 3 (H or W ∈ {8, 9, 10} ∪ {29, 30, 31}) → 24² − 18² = 252 each → 504.
  |band| = **1656** for GT and both predictions.
- D-shift: GT band slabs at D = 2, 5 + rings at 3, 4; Pred band slabs at
  D = 3, 6 + rings at 4, 5. Intersection: D = 3 ring∩slab = 252, D = 4
  ring∩ring = 252, D = 5 slab∩ring = 252 → 756. Union = 2·1656 − 756 = 2556.
  Boundary IoU = 756/2556 = **0.295775**.
- W-shift: on D = 2, 5 slab∩slab over W 9:32 → 24·23 = 552 each → 1104. On
  D = 3, 4: within H 8:32 × W 9:32, GT ring is H ∈ {8,9,10,29,30,31} or
  W ∈ {9,10,29,30,31}; Pred ring is the same H set or W ∈ {9,10,11,30,31,32}.
  Intersection = 6 H-rows · 23 + 18 other rows · |{9,10,30,31}| = 138 + 72 =
  210 each → 420. Total 1524; union 3312 − 1524 = 1788. Boundary IoU =
  1524/1788 = **0.852349**.
- With the old floor (band 1.229 mm → only the width-1 in-plane ring on all
  four slabs, 92 each, |band| = 368) the scores are D-shift **0.6** and
  W-shift **0.333** — the 3 mm shift scores better than the 1 mm shift, which is
  the review's bug 3.9. The test pins the new values and asserts
  `score_D < score_W`.
- Spacing order: the same fixture with spacing (1, 1, 3) must give different
  numbers (asserted by inequality), pinning the tuple as (D, H, W).

**F6 — Multi-label.** Volume `(4, 8, 8)` int16. Both maps: label 1 =
`[0:1, 0:2, 0:2]` (identical in GT and Pred). Label 2: GT `[1:3, 2:6, 2:6]`
(32 voxels), Pred `[1:3, 2:6, 3:7]`, overlap 2·4·3 = 24. Under `Mask(label=2)`:
Dice = 48/64 = **0.75** — in volume mode, and as `aggregate_volumes()` of the
slice-mode counts (slices 1, 2: v_pred = v_gt = 16, tp = 12; slices 0, 3 are
empty on both sides under label 2 and contribute 0). Under `Mask()` (any
non-zero) the label-1 block adds 4 matched voxels to both: Dice = (48+8)/(64+8)
= 56/72 = **0.777778** — proves `label` selects.

**F7 — Random blobs.** Three seeded volumes `(16, 32, 32)` (thresholded
smoothed noise, seed 20260913), spacings (1,1,1), (1,1,3), (2.5,0.8,0.8). No
hand value; framework and same-variant official implementation (MedPy `dc`,
`assd`) agree to `abs=1e-5` for Dice and ASSD. HD95 has no official
implementation of the MONAI variant, so it is covered by F1/F2 hand values,
the MONAI cases, and the divergent MedPy column in the report only.

**MONAI cases** (`monai_cases.py`): the spherical fixtures from MONAI 1.6.0
`tests/metrics/test_hausdorff_distance.py`, `test_surface_distance.py`,
`test_surface_dice.py`, `test_compute_meandice.py` with their expected values,
Apache-2.0 header and MONAI version noted. One-sided-empty cases (MONAI
expects `inf`) are asserted as `None` per D3 and marked as such in the file.
These prove the adapter passes `percentile`, `directed`, `symmetric`,
`class_thresholds` and `spacing` (in (D, H, W) order) through unchanged.
Reading the expected lists: `test_hausdorff_distance.py` expands each case
over `product(["euclidean", "chessboard", "taxicab"], [directed=True,
directed=False])`, so the framework's value (euclidean, undirected) is index 1;
`test_surface_distance.py` lists `[symmetric=True, symmetric=False]`, so ASSD
is index 0. MONAI runs these with `include_background=False` on a mask
repeated over three channels, which equals one channel with
`include_background=True`. Cases without a percentile use
`hausdorff95_metric(percentile=None)`. Known deviations, each asserted per D3
and marked in the file: MONAI 1.6.0 returns `nan` (not `inf`) for one-sided
HD at percentile 95 and `inf` at percentile None, `inf` for one-sided ASSD,
1.0 for both-empty Dice with `ignore_empty=False` (its TEST_CASE_11).
`test_compute_meandice.py`'s tiny 2×2 cases go through the loader as 2D
files (slice mode).

### Acceptance (mutation checks)

Performed manually at the end of the plan, each expected to turn at least one
calibration test red, then reverted:

1. `percentile=95` default removed → F2 HD95 test (5.0 ≠ 0.0).
2. Dice replaced by IoU → F1 (0.714 ≠ 0.833).
3. `symmetric=True` default removed → F2 ASSD (0.0877 ≠ 0.0442).
4. Band floor back to 1.0 mm → F5.
5. Spacing tuple reversed in `_volume_factory` → F1 aniso, F5.
6. `Mask.apply` using `== 1` instead of `!= 0` → F6 and the 0/255 PNG test.

## Documentation

- `docs/wiki-followup-kalibrierung.md` (already started) gains: the three
  calibration layers, the variant table, the test command, the MONAI-cases
  layer, `Raw()` for PQ instance maps.
- `CLAUDE.md`: `Mask()` in the normalization paragraph and the metric summary
  (masks: "load with `Mask()`; PNG 0/255 and NIfTI 0/1 are equivalent"),
  `hatch run calibration:test`, `.venv-calibration`.
- `docs/calibration/segmentation.md`: generated, committed.

## Out of scope

Review items 3.2, 3.12, 3.14, 3.15, 3.16 and test 2.4 → Part C. Items 3.4,
3.5, 3.8, 3.13 and pins 2.2 → Part B. Cleanup 4.2, 4.4–4.8 and the dependency
pin reduction 1.4 → not planned.
