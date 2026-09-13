# Metric Calibration, Part A: Segmentation Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the segmentation metrics (Dice, HD95, ASSD, NSD, PQ, Boundary IoU, VS and the voxel counts) calibrated: one loader-side foreground policy (`Mask()`), an explicit empty-mask policy without `inf`/`NaN` in the CSV, a correct physical boundary band, and a calibration suite that pins every metric to hand-derived values, to the official reference implementation of the same definitional variant, and to MONAI's own test cases.

**Architecture:** Binarization moves out of the metric adapters into a new `normalization.Mask()` strategy; the `Normalizer` protocol gains `apply()`/`empty_slices()` so `ImageLoader` no longer knows strategy semantics. The adapters validate binary input, apply one shared empty-mask policy, and lose their `threshold`/`label` knobs. A leaf module `metric_spec.py` breaks the `metrics` ↔ `segmentation_metrics` import cycle. The test suite moves into a separate Hatch environment `calibration` (own venv, optional reference packages), where `tests/calibration/` holds hand-derived cases, official-implementation adapters, copied MONAI cases and a report generator.

**Tech Stack:** Python 3.14, Hatch 1.18 + uv 0.12, pytest 9, torch 2.14, MONAI 1.6.0, scipy 1.18, nibabel 5.4, numpy 2.5; reference packages `medpy==0.5.2`, `surface-distance==0.1`, `panopticapi@7bb4655` (git), vendored `mask_to_boundary` from `boundary-iou-api@37d2558`.

**Spec:** `docs/superpowers/specs/2026-09-13-calibration-segmentation-design.md` (read it first; the plan argues from it). Review source: `docs/reviews/2026-09-13-ultrareview-kalibrierung.md`.

## Global Constraints

- Python 3.14 in `.venv` (runtime) and, from Task 1 on, `.venv-calibration` (development). After Task 1 every test command is `hatch run calibration:test <pytest args>`; before Task 1 it is `source .venv/bin/activate && pytest <args>`.
- `[project.dependencies]` pins stay exact (`==`). Do not reduce the 120-pin list beyond the dev tools named in Task 1 (review 1.4 is explicitly not done).
- `main.py` is a sandbox: no functional edits. Tests import it (`from main import evaluate`), so its re-export block may gain `Mask`; nothing else changes there.
- Imports are absolute (`from iqaevaluator.<module> import ...`). No `sys.path` hacks beyond the existing one in `tests/conftest.py`.
- Spacing tuples are `(D, H, W)` everywhere, in millimetres.
- Every score that is undefined is `None` in a record; `inf` and `NaN` never reach a record or the CSV.
- Empty-mask policy (D3): one side empty → Dice/VS/NSD/PQ/Boundary-IoU `0.0`, HD95/ASSD `None`; both empty → `None` for every score; counts `v_pred`/`v_gt`/`tp` and `vs_signed` get no policy.
- Commit messages: Conventional Commits, imperative, ≤ 72-char subject; end the body with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Docstrings and comments are written in English, as in the rest of the package.

## File Map

| File | Responsibility after this plan |
|---|---|
| `pyproject.toml` | runtime deps without dev tools; `[calibration]` extras; Hatch env `calibration`; sdist without `tests/` |
| `src/iqaevaluator/normalization.py` | `Normalizer` protocol (`range_of`, `apply`, `empty_slices`); `sparse_slices()`; `MinMax`/`Percentile`/`FixedRange`/`Raw` via `_RangeBased`; new `Mask` |
| `src/iqaevaluator/segmentation_metrics/volume.py` | numpy metrics; `as_mask` (single binarization rule, `!= 0` default); shared adapter helpers `require_binary`, `foreground_counts`, `empty_policy`, `NOT_EMPTY` |
| `src/iqaevaluator/image_loader.py` | delegates `tensor`/`empty_slice_mask` to the normalizer; `load_pair` couples only when the target produced a range |
| `src/iqaevaluator/metric_spec.py` | **new leaf**: `DEVICE`, `Spacing`, `Metric`, `MetricSpec`, `ModeSupport`, `ModeUnsupported`, `ModeCapability`, `SkippedMetric`, `REASON_*`, `MetricDirection`, `MetricChannels` |
| `src/iqaevaluator/metrics.py` | registry + pyiqa adapter; re-exports everything from `metric_spec`; plain top imports, no bottom imports |
| `src/iqaevaluator/segmentation_metrics/monai_metrics.py` | one `_monai_spec` builder, `MonaiSegmentationMetric` with policy, `MonaiPanopticQualityMetric` |
| `src/iqaevaluator/segmentation_metrics/volume_metrics.py` | `VolumeFunctionMetric` without `threshold` |
| `src/iqaevaluator/segmentation_metrics/boundary_iou.py` | adapter without `threshold`; `band_width` floor `max(spacing)` |
| `src/iqaevaluator/volume_evaluator.py`, `evaluation_result.py`, `dreamsim_metric.py` | `is_empty` no longer set in volume mode; docstring; imports from `metric_spec` |
| `tests/calibration/{__init__,cases,harness,official,monai_cases,report}.py` | fixtures + hand values + derivations; framework runner through the public path; official adapters; MONAI cases; report generator |
| `tests/calibration/test_{hand,official,official_hand,monai_cases,random}.py` | the five calibration test layers |
| `tests/test_import_order.py`, `tests/test_mask_policy.py` | cycle regression; policy table across all seven scored metrics |
| `docs/calibration/segmentation.md` | generated report, committed |
| `CLAUDE.md`, `README.md`, `docs/wiki-followup-kalibrierung.md` | env layout, test command, `Mask()` |

Verified facts the plan relies on (2026-09-13, this machine): every hand value in the spec's F1–F6 reproduces with MONAI 1.6.0 / scipy 1.18.1; `medpy==0.5.2` builds on Python 3.14; `surface-distance==0.1` on PyPI is DeepMind's upload; `boundary-iou-api` is not installable as a package (see spec D8); `panopticapi` gives 0.3 on F4 only with background encoded as a stuff segment; baseline `pytest` on the ten test files this plan edits: 335 passed.

---

### Task 1: Test suite → Hatch environment `calibration` (spec Part 0)

**Files:**
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md` (Environment section, lines 9–30)
- Modify: `README.md` (setup section around lines 17–34)

**Interfaces:**
- Produces: Hatch env `calibration` at `.venv-calibration` with scripts `test` and `report`; extras `iqaevaluator[calibration]`. Every later task runs `hatch run calibration:test ...`.

- [ ] **Step 1: Confirm the removable direct pins**

Run:
```bash
source .venv/bin/activate && for p in pytest ruff pre-commit virtualenv cfgv identify nodeenv yapf distlib iniconfig pluggy python-discovery platformdirs tomli exceptiongroup; do printf "%-17s " $p; uv pip tree --python .venv/bin/python --invert --package $p | sed -n '2,20p' | grep -oE "[a-zA-Z0-9_.-]+ v[0-9]" | sort -u | tr '\n' ' '; echo; done
```
Expected: every listed package has only `iqaevaluator`, `pyiqa`, `pytest`, `pre-commit`, `virtualenv`, `yapf` or `python-discovery` as parents — i.e. our direct pin plus dev tools (pyiqa itself declares pytest/ruff/pre-commit/yapf/tensorboard, spec D9 caveat). `filelock`, `packaging`, `pygments`, `tensorboard` have runtime parents and stay.

- [ ] **Step 2: Edit `[project.dependencies]`**

Delete exactly these 15 lines from the `dependencies` list in `pyproject.toml`:
```
  'cfgv==3.5.0',
  'distlib==0.4.3',
  'exceptiongroup==1.3.1',
  'identify==2.6.19',
  'iniconfig==2.3.0',
  'nodeenv==1.10.0',
  'platformdirs==4.11.8',
  'pluggy==1.6.0',
  'pre_commit==4.6.2',
  'pytest==9.1.1',
  'python-discovery==1.6.0',
  'ruff==0.16.6',
  'tomli==2.4.1',
  'virtualenv==21.7.9',
  'yapf==0.43.0',
```

- [ ] **Step 3: Add extras, Hatch metadata and the calibration env; drop `test` from default; drop `tests/` from the sdist**

Insert after the `[project.urls]` table:
```toml
[project.optional-dependencies]
# Official reference implementations used only by tests/calibration/ (spec D8).
calibration = [
  "surface-distance==0.1",   # DeepMind's own PyPI upload; Nikolov et al. — NSD (author implementation), HD, ASSD (area-weighted variant)
  "medpy==0.5.2",            # Dice, HD, HD95 (concatenated variant), ASSD (voxel variant); sdist only, builds on Python 3.14
  "panopticapi @ git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0",
]

[tool.hatch.metadata]
allow-direct-references = true
```
Replace the `[tool.hatch.envs.default.scripts]` table with:
```toml
[tool.hatch.envs.default.scripts]
evaluate = "python src/main.py {args}"

# Development happens here. The runtime env (.venv) stays free of our own dev
# pins so that `hatch env create` proves the package installs without them
# (pyiqa still pulls pytest/ruff/pre-commit/yapf transitively — not ours to fix).
[tool.hatch.envs.calibration]
path = ".venv-calibration"
installer = "uv"
features = ["calibration"]
dependencies = [
  "pytest==9.1.1",
  "ruff==0.16.6",
  "pre_commit==4.6.2",
  "yapf==0.43.0",
]

[tool.hatch.envs.calibration.scripts]
test = "pytest {args:tests}"
report = "python tests/calibration/report.py"
```
In `[tool.hatch.build.targets.sdist] include` delete the line `"tests/",`.

- [ ] **Step 4: Create the env and run the suite**

Run (downloads the torch stack into `.venv-calibration`, ~1.7 GB, several minutes):
```bash
hatch env create calibration && hatch run calibration:test -q -x
```
Expected: the same pass count as before (`pytest --collect-only -q | tail -1` reported 589 tests at HEAD); `hatch env find calibration` prints `.../.venv-calibration`.

- [ ] **Step 5: Verify the runtime env and the sdist**

Run:
```bash
uv pip install --python .venv/bin/python -e . && uv pip tree --python .venv/bin/python --invert --package pytest
```
Expected: `pytest` has `pyiqa` as its only parent; `iqaevaluator` no longer appears under it.
Run:
```bash
hatch build -t sdist && (tar tzf dist/iqaevaluator-0.1.0.tar.gz | grep -c '/tests/' || true) && rm -rf dist
```
Expected: `0`.

- [ ] **Step 6: Update `CLAUDE.md` and `README.md`**

In `CLAUDE.md`, replace the paragraph starting "Hatch is both the packaging **build backend**" up to and including the `hatch run evaluate` code block with:
```markdown
Hatch is both the packaging **build backend** (`hatchling`) and the **environment manager**. Two environments:

- `default` → `.venv/`: the runtime install (`hatch env create`, or `uv venv .venv && uv pip install -e .`). No dev tools of our own (pyiqa still pulls pytest/ruff/pre-commit/yapf as its own dependencies).
- `calibration` → `.venv-calibration/`: where development happens. Installs the package with the `[calibration]` extras (medpy, surface-distance, panopticapi) plus pytest/ruff. Create once with `hatch env create calibration` (downloads its own torch stack, ~1.7 GB).

Both use `installer = "uv"`; `uv` must be on PATH. `pyproject.toml`'s `[project.dependencies]` is the single source of truth for runtime versions — exact pins on purpose (pyiqa's outputs are backend-version-dependent). Two pins are load-bearing: `setuptools` must stay `<81` (`openai-clip` needs `pkg_resources`) and `pyiqa` is version-pinned deliberately. The sdist ships `src/`, `docs/`, `README.md` and `pyproject.toml` — `tests/` is not part of the runtime package.

Hatch scripts:

```bash
hatch run calibration:test [pytest args]   # the whole suite, incl. tests/calibration/
hatch run calibration:report               # regenerates docs/calibration/segmentation.md
hatch run evaluate <args>                   # src/main.py in the runtime env
```
```
In `README.md`, next to the existing `hatch env create` instructions, add one sentence: "Tests run in a second environment: `hatch env create calibration && hatch run calibration:test` (the runtime `.venv` carries no test suite)."

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml CLAUDE.md README.md
git commit -m "build: move the test suite into a hatch calibration env

Dev tools leave [project.dependencies], tests/ leaves the sdist, and the
official reference packages for the calibration suite become the
[calibration] extras of a separate .venv-calibration. boundary-iou-api is
not installable as a package (empty top-level package, unpinned
panopticapi URL) and is vendored later instead.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 2: `Normalizer` protocol gains `apply()`/`empty_slices()`; `as_mask` default is "any non-zero" (spec Part 1, D7)

**Files:**
- Modify: `src/iqaevaluator/normalization.py` (protocol at lines 39-44; strategies at lines 100-176)
- Modify: `src/iqaevaluator/segmentation_metrics/volume.py:12-36` (`as_mask`) and the `label: int = 1` defaults at lines 47, 53, 59, 67, 82
- Test: `tests/test_normalization.py`, `tests/test_volume.py`

**Interfaces:**
- Produces: `normalization.sparse_slices(raw: np.ndarray) -> torch.Tensor` (`(D,)` bool); every strategy has `apply(raw, *, source: str) -> torch.Tensor` (`(D, H, W)`) and `empty_slices(raw) -> torch.Tensor`; `volume.as_mask(x, label: Optional[int] = None, threshold: float = 0.5)` — integer input: `x != 0` when `label is None`, else `x == label`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_volume.py` (after `test_as_mask_int_label_map`):
```python
def test_as_mask_int_default_is_any_nonzero():
    x = np.array([0, 1, 2, 255], dtype=np.int16)
    np.testing.assert_array_equal(as_mask(x), [False, True, True, True])


def test_as_mask_bool_passes_through():
    x = np.array([True, False])
    assert as_mask(x) is x
```
Append to `tests/test_normalization.py` (imports at the top of the file gain `sparse_slices`, `scale`, and `torch`):
```python
class TestSparseSlices:
    def test_constant_volume_is_all_empty(self):
        assert sparse_slices(np.full((3, 8, 8), 7, dtype=np.uint8)).tolist() == [True, True, True]

    def test_blank_slice_is_flagged(self):
        vol = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32) * 100
        vol[2] = 0.0
        assert sparse_slices(vol).tolist() == [False, False, True, False]

    def test_sparse_label_map_keeps_its_one_slice(self):
        labels = np.zeros((4, 64, 64), dtype=np.int16)
        labels[1, 10, 10:13] = 1
        assert sparse_slices(labels).tolist() == [True, False, True, True]


class TestApplyAndEmptySlices:
    @pytest.mark.parametrize("strategy", [MinMax(), Percentile(), FixedRange(IntensityRange(0.0, 200.0), "fixed"), Raw()])
    def test_apply_is_scale_over_range_of(self, strategy):
        raw = (np.random.default_rng(1).random((2, 8, 8)) * 200).astype(np.float32)
        expected = scale(raw, strategy.range_of(raw), label="x")
        assert torch.equal(strategy.apply(raw, source="x"), expected)

    @pytest.mark.parametrize("strategy", [MinMax(), Percentile(), FixedRange(None, "raw"), Raw()])
    def test_empty_slices_is_the_sparse_heuristic(self, strategy):
        raw = np.random.default_rng(2).random((3, 16, 16)).astype(np.float32)
        raw[1] = 0.0
        assert torch.equal(strategy.empty_slices(raw), sparse_slices(raw))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `hatch run calibration:test tests/test_normalization.py tests/test_volume.py -q`
Expected: FAIL — `ImportError: cannot import name 'sparse_slices'`, and `test_as_mask_int_default_is_any_nonzero` fails with `[False, True, False, False]`.

- [ ] **Step 3: Implement `as_mask`'s new rule**

Replace `as_mask` in `src/iqaevaluator/segmentation_metrics/volume.py` with:
```python
from typing import Optional


def as_mask(x: np.ndarray, label: Optional[int] = None, threshold: float = 0.5) -> np.ndarray:
    """Binarize an array to a boolean mask — the framework's single binarization rule.

    `normalization.Mask()` calls this at load time; the numpy metrics below
    call it again on whatever they are handed (a no-op on the {0, 1} floats
    `Mask()` produces).

    - bool array: returned unchanged.
    - float array: values must lie in [0, 1] (probabilities); thresholded
      at `threshold`. Values outside [0, 1] raise ValueError — this usually
      means raw logits were passed without a sigmoid activation.
    - integer array (label map): every non-zero value is foreground when
      `label is None`; otherwise one-vs-rest via `x == label`. A 0/255 PNG
      mask and a 0/1 NIfTI mask therefore binarize identically.
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
        return x != 0 if label is None else x == label
    raise TypeError(f"as_mask does not support dtype {x.dtype}")
```
Change every `label: int = 1` in `v_pred`, `v_gt`, `tp`, `vs`, `vs_signed` to `label: Optional[int] = None` (the docstrings of `vs`/`vs_signed` need no change).

- [ ] **Step 4: Implement `sparse_slices`, `_RangeBased` and the protocol**

In `src/iqaevaluator/normalization.py`, replace the `Normalizer` protocol with:
```python
@runtime_checkable
class Normalizer(Protocol):
    """One strategy for turning a decoded array into the tensor metrics see.

    `range_of` says what [0, 1] stands for (None when nothing is scaled),
    `apply` produces the `(D, H, W)` tensor, `empty_slices` says which slices
    carry nothing worth scoring. `name` ends up in the report. `source` is
    the file name, used only in messages.
    """
    name: str

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]: ...
    def apply(self, raw: np.ndarray, *, source: str) -> torch.Tensor: ...
    def empty_slices(self, raw: np.ndarray) -> torch.Tensor: ...
```
Add after `scale()` (this is `ImageLoader.empty_slice_mask`'s body, moved verbatim):
```python
def sparse_slices(raw: np.ndarray) -> torch.Tensor:
    """True for slices with (almost) no content, judged on the raw intensities.

    A slice is empty when its spread is below 0.1 % of the volume's intensity
    span, or (ordinary intensity images only, see below) its lift above the
    volume floor is below that same 0.1 %. The span is measured between the
    0.5th and 99.5th percentiles, so one spike voxel cannot shrink it and mark
    healthy slices empty. For a volume whose foreground is rarer than 0.5 %
    the percentile span collapses to zero and the extremes are used instead —
    and in that fallback regime the lift test is skipped entirely, because a
    sparse image's slice mean sits only a tiny fraction of the way from floor
    to ceiling by construction. A constant volume has no span at all and
    every slice counts as empty.

    This is the heuristic for intensity images. Masks loaded with `Mask()`
    use an exact voxel count instead (`Mask.empty_slices`).
    """
    arr = raw.astype(np.float32, copy=False)
    lo, hi = (float(v) for v in np.percentile(arr, [0.5, 99.5]))
    used_extremes = hi <= lo
    if used_extremes:
        lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    if span <= 0.0:
        return torch.ones(arr.shape[0], dtype=torch.bool)
    flat = arr.reshape(arr.shape[0], -1)
    std = flat.std(axis=1)
    empty = std < 1e-3 * span
    if not used_extremes:
        lift = flat.mean(axis=1) - lo
        empty = empty | (lift < 1e-3 * span)
    return torch.from_numpy(empty)


class _RangeBased:
    """`apply`/`empty_slices` shared by every strategy that maps a range onto [0, 1]."""

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:  # overridden
        raise NotImplementedError

    def apply(self, raw: np.ndarray, *, source: str) -> torch.Tensor:
        return scale(raw, self.range_of(raw), label=source)

    def empty_slices(self, raw: np.ndarray) -> torch.Tensor:
        return sparse_slices(raw)
```
Make the four dataclasses inherit it: `class MinMax(_RangeBased):`, `class Percentile(_RangeBased):`, `class FixedRange(_RangeBased):`, `class Raw(_RangeBased):` (keep `@dataclass(frozen=True)`; each still defines its own `range_of`). In `Percentile`'s docstring replace "mirroring the same fallback in `ImageLoader.empty_slice_mask`" with "mirroring the same fallback in `sparse_slices`".

- [ ] **Step 5: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_normalization.py tests/test_volume.py tests/test_image_loader.py -q`
Expected: all pass (the loader still computes its own mask in this task; Task 4 switches it).

- [ ] **Step 6: Commit**

```bash
git add src/iqaevaluator/normalization.py src/iqaevaluator/segmentation_metrics/volume.py tests/test_normalization.py tests/test_volume.py
git commit -m "feat(normalization): apply/empty_slices on the Normalizer protocol

sparse_slices() holds the loader's empty-slice heuristic; as_mask treats
every non-zero label as foreground by default (spec D5, D7).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `Mask()` normalizer (spec Part 1, D1, D2, D6)

**Files:**
- Modify: `src/iqaevaluator/normalization.py` (new class after `Raw`; `NORMALIZER_NAMES`; module docstring)
- Test: `tests/test_normalization.py`

**Interfaces:**
- Produces: `normalization.Mask(label: Optional[int] = None, threshold: float = 0.5)`, frozen dataclass; `name` = `"mask"` or `f"mask_label_{label}"`; `range_of` → `None`; `apply` → float32 `{0.0, 1.0}` `(D, H, W)`; `empty_slices` → exact zero-foreground count; `NORMALIZER_NAMES["mask"]`.
- Consumes: `as_mask` from Task 2.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_normalization.py` (import `Mask` at the top):
```python
class TestMask:
    def test_names(self):
        assert Mask().name == "mask"
        assert Mask(label=2).name == "mask_label_2"

    def test_range_of_is_none(self):
        assert Mask().range_of(np.zeros((1, 2, 2), np.uint8)) is None

    def test_0_255_and_0_1_give_identical_tensors(self):
        png = np.array([[[0, 255], [255, 0]]], dtype=np.uint8)
        nii = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
        a, b = Mask().apply(png, source="a.png"), Mask().apply(nii, source="b.nii")
        assert torch.equal(a, b)
        assert a.dtype == torch.float32 and a.shape == (1, 2, 2)
        assert set(a.unique().tolist()) == {0.0, 1.0}

    def test_label_selects_one_class(self):
        labels = np.array([[[0, 1], [2, 2]]], dtype=np.int16)
        assert Mask(label=2).apply(labels, source="l.nii").tolist() == [[[0.0, 0.0], [1.0, 1.0]]]
        assert Mask().apply(labels, source="l.nii").tolist() == [[[0.0, 1.0], [1.0, 1.0]]]

    def test_float_map_thresholds_at_half(self):
        probs = np.array([[[0.0, 0.49], [0.5, 1.0]]], dtype=np.float32)
        assert Mask().apply(probs, source="p.nii").tolist() == [[[0.0, 0.0], [1.0, 1.0]]]
        assert Mask(threshold=0.9).apply(probs, source="p.nii").tolist() == [[[0.0, 0.0], [0.0, 1.0]]]

    def test_float_map_outside_unit_interval_names_the_source(self):
        with pytest.raises(ValueError, match=r"logits\.nii"):
            Mask().apply(np.array([[[-3.0, 4.0]]], dtype=np.float32), source="logits.nii")

    def test_empty_slices_is_an_exact_count(self):
        vol = np.zeros((3, 8, 8), dtype=np.uint8)
        vol[1, 4, 4] = 1                                # one voxel is enough
        assert Mask().empty_slices(vol).tolist() == [True, False, True]
        assert Mask(label=2).empty_slices(vol).tolist() == [True, True, True]

    def test_registered_under_its_cli_name(self):
        assert NORMALIZER_NAMES["mask"] is Mask
        assert normalizer_from_name("mask") == Mask()
```
Update the two existing expectations: in `TestFromName.test_names_match_the_registry` the set becomes `{"minmax", "percentile", "raw", "mask"}`; in `TestProtocol` the parametrize list gains `Mask(), Mask(label=2)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `hatch run calibration:test tests/test_normalization.py -q`
Expected: FAIL with `ImportError: cannot import name 'Mask'`.

- [ ] **Step 3: Implement `Mask`**

Add to `src/iqaevaluator/normalization.py` (top-level import `from iqaevaluator.segmentation_metrics.volume import as_mask` — `volume.py` has no framework imports, so this is cycle-free), after `Raw`:
```python
@dataclass(frozen=True)
class Mask:
    """Binarize at load time: the one foreground policy for segmentation runs.

    Integer and bool arrays: every non-zero value is foreground when `label`
    is None (a 0/255 PNG and a 0/1 NIfTI come out identical); `label=k`
    keeps only `== k`, so a multi-class map is scored one label per run.
    Float arrays are probability maps and are cut at `threshold`; values
    outside [0, 1] raise. The tensor is float32 with values in exactly
    {0.0, 1.0}, and a slice is empty iff it holds no foreground voxel —
    an exact count, not the intensity heuristic. Nothing is scaled, so the
    report's `scale_lo/hi` stay empty and `range_of` is None.
    """
    label: Optional[int] = None
    threshold: float = 0.5

    @property
    def name(self) -> str:
        return "mask" if self.label is None else f"mask_label_{self.label}"

    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]:
        return None

    def _binary(self, raw: np.ndarray) -> np.ndarray:
        return as_mask(raw, self.label, self.threshold)

    def apply(self, raw: np.ndarray, *, source: str) -> torch.Tensor:
        try:
            mask = self._binary(raw)
        except ValueError as exc:
            raise ValueError(f"[{source}] {exc}") from None
        return torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))

    def empty_slices(self, raw: np.ndarray) -> torch.Tensor:
        foreground = self._binary(raw).reshape(raw.shape[0], -1)
        return torch.from_numpy(~foreground.any(axis=1))
```
Register it: `NORMALIZER_NAMES` gains `"mask": Mask,`. In the module docstring's strategy list add `Mask(label=None, threshold=0.5)  binarize at load time — the way to load masks` and change the `Raw()` line to `no scaling, dtype preserved — the escape hatch for instance maps (PQ) and anything that must keep its dtype`. In `Raw`'s docstring replace the last two sentences with: "Segmentation masks belong to `Mask()`; `Raw()` is for panoptic-quality instance maps and for callers that need the stored values untouched."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_normalization.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/iqaevaluator/normalization.py tests/test_normalization.py
git commit -m "feat(normalization): Mask() strategy binarizes at load time

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Loader delegates to the strategy; `load_pair` couples only when a range exists; volume mode drops `is_empty` (spec Part 1)

**Files:**
- Modify: `src/iqaevaluator/image_loader.py:21` (import), `:257-262` (`tensor`), `:275-310` (`empty_slice_mask`), `:318-338` (`load_pair`)
- Modify: `src/iqaevaluator/volume_evaluator.py:103-111`
- Modify: `src/iqaevaluator/evaluation_result.py:93-101` (docstring)
- Modify: `src/main.py:9-11` (re-export `Mask`)
- Test: `tests/test_image_loader.py`, `tests/test_iqa_evaluator.py`, `tests/test_volume_evaluator.py`

**Interfaces:**
- Produces: `ImageLoader.tensor` = `normalizer.apply(raw, source=path.name).unsqueeze(1)`; `ImageLoader.empty_slice_mask` = `normalizer.empty_slices(raw)`; `load_pair(input_path, target_path, normalizer)` gives the input `normalizer` itself when `target.intensity_range is None`, else `FixedRange(range, name=normalizer.name)`; volume-mode records have `is_empty=False`.
- Consumes: `Mask`, `sparse_slices` (Tasks 2–3).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_loader.py` (imports gain `Mask`):
```python
class TestMaskLoading:
    def test_png_0_255_and_nifti_0_1_give_identical_tensors(self, tmp_path):
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[4:12, 4:12] = 1
        png = tmp_path / "m.png"
        Image.fromarray(arr * 255).save(png)
        nii = tmp_path / "m.nii"
        nib.save(nib.Nifti1Image(arr[:, :, None], np.eye(4)), str(nii))
        a, b = ImageLoader(png, Mask()).tensor, ImageLoader(nii, Mask()).tensor
        assert a.shape == b.shape == (1, 1, 16, 16)
        assert torch.equal(a, b)
        assert a.dtype == torch.float32 and float(a.sum()) == 64.0

    def test_label_selects_one_class_through_the_loader(self, tmp_path):
        labels = np.zeros((8, 8, 2), dtype=np.int16)
        labels[:4, :, 0] = 1
        labels[4:, :, 1] = 2
        p = tmp_path / "labels.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        assert float(ImageLoader(p, Mask(label=2)).tensor.sum()) == 32.0
        assert float(ImageLoader(p, Mask()).tensor.sum()) == 64.0

    def test_float_map_outside_unit_interval_names_the_file(self, tmp_path):
        p = tmp_path / "logits.nii"
        nib.save(nib.Nifti1Image(np.full((4, 4, 1), 3.0, dtype=np.float32), np.eye(4)), str(p))
        with pytest.raises(ValueError, match="logits.nii"):
            ImageLoader(p, Mask()).tensor

    def test_empty_slice_mask_is_exact_under_mask(self, tmp_path):
        vol = np.zeros((32, 32, 3), dtype=np.uint8)
        vol[5, 5, 1] = 1
        p = tmp_path / "one_voxel.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        assert ImageLoader(p, Mask()).empty_slice_mask.tolist() == [True, False, True]
        assert ImageLoader(p, Mask(label=2)).empty_slice_mask.tolist() == [True, True, True]

    def test_intensity_range_is_none_and_raw_range_survives(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[2:4, 2:4] = 255
        p = tmp_path / "m.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p, Mask())
        assert loader.intensity_range is None
        assert loader.raw_range == IntensityRange(0.0, 255.0)
```
Append to `TestLoadPair` in the same file:
```python
    def test_mask_pair_binarizes_each_side_independently(self, tmp_path):
        # Review 3.1: a 0/1 NIfTI prediction against a 0/255 PNG reference.
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[4:12, 4:12] = 1
        pred = tmp_path / "case_pred.nii"
        nib.save(nib.Nifti1Image(arr[:, :, None], np.eye(4)), str(pred))
        gt = tmp_path / "case_gt.png"
        Image.fromarray(arr * 255).save(gt)
        inp, tgt = load_pair(pred, gt, Mask())
        assert inp.normalizer == Mask() and tgt.normalizer == Mask()
        assert torch.equal(inp.tensor, tgt.tensor)

    def test_raw_pair_is_not_wrapped_in_a_fixed_range(self, tmp_path):
        labels = np.zeros((8, 8, 2), dtype=np.int16); labels[:4] = 1
        p1 = tmp_path / "a.nii"; p2 = tmp_path / "b.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p1))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p2))
        inp, _ = load_pair(p1, p2, Raw())
        assert inp.normalizer == Raw()
```
Append to `tests/test_iqa_evaluator.py`:
```python
class TestMaskRunEndToEnd:
    def test_perfect_prediction_scores_dice_and_vs_one(self, tmp_path):
        """Review 3.1 failure case: 0/1 NIfTI prediction vs 0/255 PNG reference."""
        import nibabel as nib
        from PIL import Image
        from iqaevaluator.normalization import Mask
        from iqaevaluator.image_loader import load_pair
        from iqaevaluator.segmentation_metrics.monai_metrics import DICE
        from iqaevaluator.segmentation_metrics.volume_metrics import VS

        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[8:24, 8:24] = 1
        pred = tmp_path / "case_pred.nii"
        nib.save(nib.Nifti1Image(arr[:, :, None], np.eye(4)), str(pred))
        gt = tmp_path / "case_gt.png"
        Image.fromarray(arr * 255).save(gt)

        inp, tgt = load_pair(pred, gt, Mask())
        records = IQAEvaluator(inp, tgt, MetricRegistry(DICE, VS)).run_evaluation()
        assert len(records) == 1
        assert records[0].extra["dice"] == pytest.approx(1.0)
        assert records[0].extra["vs"] == pytest.approx(1.0)
        assert records[0].normalization == "mask"
        assert records[0].scale_lo is None and records[0].scale_hi is None
        assert (records[0].input_min, records[0].input_max) == (0.0, 1.0)
```
Append to `tests/test_volume_evaluator.py`:
```python
class TestVolumeRowsHaveNoEmptyFlag:
    def test_all_zero_input_is_not_flagged_empty(self, tmp_path: Path, spy_registry):
        import nibabel as nib

        registry, _, _ = spy_registry
        p = tmp_path / "blank.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((8, 10, 6), dtype="float32"), np.diag([1.0, 1.0, 1.2, 1.0])), p)
        record = VolumeEvaluator(ImageLoader(p), None, registry).run_evaluation()[0]
        assert record.is_empty is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `hatch run calibration:test tests/test_image_loader.py tests/test_iqa_evaluator.py tests/test_volume_evaluator.py -q`
Expected: FAIL — `test_mask_pair_binarizes_each_side_independently` (`inp.normalizer` is a `FixedRange`), `test_perfect_prediction_scores_dice_and_vs_one` (dice 0.0), `test_all_zero_input_is_not_flagged_empty` (`is_empty` True), `test_empty_slice_mask_is_exact_under_mask` (heuristic result).

- [ ] **Step 3: Implement**

`src/iqaevaluator/image_loader.py`: change the import to `from iqaevaluator.normalization import FixedRange, IntensityRange, MinMax, Normalizer` (drop `scale`). Replace `tensor`, `empty_slice_mask` and `load_pair`:
```python
    @property
    def tensor(self) -> torch.Tensor:
        """(D, 1, H, W). float32 in [0, 1] under every strategy except `Raw()`
        (dtype preserved, unscaled); exactly {0.0, 1.0} under `Mask()`."""
        if self._tensor is None:
            self._tensor = self.normalizer.apply(self.raw, source=self.path.name).unsqueeze(1)
        return self._tensor
```
```python
    @property
    def empty_slice_mask(self) -> torch.Tensor:
        """True for slices with nothing to score, as the strategy defines it.

        Intensity strategies use `normalization.sparse_slices` (a spread/lift
        heuristic on the raw data); `Mask()` counts foreground voxels exactly.
        """
        return self.normalizer.empty_slices(self.raw)
```
```python
def load_pair(
    input_path: Path,
    target_path: Path,
    normalizer: Normalizer = MinMax(),
) -> tuple[ImageLoader, ImageLoader]:
    """Load a full-reference pair on ONE scale — the target's — when there is one.

    The target is scaled by `normalizer`; if that produced a range, the input
    is scaled by the same range, so a prediction that is uniformly too bright
    or too flat is measured as such instead of being normalized away
    (fastMRI's convention: `data_range` comes from the reference). Anything
    outside the target's range is clipped to [0, 1] in the input; the raw
    extremes remain visible as `ImageLoader.raw_range`.

    Under `Mask()` and `Raw()` the target yields no range and both sides are
    loaded with `normalizer` itself, independently — a 0/1 prediction against
    a 0/255 reference binarizes to the same tensor.

    Returns `(input, target)`.
    """
    target = ImageLoader(target_path, normalizer)
    rng = target.intensity_range
    input_normalizer = normalizer if rng is None else FixedRange(rng, name=normalizer.name)
    return ImageLoader(input_path, input_normalizer), target
```
Also update the `ImageLoader` class docstring's last sentence to "For a full-reference pair use `load_pair()`, which puts the input on the target's scale when the strategy produces one."

`src/iqaevaluator/volume_evaluator.py`: delete the line `is_empty=bool(self.input.empty_slice_mask.all().item()),` in `run_evaluation` (the record default `False` applies) and add to the module docstring's "Two deliberate differences" list: "- `is_empty` is never set: the flag describes a skipped slice, and a volume row is never skipped."

`src/iqaevaluator/evaluation_result.py`, `aggregate_volumes` docstring: replace the sentence starting "Slices flagged `is_empty` are blank on both sides" with: "Slices flagged `is_empty` hold zero foreground voxels on both sides — under `Mask()` that is an exact count, and `IQAEvaluator` only skips a slice when input and target are both empty — so their counts are real zeros and are filled in as such."

`src/main.py` line 10: add `Mask` to the re-exported names: `NORMALIZER_NAMES, Mask, MinMax, Normalizer, Percentile, Raw, normalizer_from_name,`. The CLI's `--normalization` choices come from `NORMALIZER_NAMES`, so `mask` is accepted without further edits.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_image_loader.py tests/test_iqa_evaluator.py tests/test_volume_evaluator.py tests/test_main.py tests/test_evaluation_result.py -q`
Expected: all pass (the existing `test_empty_slice_mask_*` tests pass unchanged because `sparse_slices` is the old body).

- [ ] **Step 5: Commit**

```bash
git add src/iqaevaluator/image_loader.py src/iqaevaluator/volume_evaluator.py src/iqaevaluator/evaluation_result.py src/main.py tests/test_image_loader.py tests/test_iqa_evaluator.py tests/test_volume_evaluator.py
git commit -m "fix(loader): masks binarize per side; loader defers to the strategy

load_pair no longer forces a FixedRange on the input when the target has
no range (review 3.1); volume rows never carry is_empty (review 3.11).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 5: Leaf module `metric_spec.py` ends the import cycle (spec Part 2, review 1.2)

**Files:**
- Create: `src/iqaevaluator/metric_spec.py`
- Modify: `src/iqaevaluator/metrics.py:22-115` (definitions move out), `:272-285` (bottom imports move up)
- Modify: `src/iqaevaluator/dreamsim_metric.py:30`, `src/iqaevaluator/segmentation_metrics/monai_metrics.py:54`, `src/iqaevaluator/segmentation_metrics/boundary_iou.py:34-39,49`, `src/iqaevaluator/segmentation_metrics/volume_metrics.py:22-23,31`, `src/iqaevaluator/image_loader.py:23`
- Modify: `tests/test_dreamsim_metric.py:12-16` (comment)
- Create: `tests/test_import_order.py`

**Interfaces:**
- Produces: `iqaevaluator.metric_spec` exporting `DEVICE`, `MetricDirection`, `MetricChannels`, `Spacing`, `REASON_DEEP_2D`, `REASON_NO_VOLUME_IMPL`, `ModeSupport`, `ModeUnsupported`, `ModeCapability`, `SkippedMetric`, `Metric`, `MetricSpec`. `iqaevaluator.metrics` re-exports all of them, so every existing `from iqaevaluator.metrics import MetricSpec` keeps working. `ScoringMode` stays in `metrics.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_import_order.py`:
```python
"""Every metric module must be importable first, in a fresh interpreter.

Before metric_spec.py existed, metrics.py imported the segmentation modules
and dreamsim_metric at its bottom while they imported MetricSpec back from
metrics — whichever side was imported first hit the other mid-initialisation
(review 1.2). A fresh subprocess is the only honest test: the pytest process
has already imported everything.
"""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "iqaevaluator.segmentation_metrics.boundary_iou",
    "iqaevaluator.segmentation_metrics.monai_metrics",
    "iqaevaluator.dreamsim_metric",
])
def test_module_imports_first_in_a_fresh_interpreter(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `hatch run calibration:test tests/test_import_order.py -q`
Expected: FAIL for all three with `ImportError: cannot import name ... from partially initialized module`.

- [ ] **Step 3: Create `metric_spec.py`**

Create `src/iqaevaluator/metric_spec.py` with the definitions cut from `metrics.py` (lines 22–115: `DEVICE`, the two `Literal`s, `Spacing`, both `REASON_*`, `ModeSupport`, `ModeUnsupported`, `ModeCapability`, `SkippedMetric`, `Metric`, `MetricSpec` — bodies and docstrings unchanged), under this header:
```python
"""Metric description types — a leaf module with no framework imports.

`metrics.py` (registry, pyiqa adapter, built-in specs), every module under
`segmentation_metrics/` and `dreamsim_metric.py` all need `MetricSpec` and
its companions. Keeping them here, importing only torch, means the metric
modules never import `metrics.py` and `metrics.py` can import them at its
top like any other module — no import cycle, no import-order rules.

`metrics.py` re-exports every name below, so `from iqaevaluator.metrics
import MetricSpec` keeps working.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol, Sequence, runtime_checkable

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MetricDirection = Literal["higher_is_better", "lower_is_better", "not_ranked"]
MetricChannels  = Literal["gray", "rgb"]

Spacing = tuple[float, float, float]
"""Voxel size in millimetres, ordered like the array axes: (depth, height, width)."""
```
followed by `REASON_DEEP_2D`, `REASON_NO_VOLUME_IMPL`, `ModeSupport`, `ModeUnsupported`, `ModeCapability = ModeSupport | ModeUnsupported`, `SkippedMetric`, `Metric`, `MetricSpec` exactly as they are today in `metrics.py`. In the `Metric` protocol docstring replace "Under the `Raw` strategy the batch keeps its stored dtype — only segmentation metrics are meant to receive that." with "Under `Mask()` the batch holds exactly {0, 1}; the segmentation adapters require that and reject anything else. Under `Raw()` the batch keeps its stored dtype — only the panoptic-quality adapter is meant to receive that."

- [ ] **Step 4: Rewire `metrics.py`**

Replace lines 22–47 (the imports, `DEVICE`, the `Literal`s, `Spacing`, the "Duplicated from image_loader" comment) and delete lines 49–115 (`REASON_*` through `MetricSpec`) so the top of the module reads:
```python
from typing import Callable, Literal, Optional

import pyiqa
import torch

from iqaevaluator import radimagenet_lpips  # noqa: F401 — registers RadImageNetLPIPS in pyiqa
from iqaevaluator import clip_iqa_medical   # noqa: F401 — registers ClipIQALung / ClipIQABrain in pyiqa

from iqaevaluator.constants import RESNET50
from iqaevaluator.metric_spec import (  # noqa: F401 — re-exported; the public names live here
    DEVICE, Metric, MetricChannels, MetricDirection, MetricSpec, ModeCapability,
    ModeSupport, ModeUnsupported, REASON_DEEP_2D, REASON_NO_VOLUME_IMPL,
    SkippedMetric, Spacing,
)
from iqaevaluator.segmentation_metrics.monai_metrics import (
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY,
)
from iqaevaluator.segmentation_metrics.boundary_iou import BOUNDARY_IOU
from iqaevaluator.segmentation_metrics.volume_metrics import (
    VS, VS_SIGNED, V_PRED, V_GT, TP,
)
from iqaevaluator.dreamsim_metric import DREAMSIM, dreamsim_spec

ScoringMode = Literal["slice", "volume"]
```
Delete the two bottom import blocks (lines 272–285) together with their "Imported here … circular import" comments. Everything else in `metrics.py` (`PyIQAMetric`, `MetricRegistry`, factories, built-in specs, `BUILTIN_METRICS`, `SEGMENTATION_METRICS`) is unchanged. Update the module docstring's last paragraph to mention that the description types live in `metric_spec.py`.

- [ ] **Step 5: Point the metric modules at the leaf**

- `dreamsim_metric.py:30` → `from iqaevaluator.metric_spec import DEVICE, MetricSpec, ModeSupport, ModeUnsupported, REASON_DEEP_2D`
- `segmentation_metrics/monai_metrics.py:54` → `from iqaevaluator.metric_spec import MetricSpec, ModeSupport, Spacing`
- `segmentation_metrics/boundary_iou.py:49` → `from iqaevaluator.metric_spec import MetricSpec, ModeSupport`; delete the "Usage: import `metrics` before this module …" paragraph and its code example (lines 34–39) from the module docstring.
- `segmentation_metrics/volume_metrics.py:31` → `from iqaevaluator.metric_spec import MetricSpec, ModeSupport`; delete the two-line "Usage: import `metrics` before this module" note (lines 22–23) and, in the first paragraph, the words "which also keeps it out of the circular-import dance the other segmentation modules need".
- `image_loader.py:23`: replace `Spacing = tuple[float, float, float]` with `from iqaevaluator.metric_spec import Spacing` (placed with the other package imports).
- `tests/test_dreamsim_metric.py:12-16`: delete the five comment lines about import order (keep `from iqaevaluator.metrics import DEVICE`).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_import_order.py tests/test_metrics.py tests/test_dreamsim_metric.py tests/test_segmentation_metrics.py tests/test_boundary_iou.py tests/test_volume_metrics.py tests/test_image_geometry.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/iqaevaluator/metric_spec.py src/iqaevaluator/metrics.py src/iqaevaluator/dreamsim_metric.py src/iqaevaluator/segmentation_metrics/monai_metrics.py src/iqaevaluator/segmentation_metrics/boundary_iou.py src/iqaevaluator/segmentation_metrics/volume_metrics.py src/iqaevaluator/image_loader.py tests/test_dreamsim_metric.py tests/test_import_order.py
git commit -m "refactor(metrics): move spec types to a leaf module, end the import cycle

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Shared adapter helpers — `require_binary`, `foreground_counts`, `empty_policy` (spec Part 2, D3)

**Files:**
- Modify: `src/iqaevaluator/segmentation_metrics/volume.py` (append after `as_mask`)
- Test: `tests/test_volume.py`

**Interfaces:**
- Produces:
  - `require_binary(t: torch.Tensor, *, metric: str) -> None` — raises `ValueError` unless every value is 0 or 1; message: `f"{metric} expects a binary mask with values in {{0, 1}}; load masks with normalization.Mask()"`.
  - `foreground_counts(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` — per-sample non-zero counts over every axis but the first.
  - `NOT_EMPTY` sentinel; `empty_policy(n_pred: int, n_gt: int, *, one_sided: Optional[float]) -> Optional[float] | object` — both 0 → `None`; exactly one 0 → `one_sided`; else `NOT_EMPTY`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_volume.py` (imports gain `torch` and the three helpers plus `NOT_EMPTY`):
```python
class TestAdapterHelpers:
    def test_require_binary_accepts_zero_one_floats_and_ints(self):
        require_binary(torch.tensor([[0.0, 1.0]]), metric="dice")
        require_binary(torch.tensor([[0, 1]], dtype=torch.int16), metric="dice")

    def test_require_binary_rejects_anything_else_and_names_mask(self):
        with pytest.raises(ValueError, match=r"dice expects a binary mask.*Mask\(\)"):
            require_binary(torch.tensor([[0.0, 0.5, 1.0]]), metric="dice")
        with pytest.raises(ValueError, match=r"Mask\(\)"):
            require_binary(torch.tensor([[0, 1, 2]], dtype=torch.int16), metric="dice")

    def test_foreground_counts_are_per_sample(self):
        y_pred = torch.zeros(2, 1, 4, 4); y_pred[0, 0, :2] = 1.0
        y = torch.zeros(2, 1, 4, 4); y[1, 0, 1, 1] = 1.0
        n_pred, n_gt = foreground_counts(y_pred, y)
        assert n_pred.tolist() == [8, 0] and n_gt.tolist() == [0, 1]

    def test_foreground_counts_handle_5d_volumes(self):
        vol = torch.ones(1, 1, 2, 3, 3)
        assert foreground_counts(vol, vol)[0].tolist() == [18]

    @pytest.mark.parametrize("n_pred, n_gt, one_sided, expected", [
        (0, 0, 0.0, None), (0, 0, None, None),
        (5, 0, 0.0, 0.0), (0, 5, 0.0, 0.0),
        (5, 0, None, None), (0, 5, None, None),
    ])
    def test_empty_policy_table(self, n_pred, n_gt, one_sided, expected):
        assert empty_policy(n_pred, n_gt, one_sided=one_sided) is expected

    def test_empty_policy_leaves_populated_pairs_alone(self):
        assert empty_policy(3, 4, one_sided=0.0) is NOT_EMPTY
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `hatch run calibration:test tests/test_volume.py -q`
Expected: FAIL with `ImportError: cannot import name 'require_binary'`.

- [ ] **Step 3: Implement**

Append to `src/iqaevaluator/segmentation_metrics/volume.py` (add `import torch` to the imports; the module keeps no framework imports):
```python
# ---------------------------------------------------------------------------
# Shared helpers for the metric adapters (monai_metrics, volume_metrics,
# boundary_iou). Every adapter validates its input and applies the empty-mask
# policy the same way; the numbers below never come from a metric backend.
# ---------------------------------------------------------------------------

NOT_EMPTY = object()
"""`empty_policy`'s answer when both masks have foreground: compute the metric."""


def require_binary(t: torch.Tensor, *, metric: str) -> None:
    """Raise unless every value of `t` is 0 or 1 — the contract `Mask()` fulfils.

    A probability map loaded with `MinMax()`, or a label map loaded with
    `Raw()`, would otherwise be scored on whatever the backend makes of it.
    """
    if not bool(((t == 0) | (t == 1)).all()):
        raise ValueError(
            f"{metric} expects a binary mask with values in {{0, 1}}; "
            "load masks with normalization.Mask()"
        )


def foreground_counts(y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample foreground voxel counts of `(N, C, *spatial)` batches."""
    dims = tuple(range(1, y_pred.dim()))
    return (y_pred != 0).sum(dim=dims), (y != 0).sum(dim=dims)


def empty_policy(n_pred: int, n_gt: int, *, one_sided: Optional[float]):
    """The empty-mask policy, one place for every metric.

    Both masks empty: the score is undefined → `None`. Exactly one empty:
    the metric's `one_sided` value — 0.0 for overlap-type scores (Dice, VS,
    NSD, PQ, Boundary IoU), `None` for distances (HD95, ASSD), which have no
    finite value there. Otherwise `NOT_EMPTY`: compute the metric.
    """
    if n_pred == 0 and n_gt == 0:
        return None
    if n_pred == 0 or n_gt == 0:
        return one_sided
    return NOT_EMPTY
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_volume.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/iqaevaluator/segmentation_metrics/volume.py tests/test_volume.py
git commit -m "feat(segmentation): shared binary check and empty-mask policy helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 7: MONAI adapters — one builder, binary contract, empty policy, no `threshold` (spec Part 2, D3, D5, review 3.3/3.7/4.1)

**Files:**
- Rewrite: `src/iqaevaluator/segmentation_metrics/monai_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Produces:
  - `MonaiSegmentationMetric(compute_fn, *, one_sided: Optional[float], name: Optional[str] = None, **monai_kwargs)`; `__call__(input, target) -> list[Optional[float]]`.
  - `MonaiPanopticQualityMetric(**monai_kwargs)`.
  - `_monai_spec(name, compute_fn, *, direction, one_sided, uses_spacing, description, defaults: dict, **monai_kwargs) -> MetricSpec`.
  - Builders `dice_metric(**monai_kwargs)`, `hausdorff95_metric(**monai_kwargs)`, `normalized_surface_dice_metric(**monai_kwargs)`, `average_surface_distance_metric(**monai_kwargs)`, `panoptic_quality_metric(**monai_kwargs)`; constants `DICE`, `HAUSDORFF95`, `NSD`, `ASSD`, `PANOPTIC_QUALITY` unchanged in name. Passing `threshold=` or `label=` to any of them raises `TypeError`.
- Consumes: `require_binary`, `foreground_counts`, `empty_policy`, `NOT_EMPTY` (Task 6); `metric_spec` (Task 5).

- [ ] **Step 1: Rewrite the adapter tests**

In `tests/test_segmentation_metrics.py`:

1. Every `MonaiSegmentationMetric(compute_dice, include_background=True)` gains `one_sided=0.0` (three call sites: `test_call_returns_one_score_per_sample`, `test_identical_masks_score_perfect_dice`, `TestIntegerMasks.test_identical_integer_masks_score_perfect_dice`).
2. Delete `test_threshold_binarizes_before_computing` and `TestIntegerMasks.test_threshold_zero_makes_every_label_foreground`.
3. Replace class `TestRawLabelMapEndToEnd` with the class below, and add the other new tests:

```python
class TestMonaiSegmentationMetricContract:
    def test_non_binary_input_raises_naming_mask(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, name="dice", include_background=True)
        soft = torch.full((1, 1, 8, 8), 0.7)
        with pytest.raises(ValueError, match=r"dice expects a binary mask.*Mask\(\)"):
            metric(soft, torch.ones_like(soft))

    def test_stale_threshold_kwarg_raises_type_error(self):
        from monai.metrics import compute_dice
        with pytest.raises(TypeError, match="threshold"):
            MonaiSegmentationMetric(compute_dice, one_sided=0.0, threshold=0.5)

    def test_missing_target_raises(self):
        from monai.metrics import compute_dice
        with pytest.raises(ValueError, match="target"):
            MonaiSegmentationMetric(compute_dice, one_sided=0.0)(torch.ones(1, 1, 4, 4))

    def test_one_sided_empty_uses_the_policy_value(self):
        from monai.metrics import compute_dice, compute_hausdorff_distance
        pred = torch.zeros(2, 1, 8, 8); pred[0, 0, 2:5, 2:5] = 1.0     # sample 0: gt empty; sample 1: both populated
        gt = torch.zeros(2, 1, 8, 8); gt[1, 0, 2:5, 2:5] = 1.0; pred[1, 0, 2:5, 2:5] = 1.0
        dice = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True, ignore_empty=False)
        hd = MonaiSegmentationMetric(compute_hausdorff_distance, one_sided=None, include_background=True, percentile=95)
        assert dice(pred, gt) == [0.0, pytest.approx(1.0)]
        assert hd(pred, gt) == [None, pytest.approx(0.0)]

    def test_both_empty_is_none_for_every_one_sided_value(self):
        from monai.metrics import compute_dice
        zeros = torch.zeros(1, 1, 8, 8)
        assert MonaiSegmentationMetric(compute_dice, one_sided=0.0, ignore_empty=False)(zeros, zeros) == [None]
        assert MonaiSegmentationMetric(compute_dice, one_sided=None, ignore_empty=False)(zeros, zeros) == [None]

    def test_nonfinite_backend_output_becomes_none_with_a_warning(self, capsys):
        def broken(y_pred, y):
            return torch.full((y_pred.shape[0], 1), float("inf"))
        metric = MonaiSegmentationMetric(broken, one_sided=0.0, name="broken")
        ones = torch.ones(1, 1, 4, 4)
        assert metric(ones, ones) == [None]
        assert "broken" in capsys.readouterr().out

    def test_batch_mixes_policy_and_computed_samples_in_order(self):
        from monai.metrics import compute_dice
        pred = torch.zeros(3, 1, 8, 8); gt = torch.zeros(3, 1, 8, 8)
        pred[0, 0, :4] = 1.0; gt[0, 0, :4] = 1.0        # identical → 1.0
        gt[1, 0, :4] = 1.0                              # pred empty → 0.0
        pred[2, 0, :4] = 1.0; gt[2, 0, 2:6] = 1.0       # half overlap → 0.5
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True, ignore_empty=False)
        assert metric(pred, gt) == [pytest.approx(1.0), 0.0, pytest.approx(0.5)]


@pytest.mark.parametrize("builder", [
    dice_metric, hausdorff95_metric, normalized_surface_dice_metric,
    average_surface_distance_metric, panoptic_quality_metric,
])
@pytest.mark.parametrize("stale", ["threshold", "label"])
def test_builders_reject_the_removed_knobs_loudly(builder, stale):
    with pytest.raises(TypeError, match=stale):
        builder(**{stale: 0.5})


def test_dice_is_built_without_monais_empty_skipping():
    """ignore_empty=False: the policy decides, MONAI must not answer NaN for an empty gt."""
    pred = torch.zeros(1, 1, 8, 8); pred[0, 0, :2] = 1.0
    assert DICE.slice_mode.factory()(pred, torch.zeros_like(pred)) == [0.0]


class TestIntegerMasks:
    def test_identical_integer_masks_score_perfect_dice(self):
        g = torch.Generator().manual_seed(0)
        mask = torch.randint(0, 2, (2, 1, 16, 16), generator=g)          # int64
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True)
        assert metric(mask, mask.clone()) == pytest.approx([1.0, 1.0])

    def test_multi_label_integer_map_is_rejected(self):
        labels = torch.zeros((1, 1, 8, 8), dtype=torch.int16)
        labels[0, 0, :4] = 1
        labels[0, 0, 4:] = 3
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, name="dice", include_background=True)
        with pytest.raises(ValueError, match=r"Mask\(\)"):
            metric(labels, torch.ones_like(labels))


class TestMaskLabelMapEndToEnd:
    """A real integer NIfTI label map through Mask() and the adapter."""

    def _label_map(self, tmp_path):
        import nibabel as nib
        labels = np.zeros((10, 4, 1), dtype=np.int16)
        labels[2:4, :, 0] = 1
        labels[4:7, :, 0] = 2
        labels[7:10, :, 0] = 3
        p = tmp_path / "multi_label.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        return p

    def test_any_nonzero_label_is_foreground_under_mask(self, tmp_path):
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Mask
        pred = ImageLoader(self._label_map(tmp_path), Mask()).tensor  # (1, 1, 10, 4) float {0, 1}
        gt = torch.zeros((1, 1, 10, 4)); gt[:, :, 2:10, :] = 1.0
        assert DICE.slice_mode.factory()(pred, gt) == [pytest.approx(1.0)]

    def test_one_label_under_mask_label(self, tmp_path):
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Mask
        pred = ImageLoader(self._label_map(tmp_path), Mask(label=2)).tensor
        gt = torch.zeros((1, 1, 10, 4)); gt[:, :, 4:7, :] = 1.0
        assert DICE.slice_mode.factory()(pred, gt) == [pytest.approx(1.0)]

    def test_raw_label_map_is_rejected_with_a_pointer_to_mask(self, tmp_path):
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Raw
        pred = ImageLoader(self._label_map(tmp_path), Raw()).tensor      # int16 {0,1,2,3}
        with pytest.raises(ValueError, match=r"Mask\(\)"):
            DICE.slice_mode.factory()(pred, torch.ones_like(pred))
```
Extend `TestMonaiPanopticQualityMetric`:
```python
    def test_rejects_non_integer_values(self):
        metric = MonaiPanopticQualityMetric()
        soft = torch.full((1, 1, 8, 8), 0.5)
        with pytest.raises(ValueError, match=r"Mask\(\).*Raw\(\)"):
            metric(soft, soft)

    def test_accepts_raw_instance_maps(self):
        inst = torch.zeros((1, 1, 8, 8), dtype=torch.int16)
        inst[0, 0, :3, :3] = 1
        inst[0, 0, 5:, 5:] = 2
        assert MonaiPanopticQualityMetric()(inst, inst.clone()) == [pytest.approx(1.0, abs=1e-5)]

    def test_empty_policy_applies(self):
        zeros = torch.zeros(1, 1, 8, 8)
        ones = torch.ones(1, 1, 8, 8)
        assert MonaiPanopticQualityMetric()(zeros, zeros) == [None]
        assert MonaiPanopticQualityMetric()(ones, zeros) == [0.0]

    def test_stale_threshold_raises(self):
        with pytest.raises(TypeError, match="threshold"):
            MonaiPanopticQualityMetric(threshold=0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `hatch run calibration:test tests/test_segmentation_metrics.py -q`
Expected: FAIL — `TypeError: unexpected keyword argument 'one_sided'` and friends.

- [ ] **Step 3: Rewrite `monai_metrics.py`**

Replace the whole file with:
```python
"""MONAI-backed segmentation-quality metrics for the IQA metric registry.

These metrics evaluate segmentation masks (pred vs. ground-truth label maps),
not image intensity — they answer "how good is this segmentation?" rather
than "how good is this reconstructed image?". This module performs no
segmentation itself.

Input must be binary — load masks with `normalization.Mask()`
(`ImageLoader(path, Mask())` or `load_pair(..., Mask())`). `Mask()` turns
0/255 PNGs, 0/1 NIfTIs and integer label maps (`Mask(label=k)` for one class)
into float32 {0, 1}; every adapter here checks that and rejects anything
else. The one exception is panoptic quality, which also accepts an integer
instance map loaded with `Raw()`.

Empty masks follow one policy (`volume.empty_policy`): with exactly one side
empty, Dice/NSD/PQ score 0.0 and HD95/ASSD are `None` (no finite distance
exists); with both sides empty every score is `None`. MONAI's own answers for
those cases (NaN, inf, or Dice 1.0) never reach a record.

MONAI's defaults are calibrated for the medical-imaging domain (physical voxel
spacing in millimetres, a background-class convention). Domain parameters are
forwarded unchanged via `**monai_kwargs`; each builder's docstring states
which ones matter in another domain (materials science: different physical
units via `spacing`, different class counts via `class_thresholds`).

Usage: each builder (`dice_metric()`, `hausdorff95_metric()`, ...) returns a
`MetricSpec` — pass one or more into `MetricRegistry(*specs)` and hand that
registry to `IQAEvaluator`/`VolumeEvaluator`:

    from iqaevaluator.metrics import MetricRegistry
    from iqaevaluator.segmentation_metrics.monai_metrics import DICE, HAUSDORFF95

    registry = MetricRegistry(DICE, HAUSDORFF95)

Use the pre-built constants (`DICE`, `HAUSDORFF95`, `NSD`, `ASSD`,
`PANOPTIC_QUALITY`) for defaults, or call a builder with MONAI keyword
arguments (e.g. `hausdorff95_metric(percentile=None)`) before registering.
"""

import math
from typing import Callable, Optional

import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_panoptic_quality,
    compute_surface_dice,
)

from iqaevaluator.metric_spec import MetricSpec, ModeSupport, Spacing
from iqaevaluator.segmentation_metrics.volume import (
    NOT_EMPTY, empty_policy, foreground_counts, require_binary,
)

DOMAIN_MEDICAL = "medical (MONAI)"

_REMOVED_KNOBS = ("threshold", "label")


def _reject_removed_knobs(name: str, kwargs: dict) -> None:
    """`threshold`/`label` used to be adapter parameters. A stale caller must
    fail here rather than have the keyword forwarded to MONAI (or ignored)."""
    for key in _REMOVED_KNOBS:
        if key in kwargs:
            raise TypeError(
                f"{name} no longer takes '{key}': binarization is decided by the "
                "loader — load masks with normalization.Mask() (Mask(label=k) "
                "selects one class)."
            )


class MonaiSegmentationMetric:
    """Adapter for MONAI's one-hot-batch functional metrics (Dice, HD95, NSD, ASSD).

    All four share the call signature `compute_fn(y_pred, y, **kwargs) -> (N, C)`
    tensor (batch x class channels); this adapter averages across the class
    dimension (NaN-aware) to produce one score per sample, matching the
    Metric protocol.

    Both tensors must be binary (`require_binary`); the empty-mask policy is
    applied per sample before MONAI sees anything, with `one_sided` as the
    score for exactly-one-side-empty (0.0 for Dice/NSD, None for HD95/ASSD).
    A NaN or inf that MONAI still returns for a populated pair is recorded as
    None and printed as a warning — an unexpected case must not be silent.

    In volume mode the adapter receives a single `(1, C, D, H, W)` sample and
    returns a single score; the distance metrics additionally receive
    `spacing` so their result is in millimetres rather than voxels.
    """

    def __init__(
        self,
        compute_fn: Callable[..., torch.Tensor],
        *,
        one_sided: Optional[float],
        name: Optional[str] = None,
        **monai_kwargs,
    ):
        self._name = name or compute_fn.__name__
        _reject_removed_knobs(self._name, monai_kwargs)
        self._compute_fn = compute_fn
        self._one_sided  = one_sided
        self._kwargs     = monai_kwargs

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]:
        if target is None:
            raise ValueError(f"'{self._name}' compares two masks and requires a target mask")
        require_binary(input, metric=self._name)
        require_binary(target, metric=self._name)
        # MONAI 1.6.0 happens to accept integer input; that is an implementation
        # detail of a third-party library, so the adapter guarantees floats itself.
        y_pred, y = input.float(), target.float()

        n_pred, n_gt = foreground_counts(y_pred, y)
        results: list[Optional[float]] = [None] * y_pred.shape[0]
        active: list[int] = []
        for i in range(y_pred.shape[0]):
            verdict = empty_policy(int(n_pred[i]), int(n_gt[i]), one_sided=self._one_sided)
            if verdict is NOT_EMPTY:
                active.append(i)
            else:
                results[i] = verdict

        if active:
            scores = self._compute_fn(y_pred=y_pred[active], y=y[active], **self._kwargs)  # (n, C)
            per_sample = torch.nanmean(scores.float(), dim=1)
            for i, score in zip(active, per_sample):
                value = float(score.item())
                if not math.isfinite(value):
                    print(
                        f"[WARNING] {self._name}: MONAI returned {value} for sample {i} "
                        "although both masks have foreground; recorded as None."
                    )
                    results[i] = None
                else:
                    results[i] = value
        return results


def _volume_factory(
    compute_fn: Callable[..., torch.Tensor],
    *,
    one_sided: Optional[float],
    name: str,
    uses_spacing: bool,
    **monai_kwargs,
) -> Callable[[Optional[Spacing]], MonaiSegmentationMetric]:
    """Build the volume-mode factory for one MONAI functional metric.

    `uses_spacing` marks the distance metrics (HD95, NSD, ASSD), which convert
    voxel counts to physical units. An explicit `spacing` passed to the builder
    always wins: the user pinned it deliberately, and silently replacing it with
    whatever the current file happens to say would be worse than ignoring the
    file.
    """
    def build(spacing: Optional[Spacing]) -> MonaiSegmentationMetric:
        kwargs = dict(monai_kwargs)
        if uses_spacing:
            if spacing is not None:
                kwargs.setdefault("spacing", list(spacing))
            elif "spacing" not in kwargs:
                print(
                    "[WARNING] no voxel size available, so this distance is "
                    "counted in voxels rather than millimetres. Values are "
                    "comparable between images on the same grid, but not "
                    "between images recorded at different resolutions."
                )
        return MonaiSegmentationMetric(compute_fn, one_sided=one_sided, name=name, **kwargs)

    return build


def _monai_spec(
    name: str,
    compute_fn: Callable[..., torch.Tensor],
    *,
    direction: str,
    one_sided: Optional[float],
    uses_spacing: bool,
    description: str,
    defaults: dict,
    **monai_kwargs,
) -> MetricSpec:
    """One MetricSpec for one MONAI functional; `defaults` are the domain
    defaults a caller's `monai_kwargs` may override."""
    _reject_removed_knobs(f"{name}_metric()", monai_kwargs)
    kwargs = {**defaults, **monai_kwargs}
    metric = MonaiSegmentationMetric(compute_fn, one_sided=one_sided, name=name, **kwargs)
    return MetricSpec(
        name=name,
        direction=direction,
        reference=True,
        channels="gray",
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(_volume_factory(
            compute_fn, one_sided=one_sided, name=name, uses_spacing=uses_spacing, **kwargs)),
        builtin=False,
        description=description,
        domain=DOMAIN_MEDICAL,
    )


def dice_metric(**monai_kwargs) -> MetricSpec:
    """Dice similarity coefficient: 2*|pred ∩ gt| / (|pred| + |gt|), 1.0 = perfect overlap.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI).
    Defaults assume a single foreground mask channel (include_background=True,
    since there is nothing else to include) and `ignore_empty=False`, so an
    empty reference with a non-empty prediction scores 0.0 (the empty-mask
    policy decides these cases anyway).

    Args:
        include_background (kwarg, default True): if False, drops channel 0
            (assumed background class) before scoring — only meaningful for
            multi-channel one-hot input.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_dice`.
    """
    return _monai_spec(
        "dice", compute_dice, direction="higher_is_better", one_sided=0.0, uses_spacing=False,
        defaults={"include_background": True, "ignore_empty": False},
        description=(
            "Dice similarity coefficient: overlap between predicted and "
            "ground-truth segmentation masks (1.0 = perfect overlap, 0.0 = no "
            "overlap). Domain: medical (MONAI). For other domains (e.g. "
            "materials-science multi-phase segmentation), pass a multi-channel "
            "one-hot mask and set include_background=False to exclude a real "
            "background class."
        ),
        **monai_kwargs,
    )


DICE = dice_metric()


def hausdorff95_metric(**monai_kwargs) -> MetricSpec:
    """95th-percentile Hausdorff Distance: worst-case boundary error, robust to outlier voxels.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI) —
    the returned distance is in voxel units unless `spacing` is supplied
    (physical units per voxel, e.g. mm for medical scans or µm for materials
    micrographs). Definition: max over both directions of the 95th percentile
    of surface-voxel distances (Taha & Hanbury 2015); with exactly one empty
    mask the score is None.

    Args:
        include_background (kwarg, default True): if False, drops channel 0
            (assumed background class) before scoring.
        percentile (kwarg, default 95): which percentile of boundary-point
            distances to report; None gives the plain (max) Hausdorff distance.
        directed (kwarg, default False): True measures pred→gt only.
        spacing (kwarg, default None): physical size of one voxel (scalar or
            per-axis list); converts the voxel-unit distance to real units.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_hausdorff_distance`.
    """
    return _monai_spec(
        "hausdorff95", compute_hausdorff_distance, direction="lower_is_better", one_sided=None,
        uses_spacing=True, defaults={"include_background": True, "percentile": 95},
        description=(
            "95th-percentile Hausdorff Distance: how far the predicted "
            "segmentation boundary is from the ground-truth boundary in the "
            "worst 5% of cases (lower = closer boundaries). Domain: medical "
            "(MONAI). Distance is in voxel units by default; pass "
            "`spacing=<mm-per-voxel or per-axis list>` for physical units, or "
            "the equivalent voxel size for another domain (e.g. µm for "
            "materials micrographs)."
        ),
        **monai_kwargs,
    )


HAUSDORFF95 = hausdorff95_metric()


def normalized_surface_dice_metric(**monai_kwargs) -> MetricSpec:
    """Normalized Surface Dice (NSD): fraction of the predicted/gt boundary within a tolerance distance.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI).
    `class_thresholds` is the tolerance distance per class (defaults to
    `[1.0]`) — MONAI requires it and treats it in the same units as `spacing`.

    The scoring mode therefore changes what this metric measures, not just its
    scale. Slice mode passes no spacing, so the default tolerance means one
    voxel; volume mode passes the image's voxel size, so the same default means
    one millimetre. On 0.5 mm data volume mode is twice as forgiving as slice
    mode, and on 2 mm data half as forgiving. Pass `class_thresholds`
    explicitly if the two modes have to be read on one scale, and do not
    compare an `nsd` column from a slice run against one from a volume run.

    Args:
        include_background (kwarg, default True): if False, drops channel 0.
        class_thresholds (kwarg, default [1.0]): per-class tolerance distance,
            one entry per class channel, in the units of `spacing`.
        spacing (kwarg, default None): physical size of one voxel.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_surface_dice`.
    """
    return _monai_spec(
        "nsd", compute_surface_dice, direction="higher_is_better", one_sided=0.0,
        uses_spacing=True, defaults={"include_background": True, "class_thresholds": [1.0]},
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
        **monai_kwargs,
    )


NSD = normalized_surface_dice_metric()


def average_surface_distance_metric(**monai_kwargs) -> MetricSpec:
    """Average Symmetric Surface Distance (ASSD): mean boundary distance in both directions.

    Input must be binary — load masks with `Mask()`. Domain: medical (MONAI).
    Like HD95, the result is in voxel units unless `spacing` is supplied, and
    with exactly one empty mask the score is None.

    Args:
        include_background (kwarg, default True): if False, drops channel 0.
        symmetric (kwarg, default True): mean over the pred→gt and gt→pred
            surface distances pooled (True) vs. pred→gt only (False).
        spacing (kwarg, default None): physical size of one voxel.
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_average_surface_distance`.
    """
    return _monai_spec(
        "assd", compute_average_surface_distance, direction="lower_is_better", one_sided=None,
        uses_spacing=True, defaults={"include_background": True, "symmetric": True},
        description=(
            "Average Symmetric Surface Distance: mean distance between the "
            "predicted and ground-truth boundaries, averaged in both "
            "directions (lower = closer boundaries on average). Domain: "
            "medical (MONAI). Distance is in voxel units by default; pass "
            "`spacing=<mm-per-voxel or per-axis list>` for physical units, or "
            "the equivalent voxel size for another domain (e.g. µm for "
            "materials micrographs)."
        ),
        **monai_kwargs,
    )


ASSD = average_surface_distance_metric()


class MonaiPanopticQualityMetric:
    """Adapter for MONAI's compute_panoptic_quality, which takes one integer
    instance-label map per image (no batch/channel dim, unlike the other four
    metrics) — this loops over the batch itself.

    Accepts integer-valued tensors only: `{0, 1}` from `Mask()` (a binary mask
    is scored as single-instance PQ, foreground = one instance) or an instance
    map with one integer id per object loaded with `Raw()`. The empty-mask
    policy is Dice's: one side empty → 0.0, both empty → None.
    """

    def __init__(self, **monai_kwargs):
        _reject_removed_knobs("panoptic_quality", monai_kwargs)
        self._kwargs = monai_kwargs

    @staticmethod
    def _require_integer_valued(t: torch.Tensor) -> None:
        f = t.float()
        if not bool((f == f.round()).all()):
            raise ValueError(
                "panoptic_quality expects integer-valued instance maps: load binary "
                "masks with normalization.Mask() and instance maps with normalization.Raw()"
            )

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[Optional[float]]:
        if target is None:
            raise ValueError("'panoptic_quality' compares two masks and requires a target mask")
        self._require_integer_valued(input)
        self._require_integer_valued(target)
        scores: list[Optional[float]] = []
        for i in range(input.shape[0]):
            pred_map = input[i, 0].long()
            gt_map   = target[i, 0].long()
            verdict = empty_policy(int((pred_map != 0).sum()), int((gt_map != 0).sum()), one_sided=0.0)
            if verdict is not NOT_EMPTY:
                scores.append(verdict)
                continue
            value = float(compute_panoptic_quality(pred_map, gt_map, **self._kwargs).item())
            if not math.isfinite(value):
                print(
                    f"[WARNING] panoptic_quality: MONAI returned {value} for sample {i} "
                    "although both maps have foreground; recorded as None."
                )
                scores.append(None)
            else:
                scores.append(value)
        return scores


def panoptic_quality_metric(**monai_kwargs) -> MetricSpec:
    """Panoptic Quality (PQ): detection accuracy (instances matched by IoU) times
    segmentation accuracy (mean IoU of matched instances), Kirillov et al. 2019.

    Binary masks from `Mask()` degenerate to a single-instance IoU-based score;
    for true multi-instance PQ load integer instance maps with `Raw()`. Domain:
    medical (MONAI), designed for instance segmentation (cells, lesions). For
    another domain with multiple distinct objects (grains in a micrograph),
    supply one integer id per instance and tune `match_iou_threshold`. Note
    that MONAI adds `smooth_numerator=1e-6` to the denominator.

    Args:
        match_iou_threshold (kwarg, default 0.5): minimum IoU for a predicted
            instance to count as matched; unmatched instances count against
            the score.
        metric_name (kwarg, default "pq"): "pq", "sq" or "rq".
        **monai_kwargs: forwarded verbatim to `monai.metrics.compute_panoptic_quality`.
    """
    _reject_removed_knobs("panoptic_quality_metric()", monai_kwargs)
    kwargs = {"match_iou_threshold": 0.5, **monai_kwargs}
    metric = MonaiPanopticQualityMetric(**kwargs)
    return MetricSpec(
        name="panoptic_quality",
        direction="higher_is_better",
        reference=True,
        channels="gray",
        slice_mode=ModeSupport(lambda: metric),
        volume_mode=ModeSupport(lambda spacing: MonaiPanopticQualityMetric(**kwargs)),
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_segmentation_metrics.py tests/test_metrics.py tests/test_main.py -q`
Expected: all pass. (`test_main.py` still passes because `SEGMENTATION_METRICS` is unchanged in shape.)

- [ ] **Step 5: Commit**

```bash
git add src/iqaevaluator/segmentation_metrics/monai_metrics.py tests/test_segmentation_metrics.py
git commit -m "refactor(segmentation): one MONAI builder, binary contract, empty policy

threshold/label are gone from the adapters (review 3.3, 3.6); inf/NaN
never leave the adapter (3.7); the four copy-paste builders collapse into
_monai_spec (4.1).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 8: `VolumeFunctionMetric` and `BoundaryIoUMetric` without `threshold`; physical band floor `max(spacing)` (spec Part 2, D4, D5, review 3.9)

**Files:**
- Modify: `src/iqaevaluator/segmentation_metrics/volume_metrics.py:39-141`
- Modify: `src/iqaevaluator/segmentation_metrics/boundary_iou.py:55-86` (`band_width`), `:140-192` (`boundary_iou` signature), `:195-254` (adapter), `:257-315` (builder), module docstring
- Test: `tests/test_volume_metrics.py`, `tests/test_boundary_iou.py`

**Interfaces:**
- Produces: `VolumeFunctionMetric(fn)`; `vs_metric()`, `vs_signed_metric()`, `v_pred_metric()`, `v_gt_metric()`, `tp_metric()` take no arguments. `BoundaryIoUMetric(*, dilation_ratio=DEFAULT_DILATION_RATIO, spacing=None)`; `boundary_iou_metric(*, dilation_ratio=DEFAULT_DILATION_RATIO)`; numpy `boundary_iou(pred, gt, *, dilation_ratio, label=None, threshold=0.5, spacing=None)`; `band_width(shape, dilation_ratio, spacing)` returns `max(max(spacing), ratio·‖shape·spacing‖)` when `spacing` is given.
- Consumes: `require_binary` (Task 6).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_volume_metrics.py`:
```python
class TestBinaryContract:
    def test_non_binary_input_raises_naming_mask(self):
        half = torch.full((1, 1, 8, 8), 0.5)
        with pytest.raises(ValueError, match=r"vs expects a binary mask.*Mask\(\)"):
            MetricRegistry(VS).get_metric("vs")(half, half)

    def test_removed_threshold_keyword_fails_loudly(self):
        from iqaevaluator.segmentation_metrics.volume_metrics import VolumeFunctionMetric, vs_metric
        from iqaevaluator.segmentation_metrics.volume import vs
        with pytest.raises(TypeError):
            vs_metric(threshold=0.5)
        with pytest.raises(TypeError):
            VolumeFunctionMetric(vs, threshold=0.5)

    def test_integer_zero_one_masks_are_accepted(self):
        pred, gt = _pair_4d()
        assert MetricRegistry(VS).get_metric("vs")(pred.to(torch.int16), gt.to(torch.int16))[0] == pytest.approx(1.0)
```
In `tests/test_boundary_iou.py` replace `TestBoundaryIoUMetricAdapter.test_threshold_binarizes_soft_masks` with:
```python
    def test_non_binary_input_raises_naming_mask(self):
        soft = np.where(_square(160, 120), 0.9, 0.1)
        with pytest.raises(ValueError, match=r"boundary_iou expects a binary mask.*Mask\(\)"):
            BoundaryIoUMetric()(_batch([soft]), _batch([soft]))

    def test_removed_threshold_keyword_fails_loudly(self):
        with pytest.raises(TypeError):
            BoundaryIoUMetric(threshold=0.5)
        with pytest.raises(TypeError):
            boundary_iou_metric(threshold=0.5)
```
and append to `TestBandWidth`:
```python
    def test_physical_floor_is_one_voxel_along_the_coarsest_axis(self):
        # 0.02 * |(24, 40, 40)| = 1.229 mm would be thinner than the 3 mm slice
        # spacing, so no face perpendicular to D could ever be in the band.
        assert band_width((8, 40, 40), 0.02, (3.0, 1.0, 1.0)) == 3.0

    def test_physical_floor_does_not_touch_large_volumes(self):
        assert band_width((130, 256, 256), 0.02, (1.2, 1.0, 1.0)) == pytest.approx(7.884364, abs=1e-6)

    def test_physical_floor_uses_the_largest_spacing_whatever_its_axis(self):
        assert band_width((8, 40, 40), 0.02, (1.0, 1.0, 3.0)) == 3.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `hatch run calibration:test tests/test_volume_metrics.py tests/test_boundary_iou.py -q`
Expected: FAIL — `vs_metric(threshold=0.5)` does not raise, the soft batch scores instead of raising, `band_width(...) == 1.229…`.

- [ ] **Step 3: Implement `volume_metrics.py`**

Replace `VolumeFunctionMetric`, `_both_modes` and the five builders with:
```python
class VolumeFunctionMetric:
    """Adapter turning a `volume.py` function into a Metric.

    The wrapped functions take two NumPy arrays of any matching shape, so the
    same adapter serves both scoring modes: in slice mode it sees `(N, C, H, W)`
    and scores each sample's channel 0; in volume mode it sees `(1, C, D, H, W)`
    and scores the whole `(D, H, W)` body at once.

    Input must be binary — load masks with `normalization.Mask()`. The numpy
    function's own `as_mask` is then a no-op.

    Args:
        fn: one of `vs`, `vs_signed`, `v_pred`, `v_gt`, `tp`.
    """

    def __init__(self, fn: Callable[..., float]):
        self._fn = fn

    def __call__(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> list[Optional[float]]:
        if target is None:
            raise ValueError(
                f"'{self._fn.__name__}' compares two masks and requires a target mask"
            )
        require_binary(input, metric=self._fn.__name__)
        require_binary(target, metric=self._fn.__name__)
        pred = input.detach().cpu().numpy()
        gt   = target.detach().cpu().numpy()
        scores: list[Optional[float]] = []
        for i in range(pred.shape[0]):
            value = self._fn(pred[i, 0], gt[i, 0])
            scores.append(None if np.isnan(value) else float(value))
        return scores


def _both_modes(fn: Callable[..., float]) -> tuple[ModeSupport, ModeSupport]:
    """Slice and volume capability for a function that is already shape-agnostic."""
    metric = VolumeFunctionMetric(fn)
    return ModeSupport(lambda: metric), ModeSupport(lambda spacing: metric)


def vs_metric() -> MetricSpec:
    """Volumetric Similarity (Taha & Hanbury): 1 - |Vp - Vg| / (Vp + Vg).

    Range [0, 1]; 1.0 means the two masks have the same size. Spacing-invariant:
    the voxel volume cancels in numerator and denominator. One empty mask
    scores 0.0, both empty is undefined (None).
    """
    slice_mode, volume_mode = _both_modes(_vs)
    return MetricSpec(
        name="vs", direction="higher_is_better", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            "Volumetric Similarity: how closely the predicted and reference "
            "masks agree in size (1.0 = same size). It is not an overlap "
            "measure — two masks of equal size that never touch also score 1.0 "
            "— so read it beside dice. Domain-agnostic and independent of voxel "
            "size."
        ),
        domain="",
    )


def vs_signed_metric() -> MetricSpec:
    """Signed Volumetric Similarity (SimpleITK convention), range [-2, 2].

    Negative means the prediction is smaller than the reference
    (undersegmentation), positive means larger (oversegmentation).
    """
    slice_mode, volume_mode = _both_modes(_vs_signed)
    return MetricSpec(
        name="vs_signed", direction="not_ranked", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            "Signed Volumetric Similarity: the direction of the size error "
            "(negative = the prediction is too small, positive = too large, "
            "0.0 = the sizes match). Domain-agnostic and independent of voxel size."
        ),
        domain="",
    )


def _count_metric(name: str, fn: Callable[..., float], what: str) -> MetricSpec:
    slice_mode, volume_mode = _both_modes(fn)
    return MetricSpec(
        name=name, direction="not_ranked", reference=True, channels="gray",
        slice_mode=slice_mode, volume_mode=volume_mode, builtin=False,
        description=(
            f"Raw voxel count: {what}. Not a quality score — it is reported so "
            "that per-slice runs can be summed into correct volume-level dice "
            "and volumetric similarity."
        ),
        domain="",
    )


def v_pred_metric() -> MetricSpec:
    return _count_metric("v_pred", _v_pred, "voxels marked in the prediction")


def v_gt_metric() -> MetricSpec:
    return _count_metric("v_gt", _v_gt, "voxels marked in the reference")


def tp_metric() -> MetricSpec:
    return _count_metric("tp", _tp, "voxels marked in both masks")
```
Add `from iqaevaluator.segmentation_metrics.volume import require_binary` to the imports. The constants `VS`, `VS_SIGNED`, `V_PRED`, `V_GT`, `TP`, `VOLUME_METRICS` stay.

- [ ] **Step 4: Implement `boundary_iou.py`**

`band_width`:
```python
def band_width(
    shape: tuple[int, ...],
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """Boundary band width for a mask of `shape`.

    Without `spacing` the width is a fraction of the diagonal in voxels,
    rounded to the nearest whole voxel and floored at 1 — the paper's
    definition, and the same value `dilation_pixels` returns; chessboard
    distances are integers, so thresholding at anything else would silently
    shift the band.

    With `spacing` it is a fraction of the diagonal in millimetres, left
    un-rounded, floored at `max(spacing)`: at least one voxel along every
    axis. The nearest background voxel across a face lies exactly one
    spacing away along that axis, so a band thinner than the coarsest
    spacing would contain no face perpendicular to that axis at all — on
    3 mm slices with a 1.2 mm band the whole through-plane boundary
    vanished and a one-slice shift scored better than a one-voxel in-plane
    shift.

    Args:
        shape: the mask's shape, 2D or 3D.
        dilation_ratio: fraction of the diagonal (paper default 0.02).
        spacing: physical size per axis, same length as `shape`.
    """
    extent = np.asarray(shape, dtype=float)
    if spacing is not None:
        spacing_arr = np.asarray(spacing, dtype=float)
        extent = extent * spacing_arr
        return max(float(spacing_arr.max()), dilation_ratio * float(np.linalg.norm(extent)))
    return float(max(1, round(dilation_ratio * float(np.linalg.norm(extent)))))
```
`boundary_iou`: change `label: int = 1` to `label: Optional[int] = None` and its docstring line to "label: for integer label maps, which class to score one-vs-rest; None scores every non-zero label." The adapter and builder:
```python
class BoundaryIoUMetric:
    """Adapter making `boundary_iou` satisfy the framework's Metric protocol.

    Scores channel 0 of each sample in an `(N, C, H, W)` batch (or the whole
    `(D, H, W)` body of a `(1, C, D, H, W)` volume), looping over the batch,
    and returns one score per sample. Input must be binary — load masks with
    `normalization.Mask()` (`Mask(label=k)` selects one class of a label
    map). Both masks empty → the score is undefined and comes back as `None`;
    one mask empty → 0.0.

    Args:
        dilation_ratio: boundary band width as a fraction of the image diagonal.
        spacing: physical size per axis. Without it (slice mode) the band is
            measured in voxels; with it (volume mode) it is measured in
            physical units, so it stays equally thick along every axis on
            anisotropic data and never thinner than one voxel (`band_width`).
    """

    def __init__(
        self,
        *,
        dilation_ratio: float = DEFAULT_DILATION_RATIO,
        spacing: Optional[tuple[float, ...]] = None,
    ):
        self._dilation_ratio = dilation_ratio
        self._spacing        = spacing

    def __call__(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> list[Optional[float]]:
        if target is None:
            raise ValueError(
                "boundary_iou is a full-reference metric and requires a target mask"
            )
        require_binary(input, metric="boundary_iou")
        require_binary(target, metric="boundary_iou")
        pred = input.detach().cpu().numpy()
        gt   = target.detach().cpu().numpy()
        scores: list[Optional[float]] = []
        for i in range(pred.shape[0]):
            score = boundary_iou(
                pred[i, 0], gt[i, 0],
                dilation_ratio=self._dilation_ratio, spacing=self._spacing,
            )
            scores.append(None if np.isnan(score) else float(score))
        return scores


def boundary_iou_metric(*, dilation_ratio: float = DEFAULT_DILATION_RATIO) -> MetricSpec:
    """Boundary IoU: IoU of the contour bands rather than the whole mask.

    Domain-agnostic: the band width is a fraction of the image diagonal, so
    there is no physical spacing or class count to retune between domains —
    `dilation_ratio` is the single tunable and means the same thing everywhere.
    Input must be binary — load masks with `Mask()`.

    Args:
        dilation_ratio: boundary band width as a fraction of sqrt(H^2 + W^2)
            (default 0.02, the paper's value). Larger is more forgiving; 1.0
            degenerates to plain mask IoU.
    """
    metric = BoundaryIoUMetric(dilation_ratio=dilation_ratio)

    def build_volume_metric(spacing: Optional[tuple[float, ...]]) -> BoundaryIoUMetric:
        if spacing is None:
            print(
                "[WARNING] no voxel size available, so the boundary band is "
                "measured in voxels rather than physical units. A band of the "
                "same voxel width is physically thicker along a coarse axis, "
                "so through-plane boundary errors are judged more leniently "
                "than in-plane ones. Scores stay comparable between images on "
                "the same grid. To get an evenly thick band, use a format that "
                "records the voxel size (NIfTI, NRRD, MHA or DICOM)."
            )
        return BoundaryIoUMetric(dilation_ratio=dilation_ratio, spacing=spacing)
```
The `MetricSpec(...)` returned by the builder is unchanged. Add `require_binary` to the `volume` import. In the module docstring's volume-mode paragraph append: "The physical band is never thinner than the coarsest voxel spacing (see `band_width`)."

- [ ] **Step 5: Run the tests to verify they pass**

Run: `hatch run calibration:test tests/test_volume_metrics.py tests/test_boundary_iou.py tests/test_volume.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/iqaevaluator/segmentation_metrics/volume_metrics.py src/iqaevaluator/segmentation_metrics/boundary_iou.py tests/test_volume_metrics.py tests/test_boundary_iou.py
git commit -m "fix(boundary-iou): floor the physical band at one voxel; drop threshold knobs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Policy table across every scored metric, no `inf` in a report, F6 cross-check (spec Part 2 tests, review 2.5)

**Files:**
- Create: `tests/test_mask_policy.py`
- Modify: `tests/test_volume_metrics.py` (`TestCrossCheck`)

**Interfaces:**
- Consumes: everything from Tasks 3–8; `main.evaluate(input, target, *, registry, mode, normalization)`; `EvaluationResult.aggregate_volumes()`.

- [ ] **Step 1: Write the policy-table tests**

Create `tests/test_mask_policy.py`:
```python
"""Spec D3 — the empty-mask policy table, every cell, every scored metric, both modes.

Fixture F3: volume (4, 8, 8); the populated side is the block [1:3, 2:5, 2:5]
(18 voxels). In slice mode slices 0 and 3 are empty on both sides at the
metric level; the evaluator would skip them, the metric itself answers None.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from iqaevaluator.metrics import MetricRegistry, SEGMENTATION_METRICS
from iqaevaluator.segmentation_metrics.boundary_iou import BOUNDARY_IOU
from iqaevaluator.segmentation_metrics.monai_metrics import ASSD, DICE, HAUSDORFF95, NSD, PANOPTIC_QUALITY
from iqaevaluator.segmentation_metrics.volume_metrics import TP, V_GT, V_PRED, VS, VS_SIGNED

SCORED = [DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU, VS]
ONE_SIDED = {
    "dice": 0.0, "vs": 0.0, "nsd": 0.0, "boundary_iou": 0.0, "panoptic_quality": 0.0,
    "hausdorff95": None, "assd": None,
}
ISO = (1.0, 1.0, 1.0)


def _f3(case: str) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.zeros(1, 1, 4, 8, 8)
    block[..., 1:3, 2:5, 2:5] = 1.0
    empty = torch.zeros_like(block)
    return {"gt_empty": (block, empty), "pred_empty": (empty, block), "both_empty": (empty, empty)}[case]


@pytest.mark.parametrize("case", ["gt_empty", "pred_empty", "both_empty"])
@pytest.mark.parametrize("spec", SCORED, ids=lambda s: s.name)
def test_policy_table_in_volume_mode(spec, case):
    pred, gt = _f3(case)
    metric = MetricRegistry(spec).get_metric(spec.name, "volume", ISO)
    expected = None if case == "both_empty" else ONE_SIDED[spec.name]
    assert metric(pred, gt) == [expected]


@pytest.mark.parametrize("case", ["gt_empty", "pred_empty", "both_empty"])
@pytest.mark.parametrize("spec", SCORED, ids=lambda s: s.name)
def test_policy_table_in_slice_mode(spec, case):
    pred5, gt5 = _f3(case)
    pred, gt = pred5[0].permute(1, 0, 2, 3), gt5[0].permute(1, 0, 2, 3)     # (4, 1, 8, 8)
    metric = MetricRegistry(spec).get_metric(spec.name)
    one = None if case == "both_empty" else ONE_SIDED[spec.name]
    assert metric(pred, gt) == [None, one, one, None]


def test_counts_and_vs_signed_have_no_policy():
    pred, gt = _f3("gt_empty")
    registry = MetricRegistry(V_PRED, V_GT, TP, VS_SIGNED)
    value = lambda name, p, g: registry.get_metric(name, "volume", ISO)(p, g)[0]
    assert (value("v_pred", pred, gt), value("v_gt", pred, gt), value("tp", pred, gt)) == (18.0, 0.0, 0.0)
    assert value("vs_signed", pred, gt) == pytest.approx(2.0)
    empty, _ = _f3("both_empty")
    assert value("v_pred", empty, empty) == 0.0
    assert value("vs_signed", empty, empty) is None


@pytest.mark.parametrize("spec", SEGMENTATION_METRICS, ids=lambda s: s.name)
def test_every_adapter_rejects_a_non_binary_tensor_naming_mask(spec):
    half = torch.full((1, 1, 8, 8), 0.5)
    with pytest.raises(ValueError, match=r"Mask\(\)"):
        MetricRegistry(spec).get_metric(spec.name)(half, half)


def test_a_report_with_one_sided_empty_volumes_holds_no_inf(tmp_path):
    import nibabel as nib
    from main import evaluate
    from iqaevaluator.normalization import Mask

    gt = np.zeros((16, 16, 6), dtype=np.uint8)
    gt[4:12, 4:12, 1:5] = 1
    pred_path, gt_path = tmp_path / "case_pred.nii.gz", tmp_path / "case_gt.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros_like(gt), np.eye(4)), pred_path)
    nib.save(nib.Nifti1Image(gt, np.eye(4)), gt_path)

    df = evaluate(pred_path, gt_path, registry=MetricRegistry(*SEGMENTATION_METRICS),
                  mode="volume", normalization=Mask()).to_frame()
    numeric = df.select_dtypes(include="number").to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    row = df.iloc[0]
    assert (row["dice"], row["nsd"], row["boundary_iou"], row["vs"], row["panoptic_quality"]) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert pd.isna(row["hausdorff95"]) and pd.isna(row["assd"])
    assert (row["v_pred"], row["v_gt"], row["tp"]) == (0.0, 256.0, 0.0)
    assert row["vs_signed"] == pytest.approx(-2.0)
```

- [ ] **Step 2: Add the F6 cross-check**

Append to `TestCrossCheck` in `tests/test_volume_metrics.py`:
```python
    def test_one_label_of_a_label_map_agrees_between_modes(self, tmp_path):
        """F6: a {0,1,2} int16 map under Mask(label=2). Volume dice == aggregated
        slice dice == 0.75; under Mask() the label-1 block joins in: 56/72."""
        import nibabel as nib
        from main import evaluate
        from iqaevaluator.normalization import Mask
        from iqaevaluator.segmentation_metrics.monai_metrics import DICE
        from iqaevaluator.segmentation_metrics.volume_metrics import TP, V_GT, V_PRED

        gt = np.zeros((4, 8, 8), dtype=np.int16)          # (D, H, W)
        gt[0:1, 0:2, 0:2] = 1
        gt[1:3, 2:6, 2:6] = 2
        pred = np.zeros_like(gt)
        pred[0:1, 0:2, 0:2] = 1
        pred[1:3, 2:6, 3:7] = 2
        paths = {}
        for name, arr in (("case_pred", pred), ("case_gt", gt)):
            # the NIfTI decoder yields (Z, X, Y) with spacing (dz, dx, dy): write (H, W, D)
            paths[name] = tmp_path / f"{name}.nii.gz"
            nib.save(nib.Nifti1Image(np.ascontiguousarray(np.transpose(arr, (1, 2, 0))), np.eye(4)), paths[name])

        volume = evaluate(paths["case_pred"], paths["case_gt"], registry=MetricRegistry(DICE),
                          mode="volume", normalization=Mask(label=2)).to_frame().iloc[0]["dice"]
        aggregated = evaluate(paths["case_pred"], paths["case_gt"], registry=MetricRegistry(V_PRED, V_GT, TP),
                              mode="slice", normalization=Mask(label=2)).aggregate_volumes().iloc[0]["dice"]
        assert volume == pytest.approx(0.75, abs=1e-6)
        assert aggregated == pytest.approx(0.75, abs=1e-6)

        any_label = evaluate(paths["case_pred"], paths["case_gt"], registry=MetricRegistry(DICE),
                             mode="volume", normalization=Mask()).to_frame().iloc[0]["dice"]
        assert any_label == pytest.approx(56 / 72, abs=1e-6)
```

- [ ] **Step 3: Run the tests**

Run: `hatch run calibration:test tests/test_mask_policy.py tests/test_volume_metrics.py -q`
Expected: all pass. If a cell of the policy table fails, fix the adapter from Task 7/8 that owns that metric — the table is the spec, not the test.

- [ ] **Step 4: Run the whole suite**

Run: `hatch run calibration:test -q`
Expected: all pass (previous count + the tests added in Tasks 2–9).

- [ ] **Step 5: Commit**

```bash
git add tests/test_mask_policy.py tests/test_volume_metrics.py
git commit -m "test(segmentation): empty-mask policy table, inf-free report, F6 cross-check

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 10: Calibration package — cases, harness, hand-value tests (spec Part 3, F1–F6)

**Files:**
- Create: `tests/calibration/__init__.py` (empty), `tests/calibration/cases.py`, `tests/calibration/harness.py`, `tests/calibration/test_hand.py`

**Interfaces:**
- Produces:
  - `cases.CalibrationCase(name, metric, build, expected, derivation, source, spacing=None, mode="volume", normalizer=Mask(), metric_kwargs={}, tolerance=1e-6, kind="hand")`; `cases.CASES: list[CalibrationCase]`; fixture builders `f1_block_shift()`, `f2_outlier()`, `f3_empty(kind)`, `f4_panoptic()`, `f5_boundary(shift)`, `f5_square_2d()`, `f6_multilabel()`, `f7_blobs(seed)` → `(pred, gt)` numpy `(D, H, W)` or `(H, W)`.
  - `harness.write_nifti(arr, path, spacing) -> Path`; `harness.framework_records(case, tmp_dir) -> list[ImageEvaluatorRecord]`; `harness.framework_value(case, tmp_dir) -> Optional[float]`; `harness.SPEC_BUILDERS: dict[str, Callable[..., MetricSpec]]`.
- Consumes: `Mask`, `Raw`, `load_pair`, `build_evaluator`, `MetricRegistry`, the five MONAI builders, `boundary_iou_metric`, `vs_metric`.
- Tests import the package as `from calibration.cases import ...` (pytest puts `tests/` on `sys.path` because `tests/` has no `__init__.py` and `tests/calibration/` has one).

- [ ] **Step 1: Write `cases.py`**

```python
"""Hand-derived calibration cases for the segmentation metrics (spec Part 3).

Every case is a fixture small enough for its derivation to fit in a few
lines, plus the value that derivation gives. The value is not the authority
— the derivation is; a reviewer checks the arithmetic, the test checks that
the framework reproduces it through its public path (NIfTI → load_pair →
registry → evaluator → record).

Shapes are (D, H, W); ranges are half-open Python slices; spacings are
(D, H, W) in millimetres.
"""
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from iqaevaluator.normalization import Mask, Normalizer, Raw

Spacing = tuple[float, float, float]
ISO: Spacing = (1.0, 1.0, 1.0)
Builder = Callable[[], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CalibrationCase:
    name: str
    metric: str                       # registry name, key of harness.SPEC_BUILDERS
    build: Builder                    # () -> (pred, gt)
    expected: Optional[float]         # hand value; None = undefined (policy) or no hand value (random)
    derivation: str
    source: str
    spacing: Optional[Spacing] = None
    mode: Literal["slice", "volume"] = "volume"
    normalizer: Normalizer = Mask()
    metric_kwargs: dict = field(default_factory=dict)
    tolerance: float = 1e-6
    kind: Literal["hand", "policy", "random"] = "hand"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def f1_block_shift() -> tuple[np.ndarray, np.ndarray]:
    """GT = [2:4, 3:9, 3:9], Pred = same block shifted one voxel along W."""
    gt = np.zeros((6, 12, 12), bool); gt[2:4, 3:9, 3:9] = True
    pred = np.zeros((6, 12, 12), bool); pred[2:4, 3:9, 4:10] = True
    return pred, gt


def f2_outlier() -> tuple[np.ndarray, np.ndarray]:
    """GT = 4^3 cube [2:6, 2:6, 2:6]; Pred = GT plus one voxel at (10, 3, 3), 5 voxels away."""
    gt = np.zeros((12, 10, 10), bool); gt[2:6, 2:6, 2:6] = True
    pred = gt.copy(); pred[10, 3, 3] = True
    return pred, gt


def f3_empty(kind: Literal["gt_empty", "pred_empty", "both_empty"]) -> tuple[np.ndarray, np.ndarray]:
    block = np.zeros((4, 8, 8), bool); block[1:3, 2:5, 2:5] = True
    empty = np.zeros((4, 8, 8), bool)
    return {"gt_empty": (block, empty), "pred_empty": (empty, block), "both_empty": (empty, empty)}[kind]


def f4_panoptic() -> tuple[np.ndarray, np.ndarray]:
    """(16, 16) instance maps. GT: id 1 = [0:4, 0:4], id 2 = [6:10, 6:10].
    Pred: id 1 = [0:4, 1:5] (IoU 0.6 with GT 1), id 3 = [12:16, 0:4] (FP); GT 2 unmatched (FN)."""
    gt = np.zeros((16, 16), np.int16); gt[0:4, 0:4] = 1; gt[6:10, 6:10] = 2
    pred = np.zeros((16, 16), np.int16); pred[0:4, 1:5] = 1; pred[12:16, 0:4] = 3
    return pred, gt


def f5_boundary(shift: Literal["D", "W"]) -> tuple[np.ndarray, np.ndarray]:
    """(8, 40, 40) at spacing (3, 1, 1). GT = [2:6, 8:32, 8:32]; Pred shifted one
    slice along D (3 mm) or one voxel along W (1 mm)."""
    gt = np.zeros((8, 40, 40), bool); gt[2:6, 8:32, 8:32] = True
    pred = np.zeros((8, 40, 40), bool)
    if shift == "D":
        pred[3:7, 8:32, 8:32] = True
    else:
        pred[2:6, 8:32, 9:33] = True
    return pred, gt


def f5_square_2d() -> tuple[np.ndarray, np.ndarray]:
    """(40, 40): 20x20 square [10:30, 10:30] vs the same square shifted one pixel along W."""
    gt = np.zeros((40, 40), bool); gt[10:30, 10:30] = True
    pred = np.zeros((40, 40), bool); pred[10:30, 11:31] = True
    return pred, gt


def f6_multilabel() -> tuple[np.ndarray, np.ndarray]:
    """(4, 8, 8) int16. Label 1 = [0:1, 0:2, 0:2] in both. Label 2: GT [1:3, 2:6, 2:6],
    Pred [1:3, 2:6, 3:7]."""
    gt = np.zeros((4, 8, 8), np.int16); gt[0:1, 0:2, 0:2] = 1; gt[1:3, 2:6, 2:6] = 2
    pred = np.zeros((4, 8, 8), np.int16); pred[0:1, 0:2, 0:2] = 1; pred[1:3, 2:6, 3:7] = 2
    return pred, gt


def f7_blobs(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two independent (16, 32, 32) blobs: smoothed uniform noise thresholded at 0.5."""
    rng = np.random.default_rng(seed)
    pred = gaussian_filter(rng.random((16, 32, 32)), 2) > 0.5
    gt = gaussian_filter(rng.random((16, 32, 32)), 2) > 0.5
    return pred, gt


# ---------------------------------------------------------------------------
# Derivations (checked by hand; the numbers follow from them)
# ---------------------------------------------------------------------------

_F1 = (
    "|GT| = |Pred| = 2*6*6 = 72; overlap W 4:9 -> 2*6*5 = 60. "
    "Both blocks are 2 voxels thick along D, so cross-structure erosion removes every voxel: "
    "S_gt = S_pred = all 72 voxels. gt->pred: 60 overlap voxels are also pred-surface -> 0; "
    "the 12 GT voxels at W=3 have their nearest pred-surface voxel at W=4 -> distance 1 (3 mm at spacing (1,1,3)). "
    "Symmetric for pred->gt (W=9 -> W=8)."
)
_F2 = (
    "|GT| = 64, |Pred| = 65, overlap 64. Cross erosion of a 4^3 cube removes its 2^3 interior -> S_gt = 56; "
    "the isolated voxel is its own surface -> S_pred = 57. gt->pred: all 56 distances 0. "
    "pred->gt: 56 zeros and one 5 (nearest GT surface voxel to (10,3,3) is (5,3,3))."
)
_F4 = (
    "Kirillov 2019 Eq. 1: PQ = sum_TP IoU / (|TP| + 0.5|FP| + 0.5|FN|). GT1 vs Pred1: overlap 12, union 20, "
    "IoU 0.6 > 0.5 -> TP; GT2 unmatched -> FN; Pred3 unmatched -> FP. PQ = 0.6 / (1 + 0.5 + 0.5) = 0.3 "
    "(SQ 0.6, RQ 0.5). MONAI adds smooth_numerator=1e-6 to the denominator -> 0.29999986; tolerance 1e-6."
)
_F5 = (
    "Band width: 0.02*|(24, 40, 40)| = 1.229 mm < max(spacing) = 3 mm -> floor 3.0 mm. Inside an axis-aligned box the "
    "EDT to background is the perpendicular distance to the nearest face. D=2 and D=5 slabs lie 3 mm from background "
    "along D -> whole slab in the band (576 each); on D=3,4 (6 mm from background along D) only the in-plane ring of "
    "width 3, 24^2 - 18^2 = 252 each. |band| = 1152 + 504 = 1656 for GT and both predictions. "
    "D-shift: Pred band = slabs D=3,6 + rings D=4,5. Intersection D=3 ring&slab 252, D=4 ring&ring 252, D=5 slab&ring 252 "
    "= 756; union 2*1656 - 756 = 2556 -> 756/2556. "
    "W-shift: slabs D=2,5 intersect over W 9:32 -> 24*23 = 552 each; on D=3,4 within H 8:32 x W 9:32 the 6 ring rows give "
    "6*23 = 138 and the other 18 rows share W in {9,10,30,31} -> 72; 210 each. Total 1104 + 420 = 1524; "
    "union 3312 - 1524 = 1788 -> 1524/1788. With the old 1.0 mm floor the band was the width-1 in-plane ring only "
    "(92 per slab, 368 total): D-shift 276/460 = 0.6, W-shift 184/552 = 0.333 -> the 3 mm error scored better (review 3.9)."
)
_F5_2D = (
    "Diagonal 56.57 px * 0.02 = 1.13 -> band 1 px. Ring of a 20x20 square: 2*20 + 2*18 = 76 px. "
    "Shifted ring shares rows 10 and 29 over cols 11..29 -> 2*19 = 38; the vertical sides do not coincide. "
    "Union 76 + 76 - 38 = 114 -> 38/114 = 1/3 (mask IoU would be 380/420 = 0.905)."
)
_F6 = (
    "Label 2: |GT| = |Pred| = 2*4*4 = 32, overlap W 3:6 -> 2*4*3 = 24 -> Dice 48/64. Slice counts: slices 1,2 each "
    "v_pred = v_gt = 16, tp = 12; slices 0,3 empty on both sides under label 2 -> is_empty, counts 0. "
    "Mask() (any non-zero): the identical label-1 block adds 4 to each volume and 4 to tp -> 56/72."
)
_POLICY = "Spec D3: one side empty -> 0.0 (Dice/VS/NSD/PQ/Boundary IoU) or None (HD95/ASSD); both empty -> None."

_TAHA = "Taha & Hanbury 2015, BMC Med Imaging 15:29"
_DICE_SRC = "Dice 1945; " + _TAHA + " Eq. (6)"
_HD_SRC = _TAHA + ", HD and HD_q (Eq. 22-23), voxel surfaces via cross-structure erosion (MONAI 1.6.0)"
_ASSD_SRC = _TAHA + ", ASSD (Eq. 25)"
_NSD_SRC = "Nikolov et al. 2018 (arXiv:1809.04430), voxel-surface variant as in MONAI compute_surface_dice"
_PQ_SRC = "Kirillov et al. 2019, CVPR, Eq. 1"
_BIOU_SRC = "Cheng et al. 2021, CVPR, Sec. 3; 3D physical band is this framework's extension"
_POLICY_SRC = "spec 2026-09-13-calibration-segmentation-design.md, D3"


def _case(name, metric, build, expected, derivation, source, **kw) -> CalibrationCase:
    return CalibrationCase(name, metric, build, expected, derivation, source, **kw)


CASES: list[CalibrationCase] = [
    # F1 — block shift, isotropic
    _case("F1_dice", "dice", f1_block_shift, 120 / 144, "Dice = 2*60/(72+72) = 120/144 (IoU would be 60/84 = 0.714). " + _F1, _DICE_SRC, spacing=ISO),
    _case("F1_hd95", "hausdorff95", f1_block_shift, 1.0, "P95 of 72 values with 12 ones (top 16.7 %) = 1 in both directions -> max 1.0. " + _F1, _HD_SRC, spacing=ISO),
    _case("F1_hd", "hausdorff95", f1_block_shift, 1.0, "Plain HD = max distance = 1.0. " + _F1, _HD_SRC, spacing=ISO, metric_kwargs={"percentile": None}),
    _case("F1_assd", "assd", f1_block_shift, 24 / 144, "ASSD = (12*1 + 12*1)/(72 + 72) = 24/144. " + _F1, _ASSD_SRC, spacing=ISO),
    _case("F1_nsd", "nsd", f1_block_shift, 1.0, "tau = 1 voxel: every surface voxel is within 1 -> 144/144. " + _F1, _NSD_SRC, spacing=ISO),
    # F1 — anisotropic (1, 1, 3): the W step is 3 mm
    _case("F1_aniso_hd95", "hausdorff95", f1_block_shift, 3.0, "Every non-zero distance is one W step = 3 mm -> 3.0. " + _F1, _HD_SRC, spacing=(1.0, 1.0, 3.0)),
    _case("F1_aniso_assd", "assd", f1_block_shift, 0.5, "(12*3 + 12*3)/144 = 0.5. " + _F1, _ASSD_SRC, spacing=(1.0, 1.0, 3.0)),
    _case("F1_aniso_nsd", "nsd", f1_block_shift, 120 / 144, "tau = 1.0 mm: 12 + 12 surface voxels at 3 mm fail -> 120/144. " + _F1, _NSD_SRC, spacing=(1.0, 1.0, 3.0)),
    _case("F1_aniso_dice", "dice", f1_block_shift, 120 / 144, "Dice is spacing-invariant: 120/144. " + _F1, _DICE_SRC, spacing=(1.0, 1.0, 3.0)),
    # F2 — outlier: percentile and symmetry matter
    _case("F2_dice", "dice", f2_outlier, 128 / 129, "Dice = 2*64/(65+64) = 128/129. " + _F2, _DICE_SRC, spacing=ISO),
    _case("F2_hd", "hausdorff95", f2_outlier, 5.0, "HD = max(0, 5) = 5.0. " + _F2, _HD_SRC, spacing=ISO, metric_kwargs={"percentile": None}),
    _case("F2_hd95", "hausdorff95", f2_outlier, 0.0, "P95 of [0]*56 + [5]: position 0.95*56 = 53.2 lies among the zeros -> 0; max(0, 0) = 0.0 (the percentile=95 mutation check: 5.0 vs 0.0). " + _F2, _HD_SRC, spacing=ISO),
    _case("F2_assd", "assd", f2_outlier, 5 / 113, "Symmetric: (0*56 + 0*56 + 5)/(56 + 57) = 5/113. " + _F2, _ASSD_SRC, spacing=ISO),
    _case("F2_asd_directed", "assd", f2_outlier, 5 / 57, "Directed pred->gt: 5/57 (the symmetric=True mutation check). " + _F2, _ASSD_SRC, spacing=ISO, metric_kwargs={"symmetric": False}),
    _case("F2_nsd", "nsd", f2_outlier, 112 / 113, "tau = 1: (56 + 56)/(56 + 57) = 112/113. " + _F2, _NSD_SRC, spacing=ISO),
    # F4 — panoptic quality on Raw() instance maps, 2D, slice mode
    _case("F4_pq", "panoptic_quality", f4_panoptic, 0.3, _F4, _PQ_SRC, mode="slice", normalizer=Raw()),
    # F5 — boundary IoU, physical band, spacing (D, H, W) = (3, 1, 1)
    _case("F5_biou_D_shift", "boundary_iou", lambda: f5_boundary("D"), 756 / 2556, _F5, _BIOU_SRC, spacing=(3.0, 1.0, 1.0)),
    _case("F5_biou_W_shift", "boundary_iou", lambda: f5_boundary("W"), 1524 / 1788, _F5, _BIOU_SRC, spacing=(3.0, 1.0, 1.0)),
    _case("F5_biou_2d", "boundary_iou", f5_square_2d, 38 / 114, _F5_2D, _BIOU_SRC, mode="slice"),
    # F6 — one label of a label map
    _case("F6_dice_label2", "dice", f6_multilabel, 48 / 64, _F6, _DICE_SRC, spacing=ISO, normalizer=Mask(label=2)),
    _case("F6_dice_any_label", "dice", f6_multilabel, 56 / 72, _F6, _DICE_SRC, spacing=ISO, normalizer=Mask()),
]

# F3 — the empty-mask policy, volume mode (slice mode is covered in test_hand.py directly)
_ONE_SIDED = {"dice": 0.0, "vs": 0.0, "nsd": 0.0, "boundary_iou": 0.0, "panoptic_quality": 0.0, "hausdorff95": None, "assd": None}
for _metric, _one in _ONE_SIDED.items():
    for _kind in ("gt_empty", "pred_empty", "both_empty"):
        CASES.append(_case(
            f"F3_{_metric}_{_kind}", _metric, (lambda k=_kind: f3_empty(k)),
            None if _kind == "both_empty" else _one, _POLICY, _POLICY_SRC, spacing=ISO, kind="policy",
        ))

# F7 — random blobs, no hand value: framework vs the same-variant official implementation
for _seed_offset, _spacing in enumerate([ISO, (1.0, 1.0, 3.0), (2.5, 0.8, 0.8)]):
    _seed = 20260913 + _seed_offset
    for _metric, _kwargs in (("dice", {}), ("assd", {}), ("hausdorff95", {"percentile": None})):
        CASES.append(_case(
            f"F7_{_metric}_{'x'.join(f'{s:g}' for s in _spacing)}", _metric, (lambda s=_seed: f7_blobs(s)),
            None, "no hand value — agreement with the official implementation only", "spec F7",
            spacing=_spacing, metric_kwargs=_kwargs, kind="random", tolerance=1e-5,
        ))
```

- [ ] **Step 2: Write `harness.py`**

```python
"""Runs a CalibrationCase through the framework's public path.

The path a user takes is what gets calibrated, not an adapter in isolation:
fixture -> temporary NIfTI (with spacing) -> load_pair(normalizer) ->
MetricRegistry(spec) -> IQAEvaluator / VolumeEvaluator -> record value.
"""
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np

from iqaevaluator.evaluator_factory import build_evaluator
from iqaevaluator.image_loader import load_pair
from iqaevaluator.metric_spec import MetricSpec
from iqaevaluator.metrics import MetricRegistry
from iqaevaluator.records import ImageEvaluatorRecord
from iqaevaluator.segmentation_metrics.boundary_iou import boundary_iou_metric
from iqaevaluator.segmentation_metrics.monai_metrics import (
    average_surface_distance_metric, dice_metric, hausdorff95_metric,
    normalized_surface_dice_metric, panoptic_quality_metric,
)
from iqaevaluator.segmentation_metrics.volume_metrics import (
    tp_metric, v_gt_metric, v_pred_metric, vs_metric,
)

from calibration.cases import ISO, CalibrationCase, Spacing

SPEC_BUILDERS: dict[str, Callable[..., MetricSpec]] = {
    "dice": dice_metric,
    "hausdorff95": hausdorff95_metric,
    "nsd": normalized_surface_dice_metric,
    "assd": average_surface_distance_metric,
    "panoptic_quality": panoptic_quality_metric,
    "boundary_iou": boundary_iou_metric,
    "vs": vs_metric,
    "v_pred": v_pred_metric,
    "v_gt": v_gt_metric,
    "tp": tp_metric,
}


def _storable(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int16)
    return arr.astype(np.float32)


def write_nifti(arr: np.ndarray, path: Path, spacing: Spacing = ISO) -> Path:
    """Write `arr` so that ImageLoader(path).raw == arr and .spacing == spacing.

    The NIfTI decoder transposes (X, Y, Z) -> (Z, X, Y) and reads zooms as
    (dz, dx, dy); so a (D, H, W) array is stored as (H, W, D) with an affine
    of diag(s_h, s_w, s_d). A 2D (H, W) array becomes a single slice.
    """
    if arr.ndim == 2:
        arr = arr[None]
    data = np.ascontiguousarray(np.transpose(_storable(arr), (1, 2, 0)))
    s_d, s_h, s_w = spacing
    nib.save(nib.Nifti1Image(data, np.diag([s_h, s_w, s_d, 1.0])), str(path))
    return path


def framework_records(case: CalibrationCase, tmp_dir: Path) -> list[ImageEvaluatorRecord]:
    pred, gt = case.build()
    spacing = case.spacing or ISO
    pred_path = write_nifti(pred, tmp_dir / f"{case.name}_pred.nii.gz", spacing)
    gt_path = write_nifti(gt, tmp_dir / f"{case.name}_gt.nii.gz", spacing)
    inp, tgt = load_pair(pred_path, gt_path, case.normalizer)
    spec = SPEC_BUILDERS[case.metric](**case.metric_kwargs)
    return build_evaluator(inp, tgt, MetricRegistry(spec), mode=case.mode).run_evaluation()


def framework_value(case: CalibrationCase, tmp_dir: Path) -> Optional[float]:
    """The single record value of a volume-mode or single-slice case."""
    records = framework_records(case, tmp_dir)
    assert len(records) == 1, f"{case.name}: expected one record, got {len(records)}"
    return records[0].extra.get(case.metric)
```

- [ ] **Step 3: Write `test_hand.py`**

```python
"""Layer 1: framework == hand-derived value, through the public path. No extras needed."""
import numpy as np
import pytest

from calibration.cases import CASES, f3_empty, f5_boundary
from calibration.harness import framework_records, framework_value, write_nifti
from iqaevaluator.segmentation_metrics.boundary_iou import boundary_region

HAND = [c for c in CASES if c.kind == "hand"]
POLICY = [c for c in CASES if c.kind == "policy"]


@pytest.mark.parametrize("case", HAND, ids=lambda c: c.name)
def test_framework_reproduces_the_hand_value(case, tmp_path):
    value = framework_value(case, tmp_path)
    assert value is not None, case.derivation
    assert value == pytest.approx(case.expected, abs=case.tolerance), case.derivation


@pytest.mark.parametrize("case", POLICY, ids=lambda c: c.name)
def test_framework_applies_the_empty_policy_in_volume_mode(case, tmp_path):
    value = framework_value(case, tmp_path)
    if case.expected is None:
        assert value is None, case.derivation
    else:
        assert value == case.expected, case.derivation


def test_f3_slice_mode_skips_only_slices_empty_on_both_sides(tmp_path):
    from calibration.cases import CalibrationCase
    from iqaevaluator.normalization import Mask
    base = dict(build=lambda: f3_empty("gt_empty"), expected=None, derivation="", source="", mode="slice", normalizer=Mask())
    dice_records = framework_records(CalibrationCase("F3_slice_dice", "dice", **base), tmp_path)
    hd_records = framework_records(CalibrationCase("F3_slice_hd95", "hausdorff95", **base), tmp_path)
    assert [r.is_empty for r in dice_records] == [True, False, False, True]
    assert [r.extra.get("dice") for r in dice_records] == [None, 0.0, 0.0, None]
    assert [r.extra.get("hausdorff95") for r in hd_records] == [None, None, None, None]


def test_f5_a_3mm_through_plane_shift_scores_below_a_1mm_in_plane_shift(tmp_path):
    d_case = next(c for c in CASES if c.name == "F5_biou_D_shift")
    w_case = next(c for c in CASES if c.name == "F5_biou_W_shift")
    assert framework_value(d_case, tmp_path) < framework_value(w_case, tmp_path)


def test_f5_spacing_tuple_is_read_as_depth_height_width(tmp_path):
    from dataclasses import replace
    d_case = next(c for c in CASES if c.name == "F5_biou_D_shift")
    reversed_spacing = replace(d_case, name="F5_biou_D_shift_reversed", spacing=(1.0, 1.0, 3.0), expected=None)
    assert framework_value(reversed_spacing, tmp_path) != pytest.approx(d_case.expected, abs=1e-6)


def test_f5_the_old_one_millimetre_floor_inverted_the_order():
    """Documents review 3.9: with a 1.229 mm band the D-shift scored 0.6 and the W-shift 0.333."""
    gt = f5_boundary("D")[1]
    def biou(pred, width):
        pb, gb = boundary_region(pred, width, (3.0, 1.0, 1.0)), boundary_region(gt, width, (3.0, 1.0, 1.0))
        return (pb & gb).sum() / (pb | gb).sum()
    old_band = 0.02 * float(np.linalg.norm([8 * 3.0, 40.0, 40.0]))
    assert old_band == pytest.approx(1.229, abs=1e-3)
    assert biou(f5_boundary("D")[0], old_band) == pytest.approx(0.6)
    assert biou(f5_boundary("W")[0], old_band) == pytest.approx(1 / 3)


def test_write_nifti_round_trips_shape_and_spacing(tmp_path):
    from iqaevaluator.image_loader import ImageLoader
    arr = np.zeros((3, 5, 7), np.int16); arr[1, 2, 3] = 9
    loader = ImageLoader(write_nifti(arr, tmp_path / "rt.nii.gz", (2.5, 0.8, 0.6)))
    assert loader.raw.shape == (3, 5, 7) and int(loader.raw[1, 2, 3]) == 9
    assert loader.spacing == pytest.approx((2.5, 0.8, 0.6))
```

- [ ] **Step 4: Run the layer**

Run: `hatch run calibration:test tests/calibration/test_hand.py -q`
Expected: all pass. Every hand value here was reproduced numerically on 2026-09-13 with MONAI 1.6.0 / scipy 1.18.1 (see the spec's "Fixtures and hand values"); a failure means an adapter or the harness is wrong, not the number — check `write_nifti`'s axis order first, then the adapter of the failing metric.

- [ ] **Step 5: Commit**

```bash
git add tests/calibration/__init__.py tests/calibration/cases.py tests/calibration/harness.py tests/calibration/test_hand.py
git commit -m "test(calibration): hand-derived cases F1-F6 through the public path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 11: Official reference implementations — `official.py`, layers "framework == official", "official == hand", random blobs (spec Part 3, D8, D10)

**Files:**
- Create: `tests/calibration/official.py`, `tests/calibration/test_official.py`, `tests/calibration/test_official_hand.py`, `tests/calibration/test_random.py`

**Interfaces:**
- Produces: `official.official_value(case, work_dir) -> Optional[float]` (same-variant official value; `None` when no official implementation matches the framework's variant; raises `ImportError` when the package is missing); `official.divergent_values(case, work_dir) -> dict[str, float]` (report-only columns); `official.mask_to_boundary(mask, dilation_ratio)` (vendored); `official.boundary_iou_official(pred, gt, dilation_ratio)`; `official.panopticapi_pq(pred_map, gt_map, work_dir)`; `official.reference_versions() -> dict[str, str]`; constants `PANOPTICAPI_SHA`, `BOUNDARY_IOU_API_SHA`.
- Consumes: `CalibrationCase`, `CASES`, `framework_value` (Task 10); `as_mask` (Task 2).

- [ ] **Step 1: Write `official.py`**

```python
"""Layer 2: the implementations the community cites, same definitional variant (spec D10).

Every import is local to its function so that cases.py, harness.py and the
hand-value layer work without the [calibration] extras; the tests skip on
ImportError. Variants that differ from MONAI's (spec D10 table) are exposed
only through `divergent_values` for the report, never through an equality
assertion.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from calibration.cases import ISO, CalibrationCase
from iqaevaluator.segmentation_metrics.volume import as_mask

PANOPTICAPI_SHA = "7bb4655548f98f3fedc07bf37e9040a992b054b0"     # master head, 2026-09-13
BOUNDARY_IOU_API_SHA = "37d25586a677b043ed585f10e5c42d4e80176ea9"  # master head, 2026-09-13


def _binary(case: CalibrationCase) -> tuple[np.ndarray, np.ndarray]:
    """The case's masks binarized the way its normalizer would (Mask(label=k) keeps only k)."""
    pred, gt = case.build()
    label = getattr(case.normalizer, "label", None)
    return as_mask(pred, label), as_mask(gt, label)


# --- MedPy: voxel surfaces via cross-structure erosion, same as MONAI ---------

def medpy_dice(pred, gt) -> float:
    from medpy.metric.binary import dc
    return float(dc(pred, gt))


def medpy_assd(pred, gt, spacing) -> float:
    from medpy.metric.binary import assd
    return float(assd(pred, gt, voxelspacing=spacing))


def medpy_asd(pred, gt, spacing) -> float:
    """Directed pred -> gt (MedPy: result -> reference)."""
    from medpy.metric.binary import asd
    return float(asd(pred, gt, voxelspacing=spacing))


def medpy_hd(pred, gt, spacing) -> float:
    from medpy.metric.binary import hd
    return float(hd(pred, gt, voxelspacing=spacing))


def medpy_hd95(pred, gt, spacing) -> float:
    """Divergent variant: P95 over the *concatenated* directed distances."""
    from medpy.metric.binary import hd95
    return float(hd95(pred, gt, voxelspacing=spacing))


# --- surface-distance (DeepMind): area-weighted surface elements — divergent ----

def surface_distance_values(pred, gt, spacing, tolerance_mm: float) -> dict[str, float]:
    import surface_distance as sd
    d = sd.compute_surface_distances(gt, pred, spacing_mm=spacing)
    asd_gt_pred, asd_pred_gt = sd.compute_average_surface_distance(d)
    return {
        "hd95": float(sd.compute_robust_hausdorff(d, 95)),
        "asd_gt_to_pred": float(asd_gt_pred),
        "asd_pred_to_gt": float(asd_pred_gt),
        "nsd": float(sd.compute_surface_dice_at_tolerance(d, tolerance_mm)),
    }


# --- panopticapi: Kirillov et al.'s own PQ ----------------------------------------

def panopticapi_pq(pred_map: np.ndarray, gt_map: np.ndarray, work_dir: Path) -> float:
    """PQ over the "thing" category, background encoded as a stuff segment.

    panopticapi reads RGB PNGs and COCO-panoptic annotations, so both maps are
    written to `work_dir`. Background must be a real segment (its own id,
    category isthing=0) in both maps: left as VOID (0), panopticapi ignores a
    predicted instance lying on VOID instead of counting it as a false
    positive and removes VOID pixels from the matched union — 0.5 instead of
    0.3 on fixture F4.
    """
    from PIL import Image
    from panopticapi.evaluation import pq_compute_single_core
    from panopticapi.utils import id2rgb

    THING, STUFF = 1, 2
    background_id = int(max(pred_map.max(), gt_map.max())) + 1

    def encode(m: np.ndarray) -> np.ndarray:
        out = m.astype(np.uint32).copy()
        out[out == 0] = background_id
        return out

    def segments(m: np.ndarray) -> list[dict]:
        return [
            {"id": int(i), "category_id": STUFF if i == background_id else THING,
             "area": int((m == i).sum()), "iscrowd": 0}
            for i in np.unique(m)
        ]

    gt_dir, pred_dir = work_dir / "pq_gt", work_dir / "pq_pred"
    gt_dir.mkdir(exist_ok=True); pred_dir.mkdir(exist_ok=True)
    g, p = encode(gt_map), encode(pred_map)
    Image.fromarray(id2rgb(g)).save(gt_dir / "a.png")
    Image.fromarray(id2rgb(p)).save(pred_dir / "a.png")
    categories = {
        THING: {"id": THING, "name": "instance", "isthing": 1},
        STUFF: {"id": STUFF, "name": "background", "isthing": 0},
    }
    gt_ann = {"image_id": "a", "file_name": "a.png", "segments_info": segments(g)}
    pred_ann = {"image_id": "a", "file_name": "a.png", "segments_info": segments(p)}
    stat = pq_compute_single_core(0, [(gt_ann, pred_ann)], str(gt_dir), str(pred_dir), categories)
    result, _ = stat.pq_average(categories, isthing=True)
    return float(result["pq"])


# --- boundary-iou-api: vendored (spec D8) ------------------------------------------
#
# Copyright (c) 2021, Bowen Cheng. All rights reserved. BSD-2-Clause.
# Source: https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py
# Copied verbatim (imports moved into the function) because the package's
# setup.py ships only an empty top-level package and cannot be installed as a
# dependency next to a SHA-pinned panopticapi.

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    import cv2
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou_official(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """|G_d ∩ P_d| / |G_d ∪ P_d| with the reference boundary extraction (2D only)."""
    pb = mask_to_boundary(pred.astype(np.uint8), dilation_ratio).astype(bool)
    gb = mask_to_boundary(gt.astype(np.uint8), dilation_ratio).astype(bool)
    union = int((pb | gb).sum())
    return float("nan") if union == 0 else float((pb & gb).sum()) / union


# --- dispatch -----------------------------------------------------------------------

def official_value(case: CalibrationCase, work_dir: Path) -> Optional[float]:
    """Same-variant official value for `case`, or None when none exists (spec D10)."""
    spacing = case.spacing or ISO
    if case.metric == "panoptic_quality":
        pred, gt = case.build()
        return panopticapi_pq(pred, gt, work_dir)
    pred, gt = _binary(case)
    if case.metric == "dice":
        return medpy_dice(pred, gt)
    if case.metric == "assd":
        symmetric = case.metric_kwargs.get("symmetric", True)
        return medpy_assd(pred, gt, spacing) if symmetric else medpy_asd(pred, gt, spacing)
    if case.metric == "hausdorff95":
        if case.metric_kwargs.get("percentile", 95) is None:
            return medpy_hd(pred, gt, spacing)
        return None            # MedPy hd95 and surface-distance use other variants
    if case.metric == "boundary_iou":
        return boundary_iou_official(pred, gt) if pred.ndim == 2 else None   # 3D physical band: framework extension
    return None                # nsd (author implementation is area-weighted), vs, counts


def divergent_values(case: CalibrationCase, work_dir: Path) -> dict[str, float]:
    """Report-only columns: the same metric under a different definitional variant."""
    if case.metric not in ("hausdorff95", "assd", "nsd") or case.kind == "policy":
        return {}
    spacing = case.spacing or ISO
    pred, gt = _binary(case)
    if not pred.any() or not gt.any():
        return {}
    out: dict[str, float] = {}
    tolerance = float(case.metric_kwargs.get("class_thresholds", [1.0])[0])
    try:
        sd = surface_distance_values(pred, gt, spacing, tolerance)
    except ImportError:
        sd = None
    if case.metric == "hausdorff95" and case.metric_kwargs.get("percentile", 95) == 95:
        try:
            out["MedPy hd95 (P95 over concatenated distances)"] = medpy_hd95(pred, gt, spacing)
        except ImportError:
            pass
        if sd:
            out["surface-distance robust_hausdorff(95) (area-weighted)"] = sd["hd95"]
    if case.metric == "assd" and sd:
        out["surface-distance ASD gt→pred (area-weighted)"] = sd["asd_gt_to_pred"]
        out["surface-distance ASD pred→gt (area-weighted)"] = sd["asd_pred_to_gt"]
    if case.metric == "nsd" and sd:
        out[f"surface-distance surface_dice@{tolerance:g} (area-weighted, author impl.)"] = sd["nsd"]
    return out


def reference_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for dist in ("medpy", "surface-distance"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "not installed"
    out["panopticapi"] = f"git {PANOPTICAPI_SHA[:12]}"
    out["boundary-iou-api"] = f"vendored mask_to_boundary, git {BOUNDARY_IOU_API_SHA[:12]}"
    return out
```

- [ ] **Step 2: Write the three test files**

`tests/calibration/test_official.py`:
```python
"""Layer 2: framework == official implementation of the same variant (needs the [calibration] extras)."""
import numpy as np
import pytest

from calibration.cases import CASES
from calibration.harness import framework_value
from calibration.official import boundary_iou_official, official_value

HAND = [c for c in CASES if c.kind == "hand"]


def _official_or_skip(case, tmp_path):
    try:
        value = official_value(case, tmp_path)
    except ImportError as exc:
        pytest.skip(f"reference package missing: {exc}")
    if value is None:
        pytest.skip(f"no official implementation of the framework's {case.metric} variant")
    return value


@pytest.mark.parametrize("case", HAND, ids=lambda c: c.name)
def test_framework_matches_the_official_implementation(case, tmp_path):
    official = _official_or_skip(case, tmp_path)
    assert framework_value(case, tmp_path) == pytest.approx(official, abs=case.tolerance)


def test_2d_boundary_iou_anchor_matches_the_vendored_reference():
    """The existing 0.3822 anchor (tests/test_boundary_iou.py) against Cheng et al.'s own code."""
    from iqaevaluator.segmentation_metrics.boundary_iou import boundary_iou

    def square(size, offset):
        m = np.zeros((400, 400), bool); m[offset:offset + size, offset:offset + size] = True
        return m
    a, b = square(160, 20), square(160, 25)
    assert boundary_iou(a, b) == pytest.approx(boundary_iou_official(a, b), abs=1e-9)
    assert boundary_iou(a, b) == pytest.approx(0.3822, abs=1e-4)
```

`tests/calibration/test_official_hand.py`:
```python
"""Layer 2b: official implementation == hand value — pins the reference itself."""
import pytest

from calibration.cases import CASES
from calibration.official import official_value

HAND = [c for c in CASES if c.kind == "hand"]


@pytest.mark.parametrize("case", HAND, ids=lambda c: c.name)
def test_official_implementation_reproduces_the_hand_value(case, tmp_path):
    try:
        official = official_value(case, tmp_path)
    except ImportError as exc:
        pytest.skip(f"reference package missing: {exc}")
    if official is None:
        pytest.skip(f"no official implementation of the framework's {case.metric} variant")
    assert official == pytest.approx(case.expected, abs=case.tolerance), case.derivation
```

`tests/calibration/test_random.py`:
```python
"""Layer 5: seeded random blobs, three spacings — framework == MedPy for Dice, ASSD and HD."""
import pytest

from calibration.cases import CASES
from calibration.harness import framework_value
from calibration.official import official_value

RANDOM = [c for c in CASES if c.kind == "random"]


@pytest.mark.parametrize("case", RANDOM, ids=lambda c: c.name)
def test_framework_agrees_with_medpy_on_random_blobs(case, tmp_path):
    try:
        official = official_value(case, tmp_path)
    except ImportError as exc:
        pytest.skip(f"reference package missing: {exc}")
    assert official is not None
    assert framework_value(case, tmp_path) == pytest.approx(official, abs=case.tolerance)
```

- [ ] **Step 3: Run the layers**

Run: `hatch run calibration:test tests/calibration -q -rs`
Expected: all pass; the `-rs` summary lists skips only for `hausdorff95` at percentile 95, `nsd` and the 3D boundary-IoU cases ("no official implementation of the framework's … variant"), never for a missing package (the calibration env has them all). Reference values seen on 2026-09-13: MedPy reproduces every F1/F2 number; panopticapi gives 0.3 on F4; the vendored recipe gives 38/114 on `F5_biou_2d`.

- [ ] **Step 4: Commit**

```bash
git add tests/calibration/official.py tests/calibration/test_official.py tests/calibration/test_official_hand.py tests/calibration/test_random.py
git commit -m "test(calibration): official reference implementations, same variant

MedPy (dice, assd, asd, hd), panopticapi (PQ, background as stuff) and the
vendored boundary-iou-api recipe (BSD-2) — plus seeded random blobs at
three spacings.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 12: MONAI's own test cases through `Mask()` and the adapters (spec Part 3, layer 3)

**Files:**
- Create: `tests/calibration/monai_cases.py`, `tests/calibration/test_monai_cases.py`

**Interfaces:**
- Produces: `monai_cases.MONAI_CASES: list[CalibrationCase]` (`kind="hand"` is not used here; they carry `source` naming the MONAI file and case index) and `monai_cases.create_spherical_seg_3d(...)` (copied from MONAI).
- Consumes: `CalibrationCase`, `framework_value` (Task 10). The MONAI files were fetched from `https://raw.githubusercontent.com/Project-MONAI/MONAI/1.6.0/tests/metrics/<file>` on 2026-09-13; re-fetch them to copy values, do not retype from memory.

Reading MONAI's expected lists (spec, "MONAI cases"): `test_hausdorff_distance.py` expands each case over `product(["euclidean", "chessboard", "taxicab"], [directed=True, directed=False])` → the framework's value (euclidean, undirected) is **index 1**; `test_surface_distance.py` lists `[symmetric=True, symmetric=False]` → ASSD is **index 0**. MONAI runs them with `include_background=False` on a mask repeated over three channels, which equals one channel with `include_background=True`. `seg_1` is `y_pred`, `seg_2` is `y`.

- [ ] **Step 1: Write `monai_cases.py`**

```python
"""Layer 3: MONAI 1.6.0's own metric test cases, through the loader and the adapters.

Fixtures and expected values are copied from
  tests/metrics/test_hausdorff_distance.py, test_surface_distance.py,
  test_surface_dice.py, test_compute_meandice.py, test_compute_panoptic_quality.py
of MONAI 1.6.0 (https://github.com/Project-MONAI/MONAI/tree/1.6.0/tests/metrics).
They prove the adapters pass `percentile`, `symmetric`, `class_thresholds`,
`match_iou_threshold`, `metric_name` and `spacing` (in (D, H, W) order)
through unchanged. Cases where MONAI answers inf/NaN (one or both masks
empty) or 1.0 (both-empty Dice with ignore_empty=False) are asserted per
spec D3 instead and say so in their derivation.

Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

from calibration.cases import ISO, CalibrationCase
from iqaevaluator.normalization import Mask, Raw

MONAI_VERSION = "1.6.0"
TEST_SPACING = (0.85, 1.2, 0.9)          # MONAI's `test_spacing`, per array axis


def create_spherical_seg_3d(
    radius: float = 20.0,
    centre: tuple[int, int, int] = (49, 49, 49),
    im_shape: tuple[int, int, int] = (99, 99, 99),
    im_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return a 3D image with a sphere inside. Voxel values will be 1 inside the sphere, and 0 elsewhere.
    (MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py, verbatim.)"""
    image = np.zeros(im_shape, dtype=np.int32)
    spy, spx, spz = np.ogrid[: im_shape[0], : im_shape[1], : im_shape[2]]
    spy = spy.astype(float) * im_spacing[0]
    spx = spx.astype(float) * im_spacing[1]
    spz = spz.astype(float) * im_spacing[2]
    spy -= centre[0]
    spx -= centre[1]
    spz -= centre[2]
    circle = (spx * spx + spy * spy + spz * spz) <= radius * radius
    image[circle] = 1
    image[~circle] = 0
    return image


def _tol(value: float) -> float:
    """MONAI asserts rtol 1e-5 (1e-6 for HD); an absolute tolerance of that size plus a floor."""
    return abs(value) * 1e-5 + 1e-6


def _sphere_pair(a: dict, b: dict):
    return lambda: (create_spherical_seg_3d(**a), create_spherical_seg_3d(**b))


_ZERO = np.zeros((99, 99, 99), np.int32)
_HD = "MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES[{i}], expected index 1 (euclidean, directed=False)"
_SD = "MONAI 1.6.0 tests/metrics/test_surface_distance.py TEST_CASES[{i}], expected index 0 (symmetric=True, euclidean)"
_NSD = "MONAI 1.6.0 tests/metrics/test_surface_dice.py {t}, expected_res0[0, 1] (foreground class)"
_DICE = "MONAI 1.6.0 tests/metrics/test_compute_meandice.py {t}"
_PQ = "MONAI 1.6.0 tests/metrics/test_compute_panoptic_quality.py {t}"
_DEVIATION = " — MONAI expects {m}; asserted as None per spec D3"


def _c(name, metric, build, expected, source, *, spacing=ISO, mode="volume", normalizer=Mask(), metric_kwargs=None, tolerance=None, derivation=""):
    return CalibrationCase(
        name, metric, build, expected, derivation or "copied from MONAI's test, see source", source,
        spacing=spacing, mode=mode, normalizer=normalizer, metric_kwargs=metric_kwargs or {},
        tolerance=tolerance if tolerance is not None else (_tol(expected) if expected is not None else 0.0),
    )


MONAI_CASES: list[CalibrationCase] = [
    # --- Hausdorff (percentile None unless stated; framework default is 95) ---
    _c("monai_hd_identical", "hausdorff95", _sphere_pair({}, {}), 0.0, _HD.format(i=0), metric_kwargs={"percentile": None}),
    _c("monai_hd_shift_111", "hausdorff95", _sphere_pair(dict(radius=20, centre=(20, 20, 20)), dict(radius=20, centre=(19, 19, 19))),
       1.7320508075688772, _HD.format(i=1), metric_kwargs={"percentile": None}),
    _c("monai_hd_r33_shift_1", "hausdorff95", _sphere_pair(dict(radius=33, centre=(19, 33, 22)), dict(radius=33, centre=(20, 33, 22))),
       1.0, _HD.format(i=2), metric_kwargs={"percentile": None}),
    _c("monai_hd_r20_vs_r40", "hausdorff95", _sphere_pair(dict(radius=20, centre=(20, 33, 22)), dict(radius=40, centre=(20, 33, 22))),
       20.223748416156685, _HD.format(i=3), metric_kwargs={"percentile": None}),
    _c("monai_hd_pred_empty", "hausdorff95", lambda: (_ZERO, create_spherical_seg_3d(radius=40, centre=(20, 33, 22))),
       None, _HD.format(i=4) + _DEVIATION.format(m="inf"), metric_kwargs={"percentile": None}),
    _c("monai_hd_gt_empty", "hausdorff95", lambda: (create_spherical_seg_3d(), _ZERO),
       None, _HD.format(i=5) + _DEVIATION.format(m="inf"), metric_kwargs={"percentile": None}),
    _c("monai_hd95_r20_vs_r40", "hausdorff95", _sphere_pair(dict(radius=20, centre=(20, 33, 22)), dict(radius=40, centre=(20, 33, 22))),
       20.09975124224178, _HD.format(i=6) + " (percentile 95)"),
    _c("monai_hd_spacing_shift_111", "hausdorff95",
       _sphere_pair(dict(radius=20, centre=(20, 20, 20), im_spacing=TEST_SPACING), dict(radius=20, centre=(19, 19, 19), im_spacing=TEST_SPACING)),
       2.2671568, _HD.format(i=7) + " (spacing (0.85, 1.2, 0.9))", spacing=TEST_SPACING, metric_kwargs={"percentile": None}),
    _c("monai_hd_spacing_r15_vs_r30", "hausdorff95",
       _sphere_pair(dict(radius=15, centre=(20, 33, 22), im_spacing=TEST_SPACING), dict(radius=30, centre=(20, 33, 22), im_spacing=TEST_SPACING)),
       15.62594, _HD.format(i=8) + " (spacing (0.85, 1.2, 0.9))", spacing=TEST_SPACING, metric_kwargs={"percentile": None}),
    _c("monai_hd_both_empty", "hausdorff95", lambda: (_ZERO, _ZERO), None,
       "MONAI 1.6.0 tests/metrics/test_hausdorff_distance.py TEST_CASES_NANS[0]" + _DEVIATION.format(m="nan"), metric_kwargs={"percentile": None}),
    # --- Average surface distance, symmetric, euclidean ---
    _c("monai_assd_identical", "assd", _sphere_pair({}, {}), 0.0, _SD.format(i=0)),
    _c("monai_assd_r33_shift_1", "assd", _sphere_pair(dict(radius=33, centre=(19, 33, 22)), dict(radius=33, centre=(20, 33, 22))),
       0.350217, _SD.format(i=2)),
    _c("monai_assd_r20_vs_r40", "assd", _sphere_pair(dict(radius=20, centre=(20, 33, 22)), dict(radius=40, centre=(20, 33, 22))),
       15.117741, _SD.format(i=3)),
    _c("monai_assd_pred_empty", "assd", lambda: (_ZERO, create_spherical_seg_3d(radius=40, centre=(20, 33, 22))),
       None, _SD.format(i=6) + _DEVIATION.format(m="inf")),
    _c("monai_assd_spacing", "assd",
       _sphere_pair(dict(radius=33, centre=(42, 45, 52), im_spacing=TEST_SPACING), dict(radius=33, centre=(43, 45, 52), im_spacing=TEST_SPACING)),
       0.4951, _SD.format(i=8) + " (spacing (0.85, 1.2, 0.9))", spacing=TEST_SPACING, tolerance=1e-5),
    # --- Normalized surface dice, tolerance 0 (no spacing in 2D; isotropic 1.0 in 3D) ---
    _c("monai_nsd_2d_shift_10px", "nsd", _nsd_2d, 1 - (478 + 480 + 9 * 2) / (480 * 4 + 588 * 2 + 578 * 2),
       _NSD.format(t="test_tolerance_euclidean_distance (no spacing)"), mode="slice", metric_kwargs={"class_thresholds": [0.0]}),
    _c("monai_nsd_3d_shift_10vox", "nsd", _nsd_3d,
       1 - (200 * 110 + 198 * 108 + 9 * 200 * 2 + 9 * 108 * 2) / (200 * 110 * 4 + (58 + 48) * 200 * 2 + (58 + 48) * 108 * 2),
       _NSD.format(t="test_tolerance_euclidean_distance_3d"), metric_kwargs={"class_thresholds": [0.0]}),
    # --- Dice ---
    _c("monai_dice_case_1", "dice", lambda: (np.array([[1, 0], [0, 1]], np.int16), np.array([[1, 0], [1, 1]], np.int16)),
       0.8, _DICE.format(t="TEST_CASE_1"), mode="slice"),
    _c("monai_dice_case_12_gt_empty", "dice", lambda: (np.ones((3, 3), np.int16), np.zeros((3, 3), np.int16)),
       0.0, _DICE.format(t="TEST_CASE_12 (ignore_empty=False)"), mode="slice"),
    _c("monai_dice_case_11_both_empty", "dice", lambda: (np.zeros((2, 3, 3), np.int16), np.zeros((2, 3, 3), np.int16)),
       None, _DICE.format(t="TEST_CASE_11 (ignore_empty=False)") + _DEVIATION.format(m="1.0")),
    # --- Panoptic quality on integer instance maps (Raw), slice mode ---
    _c("monai_pq_func_case_2", "panoptic_quality", lambda: (_PQ_PRED, _PQ_GT), 0.25, _PQ.format(t="TEST_FUNC_CASE_2"),
       mode="slice", normalizer=Raw(), metric_kwargs={"match_iou_threshold": 0.5}, tolerance=1e-4),
    _c("monai_pq_func_case_3_sq", "panoptic_quality", lambda: (_PQ_PRED, _PQ_GT), 0.6, _PQ.format(t="TEST_FUNC_CASE_3"),
       mode="slice", normalizer=Raw(), metric_kwargs={"metric_name": "sq", "match_iou_threshold": 0.3}, tolerance=1e-4),
    _c("monai_pq_func_case_4_rq_remap", "panoptic_quality", lambda: (_PQ_PRED_REMAP, _PQ_GT), 0.75, _PQ.format(t="TEST_FUNC_CASE_4"),
       mode="slice", normalizer=Raw(), metric_kwargs={"metric_name": "RQ", "match_iou_threshold": 0.3}, tolerance=1e-4),
]
```
with these module-level fixtures placed **above** `MONAI_CASES` (they are referenced there):
```python
def _nsd_2d():
    pred = np.zeros((480, 640), np.int16); pred[:, 50:] = 1
    gt = np.zeros((480, 640), np.int16); gt[:, 60:] = 1          # 10 px shift
    return pred, gt


def _nsd_3d():
    pred = np.zeros((200, 110, 80), np.int16); pred[:, :, 20:] = 1
    gt = np.zeros((200, 110, 80), np.int16); gt[:, :, 30:] = 1    # offset by 10
    return pred, gt


_PQ_PRED = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [2, 0, 3, 3], [4, 2, 2, 0]], np.int16)
_PQ_PRED_REMAP = np.array([[0, 7, 7, 7], [0, 0, 0, 0], [1, 0, 8, 8], [9, 1, 1, 0]], np.int16)
_PQ_GT = np.array([[1, 1, 2, 1], [0, 0, 0, 0], [1, 3, 0, 0], [4, 3, 3, 3]], np.int16)
```
Note the both-empty Dice case is a 2-slice volume: in volume mode the policy answers `None`; in slice mode the evaluator would skip both slices and no metric would run (that path is covered by `test_f3_slice_mode_skips_only_slices_empty_on_both_sides`).

- [ ] **Step 2: Write `test_monai_cases.py`**

```python
"""Layer 3: framework == MONAI's own expected values, through Mask()/Raw() and the adapters."""
import pytest

from calibration.harness import framework_value
from calibration.monai_cases import MONAI_CASES


@pytest.mark.parametrize("case", MONAI_CASES, ids=lambda c: c.name)
def test_framework_reproduces_monais_expected_value(case, tmp_path):
    value = framework_value(case, tmp_path)
    if case.expected is None:
        assert value is None, case.source
    else:
        assert value == pytest.approx(case.expected, abs=case.tolerance), case.source
```

- [ ] **Step 3: Run the layer**

Run: `hatch run calibration:test tests/calibration/test_monai_cases.py -q --durations=5`
Expected: all pass in well under two minutes (the 99³ spheres dominate). Spot values reproduced on 2026-09-13 with the functional API on one channel: HD shift (19,19,19) = 1.7320508075688772; HD at `TEST_SPACING` = 2.2671568; both NSD formulas match `compute_surface_dice` to < 1e-7; Dice TEST_CASE_1 = 0.8.

- [ ] **Step 4: Commit**

```bash
git add tests/calibration/monai_cases.py tests/calibration/test_monai_cases.py
git commit -m "test(calibration): MONAI 1.6.0 metric test cases through the loader

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 13: Report generator, `docs/calibration/segmentation.md`, documentation (spec Part 3 "report.py", "Documentation")

**Files:**
- Create: `tests/calibration/report.py`, `tests/calibration/test_report.py`, `docs/calibration/segmentation.md` (generated)
- Modify: `docs/wiki-followup-kalibrierung.md`, `CLAUDE.md` (Architecture + Metric summary), `README.md:104`

**Interfaces:**
- Produces: `report.render(cases=CASES, monai_cases=MONAI_CASES) -> str` (Markdown) and `report.main()` writing `docs/calibration/segmentation.md`; Hatch script `hatch run calibration:report` (Task 1).
- Consumes: `CASES`, `MONAI_CASES`, `framework_value`, `official_value`, `divergent_values`, `reference_versions`.

- [ ] **Step 1: Write `report.py`**

```python
"""Render docs/calibration/segmentation.md from the calibration cases.

Run with `hatch run calibration:report`. Needs the [calibration] extras for
the official columns; a missing package shows up as "n/a", never as a
silent blank. One table per metric: case | hand value | derivation |
official (package@version, variant) | framework | Δ | divergent variants.
"""
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))          # `calibration.*` when run as a script

from calibration.cases import CASES, CalibrationCase                      # noqa: E402
from calibration.harness import framework_value                          # noqa: E402
from calibration.monai_cases import MONAI_CASES, MONAI_VERSION           # noqa: E402
from calibration.official import divergent_values, official_value, reference_versions  # noqa: E402

OUT = ROOT / "docs" / "calibration" / "segmentation.md"
ORDER = ["dice", "hausdorff95", "assd", "nsd", "panoptic_quality", "boundary_iou", "vs"]
VARIANT = {
    "dice": "MedPy `dc` — voxel Dice",
    "hausdorff95": "MedPy `hd` — max over directed maxima (only at percentile=None)",
    "assd": "MedPy `assd`/`asd` — voxel surfaces, cross-structure erosion",
    "nsd": "— (author implementation `surface-distance` is area-weighted, see ¹)",
    "panoptic_quality": "panopticapi `pq_compute_single_core` — background as stuff, things-only PQ",
    "boundary_iou": "boundary-iou-api `mask_to_boundary` (vendored) — 2D only",
    "vs": "— (no reference implementation; counts-based)",
}


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.6f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _framework_versions() -> dict[str, str]:
    import monai, numpy, pyiqa, scipy, torch
    return {"monai": monai.__version__, "pyiqa": pyiqa.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "numpy": numpy.__version__}


def _official_text(case: CalibrationCase, tmp: Path) -> str:
    try:
        value = official_value(case, tmp)
    except ImportError as exc:
        return f"n/a ({exc.name} not installed)"
    if value is None:
        return "— (no same-variant implementation)"
    return f"{_fmt(value)} — {VARIANT.get(case.metric, '')}"


def _divergent_text(case: CalibrationCase, tmp: Path) -> str:
    try:
        values = divergent_values(case, tmp)
    except ImportError as exc:
        return f"n/a ({exc.name} not installed)"
    return "<br>".join(f"{k}: {_fmt(v)}" for k, v in values.items()) or "—"


def render(cases: list[CalibrationCase] = CASES, monai_cases: list[CalibrationCase] = MONAI_CASES) -> str:
    versions = ", ".join(f"{k} {v}" for k, v in _framework_versions().items())
    refs = ", ".join(f"{k} {v}" for k, v in reference_versions().items())
    lines = [
        "# Segmentation-metric calibration report",
        "",
        f"Generated {date.today().isoformat()} by `hatch run calibration:report` at framework commit `{_git_sha()}`.",
        f"Framework stack: {versions}. Reference packages: {refs}.",
        "",
        "Authority, in order: (1) the metric's definition applied by hand to a fixture small enough for the derivation "
        "to fit in a few lines — the derivation is what a reviewer checks; (2) the implementation the community cites, "
        "run on the same fixture with the same definitional variant; (3) MONAI's own test cases through this framework's "
        "loader and adapters. Δ = framework − hand value. Every framework value went through the public path "
        "(NIfTI → `load_pair` → `MetricRegistry` → evaluator → record).",
        "",
    ]
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for metric in ORDER:
            hand = [c for c in cases if c.metric == metric and c.kind == "hand"]
            policy = [c for c in cases if c.metric == metric and c.kind == "policy"]
            random_cases = [c for c in cases if c.metric == metric and c.kind == "random"]
            monai = [c for c in monai_cases if c.metric == metric]
            if not (hand or policy or random_cases or monai):
                continue
            lines += [f"## {metric}", ""]
            if hand:
                lines += ["| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |",
                          "|---|---|---|---|---|---|---|"]
                for c in hand:
                    fw = framework_value(c, tmp)
                    delta = "—" if fw is None else f"{fw - c.expected:+.2e}"
                    lines.append(f"| {c.name} | {_fmt(c.expected)} | {_cell(c.derivation)}<br>*{_cell(c.source)}* | "
                                 f"{_official_text(c, tmp)} | {_fmt(fw)} | {delta} | {_divergent_text(c, tmp)} |")
                lines.append("")
            if policy:
                lines += ["Empty-mask policy (spec D3):", "", "| case | expected | framework |", "|---|---|---|"]
                for c in policy:
                    lines.append(f"| {c.name} | {_fmt(c.expected)} | {_fmt(framework_value(c, tmp))} |")
                lines.append("")
            if random_cases:
                lines += ["Seeded random blobs — framework vs MedPy (same variant):", "",
                          "| case | spacing | framework | MedPy | Δ |", "|---|---|---|---|---|"]
                for c in random_cases:
                    fw = framework_value(c, tmp)
                    try:
                        off = official_value(c, tmp)
                    except ImportError:
                        off = None
                    delta = "—" if fw is None or off is None else f"{fw - off:+.2e}"
                    lines.append(f"| {c.name} | {c.spacing} | {_fmt(fw)} | {_fmt(off)} | {delta} |")
                lines.append("")
            if monai:
                lines += [f"MONAI {MONAI_VERSION} test cases through `Mask()`/`Raw()` and the adapters:", "",
                          "| case | MONAI expected | framework | source |", "|---|---|---|---|"]
                for c in monai:
                    lines.append(f"| {c.name} | {_fmt(c.expected)} | {_fmt(framework_value(c, tmp))} | {_cell(c.source)} |")
                lines.append("")
    lines += [
        "¹ The same metric under a different definitional variant (spec D10), reported for orientation and never "
        "used in an equality assertion: MedPy `hd95` takes the 95th percentile over the *concatenated* directed "
        "distances (MONAI/this framework: max of the two directed percentiles); DeepMind's `surface-distance` weights "
        "surface elements by area (MONAI/this framework: one surface voxel = one element). Boundary IoU's 3D physical "
        "band is this framework's extension of Cheng et al. and has a hand value only.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `test_report.py`**

```python
"""The report renders (on a two-case subset, to stay fast) and holds no blanks."""
from calibration.cases import CASES
from calibration.monai_cases import MONAI_CASES
from calibration.report import render


def test_render_produces_the_tables():
    subset = [next(c for c in CASES if c.name == "F1_dice"), next(c for c in CASES if c.name == "F3_dice_both_empty")]
    monai = [next(c for c in MONAI_CASES if c.name == "monai_dice_case_1")]
    text = render(subset, monai)
    assert "# Segmentation-metric calibration report" in text
    assert "## dice" in text and "| F1_dice | 0.833333 |" in text
    assert "| F3_dice_both_empty | — | — |" in text
    assert "| monai_dice_case_1 | 0.800000 | 0.800000 |" in text
    assert "TBD" not in text and "| None |" not in text and "| nan |" not in text
```

- [ ] **Step 3: Generate the report and update the documentation**

Run: `hatch run calibration:test tests/calibration/test_report.py -q && hatch run calibration:report`
Expected: `written .../docs/calibration/segmentation.md`; open it and check every hand-value row shows `Δ` of magnitude ≤ 1e-6 (≤ 2e-7 for `F4_pq`) and no `n/a` cell.

`docs/wiki-followup-kalibrierung.md`:
- Point 5: `match_threshold` → `match_iou_threshold`; add "Passing `threshold=`/`label=` to any builder raises `TypeError`."
- Point 9: append "Der `panopticapi`-Vergleich kodiert den Hintergrund als Stuff-Segment (eigene Kategorie, `isthing=0`) und mittelt nur über Things — als VOID (0) ignoriert panopticapi FP-Instanzen auf dem Hintergrund und liefert 0.5 statt 0.3 auf F4."
- Point 10, layer 2: "`boundary-iou-api` ist nicht als Paket installierbar (leeres Top-Level-Paket, ungepinnte panopticapi-URL); `mask_to_boundary` ist in `tests/calibration/official.py` vendored (BSD-2, SHA 37d2558)."
- Point 12: replace "(surface-distance, medpy, panopticapi@sha, boundary-iou-api@sha)" with "(surface-distance==0.1, medpy==0.5.2, panopticapi@7bb4655)" and add "pyiqa deklariert pytest/ruff/pre-commit/yapf/tensorboard selbst als Laufzeit-Abhängigkeiten — sie bleiben in `.venv` transitiv installiert; entfernt sind nur unsere direkten Pins und `tests/` aus dem sdist."
- Under "Noch offen" add: "`docs/calibration/segmentation.md` ist generiert und committed — im Wiki verlinken, nicht kopieren."

`CLAUDE.md`:
- Architecture: add an entry `**src/iqaevaluator/metric_spec.py**` — "leaf module holding `MetricSpec`, `Metric`, `ModeSupport`/`ModeUnsupported`, `SkippedMetric`, `Spacing`, `DEVICE` and the `REASON_*` strings; `metrics.py` re-exports them. Metric modules import from here, never from `metrics.py`, so there is no import cycle."
- `normalization.py` entry: add `Mask(label=None, threshold=0.5)` ("binarize at load time — the only valid way to load masks for the binary segmentation metrics; PNG 0/255 and NIfTI 0/1 are equivalent; `Mask(label=k)` scores one class") and describe `Raw()` as "escape hatch: instance maps for panoptic quality".
- `image_loader.py` entry: "`.tensor` … exactly {0, 1} under `Mask()`; `.empty_slice_mask` is the strategy's (`sparse_slices` heuristic for intensities, exact count under `Mask()`)"; `load_pair` "couples the input to the target's range only when the strategy produced one".
- Segmentation metric paragraph (after the metric summary table): replace "Load masks with `Raw()`; PNG 0/255 masks with `MinMax()`." with "Load masks with `Mask()` — PNG 0/255 and NIfTI 0/1 are equivalent, `Mask(label=k)` selects one class. The adapters reject non-binary input; `threshold`/`label` are not builder parameters. Empty masks: one side empty → 0.0 (dice, vs, nsd, panoptic_quality, boundary_iou) or None (hausdorff95, assd); both empty → None. `Raw()` is only for panoptic-quality instance maps. Calibration evidence: `docs/calibration/segmentation.md` (regenerate with `hatch run calibration:report`)."

`README.md:104`: replace "or `Raw()` (no scaling, for masks)" with "`Mask()` (binarize at load time — the way to load segmentation masks) or `Raw()` (no scaling; instance maps for panoptic quality)".

- [ ] **Step 4: Run the whole suite**

Run: `hatch run calibration:test -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/calibration/report.py tests/calibration/test_report.py docs/calibration/segmentation.md docs/wiki-followup-kalibrierung.md CLAUDE.md README.md
git commit -m "docs(calibration): generated segmentation report and Mask() documentation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Acceptance — mutation checks (spec "Acceptance")

**Files:** none change permanently. Each mutation is applied, one test command run and expected to FAIL, then the file is restored with `git checkout -- <file>`.

- [ ] **Step 1: `percentile=95` default removed**

Edit `src/iqaevaluator/segmentation_metrics/monai_metrics.py`, `hausdorff95_metric`: `defaults={"include_background": True, "percentile": 95}` → `defaults={"include_background": True}`.
Run: `hatch run calibration:test "tests/calibration/test_hand.py::test_framework_reproduces_the_hand_value[F2_hd95]" -q`
Expected: FAIL (5.0 ≠ 0.0). Restore: `git checkout -- src/iqaevaluator/segmentation_metrics/monai_metrics.py`

- [ ] **Step 2: Dice replaced by IoU**

Same file, `dice_metric`: `_monai_spec("dice", compute_dice, ...)` → `_monai_spec("dice", compute_iou, ...)` with `from monai.metrics import compute_iou` added.
Run: `hatch run calibration:test "tests/calibration/test_hand.py::test_framework_reproduces_the_hand_value[F1_dice]" -q`
Expected: FAIL (0.714286 ≠ 0.833333). Restore the file.

- [ ] **Step 3: `symmetric=True` default removed**

Same file, `average_surface_distance_metric`: `defaults={"include_background": True, "symmetric": True}` → `defaults={"include_background": True}`.
Run: `hatch run calibration:test "tests/calibration/test_hand.py::test_framework_reproduces_the_hand_value[F2_assd]" -q`
Expected: FAIL (0.087719 ≠ 0.044248). Restore the file.

- [ ] **Step 4: Band floor back to 1.0 mm**

Edit `src/iqaevaluator/segmentation_metrics/boundary_iou.py`, `band_width`: `max(float(spacing_arr.max()), ...)` → `max(1.0, ...)`.
Run: `hatch run calibration:test tests/calibration/test_hand.py -q -k "F5_biou_D_shift or F5_biou_W_shift or 3mm_through_plane"`
Expected: three FAILs (0.6 ≠ 0.295775, 0.333333 ≠ 0.852349, and the order assertion). Restore the file.

- [ ] **Step 5: Spacing tuple reversed in `_volume_factory`**

Edit `monai_metrics.py`, `_volume_factory.build`: `kwargs.setdefault("spacing", list(spacing))` → `kwargs.setdefault("spacing", list(spacing)[::-1])`.
Run: `hatch run calibration:test tests/calibration -q -k "F1_aniso or monai_hd_spacing or monai_assd_spacing"`
Expected: FAIL for `F1_aniso_hd95` (1.0 ≠ 3.0), `F1_aniso_assd`, `F1_aniso_nsd` and the MONAI spacing cases. Restore the file.

- [ ] **Step 6: `Mask.apply` binarizing with `== 1` instead of `!= 0`**

Edit `src/iqaevaluator/segmentation_metrics/volume.py`, `as_mask`: `return x != 0 if label is None else x == label` → `return x == 1 if label is None else x == label`.
Run: `hatch run calibration:test "tests/calibration/test_hand.py::test_framework_reproduces_the_hand_value[F6_dice_any_label]" "tests/test_image_loader.py::TestMaskLoading::test_png_0_255_and_nifti_0_1_give_identical_tensors" -q`
Expected: two FAILs (1.0 ≠ 0.777778; the PNG tensor sums to 0). Restore the file.

- [ ] **Step 7: Confirm a clean tree and a green suite**

Run: `git status --short && hatch run calibration:test -q`
Expected: no modified files; all tests pass. Record the six outcomes in the final PR description or hand-off note.

---

## Self-review notes (already applied while writing)

- **Spec coverage.** Part 0 → Task 1. Part 1 (`sparse_slices`, protocol, `Mask`, `as_mask`, loader, `load_pair`, `is_empty`, docstrings, tests) → Tasks 2–4. Part 2 (helpers, MONAI adapters, `_monai_spec`, PQ, `VolumeFunctionMetric`, Boundary IoU floor, import cycle, `Spacing`, cross-check F6, unit tests incl. `TypeError` on removed knobs and no-`inf` report) → Tasks 5–9. Part 3 (files, `CalibrationCase`, public-path harness, F1–F7, MONAI cases, report, variants table) → Tasks 10–13. Documentation → Tasks 1 and 13. Acceptance mutations → Task 14.
- **Names used across tasks.** `sparse_slices`, `Mask`, `as_mask(x, label=None, threshold=0.5)`, `require_binary(t, *, metric)`, `foreground_counts`, `empty_policy(n_pred, n_gt, *, one_sided)`, `NOT_EMPTY`, `MonaiSegmentationMetric(compute_fn, *, one_sided, name=None, **monai_kwargs)`, `_monai_spec(name, compute_fn, *, direction, one_sided, uses_spacing, description, defaults, **monai_kwargs)`, `MonaiPanopticQualityMetric(**monai_kwargs)`, `VolumeFunctionMetric(fn)`, `BoundaryIoUMetric(*, dilation_ratio, spacing)`, `band_width`, `CalibrationCase(..., kind)`, `CASES`, `write_nifti`, `framework_records`, `framework_value`, `official_value`, `divergent_values`, `MONAI_CASES`, `render`.
- **Known deviations from the spec, deliberate.** `MonaiSegmentationMetric` takes an extra `name=` so error messages name the metric; `CalibrationCase` has a `kind` field and there is an extra `harness.py` plus `test_report.py`; the cross-metric policy tests live in a new `tests/test_mask_policy.py` rather than being spread over three files.
