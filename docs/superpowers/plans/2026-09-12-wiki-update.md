# Wiki Update After Normalization, Packaging and Module Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the 15 published wiki pages in line with the framework on `volumetric_similarity`, and add a `Normalization.md` page for the subsystem the wiki has never documented.

**Architecture:** The wiki is a separate GitHub repository (`…IQA_MRI.wiki.git`) of flat Markdown pages, cloned into the scratchpad. Work is surgical: a verification script first produces real output for every model-free example, then each page is edited against that captured output and against `src/iqaevaluator/`. Each task ends with an executable check — a grep that must go from hits to zero, or a script whose output must match what a page claims.

**Tech Stack:** Markdown (GitHub wiki), Python 3.14 in `.venv`, Hatch + uv, nibabel/numpy for the synthetic fixtures, pandas for the report examples.

**Spec:** `docs/superpowers/specs/2026-09-12-wiki-update-design.md`

## Global Constraints

- Wiki language is **English**, prose and code alike.
- **Surgical diffs only.** Prose that is still correct is not rewritten. Do not reflow paragraphs you are not fixing — it destroys the reviewable diff.
- **Never documented, anywhere, not even in passing:** `src/main.py`, `main.evaluate()`, `src/iqaevaluator/constants.py` (and the names `INPUT`, `TARGET`, `REPORT`), `src/evaluation.ipynb`, `iqaevaluator.data`. Total silence — no "reference implementation" note, no "scheduled for replacement" note.
- Every import in every code block is package-qualified: `from iqaevaluator.<module> import …`. No bare `from metrics import …`.
- Every **full-reference** example uses `load_pair(...)`. Two independent `ImageLoader`s appear exactly once in the whole wiki, on `Normalization.md`, labelled as the trap.
- Install instructions are Hatch-primary, uv-fallback. `requirements.txt` does not exist and must not be named.
- Exact pins, copied verbatim where quoted: `setuptools==80.10.2` (must stay `<81`; `openai-clip` imports `pkg_resources`, removed in setuptools 81) and `pyiqa==0.1.15.post2` (pinned because metric outputs are backend-version-dependent).
- Weight paths are written literally: `models/RadImageNet_pytorch/ResNet50.pt`, `models/dreamsim/`.
- Model-backed examples (`dreamsim`, `musiq`, `maniqa`, `ilniqe`) stay unexecuted snippets and are marked as such. Everything else shows **captured real output**.
- **Do not push.** Work is committed in the scratchpad clone only. The push happens after the user approves the diff (Task 9).
- Wiki commits end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Working paths

Every task's bash blocks assume these two variables. Set them at the start of each block.

```bash
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
```

If `$WIKI` is empty, the clone is gone (a new session has a new scratchpad). Re-create it:

```bash
SCRATCH=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad | head -1)
git clone git@github.com:glebjan/hka.projektarbeit1.IQA_MRI.wiki.git "$SCRATCH/wiki"
```

## File structure

| File | Responsibility |
|---|---|
| `$WIKI/Normalization.md` | **New.** The `Normalizer` protocol, the four strategies with a recommendation, `load_pair`, the affine-blindness trap, `normalizer_from_name`, the five report columns |
| `$WIKI/Installation.md` | Hatch/uv install, weight locations, device selection. Loses the "there is no package" section |
| `$WIKI/Troubleshooting.md` | The pins, the import error, the corrected empty-slice entry |
| `$WIKI/Interpreting-Scores.md` | Caveats only. Loses the `TODO(norm-N)` paragraph, gains the `load_pair` reframing |
| `$WIKI/Loading-Images.md` | Formats, tensors, pairing helpers. Normalization shrinks to a pointer |
| `$WIKI/Segmentation-Metrics.md`, `$WIKI/Configuring-Metrics.md` | Mask loading with `Raw()`, corrected `threshold` guidance |
| `$WIKI/Results-and-Reports.md` | Eleven identifying columns, re-captured frame and CSV header |
| `$WIKI/Home.md`, `Core-Concepts.md`, `Registering-Metrics.md`, `Running-an-Evaluation.md`, `Custom-Metrics.md`, `Metric-Catalog.md`, `Scoring-Modes.md` | Imports, `load_pair`, exclusions |
| `$WIKI/_Sidebar.md` | Sixteen entries |
| `$SCRATCH/verify_wiki_examples.py` | Throwaway. Builds the fixtures, runs every model-free example, prints labelled blocks |
| `$SCRATCH/captured.txt` | The script's output. The source of every number the wiki prints |

No file under `$FW/src/` or `$FW/tests/` is modified by this plan.

---

### Task 1: Verification harness and captured output

**Files:**
- Create: `$SCRATCH/verify_wiki_examples.py`
- Create: `$SCRATCH/captured.txt`

**Interfaces:**
- Consumes: the installed `iqaevaluator` package in `$FW/.venv`.
- Produces: `captured.txt` with blocks delimited by lines of the form `=== <block-name> ===`. Later tasks quote from these blocks by name: `minimal-example`, `norm-minmax-vs-percentile`, `affine-trap`, `custom-metric`, `to-frame`, `csv-header`, `aggregate-volumes`, `select-skips`, `dreamsim-names`.

- [ ] **Step 1: Write the harness**

```python
# $SCRATCH/verify_wiki_examples.py
"""Throwaway: runs every model-free wiki example and prints its real output."""
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from iqaevaluator.evaluation_result import EvaluationResult
from iqaevaluator.evaluator_factory import build_evaluator
from iqaevaluator.image_loader import ImageLoader, load_pair
from iqaevaluator.metrics import LPIPS, PSNR, SSIM, MetricRegistry
from iqaevaluator.segmentation_metrics.monai_metrics import DICE
from iqaevaluator.segmentation_metrics.volume_metrics import VS, VS_SIGNED
from iqaevaluator.normalization import MinMax, Percentile, Raw
from iqaevaluator.dreamsim_metric import dreamsim_spec

TMP = Path(tempfile.mkdtemp())


def block(name: str) -> None:
    print(f"\n=== {name} ===")


def write_volume(path: Path, data: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    return path


rng = np.random.default_rng(0)
base = rng.random((8, 32, 32)) * 900 + 100          # 8 slices, range ~[100, 1000]
noisy = np.clip(base + rng.normal(0, 40, base.shape), 0, None)
spike = base.copy()
spike[0, 0, 0] = 50_000                              # one outlier voxel
bright = base * 2.0 + 500                            # uniform affine bias

target_p = write_volume(TMP / "target.nii.gz", base)
input_p = write_volume(TMP / "input.nii.gz", noisy)
spike_p = write_volume(TMP / "spike.nii.gz", spike)
bright_p = write_volume(TMP / "bright.nii.gz", bright)

registry = MetricRegistry(PSNR, SSIM)

block("minimal-example")
inp, tgt = load_pair(input_p, target_p)
records = build_evaluator(inp, tgt, registry, mode="slice").run_evaluation()
print(f"{len(records)} records, first: psnr={records[0].psnr:.3f} ssim={records[0].ssim:.4f}")
print(f"normalization={records[0].normalization} scale=({records[0].scale_lo:.1f}, {records[0].scale_hi:.1f})")

block("norm-minmax-vs-percentile")
for norm in (MinMax(), Percentile(0.5, 99.5)):
    i, t = load_pair(input_p, spike_p, norm)
    r = build_evaluator(i, t, registry, mode="slice").run_evaluation()[0]
    print(f"{r.normalization:<22} scale=({r.scale_lo:8.1f}, {r.scale_hi:9.1f})  psnr={r.psnr:7.3f}  ssim={r.ssim:.4f}")

block("affine-trap")
i_ind, t_ind = ImageLoader(bright_p), ImageLoader(target_p)
r_ind = build_evaluator(i_ind, t_ind, registry, mode="slice").run_evaluation()[0]
i_pair, t_pair = load_pair(bright_p, target_p)
r_pair = build_evaluator(i_pair, t_pair, registry, mode="slice").run_evaluation()[0]
print(f"independent loaders: psnr={r_ind.psnr:7.3f}  ssim={r_ind.ssim:.4f}")
print(f"load_pair:           psnr={r_pair.psnr:7.3f}  ssim={r_pair.ssim:.4f}")
print(f"input_max={r_pair.input_max:.1f} > scale_hi={r_pair.scale_hi:.1f}  -> overshoot was clipped")

block("volume-mode")
i, t = load_pair(input_p, target_p)
vrec = build_evaluator(i, t, registry, mode="volume").run_evaluation()
print(f"{len(vrec)} record, scoring={vrec[0].scoring}, slice_index={vrec[0].slice_index}, psnr={vrec[0].psnr:.3f}")


class MeanIntensity:
    """Custom metric: the mean of the input batch, per slice."""
    def __call__(self, input, target=None):
        return input.mean(dim=(1, 2, 3)).tolist()


block("custom-metric")
custom = MetricRegistry(PSNR)
custom.register_metric("mean_intensity", MeanIntensity(), direction="higher", reference=False, channels="gray")
i, t = load_pair(input_p, target_p)
crec = build_evaluator(i, t, custom, mode="slice").run_evaluation()
print(f"extra={ {k: round(v, 4) for k, v in crec[0].extra.items()} }")

block("to-frame")
result = EvaluationResult.from_records(crec, custom)
df = result.to_frame()
print(f"df.shape = {df.shape}")
print(list(df.columns))
print(df[["image_id", "scoring", "slice_index", "normalization", "psnr", "mean_intensity"]].head(3).to_string(index=False))

block("csv-header")
csv_path = TMP / "report.csv"
df.to_csv(csv_path, index=False)
print(csv_path.read_text().splitlines()[0])

block("select-skips")
# PSNR and SSIM have a volume implementation; LPIPS does not — that is the point.
selected, skipped = MetricRegistry(PSNR, SSIM, LPIPS).select("volume")
print("selected:", [s.name for s in selected])
for s in skipped:
    print(f"skipped: {s.name} — {s.reason}")

block("aggregate-volumes")
mask_gt = (base > base.mean()).astype(np.uint8)
mask_pred = (noisy > base.mean()).astype(np.uint8)
gt_p, pred_p = TMP / "gt.nii.gz", TMP / "pred.nii.gz"
nib.save(nib.Nifti1Image(mask_gt, np.eye(4)), gt_p)
nib.save(nib.Nifti1Image(mask_pred, np.eye(4)), pred_p)
seg = MetricRegistry(DICE, VS, VS_SIGNED)
seg_records = build_evaluator(
    ImageLoader(pred_p, Raw()), ImageLoader(gt_p, Raw()), seg, mode="slice"
).run_evaluation()
seg_result = EvaluationResult.from_records(seg_records, seg)
print(seg_result.aggregate_volumes().to_string())

block("dreamsim-names")
print(dreamsim_spec().name)
print(dreamsim_spec(dreamsim_type="dino_vitb16").name)
print(dreamsim_spec(pretrained=False).name)
print(dreamsim_spec(normalize_embeds=False).name)

block("raw-mask")
mask = (base > base.mean()).astype(np.uint8)
mask_p = TMP / "mask.nii.gz"
nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_p)   # uint8 on disk, NOT cast to float32
print("Raw():   dtype", ImageLoader(mask_p, Raw()).tensor.dtype, "unique", torch.unique(ImageLoader(mask_p, Raw()).tensor).tolist())
print("MinMax():dtype", ImageLoader(mask_p, MinMax()).tensor.dtype, "unique", torch.unique(ImageLoader(mask_p, MinMax()).tensor).tolist())
```

- [ ] **Step 2: Run it and capture**

```bash
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
SCRATCH=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad | head -1)
cd "$FW" && source .venv/bin/activate && python "$SCRATCH/verify_wiki_examples.py" 2>&1 | tee "$SCRATCH/captured.txt"
```

Expected: every block prints. `norm-minmax-vs-percentile` must show **two different psnr values** — if minmax and percentile agree to three decimals the spike fixture is not doing its job; raise the spike to `500_000` and re-run. `affine-trap` must show independent-loader psnr far **higher** than `load_pair` psnr — that is the whole point of the block.

- [ ] **Step 3: Sanity-check the numbers before anything is quoted**

Read `captured.txt` in full. Any `nan`, `inf`, `None`, or a traceback means the harness is wrong and must be fixed before any page cites it. No wiki edit may quote a number that is not in this file.

- [ ] **Step 4: Commit (framework repo — the harness is a throwaway, so nothing is committed here)**

No commit. `$SCRATCH` is outside the repository by design. Record in the task report which blocks were produced.

---

### Task 2: Installation.md and Troubleshooting.md

**Files:**
- Modify: `$WIKI/Installation.md`
- Modify: `$WIKI/Troubleshooting.md`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: the install commands every other page assumes (`hatch env create`, editable install), and the `Troubleshooting` anchors other pages link to.

- [ ] **Step 1: Prove the pages are wrong**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -n "requirements.txt\|pip install -r\|There is no package\|from metrics import" "$WIKI/Installation.md"
```

Expected: hits on all four (the current page installs from a deleted file and tells the reader there is no package).

- [ ] **Step 2: Replace the environment section of `Installation.md`**

Replace everything from `## Set up the environment` down to the end of the `## There is no package — run from src/` section with:

````markdown
## Set up the environment

The project uses [Hatch](https://hatch.pypa.io) as both build backend and
environment manager, with [uv](https://docs.astral.sh/uv/) as the installer.
`pyproject.toml` is the single source of truth for dependency versions.

```bash
cd <project-root>
hatch env create
```

`[tool.hatch.envs.default] path = ".venv"` pins that environment to the
project-local `.venv/`, so `hatch run …` and a plain `source .venv/bin/activate`
operate on the same interpreter. Confirm with `hatch env find default`.

Without Hatch:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Either way the package is installed **editable**, so `import iqaevaluator`
works from any working directory and edits in `src/` take effect immediately.

Verify:

```bash
hatch run test
python -c "from iqaevaluator.metrics import DEVICE, BUILTIN_METRICS; print(DEVICE, len(BUILTIN_METRICS))"
```

```
cpu 18
```

If installation fails, read [Troubleshooting](Troubleshooting) before changing
any pin — the common failures (`pkg_resources`, CUDA wheels, `numba`) all have
known causes.

## Model weights

Two metrics need weights on disk:

| Metric | File | How it arrives |
|---|---|---|
| `radimagenet_lpips` | `models/RadImageNet_pytorch/ResNet50.pt` | downloaded manually, once |
| `dreamsim` | `models/dreamsim/` | downloaded automatically on first use |

Every other metric downloads its weights through pyiqa on first use.
````

Keep the existing `## Requirements` and device-selection sections. Update the Python line to `Python 3.12–3.14 (`requires-python = ">=3.12"`; the pinned set was resolved against 3.14.6)`.

- [ ] **Step 3: Fix the three Troubleshooting entries**

In `$WIKI/Troubleshooting.md`:

1. The `setuptools` / `pkg_resources` entry: the fix is no longer `pip install "setuptools<81"` into a frozen requirements file — `pyproject.toml` pins `setuptools==80.10.2` and that pin must stay `<81` because `openai-clip` imports `pkg_resources`, removed in setuptools 81. Add: re-create with `hatch env prune && hatch env create` rather than hand-patching the venv.
2. Add a new entry, placed directly after the install entries:

````markdown
### `ModuleNotFoundError: No module named 'iqaevaluator'`

The package is not installed into the active environment. `src/` is a package
root, not a directory to `cd` into and hope. Fix with `hatch env create`, or
`uv pip install -e .` inside an activated venv. A `sys.path` insert is not a
substitute: the modules import each other as `iqaevaluator.<module>`.
````

3. Add, next to the pin discussion:

````markdown
### Scores changed after an upgrade

`pyiqa==0.1.15.post2` is pinned on purpose. Metric outputs depend on the
backend version, so unpinning it silently changes numbers that were already
reported. Upgrade deliberately and re-run everything you intend to compare.
````

4. The empty-slice entry (currently "below `1e-3` mean or standard deviation *after* normalization"): the test now runs on **raw** data with a spike-proof span, so a single outlier voxel no longer marks healthy slices empty. Correct the sentence and drop the "inspect the raw data range before blaming the code" framing that followed from the old behaviour.

- [ ] **Step 4: Re-run the check**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -n "requirements.txt\|pip install -r\|There is no package" "$WIKI/Installation.md" "$WIKI/Troubleshooting.md"
```

Expected: no output (exit 1).

- [ ] **Step 5: Commit**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add Installation.md Troubleshooting.md && git commit -m "$(cat <<'MSG'
Install from pyproject.toml via Hatch, not requirements.txt

requirements.txt was deleted and src/ became the iqaevaluator package, so
the install page described an install that cannot work and told readers
there was no package to import.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: New page Normalization.md, and the sidebar

**Files:**
- Create: `$WIKI/Normalization.md`
- Modify: `$WIKI/_Sidebar.md`

**Interfaces:**
- Consumes: `captured.txt` blocks `norm-minmax-vs-percentile`, `affine-trap`, `raw-mask`.
- Produces: the link target `[Normalization](Normalization)`, used by Tasks 4, 5, 6, 7.

- [ ] **Step 1: Write the page**

Structure and required content — every signature below is copied from `src/iqaevaluator/normalization.py` and must be re-checked against it, not against this plan, before committing.

````markdown
# Normalization

Every metric backend here — pyiqa, DreamSim, MONAI — takes `float32` in
`[0, 1]` and scales from there internally. None of them normalizes per image.
So the loader does, and this page is about **who decides what `[0, 1]` means**.
That decision changes every score, so it is recorded in every row of the
report.

## The pieces

```python
from iqaevaluator.normalization import (
    IntensityRange, Normalizer, MinMax, Percentile, FixedRange, Raw,
    normalizer_from_name, scale,
)
```

A `Normalizer` answers one question:

```python
class Normalizer(Protocol):
    name: str
    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]: ...
```

`IntensityRange(lo, hi)` is the raw interval mapped onto `[0, 1]`. The mapping
itself — `clip((raw - lo) / (hi - lo), 0, 1)` — lives in `scale()` and exists
in exactly one place. `range_of` returning `None` means *do not scale at all*.

## The four strategies

### `MinMax()` — the default

The image's own minimum and maximum. This is what the reported numbers use,
and what you get if you pass no normalizer.

### `Percentile(lower=0.5, upper=99.5)` — opt-in

Robust extremes, so a single spike voxel no longer compresses the whole image.
**Every score changes under it**, `psnr` and `ssim` included, so results are
only comparable with other runs that used the same percentiles. Never mix the
two strategies inside one comparison.

<!-- captured.txt block: norm-minmax-vs-percentile -->
```
<paste the captured block verbatim>
```

If the foreground is rarer than the percentiles allow — a small lesion in a
large field of view, a heavily zero-padded volume — both percentiles land on
background and the span collapses. `range_of` then falls back to the image's
own extremes, so a sparse-but-real image is scaled rather than binarized.

### `Raw()` — masks and label maps

No scaling; the decoded dtype survives, so an integer label map reaches the
segmentation metrics as integers and a single label can be selected. Note that
a PNG mask stores 0/255, not 0/1 — load those with `MinMax()`, which maps them
to exactly `{0.0, 1.0}`.

<!-- captured.txt block: raw-mask -->
```
<paste the captured block verbatim>
```

### `FixedRange(range, name)` — internal

A range decided elsewhere; the data is ignored. This is how `load_pair` puts
the input of a full-reference pair on the target's scale. You do not normally
construct it yourself.

## Full-reference runs: use `load_pair`

```python
from iqaevaluator.image_loader import load_pair
from iqaevaluator.normalization import Percentile

inp, tgt = load_pair(input_path, target_path)                      # MinMax
inp, tgt = load_pair(input_path, target_path, Percentile(0.5, 99.5))
```

The target is scaled by the normalizer; the input is then scaled by the range
the target produced (fastMRI's convention: `data_range` comes from the
reference). Anything outside that range is clipped in the input, and the raw
extremes stay visible as `ImageLoader.raw_range`.

### The trap this prevents

Load the two images independently and min-max each on its own extremes:

```python
inp = ImageLoader(input_path)     # scaled on ITS OWN range
tgt = ImageLoader(target_path)    # scaled on ITS OWN range
```

Min-max is invariant under any affine transform — `norm(x) == norm(2x + 500)`.
A generator whose output is uniformly twice as bright is therefore **not
penalised at all** by `psnr`, `ssim`, `lpips` or `dists`. The scores look fine
and measure nothing.

<!-- captured.txt block: affine-trap -->
```
<paste the captured block verbatim>
```

This is the only place in the wiki that builds a full-reference pair by hand.
Everywhere else uses `load_pair`.

## Building a normalizer from a string

```python
from iqaevaluator.normalization import normalizer_from_name

normalizer_from_name("minmax")      # MinMax()
normalizer_from_name("percentile")  # Percentile(0.5, 99.5)
normalizer_from_name("raw")         # Raw()
```

Unknown names raise `ValueError` listing the valid ones. Use it when your own
CLI or config file carries the strategy as text.

## What lands in the report

Five columns on every row:

| Column | Meaning |
|---|---|
| `normalization` | the strategy's `name` — `minmax`, `percentile_0.5_99.5`, `raw` |
| `scale_lo`, `scale_hi` | the raw interval that became `[0, 1]`. In a full-reference run this is the **target's** range |
| `input_min`, `input_max` | the input's own raw extremes |

All five are `None` under `Raw()`. `input_max > scale_hi` means the prediction
overshot the reference and was clipped — visible in the numbers rather than
normalized away.

See also: [Loading Images](Loading-Images) · [Interpreting Scores](Interpreting-Scores) · [Segmentation Metrics](Segmentation-Metrics)
````

- [ ] **Step 2: Paste the captured blocks**

Replace each `<paste the captured block verbatim>` with the matching block from `$SCRATCH/captured.txt`. No hand-typed numbers.

- [ ] **Step 3: Add to the sidebar**

In `$WIKI/_Sidebar.md`, insert `Normalization` between `Loading Images` and `Registering Metrics`, matching the existing link syntax exactly.

- [ ] **Step 4: Check**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -c "paste the captured block" "$WIKI/Normalization.md"   # expect 0
grep -n "Normalization" "$WIKI/_Sidebar.md"                    # expect 1 hit
cd "$FW" && source .venv/bin/activate && python - <<'PY'
from iqaevaluator.normalization import MinMax, Percentile, Raw, normalizer_from_name
assert MinMax().name == "minmax"
assert Percentile().name == "percentile_0.5_99.5"
assert Raw().name == "raw"
assert type(normalizer_from_name("percentile")) is Percentile
print("signatures match the page")
PY
```

- [ ] **Step 5: Commit**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add Normalization.md _Sidebar.md && git commit -m "$(cat <<'MSG'
Document intensity normalization on its own page

Normalization became a per-run choice with four strategies and five report
columns, and load_pair now puts a full-reference pair on the target's scale.
None of that was documented, and the affine-blindness trap it prevents was
described as an unavoidable property.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: Interpreting-Scores.md and Loading-Images.md

**Files:**
- Modify: `$WIKI/Interpreting-Scores.md`
- Modify: `$WIKI/Loading-Images.md`

**Interfaces:**
- Consumes: `[Normalization](Normalization)` from Task 3.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Prove the claims are stale**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -n "TODO(norm\|normalized \*\*independently\*\*\|after\*\* normalization\|after normalization\|min-max normalized" "$WIKI/Interpreting-Scores.md" "$WIKI/Loading-Images.md"
```

Expected: hits including the `TODO(norm-N)` paragraph, which names markers that no longer exist in `src/`.

- [ ] **Step 2: Fix `Interpreting-Scores.md`**

1. Opening sentence ("Every decoder ends with a min-max normalization of the whole file to `[0, 1]`") → the decoders return raw data and `ImageLoader` applies a `Normalizer`; min-max is the default. Link to [Normalization](Normalization).
2. The independence caveat: keep the affine-invariance explanation — it is correct and it is the most important warning in the wiki — but reframe it. It describes what happens when a full-reference pair is built from two independent loaders, and `load_pair` is what prevents it. One sentence stating that every example in this wiki uses `load_pair`.
3. The empty-slice sentence ("below `1e-3` after normalization") → the test runs on raw data with a spike-proof span; slices are dropped in slice mode, kept in volume mode.
4. **Delete the whole `TODO(norm-N)` paragraph.** It promises fixes that shipped. Replace with two sentences: the strategies exist, the scale is recorded per row, see [Normalization](Normalization).

- [ ] **Step 3: Fix `Loading-Images.md`**

1. All imports → `from iqaevaluator.image_loader import …`.
2. `ImageLoader(path)` → document the second parameter: `ImageLoader(path, normalizer=MinMax())`.
3. Add `.raw`, `.raw_range`, `.intensity_range` to the attribute list, with `.tensor` noted as `(D, 1, H, W)` float32 in `[0, 1]` **under every strategy except `Raw()`**, where the stored dtype survives unscaled.
4. Shrink the "Every decoder ends in the same step…" normalization section to two sentences plus a link to [Normalization](Normalization).
5. Correct the empty-slice paragraphs the same way as step 2.4 — the "cruder than it looks / a single bright outlier voxel" warning is obsolete.
6. `main.evaluate()` citation on the pairing helpers → "Two helpers build input/target pairs from directories." Nothing more.
7. Add `load_pair` to the pairing section as the full-reference entry point, linking to [Normalization](Normalization) for why.

- [ ] **Step 4: Check**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -n "TODO(norm\|main\.evaluate\|main\.py" "$WIKI/Interpreting-Scores.md" "$WIKI/Loading-Images.md"   # expect no output
grep -c "Normalization)" "$WIKI/Interpreting-Scores.md" "$WIKI/Loading-Images.md"                          # expect >= 1 each
```

- [ ] **Step 5: Commit**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add Interpreting-Scores.md Loading-Images.md && git commit -m "$(cat <<'MSG'
Correct the normalization claims in the caveats and loader pages

Both pages described the old hardcoded min-max: normalization applied by the
decoder, independent scaling as an unavoidable property, emptiness tested
after scaling, and a list of TODO(norm-N) markers that have since shipped.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: Segmentation-Metrics.md and Configuring-Metrics.md

**Files:**
- Modify: `$WIKI/Segmentation-Metrics.md`
- Modify: `$WIKI/Configuring-Metrics.md`

**Interfaces:**
- Consumes: `captured.txt` block `raw-mask`; `[Normalization](Normalization)`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Prove the mask advice is wrong**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -n "min-max normalized and may be soft\|arrives min-max\|^from segmentation_metrics\|^from metrics" "$WIKI/Segmentation-Metrics.md" "$WIKI/Configuring-Metrics.md"
```

Expected: the `threshold` paragraph claiming a loaded mask arrives min-max normalized, plus flat imports.

- [ ] **Step 2: Rewrite the mask-loading story**

Both pages: masks are loaded with `Raw()`, which preserves the integer dtype, and MONAI accepts raw integer masks. PNG masks store 0/255 and are loaded with `MinMax()`, which maps them to exactly `{0.0, 1.0}`. Paste the `raw-mask` block as evidence in `Segmentation-Metrics.md`.

The `threshold` paragraph becomes: `threshold` (default `None`) binarizes both masks as `value > threshold` before scoring. With `Raw()`-loaded integer masks you do not need it. Set it only for genuinely soft masks (probability maps); setting it on data that is not a probability map silently corrupts scores.

- [ ] **Step 3: Package-qualify every import on both pages**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && sed -i '' -E 's/^from (metrics|image_loader|records|evaluation_result|evaluator_factory|normalization|iqa_evaluator|volume_evaluator|volumetric_iqa|constants) import/from iqaevaluator.\1 import/; s/^from segmentation_metrics\./from iqaevaluator.segmentation_metrics./' Segmentation-Metrics.md Configuring-Metrics.md
```

Then read the diff — `sed` does not catch indented or multi-line imports. Fix those by hand.

- [ ] **Step 4: Check**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -nE "from (metrics|image_loader|records|evaluation_result|evaluator_factory|normalization|segmentation_metrics)" "$WIKI/Segmentation-Metrics.md" "$WIKI/Configuring-Metrics.md" | grep -v iqaevaluator   # expect no output
grep -n "min-max normalized and may be soft" "$WIKI"/*.md                                                                                                                                     # expect no output
```

- [ ] **Step 5: Commit**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add Segmentation-Metrics.md Configuring-Metrics.md && git commit -m "$(cat <<'MSG'
Load masks with Raw(); correct the threshold guidance

MONAI now accepts raw integer masks and Raw() keeps the dtype, so the advice
to threshold because "a mask arrives min-max normalized and may be soft" no
longer describes what happens.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: Results-and-Reports.md

**Files:**
- Modify: `$WIKI/Results-and-Reports.md`

**Interfaces:**
- Consumes: `captured.txt` blocks `custom-metric`, `to-frame`, `csv-header`, `aggregate-volumes`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Prove the column count is stale**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -n "6 identifying columns\|(8, 25)\|image_id, source_model, mode, scoring, slice_index, is_empty" "$WIKI/Results-and-Reports.md"
```

Expected: hits. The record now carries eleven identifying fields — `image_id`, `source_model`, `mode`, `scoring`, `slice_index`, `is_empty`, `normalization`, `scale_lo`, `scale_hi`, `input_min`, `input_max`.

- [ ] **Step 2: Replace every printed output with the captured one**

Swap the `df.shape` line, the column list, the printed frame and the CSV header for the `to-frame` and `csv-header` blocks from `captured.txt`. Update the prose "6 identifying columns + 19 built-in metric columns" to the real counts from that block.

- [ ] **Step 3: Add the intensity columns to the column-order explanation**

One paragraph: the five intensity columns sit with the identifying columns, before the metric columns; they say what `[0, 1]` meant for that row; they are `None` under `Raw()`; link to [Normalization](Normalization).

- [ ] **Step 4: Package-qualify the imports, and check**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
grep -nE "from (metrics|records|evaluation_result|evaluator_factory|image_loader)" "$WIKI/Results-and-Reports.md" | grep -v iqaevaluator   # expect no output
grep -n "6 identifying columns\|(8, 25)" "$WIKI/Results-and-Reports.md"                                                                     # expect no output
```

- [ ] **Step 5: Commit**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add Results-and-Reports.md && git commit -m "$(cat <<'MSG'
Re-capture the report examples with the five intensity columns

Every row now records normalization, scale_lo/hi and input_min/max, so the
column list, the frame shape and the CSV header on this page were all short
by five.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 7: The remaining seven pages — imports, `load_pair`, exclusions

**Files:**
- Modify: `$WIKI/Home.md`, `$WIKI/Core-Concepts.md`, `$WIKI/Registering-Metrics.md`, `$WIKI/Running-an-Evaluation.md`, `$WIKI/Custom-Metrics.md`, `$WIKI/Metric-Catalog.md`, `$WIKI/Scoring-Modes.md`

**Interfaces:**
- Consumes: `captured.txt` blocks `minimal-example`, `volume-mode`, `select-skips`, `custom-metric`, `dreamsim-names`; `[Normalization](Normalization)`.
- Produces: the final state of the page set for Task 8's global sweep.

- [ ] **Step 1: Sweep the imports**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && sed -i '' -E 's/^from (metrics|image_loader|records|evaluation_result|evaluator_factory|normalization|iqa_evaluator|volume_evaluator|volumetric_iqa|dreamsim_metric) import/from iqaevaluator.\1 import/; s/^from segmentation_metrics\./from iqaevaluator.segmentation_metrics./' Home.md Core-Concepts.md Registering-Metrics.md Running-an-Evaluation.md Custom-Metrics.md Metric-Catalog.md Scoring-Modes.md
git diff --stat
```

Then read the diff and fix indented or parenthesised imports `sed` missed.

- [ ] **Step 2: Switch every full-reference example to `load_pair`**

Pattern to replace, wherever a page builds an FR pair:

```python
# before
inp = ImageLoader(input_path)
tgt = ImageLoader(target_path)

# after
from iqaevaluator.image_loader import load_pair
inp, tgt = load_pair(input_path, target_path)
```

`Home.md`'s minimal example is the most visible one — rebuild it against the `minimal-example` block and paste that block's output under it.

- [ ] **Step 3: Apply the exclusions**

- `Core-Concepts.md`: delete the `main.py   reference CLI on top of all of it` line from the module map. Paths in that map become `iqaevaluator/…`.
- Any page: remove mentions of `constants.py`, `INPUT`, `TARGET`, `REPORT`, `evaluation.ipynb`, `iqaevaluator.data`. Where a page needed `REPORT` to explain `generate_report`, state that the path is a parameter with a default instead.
- `Home.md`: no status note about `main.py`. Silence.

- [ ] **Step 4: Refresh the captured snippets**

`Registering-Metrics.md` → `select-skips`. `Running-an-Evaluation.md` and `Scoring-Modes.md` → `volume-mode`. `Custom-Metrics.md` → `custom-metric`. `Configuring-Metrics.md` was done in Task 5; if its `dreamsim` variant names differ from the `dreamsim-names` block, fix them there too.

- [ ] **Step 5: Check**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI"
grep -nE "from (metrics|image_loader|records|evaluation_result|evaluator_factory|normalization|iqa_evaluator|volume_evaluator|volumetric_iqa|dreamsim_metric|segmentation_metrics)" *.md | grep -v iqaevaluator   # expect no output
grep -n "main\.py\|main\.evaluate\|constants\.py\|evaluation\.ipynb\|iqaevaluator\.data\|\bINPUT\b\|\bTARGET\b" *.md                                                     # expect no output
```

- [ ] **Step 6: Commit**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add Home.md Core-Concepts.md Registering-Metrics.md Running-an-Evaluation.md Custom-Metrics.md Metric-Catalog.md Scoring-Modes.md && git commit -m "$(cat <<'MSG'
Package-qualify every import, pair through load_pair, drop the CLI

src/ became the iqaevaluator package, so every example imported a module
that no longer resolves. Full-reference examples now go through load_pair,
and main.py, constants.py, the notebook and the survey script are out of the
documented surface.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 8: Global sweep and signature self-review

**Files:**
- Modify: any page a check flags

**Interfaces:**
- Consumes: the full page set.
- Produces: a clean sweep, and the diff Task 9 shows the user.

- [ ] **Step 1: Run the full stale-pattern sweep**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI"
echo "--- flat imports ---";        grep -nE "^ *from (metrics|image_loader|records|evaluation_result|evaluator_factory|normalization|iqa_evaluator|volume_evaluator|volumetric_iqa|dreamsim_metric|clip_iqa_medical|radimagenet_lpips|constants|segmentation_metrics)" *.md | grep -v iqaevaluator
echo "--- excluded surface ---";    grep -n "main\.py\|main\.evaluate\|constants\.py\|evaluation\.ipynb\|iqaevaluator\.data" *.md
echo "--- dead packaging ---";      grep -n "requirements.txt\|pip install -r\|cd src\|There is no package\|sys\.path" *.md
echo "--- retired markers ---";     grep -n "TODO(norm" *.md
echo "--- stale norm claims ---";   grep -n "after normalization\|normalized \*\*independently\*\*\|min-max normalized and may be soft" *.md
```

Expected: every section empty. Fix anything that prints, then re-run until clean.

- [ ] **Step 2: Verify every signature the wiki states**

```bash
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
cd "$FW" && source .venv/bin/activate && python - <<'PY'
import inspect
from iqaevaluator import image_loader, evaluator_factory, normalization, metrics
from iqaevaluator.evaluation_result import EvaluationResult
for f in (image_loader.load_pair, evaluator_factory.build_evaluator,
          normalization.normalizer_from_name, EvaluationResult.from_records,
          metrics.MetricRegistry.register_metric, metrics.MetricRegistry.select):
    print(f.__qualname__, inspect.signature(f))
print("ImageLoader:", inspect.signature(image_loader.ImageLoader.__init__))
print("BUILTIN:", len(metrics.BUILTIN_METRICS), "SEG:", len(metrics.SEGMENTATION_METRICS))
PY
```

Compare each line against what the pages print. Any mismatch is a wiki bug — fix the wiki, never the code.

- [ ] **Step 3: Check every internal link resolves**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && grep -oE "\]\([A-Z][A-Za-z-]+\)" *.md | sed -E 's/.*\(([^)]+)\)/\1/' | sort -u | while read p; do [ -f "$p.md" ] || echo "BROKEN: $p"; done
```

Expected: no `BROKEN:` lines.

- [ ] **Step 4: Commit any fixes**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git add -A && git commit -m "$(cat <<'MSG'
Fix what the global sweep found

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)" || echo "nothing to fix"
```

---

### Task 9: Local docs re-check, diff presentation, push gate

**Files:**
- Modify (only if they contradict the wiki): `$FW/README.md`, `$FW/CLAUDE.md`

**Interfaces:**
- Consumes: the committed wiki clone.
- Produces: the final diff and the push, after approval.

- [ ] **Step 1: Check the local docs against the exclusion decisions**

```bash
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
cd "$FW" && grep -n "requirements.txt\|pip install -r" README.md CLAUDE.md
```

`CLAUDE.md` legitimately documents `main.py` and `constants.py` — it is the maintainer's file, not the wiki, and stays as is. Fix only genuine contradictions (a dead `requirements.txt` reference, a wrong install command). If the grep is empty, change nothing.

- [ ] **Step 2: Show the user the full diff**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git log --oneline origin/master..HEAD && git diff origin/master --stat && git diff origin/master
```

Present the file list and the diff. **Stop here.** Do not push.

- [ ] **Step 3: Push, only after the user says go**

```bash
WIKI=$(ls -d /private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework/*/scratchpad/wiki | head -1)
cd "$WIKI" && git push origin master
```

- [ ] **Step 4: Commit any local doc fix in the framework repo**

Only if Step 1 found a contradiction:

```bash
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
cd "$FW" && git add README.md CLAUDE.md && git commit -m "$(cat <<'MSG'
docs: align README/CLAUDE.md with the updated wiki

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

Note: the repository has 36 files staged from the Hatch migration. Commit with an explicit pathspec so that work is not swept in.
