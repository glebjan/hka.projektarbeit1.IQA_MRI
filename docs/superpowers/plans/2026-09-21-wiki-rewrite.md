# Wiki Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the framework's GitHub wiki with 15 new English, usage-oriented pages plus a sidebar, every example executed against mock data, in a single orphan commit that is not pushed.

**Architecture:** Three scratchpad tools make the prose verifiable. `make_mock_data.py` builds a deterministic dataset, `run_examples.py` executes every `python`/`bash` block of a page in document order and checks the `text` block that follows it, `lint_wiki.py` enforces the style rules and internal links. Pages are written in dependency order (concepts before details), each task ends green on runner and linter. The wiki repository is reset to an orphan branch first and committed once at the end.

**Tech Stack:** Markdown (GitHub wiki), Python 3.14 in `framework/.venv` (numpy, nibabel, SimpleITK, pydicom, Pillow, scipy, pandas, torch, pyiqa, MONAI), bash, git.

**Spec:** `framework/docs/superpowers/specs/2026-09-21-wiki-rewrite-design.md`

## Global Constraints

- Paths used in every command:
  - `W=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework.wiki` (wiki repo)
  - `FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework` (framework repo)
  - `T=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/wikitools` (tools, sandbox, runs, discrepancy list)
  - `THESIS=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/thesis.txt` (report text, `pdftotext -layout $W/main.pdf $THESIS` regenerates it)
- Shell state does not persist between commands; every command sets the variables it needs.
- Code is the source of truth, the report must not be contradicted. Every disagreement goes into `$T/discrepancies.md`, never into the wiki.
- Prose, headings, table cells: English (American spelling), impersonal, present tense, no "you/your/we/our/I", no `–`, `—`, `:`, `;`, no bullet or numbered lists. Exempt are code, inline code, URLs, link syntax, HTML comments, table separator rows.
- Every table is introduced by one sentence. Abbreviations are written out at first use on each page, in the report's form (e.g. "Peak Signal-to-Noise Ratio (PSNR)", "Full-Reference (FR)", "No-Reference (NR)").
- Terms fixed across all pages: Full-Reference (FR), No-Reference (NR), input, target, slice mode, volume mode, registry, metric specification (`MetricSpec`), record (`ImageEvaluatorRecord`), normalization strategy, report.
- Examples assume the framework repository root as working directory, `.venv` activated and `PYTHONPATH=src`. Mock paths start with `data/`. DataFrames are printed with `.to_string()` (optionally `.round(3)`) so output does not depend on pandas display options.
- Fenced languages allowed: `python`, `bash`, `text`. A non-executed block is preceded by the line `<!-- example: norun -->`. A `text` block directly after a code block (only blank lines between) is its expected output, copied from `$T/runs/<Page>/block<N>.out`, elided with a line `...` where long.
- The metric total is not stated as a single number (report says 29, Tabelle A.4 lists 30). Pages state the breakdown: 18 image quality metrics in `BUILTIN_METRICS`, DreamSim, 11 segmentation metrics in `SEGMENTATION_METRICS`.
- No intermediate commits in the wiki repo; exactly one commit in Task 13. Nothing is pushed in either repository. No framework code is changed.

---

### Task 0: Prepare the wiki repository and the discrepancy list

**Files:**
- Modify: `$W/.git/info/exclude`
- Create: `$T/discrepancies.md`

**Interfaces:**
- Produces: wiki working tree on orphan branch `wiki-rewrite` with an empty index and only `AGENTS.md`, `main.pdf`, `.DS_Store` present (all excluded). Task 13 renames the branch to `master`.

- [ ] **Step 1: Inspect before destroying**

Run: `cd $W && git status && git stash list && git log --oneline -3 && git reflog -5`
Expected: interactive rebase in progress onto `ec7c7da`, conflicts in `Configuring-Metrics.md`, `Core-Concepts.md`. If anything else is found (stashes, extra branches), stop and report.

- [ ] **Step 2: Keep a local safety ref, abort the rebase, create the orphan branch**

The working-tree `AGENTS.md` may differ from any committed version (the author edits it locally), and `git rebase --abort` rewrites tracked files. Both local files are therefore copied out first and restored unconditionally afterwards.

```bash
W=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework.wiki
T=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/wikitools
mkdir -p "$T/keep"
cd "$W"
cp -p AGENTS.md main.pdf "$T/keep/"
git rebase --abort
git branch -f backup/pre-rewrite HEAD
git checkout --orphan wiki-rewrite
git rm -r -q --cached .
git ls-files --others | grep -v -E '^(AGENTS\.md|main\.pdf|\.DS_Store)$' | while read -r f; do rm -f "$f"; done
find . -type d -empty -not -path './.git*' -delete
printf 'AGENTS.md\nmain.pdf\n.DS_Store\n' >> .git/info/exclude
cp -p "$T/keep/AGENTS.md" "$T/keep/main.pdf" .
cmp AGENTS.md "$T/keep/AGENTS.md" && echo "AGENTS.md preserved"
ls -A
git status --short
```

Expected: `AGENTS.md preserved`, `ls -A` shows `.git AGENTS.md main.pdf` (and `.DS_Store` if present). `git status --short` prints nothing. `backup/pre-rewrite` is local only and is deleted in Task 13 only if the author agrees (it is mentioned in the final report).

- [ ] **Step 3: Seed the discrepancy list**

Create `$T/discrepancies.md` with this content (the tools directory is created here):

```markdown
# Discrepancies between report (main.pdf), code and wiki

Format: ID | where | report says | code / fact | wiki handling | suggested fix

D1 | Report 1.1, Zusammenfassung, 6.1 vs Tabelle A.4 | 29 metrics | A.4 and code list 30 registrable metrics (18 BUILTIN_METRICS + DreamSim + 11 SEGMENTATION_METRICS) | wiki gives the breakdown, no total | change 29 to 30 in the report, or define what is excluded
D2 | Report 1.1 | three main categories | report 2.3 and wiki use four families | wiki uses four families | align 1.1 with 2.3
D3 | Report 5.4 | clip_iqa_lung/brain use "je zehn medizinischen Promptpaaren" | clip_iqa_medical.py holds 10 prompts = 5 positive/negative pairs per organ | wiki says five prompt pairs | change to "je fünf Promptpaaren (zehn Prompts)"
D4 | Report 4.3 | pairing needs a prefix that "eine Mindestlänge überschreitet" | image_loader.find_matching_target accepts length >= 4 (reaching, not exceeding) | wiki says at least four characters | "eine Mindestlänge von vier Zeichen erreicht"
D5 | Code src/main.py evaluate() docstring and --normalization help | Raw() "is the right choice for mask files", "(raw, for masks)" | report 4.4 and normalization.py: masks belong to Mask(), Raw() only for instance maps | wiki follows report | fix docstring and CLI help
D6 | Code src/iqaevaluator/data.py main() | README/CLAUDE.md: `python -m iqaevaluator.data` surveys a dataset | main() only prints "test"; analyze() is commented out | wiki documents analyze(path, report_path) | restore main()
D7 | pyproject.toml [project.urls] | — | points to github.com/genizki/iqaevaluator, remote is github.com/glebjan/hka.projektarbeit1.IQA_MRI | wiki uses the real remote | fix URLs
D8 | Repository state | — | documented code is branch volumetric_similarity (125 commits ahead of origin/main) | wiki describes that state | merge into main before the wiki is published
D9 | framework/README.md | — | outdated (global registry, mask_writer.py, registry.register) | wiki ignores it | rewrite or point README to the wiki
```

- [ ] **Step 4: Verify**

Run: `cd $W && git branch --show-current && git status --short | wc -l && tail -3 .git/info/exclude`
Expected: `wiki-rewrite`, `0`, the three exclude lines.

---

### Task 1: Mock dataset generator

**Files:**
- Create: `$T/make_mock_data.py`

**Interfaces:**
- Produces: `python make_mock_data.py <sandbox>` writes `<sandbox>/data/` with exactly this tree (all later tasks reference these paths verbatim):

```text
data/reference/subject01_t1.nii.gz          int16, 12 slices of 128x128, spacing (3.0, 0.9, 0.9), slices 0 and 11 empty
data/reference/subject02_t1.nii.gz          same geometry, shifted phantom
data/generated/model_a/subject01_t1.nii.gz  blurred reference + mild noise, empty ends kept
data/generated/model_a/subject02_t1.nii.gz
data/generated/model_b/subject01_t1.nii.gz  10 % brighter + noise everywhere (overshoot, non-empty ends)
data/generated/model_b/subject02_t1.nii.gz
data/generated/model_b/unpaired_scan.nii.gz no target shares 4 prefix characters -> NR rows
data/png/reference/case01.png, case02.png   uint8 128x128
data/png/generated/case01.png, case02.png
data/masks/reference/subject01_seg.nii.gz   uint8 label map, 1 = brain, 2 = lesion, spacing as above
data/masks/predicted/subject01_seg.nii.gz   eroded brain, shifted lesion
data/masks/probability/subject01_seg.nii.gz float32 probability map in [0, 1]
data/masks/png/reference/case01_mask.png    0/255
data/masks/png/predicted/case01_mask.png    0/255
data/instances/reference/cells01.nii.gz     int16 instance map, 4 slices of 96x96, ids 1..6, spacing (2.0, 0.5, 0.5)
data/instances/predicted/cells01.nii.gz     one cell missing, one shrunk, all shifted
data/dicom/ct_slice.dcm                     one CT slice, RescaleIntercept -1024, PixelSpacing 0.7, SliceThickness 2.5
data/other/subject01_t1.nrrd, subject01_t1.mha
data/other/timeseries_4d.nii.gz             4D (64, 64, 6, 3)
data/broken/shape/reference/subject01_t1.nii.gz, data/broken/shape/generated/subject01_t1.nii.gz (10 slices)
data/broken/spacing/reference/subject01_t1.nii.gz, data/broken/spacing/generated/subject01_t1.nii.gz (spacing 2.5 in depth)
data/pitfalls/generated/subject01_t1.nii.gz, subject01_t1_v2.nii.gz (both match reference subject01)
```

- [ ] **Step 1: Write the generator**

```python
"""Deterministic mock dataset for the wiki examples.

Usage: python make_mock_data.py <sandbox_dir>
Replaces <sandbox_dir>/data.
"""
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from scipy.ndimage import binary_erosion, gaussian_filter

SHAPE = (128, 128, 12)   # NIfTI array axes (X, Y, Z): the loader yields 12 slices of 128 x 128
ZOOMS = (0.9, 0.9, 3.0)  # mm, anisotropic


def grid(shape=SHAPE):
    return np.meshgrid(*[np.linspace(-1.0, 1.0, n) for n in shape], indexing="ij")


def regions(offset=(0.0, 0.0), lesion_shift=0.0):
    x, y, z = grid()
    x, y = x - offset[0], y - offset[1]
    head = (x / 0.80) ** 2 + (y / 0.90) ** 2 + (z / 0.95) ** 2 <= 1.0
    brain = (x / 0.65) ** 2 + (y / 0.75) ** 2 + (z / 0.85) ** 2 <= 1.0
    lesion = ((x - 0.25 - lesion_shift) / 0.15) ** 2 + ((y + 0.20) / 0.15) ** 2 + (z / 0.5) ** 2 <= 1.0
    return head, brain, lesion & brain


def phantom(seed, offset=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    head, brain, lesion = regions(offset)
    img = np.zeros(SHAPE, np.float32)
    img[head], img[brain], img[lesion] = 600.0, 900.0, 1200.0
    texture = gaussian_filter(rng.normal(0.0, 1.0, SHAPE), sigma=(2, 2, 0)) * 120.0
    return np.clip(np.where(head, img + texture, 0.0), 0, None).astype(np.int16)


def blurred(ref, seed):
    rng = np.random.default_rng(seed)
    out = gaussian_filter(ref.astype(np.float32), sigma=(1.2, 1.2, 0))
    out = np.where(out > 0, out + rng.normal(0.0, 15.0, SHAPE), 0.0)
    return np.clip(out, 0, None).astype(np.int16)


def brighter(ref, seed):
    rng = np.random.default_rng(seed)
    out = ref.astype(np.float32) * 1.1 + rng.normal(0.0, 40.0, SHAPE)
    return np.clip(out, 0, None).astype(np.int16)


def save_nifti(arr, path, zooms=ZOOMS):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr, np.diag([*zooms[:3], 1.0]))
    img.header.set_zooms(zooms)
    nib.save(img, str(path))


def save_png(arr2d, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr2d.astype(np.uint8)).save(path)


def to_uint8(slice2d):
    return np.clip(slice2d.astype(np.float32) / 1400.0 * 255.0, 0, 255)


def labels(offset=(0.0, 0.0), erode=0, lesion_shift=0.0):
    _, brain, lesion = regions(offset, lesion_shift)
    if erode:
        brain = binary_erosion(brain, structure=np.ones((3, 3, 1)), iterations=erode)
        lesion = lesion & brain
    lab = np.zeros(SHAPE, np.uint8)
    lab[brain] = 1
    lab[lesion] = 2
    return lab


def instances(shift=0, drop_last=False, shrink=None):
    shape = (96, 96, 4)
    x, y = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    centers = [(20, 20), (20, 70), (48, 45), (75, 20), (75, 70), (48, 85)]
    out = np.zeros(shape, np.int16)
    for ident, (cx, cy) in enumerate(centers, start=1):
        if drop_last and ident == len(centers):
            continue
        radius = 5 if shrink == ident else 8
        disc = (x - cx - shift) ** 2 + (y - cy) ** 2 <= radius ** 2
        for z in range(shape[2]):
            out[..., z][disc] = ident
    return out


def save_dicom(path, ref):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 128, 128
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    ds.PixelSpacing = [0.7, 0.7]
    ds.SliceThickness = 2.5
    ds.WindowCenter, ds.WindowWidth = 40, 400
    hu = np.where(ref[:, :, 6] > 0, ref[:, :, 6] / 20.0, -1000.0)
    ds.PixelData = (hu + 1024).astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def save_sitk(arr_xyz, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.transpose(arr_xyz, (2, 1, 0)))
    img.SetSpacing(ZOOMS)
    sitk.WriteImage(img, str(path))


def main(sandbox: Path) -> None:
    data = sandbox / "data"
    shutil.rmtree(data, ignore_errors=True)
    ref1, ref2 = phantom(1), phantom(2, offset=(0.08, -0.05))
    save_nifti(ref1, data / "reference/subject01_t1.nii.gz")
    save_nifti(ref2, data / "reference/subject02_t1.nii.gz")
    save_nifti(blurred(ref1, 11), data / "generated/model_a/subject01_t1.nii.gz")
    save_nifti(blurred(ref2, 12), data / "generated/model_a/subject02_t1.nii.gz")
    save_nifti(brighter(ref1, 21), data / "generated/model_b/subject01_t1.nii.gz")
    save_nifti(brighter(ref2, 22), data / "generated/model_b/subject02_t1.nii.gz")
    save_nifti(blurred(phantom(5), 25), data / "generated/model_b/unpaired_scan.nii.gz")

    save_png(to_uint8(ref1[:, :, 6]), data / "png/reference/case01.png")
    save_png(to_uint8(ref2[:, :, 4]), data / "png/reference/case02.png")
    save_png(to_uint8(blurred(ref1, 31)[:, :, 6]), data / "png/generated/case01.png")
    save_png(to_uint8(blurred(ref2, 32)[:, :, 4]), data / "png/generated/case02.png")

    save_nifti(labels(), data / "masks/reference/subject01_seg.nii.gz")
    save_nifti(labels(erode=2, lesion_shift=0.08), data / "masks/predicted/subject01_seg.nii.gz")
    _, brain, _ = regions()
    prob = gaussian_filter(brain.astype(np.float32), sigma=(1.5, 1.5, 0)).astype(np.float32)
    save_nifti(np.clip(prob, 0.0, 1.0), data / "masks/probability/subject01_seg.nii.gz")
    save_png((labels()[:, :, 6] > 0) * 255, data / "masks/png/reference/case01_mask.png")
    save_png((labels(erode=2)[:, :, 6] > 0) * 255, data / "masks/png/predicted/case01_mask.png")

    inst_zooms = (0.5, 0.5, 2.0)
    save_nifti(instances(), data / "instances/reference/cells01.nii.gz", inst_zooms)
    save_nifti(instances(shift=2, drop_last=True, shrink=5), data / "instances/predicted/cells01.nii.gz", inst_zooms)

    save_dicom(data / "dicom/ct_slice.dcm", ref1)
    save_sitk(ref1, data / "other/subject01_t1.nrrd")
    save_sitk(ref1, data / "other/subject01_t1.mha")
    small = phantom(3)[32:96, 32:96, 3:9]
    save_nifti(np.stack([small] * 3, axis=-1), data / "other/timeseries_4d.nii.gz", (1.0, 1.0, 3.0, 2.0))

    save_nifti(ref1, data / "broken/shape/reference/subject01_t1.nii.gz")
    save_nifti(blurred(ref1, 41)[:, :, :10], data / "broken/shape/generated/subject01_t1.nii.gz")
    save_nifti(ref1, data / "broken/spacing/reference/subject01_t1.nii.gz")
    save_nifti(blurred(ref1, 42), data / "broken/spacing/generated/subject01_t1.nii.gz", (0.9, 0.9, 2.5))

    save_nifti(blurred(ref1, 51), data / "pitfalls/generated/subject01_t1.nii.gz")
    save_nifti(brighter(ref1, 52), data / "pitfalls/generated/subject01_t1_v2.nii.gz")

    for p in sorted(data.rglob("*")):
        if p.is_file():
            print(p.relative_to(sandbox))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

- [ ] **Step 2: Generate and check shapes, spacing, empty slices through the framework's own loader**

```bash
T=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/wikitools
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
mkdir -p $T/sandbox && ln -sfn $FW/src $T/sandbox/src && ln -sfn $FW/models $T/sandbox/models
$FW/.venv/bin/python $T/make_mock_data.py $T/sandbox | wc -l
cd $T/sandbox && PYTHONPATH=src PYTHONWARNINGS=ignore $FW/.venv/bin/python - <<'EOF'
from pathlib import Path
from iqaevaluator.image_loader import ImageLoader
from iqaevaluator.normalization import Mask, Raw
for p in sorted(Path("data").rglob("*")):
    if p.is_file():
        norm = Raw() if "instances" in str(p) else (Mask() if "masks" in str(p) and "probability" not in str(p) else None)
        l = ImageLoader(p, norm) if norm else ImageLoader(p)
        print(p, tuple(l.tensor.shape), l.spacing, l.is_volumetric, int(l.empty_slice_mask.sum()))
EOF
```

Expected: 31 files. `reference/subject01_t1` → `(12, 1, 128, 128) (3.0, 0.9, 0.9) True 2`. `model_b/subject01_t1` → 0 empty slices. `dicom/ct_slice.dcm` → `(1, 1, 128, 128) (2.5, 0.7, 0.7) False`. `timeseries_4d` → `None False`. `png/...` → `None False`. `other/*.nrrd` → spacing `(3.0, 0.9, 0.9)`. If any line differs, fix the generator before continuing.

- [ ] **Step 3: Smoke test all 18 built-in metrics and DreamSim at mock size**

```bash
T=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/wikitools
FW=/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework
cd $T/sandbox && PYTHONPATH=src PYTHONWARNINGS=ignore $FW/.venv/bin/python - <<'EOF'
from pathlib import Path
from main import evaluate, MetricRegistry, BUILTIN_METRICS, DREAMSIM
r = evaluate(Path("data/generated/model_a/subject01_t1.nii.gz"), Path("data/reference/subject01_t1.nii.gz"),
             registry=MetricRegistry(*BUILTIN_METRICS, DREAMSIM))
df = r.to_frame()
print(df.loc[~df.is_empty].drop(columns=["image_id"]).notna().all().to_string())
EOF
```

Expected: every metric column `True` (weights may download on first use, which needs network). A metric that fails at 128x128 prints `metric '<name>' batch failed`. In that case raise `SHAPE` to `(224, 224, 8)` in the generator, regenerate, re-run Step 2 and Step 3, and note the change in the Task 1 report. Record the wall time of this step (it tells how long full-builtin examples take).

---

### Task 2: Example runner

**Files:**
- Create: `$T/run_examples.py`
- Test fixture: `$T/fixture_wiki/Fixture.md`

**Interfaces:**
- Consumes: `$T/make_mock_data.py` (Task 1), sandbox layout `$T/sandbox/{src,models,data}`.
- Produces: `python run_examples.py [--fresh-data] (--all | PAGE ...)`, exit 0 when all blocks pass. Env var `WIKI_DIR` overrides the wiki directory. Outputs in `$T/runs/<PAGE>/block<N>.out`, stderr in `$T/runs/<PAGE>/stderr.txt`. Status lines `PASS`, `RAN`, `FAIL`, `MISMATCH`, `NOT RUN` with `<Page>.md:<line>`.

- [ ] **Step 1: Write the fixture page (the test)**

`$T/fixture_wiki/Fixture.md`:

````markdown
# Fixture

```python
x = 21 * 2
print(f"answer {x}")
```

```text
answer 42
```

<!-- example: norun -->
```python
raise SystemExit("must not run")
```

```bash
echo "cwd $(basename "$PWD")"
ls data/reference
```

```text
cwd sandbox
...
subject02_t1.nii.gz
```

```python
print(f"still {x}")
```

```text
still 43
```
````

- [ ] **Step 2: Write the runner**

```python
"""Run the code examples of wiki pages against the mock dataset.

Usage:
    python run_examples.py [--fresh-data] PAGE [PAGE ...]
    python run_examples.py [--fresh-data] --all

Rules:
- ```python and ```bash blocks run in document order inside one driver
  process per page. Python blocks share one namespace.
- A block whose preceding non-blank line is <!-- example: norun --> is skipped.
- A ```text block separated from the previous code block only by blank lines is
  that block's expected output. Every non-blank expected line must occur,
  whitespace-collapsed, as a substring of a stdout line, in order. A line "..."
  is skipped.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMEWORK = Path("/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework")
WIKI = Path(os.environ.get("WIKI_DIR", "/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework.wiki"))
SANDBOX = HERE / "sandbox"
RUNS = HERE / "runs"
PYTHON = FRAMEWORK / ".venv" / "bin" / "python"
FENCE = re.compile(r"^```([A-Za-z]*)[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)
NORUN = "<!-- example: norun -->"

DRIVER = r'''
import json, subprocess, sys, traceback
blocks = json.load(open(sys.argv[1]))
namespace = {"__name__": "__main__"}
for index, (lang, source) in enumerate(blocks):
    print(f"@@BLOCK {index}@@", flush=True)
    try:
        if lang == "python":
            exec(compile(source, f"<block {index}>", "exec"), namespace)
        else:
            done = subprocess.run(["bash", "-eo", "pipefail", "-c", source],
                                  stdout=subprocess.PIPE, text=True)
            sys.stdout.write(done.stdout)
            if done.returncode != 0:
                raise RuntimeError(f"bash exited with {done.returncode}")
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc(file=sys.stdout)
        print(f"@@FAILED {index}@@", flush=True)
        sys.exit(1)
    sys.stdout.flush()
'''


def parse(text: str) -> list[dict]:
    items: list[dict] = []
    last_end, last_code = 0, None
    for m in FENCE.finditer(text):
        between = [l for l in text[last_end:m.start()].splitlines() if l.strip()]
        lang, body = m.group(1).lower(), m.group(2)
        if lang == "text" and last_code is not None and not between:
            last_code["expected"] = body
            last_code = None
        elif lang in ("python", "bash"):
            last_code = {
                "lang": lang, "source": body, "expected": None,
                "norun": bool(between) and between[-1].strip() == NORUN,
                "line": text.count("\n", 0, m.start()) + 1,
            }
            items.append(last_code)
        else:
            last_code = None
        last_end = m.end()
    return items


def collapse(s: str) -> str:
    return " ".join(s.split())


def first_missing(expected: str, actual: str):
    lines = [collapse(l) for l in actual.splitlines()]
    pos = 0
    for want in (collapse(l) for l in expected.splitlines()):
        if not want or want == "...":
            continue
        while pos < len(lines) and want not in lines[pos]:
            pos += 1
        if pos == len(lines):
            return want
        pos += 1
    return None


def ensure_sandbox(fresh: bool) -> None:
    SANDBOX.mkdir(parents=True, exist_ok=True)
    for name in ("src", "models"):
        link = SANDBOX / name
        if not link.exists():
            link.symlink_to(FRAMEWORK / name)
    if fresh or not (SANDBOX / "data").exists():
        subprocess.run([str(PYTHON), str(HERE / "make_mock_data.py"), str(SANDBOX)],
                       check=True, stdout=subprocess.DEVNULL)


def run_page(page: str) -> bool:
    items = [it for it in parse((WIKI / f"{page}.md").read_text()) if not it["norun"]]
    out_dir = RUNS / page
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    if not items:
        print(f"{page}: no runnable blocks")
        return True
    shutil.rmtree(SANDBOX / "report", ignore_errors=True)
    blocks_file = out_dir / "blocks.json"
    blocks_file.write_text(json.dumps([[it["lang"], it["source"]] for it in items]))
    driver = out_dir / "driver.py"
    driver.write_text(DRIVER)
    venv_bin = FRAMEWORK / ".venv" / "bin"
    env = dict(os.environ, PYTHONPATH="src", PYTHONWARNINGS="ignore",
               VIRTUAL_ENV=str(FRAMEWORK / ".venv"), PATH=f"{venv_bin}:{os.environ['PATH']}")
    done = subprocess.run([str(PYTHON), str(driver), str(blocks_file)], cwd=SANDBOX, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (out_dir / "stderr.txt").write_text(done.stderr)
    chunks = re.split(r"^@@BLOCK (\d+)@@\n", done.stdout, flags=re.M)
    outputs = {int(chunks[i]): chunks[i + 1] for i in range(1, len(chunks), 2)}
    ok = True
    for index, it in enumerate(items):
        label = f"{page}.md:{it['line']} [{it['lang']}]"
        actual = outputs.get(index)
        if actual is None:
            print(f"NOT RUN  {label}")
            ok = False
            continue
        (out_dir / f"block{index}.out").write_text(actual)
        if f"@@FAILED {index}@@" in actual:
            print(f"FAIL     {label}\n{actual}")
            ok = False
        elif it["expected"] is None:
            print(f"RAN      {label}")
        elif (missing := first_missing(it["expected"], actual)) is not None:
            print(f"MISMATCH {label}: expected line not found: {missing!r}")
            ok = False
        else:
            print(f"PASS     {label}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fresh-data", action="store_true")
    args = ap.parse_args()
    pages = (sorted(p.stem for p in WIKI.glob("*.md") if not p.stem.startswith("_") and p.stem != "AGENTS")
             if args.all else args.pages)
    ensure_sandbox(args.fresh_data)
    results = [run_page(p) for p in pages]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the fixture and verify the expected failure pattern**

Run: `T=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/wikitools; cd $T && WIKI_DIR=$T/fixture_wiki /Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework/.venv/bin/python run_examples.py Fixture; echo "exit $?"`
Expected:
```text
PASS     Fixture.md:3 [python]
PASS     Fixture.md:17 [bash]
MISMATCH Fixture.md:28 [python]: expected line not found: 'still 43'
exit 1
```
The norun block does not appear.

- [ ] **Step 4: Fix the fixture's last expectation to `still 42`, re-run**

Expected: three `PASS` lines, `exit 0`.

---

### Task 3: Style and link linter

**Files:**
- Create: `$T/lint_wiki.py`
- Test fixture: `$T/fixture_wiki/Bad-Style.md`

**Interfaces:**
- Produces: `python lint_wiki.py [PAGE ...]` (default every `*.md` in the wiki except `AGENTS.md`, including `_Sidebar.md`), output lines `<file>:<line>: <rule>: <excerpt>`, exit 1 on findings. `WIKI_DIR` overrides the directory.

- [ ] **Step 1: Write the fixture (the test)**

`$T/fixture_wiki/Bad-Style.md`:

````markdown
# Bad Style: Heading

This sentence has a dash – here.
Another sentence; with a semicolon.
- a bullet
1. a numbered item
You should not address the reader.
Allowed are `a:b` inline code, [a link](Fixture#fixture) and https://example.org/x:y.
Broken [page](Missing-Page) and [anchor](Fixture#nope).

| Column | Other |
|:---|---:|
| cell: bad | fine |

<!-- note: comments are exempt -->
```python
x = {"a": 1}; y = 2  # code is exempt
```
````

- [ ] **Step 2: Write the linter**

```python
"""Style and link lint for the wiki pages (AGENTS.md rules, see the spec).

Usage: python lint_wiki.py [PAGE ...]
"""
import os
import re
import sys
from pathlib import Path

WIKI = Path(os.environ.get("WIKI_DIR", "/Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework.wiki"))
FENCE = re.compile(r"^```.*?^```[ \t]*$", re.M | re.S)
FENCE_OPEN = re.compile(r"^```([A-Za-z]*)", re.M)
COMMENT = re.compile(r"<!--.*?-->", re.S)
INLINE = re.compile(r"`[^`\n]+`")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
URL = re.compile(r"<?https?://[^\s>)]+>?")
TABLE_SEP = re.compile(r"^[\s|:\-]*---[\s|:\-]*$")
LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
FORBIDDEN = re.compile(r"[–—:;]")
ADDRESS = re.compile(r"\b(?:[Yy]ou|[Yy]our|[Yy]ours|[Ww]e|[Oo]ur|[Uu]s)\b|\bI\b")
LANGS = {"python", "bash", "text"}


def blank(pattern, text, keep=None):
    def repl(m):
        kept = keep(m) if keep else ""
        return kept + re.sub(r"[^\n]", " ", m.group(0)[len(kept):])
    return pattern.sub(repl, text)


def slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(page: str) -> set[str]:
    body = FENCE.sub("", (WIKI / f"{page}.md").read_text())
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.*)$", body, re.M)}


def lint(path: Path) -> list[str]:
    text = path.read_text()
    findings = []
    for m in FENCE_OPEN.finditer(text):
        if m.start() == 0 or text[m.start() - 1] == "\n":
            lang = m.group(1).lower()
            if lang and lang not in LANGS:
                findings.append((text.count("\n", 0, m.start()) + 1, "fence-language", lang))
    prose = blank(FENCE, text)
    prose = blank(COMMENT, prose)
    no_code = prose
    prose = blank(INLINE, prose)
    prose = blank(LINK, prose, keep=lambda m: "[" + m.group(1) + "]")
    prose = blank(URL, prose)
    for number, line in enumerate(prose.splitlines(), start=1):
        if TABLE_SEP.match(line) and "|" in line:
            continue
        if LIST.match(line):
            findings.append((number, "list-marker", line.strip()))
        if FORBIDDEN.search(line):
            findings.append((number, "forbidden-char", line.strip()))
        if ADDRESS.search(line):
            findings.append((number, "direct-address", line.strip()))
    for m in LINK.finditer(blank(INLINE, no_code)):
        target = m.group(2)
        number = no_code.count("\n", 0, m.start()) + 1
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        page, _, anchor = target.partition("#")
        page = page or path.stem
        if page.endswith(".md") or not (WIKI / f"{page}.md").exists():
            findings.append((number, "broken-link", target))
        elif anchor and anchor not in anchors(page):
            findings.append((number, "broken-anchor", target))
    return [f"{path.name}:{n}: {rule}: {excerpt}" for n, rule, excerpt in findings]


def main() -> None:
    pages = sys.argv[1:] or sorted(p.stem for p in WIKI.glob("*.md") if p.stem != "AGENTS")
    findings = [f for page in pages for f in lint(WIKI / f"{page}.md")]
    print("\n".join(findings) if findings else "lint clean")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run on the fixture**

Run: `T=/private/tmp/claude-501/-Users-gl-Documents-2-HKA-Studium-M-Projektarbeit1-framework-wiki/12aabe39-c782-482c-90b1-c1a5b60a7577/scratchpad/wikitools; cd $T && WIKI_DIR=$T/fixture_wiki python3 lint_wiki.py Bad-Style; echo "exit $?"`
Expected findings, each exactly once, at these lines: 1 forbidden-char (heading colon), 3 forbidden-char, 4 forbidden-char, 5 list-marker, 6 list-marker, 7 direct-address, 9 broken-link `Missing-Page`, 9 broken-anchor `Fixture#nope`, 13 forbidden-char (`cell: bad`). No finding for line 8, the separator row, the comment or the code block. `exit 1`.

- [ ] **Step 4: Run on the good fixture**

Run: `WIKI_DIR=$T/fixture_wiki python3 lint_wiki.py Fixture` → `lint clean`, exit 0. Fix the linter if either step deviates.

---

### Task 4: Home, Installation, Quick Start

**Files:**
- Create: `$W/Home.md`, `$W/Installation.md`, `$W/Quick-Start.md`

**Interfaces:**
- Consumes: runner and linter (Tasks 2 and 3), mock tree (Task 1).
- Produces: page names and anchors referenced by later pages: `Installation#working-directory`, `Quick-Start`.

Content requirements (facts verified against code, report sections in brackets):

**Home.md** (no code blocks)
- One paragraph on purpose: evaluation of synthetic or reconstructed images and volumes, with or without a reference image, medical CT/MRI as starting point, ordinary 2D images take the same path [report 4.1].
- Four metric families [report 2.3] as a table with columns Family, Needs target, Examples, Question answered: signal-based FR (psnr, ssim, fsim, gmsd, vsi), learning-based FR (lpips, dists, radimagenet_lpips, dreamsim), NR (clipiqa, clip_iqa_lung, clip_iqa_brain, brisque, niqe, ilniqe, piqe, musiq, maniqa, paq2piq), segmentation (dice, hausdorff95, nsd, assd, panoptic_quality, boundary_iou, vs, vs_signed, v_pred, v_gt, tp).
- Scope limits in prose: no graphical interface, no statistical analysis of the result table, no training of metric models, no resampling of inputs and targets on different grids [report 3, after Tabelle 3.1].
- Properties in prose: explicit metric selection (registry starts empty), per-row logging of normalization and range, fault tolerance (a failing metric or file leaves empty values), backend independence via a narrow protocol [report NFA-1..5].
- Page guide table (Page, Purpose) linking all 14 other pages.
- One sentence naming the report as background source: "The theoretical background and the design rationale are described in the project report *Framework für die Bildqualitätsbewertung anhand referenzbasierter und referenzloser Metriken* (Gleb Janickij, Hochschule Karlsruhe, summer semester 2026)." Rephrase to avoid colons.

**Installation.md**
- Requirements prose: Python 3.12 or newer (`requires-python >=3.12`, developed on 3.14 [report 5.1]), Hatch and uv on `PATH`, git. Network access on first use of a network-backed metric (pyiqa weights into the torch hub cache, CLIP RN50 for the CLIP metrics, DreamSim weights about 1 GB into `models/dreamsim`).
- Clone (norun): `git clone https://github.com/glebjan/hka.projektarbeit1.IQA_MRI.git` then `cd hka.projektarbeit1.IQA_MRI`.
- Two environments as table [report Tabelle A.2, pyproject]: `default` → `.venv/` runtime (`hatch env create`), `calibration` → `.venv-calibration/` with extras `calibration` (medpy 0.5.2, surface-distance 0.1, panopticapi commit 7bb4655) plus pytest, ruff, pre-commit, yapf (`hatch env create calibration`). Alternative without Hatch (norun): `uv venv .venv && uv pip install -e .`.
- Why exact pins: pyiqa scores depend on backend versions [report NFA-1, 5.2]. `setuptools<81` is required because `openai-clip` imports `pkg_resources`.
- Model weights table: `radimagenet_lpips` needs `models/RadImageNet_pytorch/ResNet50.pt`, which is part of the repository (90 MB, tracked in git), `dreamsim` downloads into `models/dreamsim` on first use (git-ignored), CLIP and pyiqa weights download to the user cache on first use.
- Section `## Working Directory` (anchor `working-directory`): all relative paths in `iqaevaluator/constants.py` (`RESNET50`, `DREAMSIM_CACHE`, `REPORT = report/IXI661_report.csv`) resolve against the current directory, so commands run from the repository root. `evaluate()` lives in the reference script `src/main.py`, which is not part of the installed package, hence `export PYTHONPATH=src` (norun block: `source .venv/bin/activate` and `export PYTHONPATH=src`).
- Device: `DEVICE` is `cuda` when available, otherwise `cpu`, resolved once at import and shared by all adapters [metric_spec.py, report 5.1]. DreamSim alone accepts a `device` argument (e.g. `"mps"`).
- Runnable check block (python):

```python
from iqaevaluator.metrics import BUILTIN_METRICS, SEGMENTATION_METRICS, DEVICE
print(len(BUILTIN_METRICS), len(SEGMENTATION_METRICS), DEVICE)
```
with its expected output from the run (`18 11 cpu` on the author's machine; the prose notes that the device differs per machine).

**Quick-Start.md**
- Mock data note in prose: the examples use files under `data/`, which stand for the user's own reference and generated volumes.
- CLI run (bash): `python src/main.py data/generated/model_a data/reference` then show the tail of the output from the run (skipped-metric message is absent in slice mode for BUILTIN metrics; `CSV written`, `Report written: report/IXI661_report.csv`). Explain in prose that `main.py` registers all 18 `BUILTIN_METRICS`, writes the CSV to `constants.REPORT` and prints `describe()`. Mention `--mode {slice,volume}` and `--normalization {minmax,percentile,raw,mask}` and `hatch run evaluate <args>` (norun).
- Because a full BUILTIN run is slow on CPU (measure it in Task 1 Step 3 and state the measured order of magnitude only if it is reproducible, else omit), the Python example uses a small registry:

```python
from pathlib import Path
from main import evaluate, MetricRegistry, PSNR, SSIM, NIQE

registry = MetricRegistry(PSNR, SSIM, NIQE)
result = evaluate(Path("data/generated/model_a"), Path("data/reference"), registry=registry)
df = result.to_frame()
print(df[["image_id", "mode", "is_empty", "psnr", "ssim", "niqe"]].head(4).round(3).to_string())
```
- Then the NR-only call (`evaluate(Path("data/generated/model_b/unpaired_scan.nii.gz"), registry=registry)`) showing that psnr/ssim stay empty and `mode` is `no_reference`.
- Then `result.generate_report(Path("report/quickstart.csv"))`.
- Closing paragraph with links to Core-Concepts, Selecting-Metrics, Results-and-Reports.

- [ ] **Step 1: Write the three pages** following the requirements, with expected-output blocks left empty for now.
- [ ] **Step 2: Run** `cd $T && $FW/.venv/bin/python run_examples.py Home Installation Quick-Start`, copy outputs from `$T/runs/<Page>/block<N>.out` into the `text` blocks (elide with `...`), re-run until every block is `PASS` or deliberately `RAN`.
- [ ] **Step 3: Lint** `python3 $T/lint_wiki.py Home Installation Quick-Start` → `lint clean` except broken links to pages not yet written (allowed until Task 12, list them in the task report).
- [ ] **Step 4: Report alignment check.** Compare every claim against report sections 1, 3, 4.1, 5.1, 5.2, A.2, A.3. New disagreements go into `$T/discrepancies.md`.

---

### Task 5: Core Concepts, Loading Images

**Files:**
- Create: `$W/Core-Concepts.md`, `$W/Loading-Images.md`

**Content requirements**

**Core-Concepts.md**
- Input and target, FR and NR: with a target both metric kinds run, without one FR metrics are skipped per record (`spec.reference and not has_target → continue` in both evaluators), `mode` column `full_reference` or `no_reference`.
- Component roles as a table (Component, Responsibility, Module) following report 4.2 and Abbildung 4.1: `ImageLoader`/`LoadedImage` (image_loader.py), `Normalizer` (normalization.py), `MetricSpec`/`Metric` (metric_spec.py), `MetricRegistry` (metrics.py), `IQAEvaluator`/`VolumeEvaluator`/`build_evaluator` (iqa_evaluator.py, volume_evaluator.py, evaluator_factory.py), `ImageEvaluatorRecord` (records.py), `EvaluationResult` (evaluation_result.py), `evaluate()` (src/main.py, reference script).
- Registry per run: starts empty, `MetricRegistry(*specs)`, instances built lazily on first use and cached per (name, mode, spacing), two registries never share state [report 4.2, A.1]. An empty registry is a valid selection for the evaluators, but `evaluate()` raises `ValueError` when no registered metric can serve the chosen mode [main.py].
- Data flow in prose following Abbildung 4.2: pairing → loading and normalization (target scale for FR, own scale for NR) → `registry.select(mode)` → evaluator by mode → records → `EvaluationResult` → optional CSV.
- Fault tolerance: a failing metric prints `[<path>] metric '<name>' batch failed: ...` and leaves `None` for that batch, a failing file prints `[<path>] evaluation failed: ...` and the run continues. A missing value therefore means "not registered" or "failed here", only the run output distinguishes the two [report 4.3].
- Example (python): the lower-level path without `evaluate()`:

```python
from pathlib import Path
from iqaevaluator.image_loader import load_pair
from iqaevaluator.evaluator_factory import build_evaluator
from iqaevaluator.evaluation_result import EvaluationResult
from iqaevaluator.metrics import MetricRegistry, PSNR, SSIM

registry = MetricRegistry(PSNR, SSIM)
inp, tgt = load_pair(Path("data/generated/model_a/subject01_t1.nii.gz"), Path("data/reference/subject01_t1.nii.gz"))
records = build_evaluator(inp, tgt, registry, mode="slice", source_model="model_a").run_evaluation()
result = EvaluationResult.from_records(records, registry)
print(result.to_frame()[["image_id", "source_model", "slice_index", "is_empty", "psnr", "ssim"]].head(3).round(3).to_string())
```
- A short example showing `registry.specs` names and `registry.direction`.

**Loading-Images.md**
- Formats table (Extension, Decoder library, Decoding steps, Spacing source, Volumetric when) from image_loader.py and report 5.3: `.png .jpg .jpeg` Pillow, converted to 8-bit grayscale, depth 1, no spacing, never volumetric. `.dcm` pydicom, RescaleSlope and RescaleIntercept applied, MONOCHROME1 inverted, RGB converted to luminance, WindowCenter/WindowWidth ignored, spacing from SliceThickness and PixelSpacing, volumetric with more than one frame and spacing and no FrameTime/CineRate. `.nii .nii.gz` nibabel, `as_closest_canonical`, stored dtype kept, 3D transposed to (Z, X, Y) with spacing (dz, dx, dy), 4D flattened with no spacing and never volumetric. `.nrrd .mha .mhd` SimpleITK, spacing reordered to array axes.
- Tensor contract: `(D, 1, H, W)`, float32 in [0, 1] under every strategy except `Raw()` (dtype kept) and exactly {0, 1} under `Mask()`. `rgb_tensor` repeats the channel three times. Spacing is `(depth, height, width)` in mm, ordered like the array axes.
- `ImageLoader` is lazy (decodes on first property access). Properties list in prose: `raw`, `raw_range`, `intensity_range`, `tensor`, `rgb_tensor`, `spacing`, `is_volumetric`, `empty_slice_mask`, `log_tensor_shape()`.
- Examples (python): load NIfTI and print shape, spacing, is_volumetric, raw dtype, empty slices, raw_range. Load DICOM and show raw_range reflects the −1024 intercept. Load NRRD and show identical tensor to NIfTI (`torch.equal`). Load 4D and show `is_volumetric False`, spacing `None`. PNG shape.
- Pairing section: `list_images()` walks the directory recursively and keeps supported files sorted, `find_matching_target()` strips all extensions (`name.split(".")[0]`) and picks the target with the longest shared prefix, which must be at least four characters. Unmatched inputs run as NR. `evaluate()` warns when one target matches several inputs and when targets remain unmatched.
- Pitfall example (python) with `data/pitfalls/generated` vs `data/reference`, showing both warnings and that `subject01_t1_v2` was paired with `subject01_t1`. Advice in prose: give each generated file exactly the base name of its reference, keep variants in separate directories.
- Dataset survey example: `from iqaevaluator.data import analyze` then `analyze(Path("data/reference"), Path("report/tensor_sizes.csv"))` (not `python -m iqaevaluator.data`, see D6).

- [ ] **Step 1: Write both pages.**
- [ ] **Step 2: Run** `run_examples.py Core-Concepts Loading-Images`, fill expected outputs, re-run to green.
- [ ] **Step 3: Lint** both pages (only not-yet-written link targets may remain).
- [ ] **Step 4: Report alignment** against report 4.2, 4.3, 5.3, Abbildungen 4.1 and 4.2. Log disagreements.

---

### Task 6: Normalization

**Files:**
- Create: `$W/Normalization.md`

**Content requirements**
- Why: every backend expects [0, 1] and none normalizes per image, so the loader decides [normalization.py docstring].
- The single mapping `scale()` in prose: `clip((x - lo) / (hi - lo), 0, 1)`, shown as a code line, not as math.
- Strategies table (Strategy, Range, Empty-slice rule, Report name, Use): `MinMax()` own min/max, heuristic `sparse_slices`, `minmax`, default. `Percentile(lower=0.5, upper=99.5)` robust percentiles with fallback to extremes when the span collapses, heuristic, `percentile_0.5_99.5`, opt-in, changes every score. `FixedRange(range, name)` a given `IntensityRange`, heuristic, the given name, used internally by `load_pair`. `Mask(label=None, threshold=0.5)` none (binarizes), exact voxel count, `mask` or `mask_label_<k>`, segmentation masks. `Raw()` none, heuristic, `raw`, instance maps only.
- Coupling to the target: `load_pair` scales the target by the strategy and gives its range to the input as `FixedRange` with the same name, values outside are clipped, so a uniformly too bright prediction shows as a difference instead of being normalized away (fastMRI convention) [report 4.3, 5.3, A.1]. Under `Mask()` and `Raw()` both sides are loaded independently. Scaling is per volume, never per slice, so slice rows of one volume are not independent samples [main.py docstring].
- Logged columns: `normalization`, `scale_lo`, `scale_hi` (empty under `Mask()` and `Raw()`), `input_min`, `input_max` (raw input extremes). `input_max > scale_hi` signals overshoot.
- Empty slices: `sparse_slices` marks a slice empty when its spread, or its lift above the volume floor, is below 0.1 % of the 0.5 to 99.5 percentile span. Report 5.3 mentions only the spread criterion, verify and log as a discrepancy if the report must mention lift (the report says "deren Streuung unter 0,1 % der globalen Perzentilspanne liegt").
- Degenerate ranges print a warning (`constant image` or `degenerate range`) and keep values clipped.
- Examples (python): model_b vs reference under MinMax, print `normalization, scale_lo, scale_hi, input_min, input_max` for one row, show `input_max > scale_hi`. Same under `Percentile()`, show different name and psnr change. Direct use of `scale()` and `IntensityRange` on a tiny numpy array. CLI `--normalization percentile` (bash) head of output. `normalizer_from_name("percentile")`.
- Custom strategy example: a class `ZScoreClip` (frozen dataclass) implementing `name`, `range_of` (mean ± 3 std), `apply` (via `scale`), `empty_slices` (via `sparse_slices`), passed as `normalization=` to `evaluate()`; show its name in the `normalization` column. Note that `isinstance(obj, Normalizer)` holds because the protocol is runtime-checkable.

- [ ] **Step 1: Write the page.** - [ ] **Step 2: Run and fill outputs.** - [ ] **Step 3: Lint.** - [ ] **Step 4: Report alignment** against report 4.2, 4.3, 5.3, A.1, NFA-1.

---

### Task 7: Selecting Metrics, Metric Catalog

**Files:**
- Create: `$W/Selecting-Metrics.md`, `$W/Metric-Catalog.md`

**Content requirements**

**Selecting-Metrics.md**
- Import locations: everything is re-exported from `main` (for users of `evaluate`) and from `iqaevaluator.metrics` (`BUILTIN_METRICS`, `SEGMENTATION_METRICS`, the constants, `DREAMSIM`, `dreamsim_spec`, `MetricRegistry`, `MetricSpec`, `ModeSupport`, `ModeUnsupported`). Builders live in `iqaevaluator.segmentation_metrics.monai_metrics` (`dice_metric`, `hausdorff95_metric`, `normalized_surface_dice_metric`, `average_surface_distance_metric`, `panoptic_quality_metric`) and `iqaevaluator.segmentation_metrics.boundary_iou` (`boundary_iou_metric`).
- Registry construction examples: `MetricRegistry(PSNR, SSIM)`, `MetricRegistry(*BUILTIN_METRICS)`, `MetricRegistry(*BUILTIN_METRICS, DREAMSIM)`, `registry.register(LPIPS)` later, re-registering a name replaces the spec and drops its cached instances.
- Selecting by property: comprehension over `BUILTIN_METRICS` filtering `spec.reference is False` (all NR metrics) or `spec.direction`. Print the resulting names.
- Why DreamSim is opt-in: about 1 GB download, six ViT forward passes per slice pair for the ensemble [dreamsim_metric.py, report A.1]. `dreamsim_spec(dreamsim_type=..., pretrained=..., device=..., cache_dir=..., normalize_embeds=..., use_patch_model=..., name=...)`, valid types table (`ensemble`, `dino_vitb16`, `clip_vitb32`, `open_clip_vitb32`, `dinov2_vitb14`, `synclr_vitb16`), patch model only for `dino_vitb16` and `dinov2_vitb14`, derived names (`dreamsim`, `dreamsim_dino_vitb16`, `dreamsim_dino_vitb16_patch`, `dreamsim_scratch`, `dreamsim_rawembeds`), only the default name has a dedicated record field, variants land in extra columns. Example printing `.name` and `.builtin` for three configurations (no weights needed, construction is lazy). One run with `dreamsim_spec(dreamsim_type="dino_vitb16")` on one PNG pair (weights are present locally).
- Configurable segmentation builders: `hausdorff95_metric(percentile=None)` gives plain Hausdorff (name stays `hausdorff95`, so register only one variant per registry), `normalized_surface_dice_metric(class_thresholds=[2.0])`, `boundary_iou_metric(dilation_ratio=0.05)`, `panoptic_quality_metric(match_iou_threshold=0.3)`. Passing `threshold` or `label` raises `TypeError` (removed knobs, binarization belongs to `Mask()`), show the message.
- Mode filter: `registry.select("volume")` returns applicable specs and `SkippedMetric` entries, `evaluate()` prints the grouped message (copy the exact text from a run with `MetricRegistry(PSNR, LPIPS, NIQE)` in volume mode on `data/reference/subject01_t1.nii.gz`), and raises `ValueError` when nothing applies (show the message with `MetricRegistry(LPIPS)`).

**Metric-Catalog.md**
- First read `$THESIS` lines 400 to 977 (report 2.3.2 to 2.4) so interpretation sentences align with the report's descriptions and value ranges.
- Main table, one row per metric in the order of report Tabelle A.4, columns: Name, Constant, Family, Target, Channels, Direction, Slice, Volume, Backend. Values from `metrics.py`, `dreamsim_metric.py`, segmentation modules: channels `gray` for psnr, ssim and all segmentation metrics, `rgb` for the rest. Direction `higher`, `lower` or `not ranked` (vs_signed, v_pred, v_gt, tp). Volume support only psnr, ssim and all 11 segmentation metrics. Backend pyiqa, pyiqa plugin (radimagenet_lpips, clip_iqa_lung, clip_iqa_brain), DreamSim, MONAI (dice, hausdorff95, nsd, assd, panoptic_quality, and psnr/ssim in volume mode), own function (boundary_iou, vs, vs_signed, v_pred, v_gt, tp).
- Per-family sections with one or two sentences per metric (principle, interpretation, domain caveat) and a source link. Sources:
  - psnr, MSE: Wang and Bovik 2009, https://doi.org/10.1109/MSP.2008.930649
  - ssim: Wang et al. 2004, https://doi.org/10.1109/TIP.2003.819861
  - fsim: Zhang et al. 2011, https://doi.org/10.1109/TIP.2011.2109730
  - gmsd: Xue et al. 2014, https://doi.org/10.1109/TIP.2013.2293423
  - vsi: Zhang, Shen and Li 2014, https://doi.org/10.1109/TIP.2014.2346028
  - lpips: Zhang et al. 2018, https://doi.org/10.1109/CVPR.2018.00068
  - dists: Ding et al. 2022, https://doi.org/10.1109/TPAMI.2020.3045810
  - radimagenet_lpips: LPIPS with a RadImageNet ResNet50 backbone, Mei et al. 2022, https://doi.org/10.1148/ryai.210315
  - dreamsim: Fu et al. 2023, https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html
  - clipiqa: Wang, Chan and Loy 2023, https://doi.org/10.1609/aaai.v37i2.25353, CLIP backbone Radford et al. 2021, https://proceedings.mlr.press/v139/radford21a.html
  - clip_iqa_lung, clip_iqa_brain: CLIP RN50 with five organ-specific positive/negative prompt pairs each (see D3), list the pairs in a table (Organ, Positive prompt, Negative prompt) copied from `clip_iqa_medical.py`.
  - brisque: Mittal, Moorthy and Bovik 2012, https://doi.org/10.1109/TIP.2012.2214050
  - niqe: Mittal, Soundararajan and Bovik 2013, https://doi.org/10.1109/LSP.2012.2227726
  - ilniqe: Zhang, Zhang and Bovik 2015, https://doi.org/10.1109/TIP.2015.2426416
  - piqe: Venkatanath et al. 2015, https://doi.org/10.1109/NCC.2015.7084843
  - musiq: Ke et al. 2021, https://doi.org/10.1109/ICCV48922.2021.00510
  - maniqa: Yang et al. 2022, https://doi.org/10.1109/CVPRW56347.2022.00126
  - paq2piq: Ying et al. 2020, https://doi.org/10.1109/CVPR42600.2020.00363
  - dice, hausdorff95, nsd, assd: Maier-Hein et al. 2024 (Metrics Reloaded), https://doi.org/10.1038/s41592-023-02151-z, nsd additionally Nikolov et al. 2021, https://doi.org/10.2196/26151, Hausdorff distance Huttenlocher et al. 1993, https://doi.org/10.1109/34.232073
  - boundary_iou: Cheng et al. 2021, https://doi.org/10.1109/CVPR46437.2021.01508
  - panoptic_quality: Kirillov et al. 2019, https://doi.org/10.1109/CVPR.2019.00963
  - vs, vs_signed: Taha and Hanbury 2015, BMC Medical Imaging 15:29. Verify the DOI with WebFetch on `https://doi.org/10.1186/s12880-015-0068-x` before using it, else cite without link. Not in the report's bibliography, so note in the discrepancy list as "wiki cites a source absent from the report" (informational).
  - backends: pyiqa https://github.com/chaofengc/IQA-PyTorch, MONAI https://monai.io
- Interpretation notes: NR metrics and DreamSim were trained or tuned on natural photographs, use them for relative comparison within one study only [report 6.1, dreamsim_metric.py]. maniqa is the slowest pyiqa metric on CPU (measure on the mock volume in this task and state it only as a relative statement, e.g. "several times slower than the other metrics", unless the measurement is stable). Segmentation value ranges and units (distance metrics in voxels in slice mode, millimetres in volume mode) with link to Scoring-Modes.
- Example (python): print a generated table of all specs from `BUILTIN_METRICS + (DREAMSIM,) + SEGMENTATION_METRICS` (name, reference, channels, direction, slice and volume support via `isinstance(spec.volume_mode, ModeSupport)`) and assert it matches the page table (the example prints, the author compares). Print `SEGMENTATION_METRICS` descriptions for two metrics via `spec.description`.

- [ ] **Step 1: Write both pages.** - [ ] **Step 2: Run and fill outputs.** - [ ] **Step 3: Lint.** - [ ] **Step 4: Report alignment** against report 2.3 (Tabellen 2.1 to 2.4 directions and ranges), 5.4, A.4. Log disagreements.

---

### Task 8: Scoring Modes, Segmentation Evaluation

**Files:**
- Create: `$W/Scoring-Modes.md`, `$W/Segmentation-Evaluation.md`

**Content requirements**

**Scoring-Modes.md**
- Slice mode is the default and the only mode for 2D methods, one row per slice, `image_id` = `<stem>_s<NNN>` (three digits) prefixed by `<source_model>/` when set, `slice_index` set. Only slices empty on both sides are skipped (`is_empty True`, metrics empty), NR runs decide by the input alone, one-sided emptiness is scored as a real failure [iqa_evaluator.py, report 4.3]. Batches of `BATCH_SIZE = 32` slices, lower it on GPU memory errors (`iqaevaluator.iqa_evaluator.BATCH_SIZE`).
- Volume mode: one row per volume, tensor `(1, C, D, H, W)`, no empty-slice filtering, `slice_index` empty, `is_empty` always False, `scoring` = `volume`. Requires a real spatial depth axis (`is_volumetric`), otherwise the file is skipped with a message. Input and target must have identical spacing, otherwise `ValueError` (resampling is out of scope) [volume_evaluator.py, report 4.3, 5.6].
- Metric support: only psnr, ssim and the 11 segmentation metrics serve volume mode. Volume psnr and ssim come from MONAI (`MonaiPSNRMetric` with `max_val=1.0`, `MonaiSSIMMetric` with a 3D Gaussian window of up to 11 voxels shrunk to the largest odd size that fits every axis, at least 3), so they are different estimators from the slice-mode pyiqa ones [volumetric_iqa.py, report 5.4].
- Comparability table (Metric, Slice mode unit, Volume mode unit): hausdorff95, assd in voxels vs millimetres, nsd tolerance `class_thresholds=[1.0]` means one voxel vs one millimetre, boundary_iou chessboard band in voxels vs Euclidean band in physical units, psnr/ssim different estimators. Conclusion sentence: compare a column only within one mode [main.py docstring, report 4.3].
- Examples: same pair in slice and volume mode with `MetricRegistry(PSNR, SSIM)`, printing both frames. Volume mode on a PNG (skip message) and on `timeseries_4d` (skip message). Spacing mismatch with `data/broken/spacing` (message printed by `evaluate` as `evaluation failed: ...`). CLI `--mode volume` (bash).
- `aggregate_volumes()` summary sentence with link to Segmentation-Evaluation.

**Segmentation-Evaluation.md**
- Masks load with `Mask()`, the only valid strategy for binary segmentation metrics [report 4.4]. Rules from `volume.as_mask`: bool as is, integer maps and float maps with only finite whole non-negative values are label maps (non-zero = foreground, `label=k` selects one class), other float arrays are probability maps cut at `threshold` (values outside [0, 1] raise, `label` with a probability map raises). A 0/255 PNG and a 0/1 NIfTI binarize identically.
- Multi-class: one class per run via `Mask(label=k)`, no one-hot path, a multi-channel batch raises [require_single_channel].
- Instance maps: `Raw()` for `panoptic_quality` only, ids preserved, binary masks under `Mask()` score single-instance PQ.
- Empty-mask policy table (Metric, One side empty, Both empty): dice, nsd, panoptic_quality, vs, boundary_iou → 0.0 / empty. hausdorff95, assd → empty / empty. vs_signed → ±2 by arithmetic, verify by running (`2*(vp-vg)/(vp+vg)` gives −2.0 for an empty prediction) / empty. v_pred, v_gt, tp → counts (0 on the empty side) / 0 (on slices both empty are skipped in slice mode, counts filled as 0 by `aggregate_volumes`). Never infinite [report 4.4, volume.empty_policy, adapters]. Verify every cell by running a small example with synthetic masks written to `report/` or via direct metric calls, and correct the table from the run.
- Distances without spacing print a warning and are in voxels, an explicit `spacing=` passed to a builder wins over the file's spacing [monai_metrics._volume_factory].
- `aggregate_volumes()`: sums v_pred, v_gt, tp per volume (image_id without `_sNNN`), then dice, vs, vs_signed. Requires the three count metrics (else `ValueError` with the message), refuses volume-mode results (`ValueError`), empty slices contribute zeros, an uncounted slice makes the volume NaN with a warning. Distance and perceptual metrics are not aggregated, run volume mode for them [evaluation_result.py, report 4.3, 5.6].
- Examples (python): full `SEGMENTATION_METRICS` run in slice mode on `data/masks/predicted` vs `data/masks/reference` with `normalization=Mask()`; `Mask(label=2)` run for the lesion; `aggregate_volumes()` output; volume-mode run and comparison of volume dice with aggregated dice (they agree, show both). PNG masks 0/255 with `Mask()`. Probability map with `Mask(threshold=0.5)` vs `Mask(threshold=0.8)` (dice changes). Instance maps with `Raw()` and `MetricRegistry(PANOPTIC_QUALITY)`, plus `panoptic_quality_metric(metric_name="sq")`. Error example: dice on masks loaded with `MinMax()` prints `dice expects a binary mask with values in {0, 1}; load masks with normalization.Mask()` from the batch failure line (probability map under MinMax is non-binary; copy the exact message).
- Calibration pointer: link to Testing-and-Calibration.

- [ ] **Step 1: Write both pages.** - [ ] **Step 2: Run and fill outputs.** - [ ] **Step 3: Lint.** - [ ] **Step 4: Report alignment** against report 2.3.4 (Tabelle 2.4), 4.3, 4.4, 5.5, 5.6. Log disagreements.

---

### Task 9: Results and Reports, Extending the Framework

**Files:**
- Create: `$W/Results-and-Reports.md`, `$W/Extending-the-Framework.md`

**Content requirements**

**Results-and-Reports.md**
- `EvaluationResult` methods: `to_frame()` (no I/O, fixed columns in record order plus one column per non-builtin spec of the registry), `generate_report(path=constants.REPORT)` (creates parent directories, writes CSV without index, prints `CSV written: <path>`, returns the frame), `aggregate_volumes()`, `EvaluationResult.from_records(records, registry)`.
- Columns table following report Tabelle A.5, verified against `records.py`: image_id, source_model, mode, scoring, slice_index, is_empty, normalization, scale_lo, scale_hi, input_min, input_max, 19 built-in metric fields (18 builtins plus `dreamsim`), then extra columns. State that segmentation metrics and DreamSim variants appear as extra columns (builtin=False) after the fixed ones, and that built-in columns exist even when not registered (empty).
- `source_model`: `evaluate()` does not set it, `build_evaluator(..., source_model=...)` does, and it prefixes `image_id`. Report 4.2 describes the factory parameter, consistent.
- Examples: column list of a frame; reading the CSV back with pandas; per-volume means for an IQA metric with `df[~df.is_empty].groupby(df.image_id.str.replace(r"_s\d+$", "", regex=True))[...]`, with a sentence that per-slice rows are not independent and that ratio metrics must not be averaged (link to Segmentation-Evaluation); ranking two models by mean psnr with direction from `registry.direction`; filtering rows where `input_max > scale_hi`.

**Extending-the-Framework.md**
- Three extension paths without touching evaluator, record or result [report 4.4]: custom metric, custom backend adapter, custom normalization strategy (the last one already on Normalization, link it).
- Metric protocol: `__call__(input, target=None) -> Sequence[float]`, batches `(N, C, H, W)` float32 in [0, 1], one score per sample, runtime-checkable, no base class [metric_spec.py, report 5.4].
- `registry.register_metric(name, metric, *, direction, reference, channels="rgb", volume_factory=None, volume_reason=REASON_NO_VOLUME_IMPL)`. Note that the default channels is `rgb`. Scores go to `record.extra[name]` and appear as a column.
- Example 1 (python): `MeanIntensity` NR metric (the report's own example in 5.4) registered with `direction="higher_is_better", reference=False, channels="gray"`, run on `data/reference`, show the column.
- Example 2: FR metric `MeanAbsoluteError` with `volume_factory=lambda spacing: VolumeMAE()` that handles `(1, C, D, H, W)`, run in both modes.
- Example 3: hand-built `MetricSpec(name, direction, reference, channels, slice_mode=ModeSupport(factory), volume_mode=ModeUnsupported("needs 2D input"), builtin=False, description=...)`, factory called lazily on first use (demonstrate with a print in the factory, called once for many slices).
- Example 4: backend adapter wrapping a third-party function (use `skimage` only if installed in `.venv`, else scipy or numpy based sharpness e.g. variance of the Laplacian from `scipy.ndimage.laplace`), showing how an adapter translates the tensor batch.
- Failure behaviour: an exception inside a custom metric is printed and yields empty values for that batch (demonstrate with a metric that raises for one batch).
- Rules for authors in prose: return exactly N values, keep the metric stateless or cache inside, do not load weights in `__init__` if the registry should stay cheap (the framework builds instances lazily), respect `reference`.

- [ ] **Step 1: Write both pages.** - [ ] **Step 2: Run and fill outputs.** - [ ] **Step 3: Lint.** - [ ] **Step 4: Report alignment** against report 4.2, 4.4, 5.4, 5.6, A.5. Log disagreements.

---

### Task 10: Worked Examples

**Files:**
- Create: `$W/Worked-Examples.md`

**Content requirements** (each scenario: goal paragraph, complete runnable code, output, two to four sentences of reading the result)
1. Benchmarking two generative models against one reference set. Loop over `data/generated/model_a` and `model_b`, run `evaluate` per model with `MetricRegistry(PSNR, SSIM, LPIPS, NIQE)`, add a `model` column, concatenate, per-model per-volume means on non-empty FR rows, show that model_b's `input_max > scale_hi` explains lower psnr, save CSV. Mention that `unpaired_scan` appears as NR rows.
2. Same benchmark through `build_evaluator(..., source_model=...)` and `EvaluationResult.from_records` so `source_model` is filled natively.
3. No-reference screening of synthetic volumes without ground truth with `MetricRegistry(NIQE, BRISQUE, CLIP_IQA_BRAIN)` on `data/generated/model_b`, with the caveat about natural-image training.
4. Downstream segmentation check: masks of a generated volume vs reference masks, class-wise (`Mask(label=1)`, `Mask(label=2)`) in slice mode with aggregation, then volume mode for hausdorff95 and nsd in millimetres.
5. PNG image pairs (2D) with FR and NR metrics and the DreamSim single-backbone variant.
6. Normalization sensitivity: the same pair under `MinMax()` and `Percentile()` with psnr side by side, conclusion that runs are comparable only under the same strategy.

- [ ] **Step 1: Write the page.** - [ ] **Step 2: Run and fill outputs** (this page is slow, run it alone). - [ ] **Step 3: Lint.** - [ ] **Step 4: Check that every statement repeats facts already established on the topic pages and link back to them.**

---

### Task 11: Testing and Calibration, Troubleshooting

**Files:**
- Create: `$W/Testing-and-Calibration.md`, `$W/Troubleshooting.md`

**Content requirements**

**Testing-and-Calibration.md**
- Test suite: one test file per source module, synthetic fixtures without network in `tests/conftest.py` (including anisotropic volumes), 858 tests (verify with `cd $FW && .venv-calibration/bin/python -m pytest --collect-only -q | tail -1`, expected `858 tests collected`), run with `hatch run calibration:test` or a subset `hatch run calibration:test tests/test_normalization.py` (norun blocks, but execute them once manually in `$FW` and paste the real summary line).
- Calibration: `tests/calibration/` with `cases.py` (hand-derived cases with derivation and source), `harness.py` (runs through the public path `load_pair` → `MetricRegistry` → `build_evaluator`), `official.py` (MedPy 0.5.2, surface-distance 0.1, panopticapi commit 7bb4655, same-variant vs divergent variants, surface-distance area-weighted vs voxel counting), `monai_cases.py` (MONAI 1.6.0 test cases), missing packages give a named skip [report 5.7]. Report regeneration `hatch run calibration:report` writes `docs/calibration/segmentation.md`. Link to it on GitHub: `https://github.com/glebjan/hka.projektarbeit1.IQA_MRI/blob/main/docs/calibration/segmentation.md` (branch caveat D8).
- Table of what is calibrated (Metric, Hand cases, Official reference, MONAI cases) derived from the section headings of `$FW/docs/calibration/segmentation.md`.
- Reading the calibration report: authority order (hand value, same-variant official implementation, MONAI cases), Δ column, divergent variants column.

**Troubleshooting.md**
- Table-free structure: one `###` section per symptom, each with the exact message in a `text` block (copied from a run), then cause and remedy in prose. Every message must be produced by a runnable example on this page or already verified on another page (reference that page).
- Required symptoms and messages (verify exact wording by running): shape mismatch (`evaluation failed: shape mismatch: input (10, 1, 128, 128) vs target (12, 1, 128, 128)` from `data/broken/shape`), different voxel sizes in volume mode, not a 3D volume in volume mode, no metric applicable in the chosen mode, metrics skipped in the chosen mode, target matched multiple inputs, targets without input, no supported files in a directory, `Unsupported format`, binary mask required (`dice expects a binary mask ...`), probability map out of range (`as_mask expects float values in [0, 1] ...` via `Mask()` on a MinMax-incompatible file, construct with a small NIfTI written to `report/` in the example), `label` with a probability map, removed `threshold`/`label` keyword on a builder (`TypeError`), unknown DreamSim type, patch model on unsupported type, unknown normalization name, distance metrics without spacing (warning), degenerate range / constant image warnings, `aggregate_volumes` without count metrics and on a volume-mode result, GPU out of memory (lower `BATCH_SIZE`, no runnable example, norun), network needed for first weight download (no runnable example), working directory wrong so `models/...` not found (explain `constants.py`, no example).

- [ ] **Step 1: Write both pages.** - [ ] **Step 2: Run and fill outputs; run the norun test commands manually in `$FW` and paste real output.** - [ ] **Step 3: Lint.** - [ ] **Step 4: Report alignment** against report 5.7 and NFA-3, NFA-5.

---

### Task 12: Sidebar and full verification

**Files:**
- Create: `$W/_Sidebar.md`
- Modify: any page with findings
- Modify: `$T/discrepancies.md`

- [ ] **Step 1: Write `_Sidebar.md`** without list markers:

```markdown
**Getting Started**<br>
[Home](Home)<br>
[Installation](Installation)<br>
[Quick Start](Quick-Start)

**Concepts**<br>
[Core Concepts](Core-Concepts)<br>
[Loading Images](Loading-Images)<br>
[Normalization](Normalization)<br>
[Selecting Metrics](Selecting-Metrics)<br>
[Metric Catalog](Metric-Catalog)<br>
[Scoring Modes](Scoring-Modes)<br>
[Segmentation Evaluation](Segmentation-Evaluation)<br>
[Results and Reports](Results-and-Reports)

**Going Further**<br>
[Extending the Framework](Extending-the-Framework)<br>
[Worked Examples](Worked-Examples)<br>
[Testing and Calibration](Testing-and-Calibration)<br>
[Troubleshooting](Troubleshooting)
```

- [ ] **Step 2: Full run on fresh data**

Run: `cd $T && $FW/.venv/bin/python run_examples.py --fresh-data --all; echo "exit $?"`
Expected: no `FAIL`, `MISMATCH`, `NOT RUN`, `exit 0`.

- [ ] **Step 3: Full lint**

Run: `python3 $T/lint_wiki.py; echo "exit $?"` → `lint clean`, `exit 0`.

- [ ] **Step 4: Cross-page consistency sweep**

Run: `cd $W && grep -n -E "\b(18|19|11|30|29|858|BATCH_SIZE|0\.5|99\.5|0\.02|224)\b" *.md | grep -v '^_'` and check that every number means the same thing everywhere and matches code and report. Run `grep -n -i -E "registry\.register\(\*|mask_writer|_mask\.png|overlay" *.md` → no hits (outdated README concepts).

- [ ] **Step 5: Report alignment sweep**

Read every page once against `$THESIS` chapters 3 to 6 and Anhang A. Each claim either agrees with the report, goes beyond it without contradiction, or is listed in `$T/discrepancies.md`. Finalize the discrepancy list (sorted by importance, each with a concrete suggested fix).

---

### Task 13: Single orphan commit

**Files:**
- Commit: all 16 files in `$W`

- [ ] **Step 1: Verify the tree**

Run: `cd $W && git status --short && ls`
Expected: exactly the 15 pages and `_Sidebar.md` untracked, `AGENTS.md` and `main.pdf` not listed by `git status`.

- [ ] **Step 2: Commit and replace master locally**

```bash
cd /Users/gl/Documents/2_HKA_Studium/M_Projektarbeit1/framework.wiki
git add Home.md Installation.md Quick-Start.md Core-Concepts.md Loading-Images.md Normalization.md \
  Selecting-Metrics.md Metric-Catalog.md Scoring-Modes.md Segmentation-Evaluation.md \
  Results-and-Reports.md Extending-the-Framework.md Worked-Examples.md Testing-and-Calibration.md \
  Troubleshooting.md _Sidebar.md
git commit -m "Rewrite wiki from scratch

New usage-oriented documentation of the iqaevaluator framework, aligned
with the project report. Every example was executed against mock data.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git branch -M wiki-rewrite master
git log --oneline --all | head
git rev-list --count master
```

Expected: `git rev-list --count master` prints `1`. `backup/pre-rewrite` still exists locally. Nothing is pushed (`git push` is not run).

- [ ] **Step 3: Final report to the author**

State: commit hash, page list, runner and lint results, the discrepancy list (full content of `$T/discrepancies.md`), that `backup/pre-rewrite` holds the old local state and can be deleted with `git branch -D backup/pre-rewrite`, and that publishing requires `git push --force origin master` (explicitly the author's decision, it overwrites the remote wiki history).
