# Wiki Update After Normalization, Packaging and Module Restructure — Design

**Date:** 2026-09-12
**Status:** approved
**Target repo:** `https://github.com/glebjan/hka.projektarbeit1.IQA_MRI.wiki.git` (branch `master`)
**Supersedes parts of:** `2026-09-11-framework-wiki-documentation-design.md`

## Goal

Bring the 15 published wiki pages back in line with the framework as it stands
on `volumetric_similarity` (HEAD `e3c0527` plus the staged working tree), and
document intensity normalization — a user-facing subsystem the wiki has never
described. A reader following the wiki must end up with a working environment
and code that runs, which today they do not: every import statement, the whole
installation page, and the central scoring caveat are wrong.

## Drift being corrected

The wiki was last written on 2026-09-11 12:51 (`0cf16ff`), with two markdown
touch-ups at 18:15. Everything below landed after the rewrite.

| Change | Commits | Wiki impact |
|---|---|---|
| Module restructure into the `iqaevaluator` package | `458576e`, staged tree | ~35 flat imports wrong across 10 pages |
| Hatch as build backend and env manager; `requirements.txt` deleted | `b2f3f06`, staged tree | `Installation.md` describes an install that cannot work |
| Intensity normalization subsystem | `d2dbfd5`…`c54563b`, `8aaf4f0` | Undocumented; three pages describe the superseded hardcoded min-max |
| `load_pair` scales a full-reference pair on the target's range | `6aa3db2` | `Interpreting-Scores.md`'s headline caveat is now conditional, not absolute |
| Five intensity columns written into every row | `de159fd` | `Results-and-Reports.md` column count and CSV header stale |
| Empty-slice test moved to raw data with a spike-proof span | `cf54f7e`, `0589661` | Claim "below `1e-3` after normalization" wrong on three pages |
| MONAI accepts raw integer masks | `4f90160`, `5d1e415` | `Segmentation-Metrics.md` threshold advice wrong |
| `TODO(norm-N)` markers retired | `c54563b` | `Interpreting-Scores.md:103` cites markers that no longer exist |

## Decisions

| Decision | Choice |
|---|---|
| Language | English (unchanged) |
| Edit style | Surgical per-page diffs. Prose still correct is left untouched — the pages were verified line by line on 2026-09-11 and a rewrite would discard that |
| Normalization placement | Its own page, `Normalization.md`. It is a peer of `Registering-Metrics`: something the reader chooses and configures, not a footnote |
| Normalization stance | The page recommends, it does not merely catalogue (see below) |
| Full-reference pairing | `load_pair()` is THE documented path. Two independent `ImageLoader`s appear exactly once, on `Normalization.md`, labelled as the trap |
| `normalizer_from_name()` | Documented as library surface, for readers wiring their own CLI or config file |
| Install path | Hatch primary (`uv` + `hatch env create` + `hatch run test`), `uv venv .venv && uv pip install -e .` as the fallback box |
| Dependency pins | Documented on `Troubleshooting.md`, not on `Installation.md`. A reader reaches them after the failure, not before |
| `src/main.py` | Out of the wiki entirely. Total silence — no status note, no "scheduled for replacement" |
| `constants.py`, `src/evaluation.ipynb`, `iqaevaluator.data` | Out as well. Weight locations are stated literally on `Installation.md` instead of through `constants.py` |
| Example verification | The scratchpad harness is rebuilt and every model-free example re-executed. Normalization changed the numbers, so previously captured output is wrong, not merely stale |
| Delivery | Write and commit in a scratchpad clone, show file list and full diff, push only after explicit approval |
| Framework code changes | None. This update needs no counterpart in `src/` |

## New page: `Normalization.md`

Placed between `Loading-Images` and `Registering-Metrics` in `_Sidebar.md`.

1. **What `[0, 1]` means and who decides it** — `IntensityRange`, the
   `Normalizer` protocol (`name`, `range_of(raw) -> Optional[IntensityRange]`),
   and the single `scale()` mapping every strategy funnels through.
2. **The four strategies, with a stance.**
   - `MinMax()` — the default, and what the reported numbers use.
   - `Percentile(0.5, 99.5)` — opt-in for volumes with outlier spikes. It
     changes every score; never mix the two strategies inside one comparison.
   - `Raw()` — masks. dtype preserved, unscaled, `range_of` returns `None`.
   - `FixedRange` — what `load_pair` uses internally. Not something a caller
     passes by hand.
   A neutral catalogue was rejected: a reader facing four classes without
   guidance picks arbitrarily, and their scores stop being comparable to ours.
3. **`load_pair(input_path, target_path, normalizer)`** — the target sets the
   scale, the input is scaled by the range the target produced and clipped into
   it (fastMRI's convention: `data_range` comes from the reference). Raw
   extremes stay visible as `ImageLoader.raw_range`.
4. **The trap, shown once.** Two independent `ImageLoader`s: min-max is
   invariant under any affine transform, so a generator whose output is
   uniformly too bright is not penalised at all by `psnr`, `ssim`, `lpips` or
   `dists`. This is what `load_pair` exists to prevent. The failure is
   invisible in the score, which is why it is documented at the point of use
   and not only under caveats.
5. **`normalizer_from_name(name)`** — string to `Normalizer`.
6. **What lands in the report** — `normalization`, `scale_lo`, `scale_hi`,
   `input_min`, `input_max`; `None` under `Raw()`. How to read
   `input_max > scale_hi` as "the prediction overshot the reference and was
   clipped".

## Per-page changes

| Page | Change |
|---|---|
| `Installation.md` | Rewrite the environment section: `uv` plus `hatch env create` as the primary path, `uv venv .venv && uv pip install -e .` as the fallback, `hatch run test` as the verification. Delete the section "There is no package — run from `src/`" in full; replace it with the editable install and `import iqaevaluator...` working from any working directory. State the weight paths literally (`models/RadImageNet_pytorch/ResNet50.pt`, `models/dreamsim/`) |
| `Troubleshooting.md` | Rewrite the `setuptools` / `pkg_resources` entry against `pyproject.toml` (`setuptools==80.10.2`, `<81`, `openai-clip`); add the deliberate `pyiqa` pin and why unpinning changes scores; add `ModuleNotFoundError: iqaevaluator` → the package is not installed editable; correct the empty-slice entry to the raw-data span test |
| `Interpreting-Scores.md` | Reframe the independence caveat as the trap `load_pair` prevents, and say which path the reader is on. Correct the empty-slice threshold to raw data. Delete the `TODO(norm-N)` paragraph; replace it with a pointer to `Normalization` and a statement that the strategies shipped |
| `Loading-Images.md` | Imports; `ImageLoader(path, normalizer=MinMax())`; add `.raw`, `.raw_range`, `.intensity_range`; shrink the normalization section to a pointer; drop the `main.evaluate()` citation from the pairing helpers |
| `Segmentation-Metrics.md` | Imports; correct the mask story — masks load with `Raw()` and MONAI accepts raw integer masks; PNG 0/255 masks with `MinMax()`; `threshold` guidance rewritten accordingly |
| `Configuring-Metrics.md` | Imports; the same `threshold` correction for the MONAI factories |
| `Results-and-Reports.md` | Six identifying columns become eleven; re-capture `df.shape`, the printed frame and the CSV header from a real run; update the column-order paragraph |
| `Core-Concepts.md` | Imports; module map drops `main.py`; the object-flow diagram gains the `Normalizer` |
| `Home.md` | Imports in the minimal example; the example switches to `load_pair`; object-flow diagram matched to `Core-Concepts` |
| `Registering-Metrics.md`, `Running-an-Evaluation.md`, `Custom-Metrics.md`, `Metric-Catalog.md` | Imports; full-reference examples switch to `load_pair` |
| `Scoring-Modes.md` | Re-check only — no known drift |
| `_Sidebar.md` | Add `Normalization` |

## Verification

A scratchpad script builds a synthetic NIfTI volume and a mask pair, then runs
and captures:

- registry construction and `select(mode)` skip output
- `PSNR` / `SSIM` in slice and volume mode
- `load_pair` under `MinMax` and under `Percentile(0.5, 99.5)`, side by side,
  to show that the scores genuinely differ
- the affine-bias case: independent loaders versus `load_pair` on the same
  uniformly brightened input
- a custom metric end to end into `record.extra`
- `from_records` → `to_frame` → CSV, including the full header
- `aggregate_volumes`
- `dreamsim_spec` name derivation without loading weights

Captured output is what the pages show. Model-backed examples (`dreamsim`,
`musiq`, `maniqa`, `ilniqe`) stay snippets, checked against the test suite and
marked as not executed.

## Order of work

1. Clone the wiki into the scratchpad (already done for this design).
2. Write and run the verification script; capture real output.
3. Write `Normalization.md`.
4. Apply the per-page diffs in the table above.
5. Self-review: every signature, parameter name and default re-checked against
   `src/iqaevaluator/`; grep for surviving flat imports, `requirements.txt`,
   `main.py`, `constants.py`.
6. Commit in the clone, show the file list and full diff, stop.
7. Push after approval.
8. Re-check local `README.md` and `CLAUDE.md` for the exclusion decisions; fix
   only where they contradict the wiki.

## Out of scope

- Any change to framework behaviour or to `src/`
- Documenting `main.py`, `constants.py`, `evaluation.ipynb` or `iqaevaluator.data`
- Rewriting pages that are still correct
- Translating the wiki into German
