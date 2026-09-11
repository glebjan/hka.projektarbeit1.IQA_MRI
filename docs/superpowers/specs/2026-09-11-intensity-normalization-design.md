# Intensity Normalization as a Per-Run Strategy — Design

Date: 2026-09-11
Branch: `volumetric_similarity`

## Purpose

`ImageLoader` min-max-normalizes every decoded image over its whole stack, with
input and target scaled independently, and the decoders throw the raw
intensities away. Commit `ab113e1` recorded the consequences as ten
`TODO(norm-N)` markers. This design resolves the ones that distort the
evaluation and states which ones are deliberately left as they are.

The reference conventions this design follows are the ones quantitative MRI
evaluation already uses:

- **One scale per full-reference pair, the reference's.** fastMRI evaluates
  PSNR/SSIM with `data_range = target.max()` per volume and never rescales the
  prediction on its own. Independent scaling is invariant under any affine
  transform, so `norm(x) == norm(2x + 500)`: a generator that is uniformly too
  bright is scored as perfect by every full-reference metric. (norm-1)
- **One scale per volume, never per slice.** A scan is one acquisition. Scaling
  slices individually amplifies noise in near-empty edge slices and makes
  scores incomparable within a stack. This is how the loader already behaves;
  it is kept. (norm-4 — not a defect)
- **Masks are never intensity-normalized.** MONAI (`AsDiscrete`) and nnU-Net
  keep label maps as integers; binarization is the metric's job. Today a
  `{0,1,2,3}` label map is scaled to `[0, .333, .667, 1]` and a 0.5 cutoff
  drops label 1 into the background while merging 2 and 3, silently. (norm-6,
  norm-7)
- **Robust (percentile) scaling is a preprocessing convention, not an
  evaluation one.** TorchIO `RescaleIntensity(percentiles=(0.5, 99.5))` and
  MONAI `ScaleIntensityRangePercentiles` exist for training. For evaluation it
  is legitimate only when the reference defines the percentiles and both
  images use them; clipping the prediction on its own would hide overshoot
  artefacts. Offered as an opt-in strategy, not the default. (norm-2)
- **The scale is part of the result.** PSNR against `data_range = 1.0` is only
  interpretable when the reader knows what `1.0` stood for. (norm-5)

Every backend the framework uses — pyiqa, DreamSim, MONAI's PSNR/SSIM — takes
`[0, 1]` input and scales from there internally (`×255` for brisque/niqe/piqe/
vsi, `[-1, 1]` for lpips/musiq, ImageNet/CLIP statistics for maniqa/clipiqa/
dists, `max_val = 1.0` for MONAI PSNR). None of them normalizes per image.
The `Metric` contract "float32 in `[0, 1]`" therefore stays. What changes is
*who decides how the raw intensities reach `[0, 1]`*: the decoder no longer
does; a strategy chosen per run does. (norm-10)

## Scope

**In scope**

- Decoders return raw intensities; normalization moves out of them.
- A `Normalizer` strategy on `ImageLoader`, chosen per run: `MinMax` (default,
  today's behaviour), `Percentile`, `FixedRange`, `Raw`.
- Full-reference runs scale input and target on the target's range.
- The range used, and the input's raw extremes, in every report row.
- `empty_slice_mask` computed on raw data, robust to a single spike.
- Full-reference emptiness: a slice is empty only if input *and* target are
  blank. `aggregate_volumes()` simplified accordingly.
- Constant images keep their value (clipped), with a warning, instead of
  becoming zeros. The `+1e-8` epsilon goes.
- Tests for each strategy and each behaviour change; documentation in
  `CLAUDE.md`, `README.md`, the notebook, and the wiki.

**Out of scope**

- norm-4 (per-slice scaling): rejected, see Purpose. One sentence in the wiki's
  *Interpreting Scores* page names the slice dependence.
- norm-9 (DICOM `WindowCenter`/`WindowWidth`): a display setting, not a
  measurement scale; no use case in this project. `RescaleSlope`/`Intercept`
  stay applied in the decoder because they are part of decoding, not of
  normalization.
- Z-score normalization: does not produce `[0, 1]`, so it cannot satisfy the
  `Metric` contract.
- A `label=` parameter for the MONAI adapter (one-vs-rest on multi-label
  masks). With `Raw`, the MONAI metrics binarize with `value > threshold`, so
  every non-zero label is foreground — defined behaviour, documented. The
  numpy-backed volume metrics (`vs`, `v_pred`, …) already take `label=`.

## Design

### 1. `LoadedImage` carries raw data

```python
@dataclass(frozen=True)
class LoadedImage:
    raw:           np.ndarray        # (D, H, W), dtype as decoded (float32 for
                                     # images; the file's integer dtype for
                                     # integer NIfTI/NRRD label maps)
    spacing:       Optional[Spacing] = None
    is_volumetric: bool = False
```

`tensor` is removed from `LoadedImage`. Decoders keep everything that is part
of decoding — DICOM slope/intercept and MONOCHROME1 inversion, NIfTI canonical
reorientation, axis transposition — and stop there. `_load_nifti` uses
`np.asanyarray(image.dataobj)` instead of `get_fdata()` so an integer label
map keeps its dtype; float scaling (`scl_slope`) is applied only when the
header declares it. PIL grayscale arrives as `uint8` and is kept as such.

### 2. `Normalizer` strategy

```python
@dataclass(frozen=True)
class IntensityRange:
    lo: float
    hi: float

class Normalizer(Protocol):
    name: str
    def range_of(self, raw: np.ndarray) -> Optional[IntensityRange]: ...
```

A strategy decides only *which* `(lo, hi)` is mapped onto `[0, 1]`. The
mapping itself is fixed and lives in one function:

```python
def scale(raw, rng: Optional[IntensityRange]) -> torch.Tensor:
    # rng None       -> raw as-is, dtype preserved          (masks)
    # rng.hi > lo    -> clip((raw - lo) / (hi - lo), 0, 1)  (no epsilon)
    # rng.hi == lo   -> clip(raw, 0, 1), warning printed   (norm-7)
```

Strategies shipped:

| Strategy | `range_of` | `name` | Use |
|---|---|---|---|
| `MinMax()` | `(raw.min(), raw.max())` | `minmax` | default; today's behaviour |
| `Percentile(lower=0.5, upper=99.5)` | `np.percentile(raw, [lower, upper])` | `percentile_0.5_99.5` | opt-in robust scaling |
| `FixedRange(rng, name)` | `rng`, ignores `raw` | `name` as given — the strategy that produced `rng` | reference scale for the input of an FR pair |
| `Raw()` | `None` | `raw` | masks |

`FixedRange(None)` is equivalent to `Raw()`; this is what an FR pair under
`Raw` degenerates to, so the pairing logic below has no special case.

### 3. `ImageLoader`

```python
ImageLoader(path, normalizer: Normalizer = MinMax())
```

- `.raw` — the decoded array (lazy, cached).
- `.raw_range` — `IntensityRange(raw.min(), raw.max())`, the image's own
  extremes regardless of strategy.
- `.intensity_range` — `normalizer.range_of(raw)`: what `[0, 1]` stands for.
  `None` under `Raw`.
- `.tensor` — `scale(raw, intensity_range).unsqueeze(1)`, `(D, 1, H, W)`,
  cached. float32 for every strategy except `Raw`, which preserves dtype.
- `.rgb_tensor`, `.spacing`, `.is_volumetric` — unchanged.
- `.empty_slice_mask` — see §5.

A pairing helper keeps the reference-scale rule in one place, so users who
build evaluators by hand get it too:

```python
def load_pair(input_path, target_path, normalizer=MinMax()) -> tuple[ImageLoader, ImageLoader]:
    target = ImageLoader(target_path, normalizer)
    inp    = ImageLoader(input_path, FixedRange(target.intensity_range, name=normalizer.name))
    return inp, target
```

`main.evaluate()` gains `normalization: Normalizer = MinMax()` and uses
`load_pair` when a target exists, `ImageLoader(inp, normalization)` otherwise.
The CLI gains `--normalization {minmax,percentile,raw}`.

### 4. Report columns (norm-5)

`ImageEvaluatorRecord` gains five fields, filled by both evaluators from the
loaders:

| Field | Meaning |
|---|---|
| `normalization` | strategy name (`minmax`, `percentile_0.5_99.5`, `raw`) |
| `scale_lo`, `scale_hi` | the range mapped to `[0, 1]`; in an FR run this is the target's. NaN under `Raw` |
| `input_min`, `input_max` | the input's raw extremes |

In an FR run, `input_max > scale_hi` means the prediction overshot the
reference and was clipped; `input_min`/`input_max` inside a narrow part of
`[scale_lo, scale_hi]` means it is too dark or too flat. This makes the bias
that norm-1 used to hide readable from the report without a new metric.

### 5. Empty slices (norm-3, and FR emptiness)

`empty_slice_mask` is computed on `raw`, relative to a robust range of the
volume:

```
span  = p99.5(raw) - p0.5(raw)          # spike-proof, unlike max - min
empty = (slice.std() < 1e-3 * span) | (slice.mean() - p0.5(raw) < 1e-3 * span)
```

Under `MinMax` on spike-free data this is the same test as today; with a spike
it no longer collapses. When the percentile span is zero because the
foreground is rarer than 0.5 % of the volume (a small lesion mask loaded with
`Raw`), the span falls back to `max - min`, so such masks are not skipped
wholesale. `span == 0` after that (a constant volume) marks every slice empty,
as today.

`IQAEvaluator` (slice mode) sets `is_empty = input_empty & target_empty` when a
target exists, `input_empty` otherwise. A blank prediction over an occupied
reference is now scored — as the failure it is — instead of being dropped.
`VolumeEvaluator` is unchanged (`all()` over the input's mask).

`EvaluationResult.aggregate_volumes()` follows: `is_empty` on an FR row now
means both masks are blank, so `v_pred`, `v_gt` and `tp` are all filled with
`0` on such rows. The `blank_prediction` branch, its warning and the NaN
propagation for that case are removed; the "metric raised" case (all three
NaN) stays as it is.

### 6. Segmentation path (norm-6, norm-7)

Mask files are loaded with `Raw()`. `as_mask` in `segmentation_metrics/volume.py`
then receives the integer dtype it was written for and its `label=` branch
becomes reachable; no change there beyond deleting the TODO. The MONAI adapter
casts both tensors to `float32` after `_binarize` so an integer input is
accepted by MONAI's functionals. The three `TODO(norm-6/7)` comments in
`monai_metrics.py` and `volume.py` are replaced by a note that masks must be
loaded with `Raw()` and that `threshold` binarizes every non-zero label as
foreground.

### 7. What each TODO becomes

| TODO | Resolution |
|---|---|
| norm-1 | §3 `load_pair`: FR pairs scale on the target's range |
| norm-2 | §2 `Percentile`, opt-in |
| norm-3 | §5 raw, percentile-span test |
| norm-4 | not a defect; one sentence in *Interpreting Scores* |
| norm-5 | §4 report columns |
| norm-6 | §6 `Raw` + dtype-preserving decoders |
| norm-7 | §2 constant image keeps its clipped value, warning |
| norm-8 | §2 no epsilon |
| norm-9 | out of scope, comment replaced by one line saying so |
| norm-10 | §2–§3; the `Metric` docstring states that `[0, 1]` is produced by a per-run `Normalizer` and names the default |

All ten `TODO(norm-N)` comments are removed; where a decision needs recording
it becomes a normal comment or docstring sentence.

## Behaviour changes a user will notice

- **FR runs scale on the target.** For SMORE outputs of the same source
  volume the ranges nearly coincide, so `psnr`/`ssim` move very little; where a
  generator has a systematic intensity bias, scores drop and the report shows
  why. Reports produced before this change are not row-for-row comparable with
  reports produced after it.
- **NR runs are numerically unchanged** under the default strategy on
  spike-free data (only the epsilon differs, below float32 resolution for any
  real range).
- **Constant images** now yield their clipped value, not zeros. A `PNG` of all
  `42` becomes `1.0`, not `0.0`. `test_constant_array_becomes_zeros` is
  replaced.
- **Blank-prediction slices over an occupied reference** are now scored.
  `aggregate_volumes()` no longer returns NaN for volumes that contain such
  slices.
- **Five new report columns.** Column order of the existing ones is kept.

## Error handling

- `Percentile` with `lower >= upper` or outside `[0, 100]` raises `ValueError`
  at construction.
- `FixedRange` with `hi < lo` raises `ValueError` at construction.
- A constant image (`hi == lo`) prints one warning naming the file and the
  value; it does not raise.
- `load_pair` with a target that yields `intensity_range is None` (i.e. `Raw`)
  produces `FixedRange(None)` for the input, so a `Raw` FR pair behaves like
  two `Raw` loads. No error.
- `IQAEvaluator`'s existing shape check is unchanged; a dtype mismatch between
  input and target (e.g. an integer mask against a float mask under `Raw`) is
  not an error — `_binarize` and `as_mask` handle both.

## Testing

Unit tests, offline, in the existing files plus `tests/test_normalization.py`:

- `scale()`: `[0, 1e-8]` maps to exactly `[0.0, 1.0]`; constant `42` maps to
  `1.0` with a warning; `None` range preserves dtype and values.
- `MinMax` reproduces the previous tensor bit-for-bit on the existing PNG,
  NIfTI and DICOM fixtures (epsilon aside).
- `Percentile`: a volume with one spike at `4000` over `[100, 140]` keeps the
  bulk spread across `[0, 1]`; the spike clips to `1.0`.
- `FixedRange`: an input equal to `2 * target + 500` scores a finite, non-perfect
  `psnr` under `load_pair`; the same pair scores perfect `psnr` when both are
  loaded independently with `MinMax` (the regression this design fixes).
- `Raw`: an integer NIfTI `{0,1,2,3}` label map reaches `vs(label=2)` with the
  right voxel count; the MONAI `dice` adapter accepts the integer tensor.
- `empty_slice_mask`: unchanged verdicts on the existing fixtures; a volume with
  one spike voxel does not mark its normal slices empty.
- `IQAEvaluator`: FR slice with blank input and occupied target is scored;
  NR slice with blank input is still empty. `_make_loader` builds
  `LoadedImage(raw=np.ndarray)`.
- `aggregate_volumes`: a blank-both slice contributes zeros to all three
  counts; the "metric raised" case still yields NaN.
- Records/report: the five new columns are present with the expected values
  in slice and volume mode; `to_frame()` column order unchanged for the
  existing fields.
- `main.evaluate`: `normalization=` reaches both loaders; the CLI flag parses.

The full suite must pass; `test_dreamsim_integration.py` stays gated as today.

## Documentation

- `CLAUDE.md`: `image_loader.py` paragraph describes `LoadedImage.raw`, the
  strategies and `load_pair`; the metric summary notes that FR runs scale on
  the target.
- `README.md` line 93 (normalization sentence) rewritten.
- `evaluation.ipynb`: the path cell gains `NORMALIZATION = MinMax()` and one
  markdown sentence on what the reported `scale_lo/hi` mean.
- Wiki (separate `*.wiki` repository): *Loading Images* (strategies,
  `load_pair`, masks with `Raw`), *Results and Reports* (new columns,
  `aggregate_volumes` change), *Interpreting Scores* (reference scale,
  percentile caveat, slice dependence of volume-global scaling),
  *Segmentation Metrics* (load masks with `Raw`).
