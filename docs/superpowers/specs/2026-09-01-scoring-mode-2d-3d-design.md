# Scoring Mode: 2D Slices and 3D Volumes — Design

Date: 2026-09-01
Branch: `volumetric_similarity`

## Purpose

Every metric in the framework is currently scored one 2D slice at a time.
`ImageLoader.tensor` yields `(D, 1, H, W)` and `IQAEvaluator` treats `D` as a
batch axis, so a volume is evaluated as a stack of unrelated pictures. For some
metrics that is correct; for others it produces a number that is systematically
wrong, and no amount of post-processing in the notebook can repair it.

Two distinct defects follow from slice-only scoring:

**Aggregation bias.** The metric is fine, the mean over slices is not.
`PSNR = 10·log10(MAX²/MSE)` is non-linear in MSE, so by Jensen's inequality
`mean_i(PSNR_i) ≥ PSNR(mean_i MSE_i)` — the slice mean always overstates
quality, and near-empty border slices with `MSE ≈ 0` inflate it further. Dice
and VS are ratios of sums; the mean of per-slice ratios is not the ratio of the
summed volume, and a slice holding 3 voxels counts as much as one holding 3000.

**Missing third dimension.** The metric measures something else entirely.
Surface distance is a 3D quantity: the nearest surface point often lies in the
neighbouring slice, which per-slice scoring is not allowed to find, so HD95,
ASSD and NSD systematically overstate the error. Panoptic Quality counts one
object spanning 20 slices as 20 instances. Boundary IoU's bands are in-plane
contours only, so the caps of a structure — the surfaces parallel to the slice
plane — are never scored.

| Metric | Per-slice distorted? | Direction | Repairable by aggregation alone |
|---|---|---|---|
| `psnr` | yes (mean) | too good | yes — aggregate MSE |
| `ssim` | no (mean) / yes (blind through-plane) | — | no |
| `lpips`, `dists`, `radimagenet_lpips` | no — 2D by definition | — | n/a |
| `clipiqa`, `clip_iqa_lung`, `clip_iqa_brain`, `brisque`, `niqe` | no — 2D by definition | — | n/a |
| `dice`, `vs` | yes (mean) | either way | yes — aggregate counts |
| `hausdorff95`, `assd`, `nsd` | **yes, fundamentally** | error too large | **no** |
| `panoptic_quality` | **yes, semantically** | — | **no** |
| `boundary_iou` | **yes, partly blind** | too good | **no** |

This design makes the scoring granularity an explicit choice per evaluation run,
adds real volumetric computation where slicing breaks the definition, and gives
per-slice runs a correct aggregation path.

## Scope

**In scope**

- A per-run scoring mode: `"slice"` or `"volume"`.
- Real 3D computation for `dice`, `hausdorff95`, `nsd`, `assd`,
  `panoptic_quality`, `boundary_iou`, and volumetric `psnr` / `ssim`.
- Voxel geometry (`spacing`) read from the image header instead of discarded.
- `vs` / `vs_signed` promoted to registrable `MetricSpec`s.
- Correct volume-level aggregation of per-slice runs.

**Out of scope**

- Reslicing a volume to another orientation (sagittal/coronal) for evaluation.
  Worth doing for SMORE — through-plane super-resolution is invisible in axial
  slices — but it is an independent change.
- Resampling one image onto the other's grid. Rejected inside the evaluator
  (see *Errors and skips*); it belongs to the planned multi-volume loading class.
- The multi-volume loading class itself, which will replace `main.evaluate()`.
- `avd` as a registrable metric. It would need mm³ units and a rename of the
  existing signed `avd_voxels`; dropped to avoid breaking notebook code.

## Core mechanism

An evaluation run has exactly one mode.

| Mode | Tensor handed to the metric | Records produced |
|---|---|---|
| `"slice"` | `(D, C, H, W)` — `D` is the batch axis | one per slice |
| `"volume"` | `(1, C, D, H, W)` — one sample | one per volume |

A volume score is the *same metric* called once with a 5D sample. There is no
sufficient-statistics protocol: MONAI's functional metrics accept both ranks
natively, and for `dice` the 5D call *is* the ratio of sums, so correct
aggregation falls out of the same mechanism that provides 3D surface distance.

There is no `"both"` mode. Comparing per-slice against volumetric scoring means
two runs.

The `Metric` protocol is **unchanged**. Only its docstring widens from
`(N, C, H, W)` to `(N, C, *spatial)`.

In volume mode:

- `empty_slice_mask` is **not** applied. Removing slices from a volume is not
  filtering, it is a change of geometry — the body gains holes and surface
  distances measure edges that do not exist. `is_empty` means the whole volume
  is empty.
- `BATCH_SIZE` is unused; one call carries the whole volume.

## Capability model

`MetricSpec` declares what it can do per mode, with illegal states
unrepresentable:

```python
@dataclass(frozen=True)
class ModeSupport:
    factory: Callable[..., Metric]

@dataclass(frozen=True)
class ModeUnsupported:
    reason: str        # plain language, shown in the skip message

slice_mode:  ModeSupport | ModeUnsupported
volume_mode: ModeSupport | ModeUnsupported
```

A metric either supports a mode (there is a factory) or does not (there is a
reason). Never both, never neither.

The volume factory receives the image's voxel spacing:

```python
ModeSupport.factory: Callable[[], Metric]                      # slice_mode
ModeSupport.factory: Callable[[Optional[Spacing]], Metric]     # volume_mode
```

`spacing` is a property of the *file*, read at runtime, but a *computation
parameter* for HD95, ASSD, NSD and Boundary IoU. Passing it through the factory
rather than through `Metric.__call__` keeps the adapter boundary — the
framework's central abstraction — free of an argument that only four of sixteen
metrics use, and keeps metrics that ignore spacing unchanged. It is
configuration of the metric *instance*, exactly like `threshold` and
`dilation_ratio` today.

`MetricRegistry` caches on `(name, mode, spacing)`. `get_metric` gains defaults
so the existing call site in `IQAEvaluator` stays valid verbatim:

```python
def get_metric(self, name, mode="slice", spacing=None) -> Metric
```

Mode selection is a query on the registry, not a free function, and performs no
I/O:

```python
def select(self, mode) -> tuple[list[MetricSpec], list[SkippedMetric]]
```

The caller formats and prints the skip message. This preserves the framework's
existing separation: `IQAEvaluator` does no I/O, output belongs to
`EvaluationResult` and to the top-level entry point.

`register_metric()` gains an optional `volume_factory=None`; omitting it yields
a slice-only metric, so every existing call stays valid.

## Metric capability matrix

| Metric | slice | volume | Volume backend |
|---|---|---|---|
| `psnr` | ✓ | ✓ | `monai.metrics.PSNRMetric(max_val=1.0)` |
| `ssim` | ✓ | ✓ | `monai.metrics.SSIMMetric(spatial_dims=3)` |
| `lpips`, `dists`, `radimagenet_lpips` | ✓ | — | — |
| `clipiqa`, `clip_iqa_lung`, `clip_iqa_brain`, `brisque`, `niqe` | ✓ | — | — |
| `dice`, `hausdorff95`, `nsd`, `assd` | ✓ | ✓ | same MONAI function, 5D sample |
| `panoptic_quality` | ✓ | ✓ | same adapter — `y_pred[i, 0]` yields `(D,H,W)` |
| `boundary_iou` | ✓ | ✓ | nD kernel (below) |
| `vs`, `vs_signed` | ✓ | ✓ | new adapter over `volume.py` |
| `v_pred`, `v_gt`, `tp` | ✓ | ✓ | new adapter over `volume.py` |

`PSNRMetric` and `SSIMMetric(spatial_dims=3)` are present in the installed MONAI
1.6.0 and accept 5D input; `compute_panoptic_quality` was verified to accept 3D
spatial input unchanged.

The eight deep-2D metrics are bound to CNN/CLIP backbones trained on flat
pictures and have no volumetric counterpart; they are skipped in volume mode.

**Known consequence.** `psnr`, `ssim` and `boundary_iou` are backed by different
estimators in the two modes (pyiqa vs MONAI; chessboard vs euclidean band). A
column of the same name from a slice run and from a volume run is not directly
comparable. Documented once, centrally, rather than three times.

`v_pred`, `v_gt` and `tp` are raw voxel counts, not quality measures, so
`MetricDirection` widens to
`Literal["higher_is_better", "lower_is_better", "not_ranked"]`. This is a
deliberate compromise: modelling them as specs lets them flow through the
existing evaluator untouched and reach the report as ordinary columns, at the
cost of one enum value that does not describe a ranking. The alternative — a
second per-slice computation mechanism beside the registry — costs more concepts
for less benefit.

## Image geometry

Decoders currently return a bare tensor and discard the header:
`_load_nifti` calls `get_fdata()`, `_load_sitk` calls `GetArrayFromImage()`.
Both voxel spacing and the question "is the depth axis spatial?" are born and
die there.

Decoders return a value object instead:

```python
@dataclass(frozen=True)
class LoadedImage:
    tensor:        torch.Tensor          # (D, 1, H, W), float32 in [0,1]
    spacing:       Optional[tuple[float, float, float]]   # (dz, dy, dx) in mm
    is_volumetric: bool
```

`_LOADERS` changes from `Callable[[Path], torch.Tensor]` to
`Callable[[Path], LoadedImage]`. `ImageLoader` exposes `.spacing` and
`.is_volumetric` alongside `.tensor`.

**Axis order is the easiest bug here.** Spacing must be reordered to match the
tensor's axes:

| Format | Header gives | Array axes after decoding | Spacing as stored |
|---|---|---|---|
| NIfTI 3D | `get_zooms()` = `(dx, dy, dz)`, array `(X, Y, Z)` | transposed `(2,0,1)` → `(Z, X, Y)` | `(dz, dx, dy)` |
| SimpleITK | `GetSpacing()` = `(sx, sy, sz)`, array `(z, y, x)` | `(Z, Y, X)` | `(sz, sy, sx)` — reversed |
| DICOM | `PixelSpacing` = `(dy, dx)`, `SliceThickness` = `dz` | `(D, H, W)` | `(dz, dy, dx)` |
| PNG / JPEG | none | `(1, H, W)` | `None` |

**`is_volumetric` is decided by the decoder, not sniffed afterwards.** The
dangerous case is 4D NIfTI: `_load_nifti` flattens time into depth
(`transpose(3,2,0,1).reshape(-1, X, Y)` → `D = t·z`), and volume-mode HD95 over
a half-temporal axis is meaningless, not merely imprecise. That file has a valid
spacing and `D > 1`, so any heuristic of the form
`spacing is not None and D > 1` would pass it — the guard must come from the
decoder that built the axis.

| Source | `is_volumetric` |
|---|---|
| NIfTI, 3D | `True` |
| NIfTI, 4D | `False` — time flattened into depth |
| SimpleITK, 3D | `True` |
| DICOM, multi-slice per header | `True` |
| DICOM, single frame / cine | `False` |
| PNG, JPEG | `False` |
| any source with `D == 1` | `False` |

`D == 1` counts as non-volumetric: a 3D metric on a single slice equals its 2D
version, and letting it through invites the belief that something volumetric was
measured.

## Boundary IoU in 3D

The 2D path is unchanged and stays bit-identical to Cheng et al.: chessboard
distance transform (`distance_transform_cdt`), band width
`max(1, round(dilation_ratio · √(H² + W²)))` in pixels.

The volume path measures the band **physically**, in millimetres:

```python
dist = distance_transform_edt(padded, sampling=spacing)
band = mask & (dist <= d_mm)
d_mm = dilation_ratio * norm(physical extent of the volume)
```

Rationale: with a voxel-counted band the width is `d` voxels in every direction,
which on 1×1×1.2 mm data is 8 mm in-plane but 9.6 mm through-plane — the metric
becomes 20 % more forgiving in exactly the axis where through-plane
super-resolution makes its errors, and a factor of 3 on 1×1×3 mm data. That is
the same class of error as running HD95 without spacing, only harder to see
because no millimetre figure is claimed.

When spacing is unknown (PNG masks), the volume path falls back to the
voxel/chessboard definition with a warning.

`boundary_region` drops its `ndim != 2` check and takes an optional `sampling`
argument; `distance_transform_cdt` already handles nD, and
`distance_transform_edt` supports anisotropic sampling. The band is a cube in the
2D path and a ball in the volume path — the second reason the two modes are not
directly comparable.

In 3D the band automatically includes the caps of a structure, which per-slice
scoring could not see at all.

## Volumetric similarity as metrics

`vs` and `vs_signed` become registrable specs in **both** modes, like `dice`.

Per-slice VS is not a broken measurement, it is a complementary one. The
implication runs one way only: if every slice has the correct area then the
total volume is correct, but a correct total volume does not imply correct
per-slice areas. A model that moves mass from slice A to slice B leaves the sums
untouched — volume VS reports 1.0 while the structure is smeared along z. That
is precisely the failure mode of through-plane super-resolution, and volume VS
is blind to it while per-slice VS is not.

VS is spacing-invariant: with `V = n·(dx·dy·dz)` the voxel factor cancels in
both numerator and denominator, so it needs no geometry.

Note that VS is a size measure, not an overlap measure — two disjoint masks of
equal size score 1.0. It is only meaningful beside Dice or IoU. This is
pre-existing behaviour, documented in `volume.py`.

The specs live in a new `segmentation_metrics/volume_metrics.py`. `volume.py`
stays pure NumPy with no `MetricSpec` import, which keeps it free of the
circular-import dance the other segmentation modules need.

## Aggregating a per-slice run

The averaging trap is handled once, centrally, rather than avoided per metric:

```python
result.to_frame()          # one row per slice, unchanged
result.aggregate_volumes() # one row per volume, correctly aggregated
```

`aggregate_volumes()` reconstructs `dice`, `vs` and `vs_signed` exactly from the
per-slice `v_pred` / `v_gt` / `tp` columns — ratios of sums, never means of
ratios. This is the same computation `volume.aggregate_patient()` performs; that
function stays as the standalone notebook tool and is documented as the correct
path for per-slice runs.

Metrics that cannot be reconstructed from per-slice values — `hausdorff95`,
`assd`, `nsd`, `panoptic_quality`, `boundary_iou`, and every deep-2D metric —
return **no value** rather than a biased mean, with a note pointing at
`mode="volume"`. The boundary of what aggregation can achieve is visible in the
tool itself, not only in documentation.

If the counting specs were not registered, `aggregate_volumes()` reports that in
plain language instead of failing.

## Evaluator extension

`IQAEvaluator` is frozen: extended, never modified. The volume path is a
subclass in a new module:

```python
class VolumeEvaluator(IQAEvaluator):
    def __init__(...)            # super().__init__, then geometry checks
    def _pick_tensor_batch(...)  # (D,C,H,W) -> (1,C,D,H,W)
    def run_evaluation(...)      # one record per volume
```

`base.permute(1, 0, 2, 3).unsqueeze(0)` produces the 5D sample. The subclass
inherits `_compute_batch()` — whose per-metric `try/except` behaviour must be
identical in both modes — and `_format_slice_id()`.

The class split is an implementation detail and must not reach the user, who
already selects behaviour by mode everywhere else. A factory function keeps the
public surface uniform:

```python
build_evaluator(input_loader, target_loader, registry, mode="slice")
```

Both classes expose the same `run_evaluation() -> list[ImageEvaluatorRecord]`,
so calling code is unaffected by which one it receives.

| Caller | What they write |
|---|---|
| CLI | `--mode volume` |
| `evaluate()` | `evaluate(..., mode="volume")` |
| Notebook, evaluator directly | `build_evaluator(..., mode="volume")` |
| Existing code building `IQAEvaluator(...)` | unchanged, still valid |

`build_evaluator()` lives in its own module: putting it in `iqa_evaluator.py`
would require importing `VolumeEvaluator`, which imports `IQAEvaluator` — the
same circular-import problem `metrics.py` already works around with a late
import. `main.py` is not a home either, since it is scheduled for replacement.

A free factory function matches the existing convention: `dice_metric()`,
`boundary_iou_metric()` and the other builders are already free functions
returning objects. A single-method stateless factory *class* would be a function
in disguise.

## Records and report

`ImageEvaluatorRecord` gains:

- `scoring: Literal["slice", "volume"]` — `mode` is already taken by
  `full_reference` / `no_reference`.
- `slice_index: Optional[int]` — `None` in volume mode.

In volume mode `image_id` is the volume's file name without the `_sNNN` suffix,
and `is_empty` refers to the whole volume.

One record type serves both modes. Because a run has exactly one mode, a report
holds exactly one kind of row, so the CSV column set is unchanged and existing
notebook code keeps working with fewer rows.

## Errors and skips

| Situation | Behaviour |
|---|---|
| Metric cannot serve the chosen mode | Skip. One collected message before the first tensor, naming every skipped metric grouped by reason |
| *No* metric can serve the mode | Hard error — the run has nothing to compute |
| Image not volumetric in volume mode (PNG, 4D NIfTI, `D == 1`) | Skip the file, message, summary line at the end |
| `spacing` differs between input and target | Abort, message names resampling as the user's decision |
| `spacing` missing, metric requires it | Skip that metric with its reason |
| `spacing` missing, `boundary_iou` | Fall back to the voxel/chessboard band, warn |

Configuration problems are detected before any computation: HD95 over a full
volume runs for minutes, and failing after twenty of them because LPIPS cannot
score volumes is not acceptable. All offending metrics are reported at once, not
the first one found.

Data problems skip one file and let the run continue — one bad file must not
kill a 200-file directory run.

**Message pattern:** what happened → why it is a property of the thing, not the
user's mistake → what to type next. No shape tuples, no class names, no
"unsupported".

```
Cannot score in volume mode: 8 of the 10 selected metrics can only look at
one 2D slice at a time.

  lpips, dists, radimagenet_lpips, clipiqa, clip_iqa_lung,
  clip_iqa_brain, brisque, niqe

These metrics compare images with neural networks trained on flat 2D
pictures. They have no way to see a stack of slices as one 3D body, so
there is no meaningful score to compute.

They will be skipped. To score them, run again with mode="slice".
```

The reason text lives on `ModeUnsupported`, next to the missing factory, so it
cannot drift away from the capability it explains.

**Resampling is deliberately not performed by the evaluator.** Two images on
different grids can only be aligned through their affine (origin and
orientation), which the decoders currently discard; the direction of resampling
is itself part of what is being measured — downsampling the reference makes a
super-resolution model look better, upsampling the prediction pits the model
against the interpolator; and masks require nearest-neighbour interpolation
while images do not, a distinction `ImageLoader` cannot make. Silent resampling
would produce a plausible-looking number from modified data, which is the class
of failure this design exists to remove. The evaluator aborts and names the
option; resampling belongs to the future loading class, where preprocessing is
at home and mask-vs-image is known, and it must be recorded in the report when
it happens.

## Module layout

| File | Change | Why here |
|---|---|---|
| `src/image_loader.py` | `LoadedImage`; all four decoders return spacing + `is_volumetric`; `_LOADERS` signature; `ImageLoader.spacing` / `.is_volumetric` | The only place headers are read. Both facts are born and discarded here. Most invasive single cut. |
| `src/metrics.py` | `ModeSupport` / `ModeUnsupported`; `MetricSpec.slice_mode` / `.volume_mode`; `MetricDirection` gains `not_ranked`; registry cache on `(name, mode, spacing)`; `MetricRegistry.select()`; `get_metric` defaults; `register_metric(volume_factory=…)`; `PSNR` / `SSIM` volume factories | The registry maps capability to instance; mode selection belongs where the specs live. |
| `src/volumetric_iqa.py` *(new)* | `MonaiPSNRMetric`, `MonaiSSIMMetric` | Adapter classes, same pattern as `PyIQAMetric`. Not under `segmentation_metrics/` — they are IQA metrics. Keeps `metrics.py` from growing. |
| `src/volume_evaluator.py` *(new)* | `VolumeEvaluator(IQAEvaluator)` | Extension, not modification. |
| `src/evaluator_factory.py` *(new)* | `build_evaluator()` | Avoids the import cycle; keeps the class split invisible to callers. |
| `src/iqa_evaluator.py` | **none** | Frozen by requirement. Registry defaults keep its call sites valid. |
| `src/records.py` | `scoring` field; `slice_index: Optional[int]` | Row shape is defined here. |
| `src/evaluation_result.py` | `aggregate_volumes()` | The result object owns its output and analysis. |
| `src/segmentation_metrics/boundary_iou.py` | nD `boundary_region` with `sampling`; physical band width; `edt` for volumes, `cdt` retained for 2D; adapter gains spacing; volume factory | The only genuinely 2D-bound compute kernel in the project. |
| `src/segmentation_metrics/monai_metrics.py` | Volume factory on all five specs; spacing threaded to HD95 / NSD / ASSD | The specs live here. PQ needs no adapter change. |
| `src/segmentation_metrics/volume_metrics.py` *(new)* | Adapters and specs `VS`, `VS_SIGNED`, `V_PRED`, `V_GT`, `TP` | Keeps `volume.py` free of `MetricSpec` and of the circular import. |
| `src/segmentation_metrics/volume.py` | Docstring only: position `aggregate_patient()` as the correct path for per-slice runs | Its functions are already shape-agnostic. |
| `src/main.py` | `evaluate(..., mode)`, CLI `--mode`, re-exports | Minimal — scheduled for replacement by the loading class. |
| `src/data.py` | Follow the `LoadedImage` signature | Depends on the changed decoder contract. |

## Testing

The 274 existing tests must stay green — that is the evidence the slice path is
untouched.

| Suite | Added coverage |
|---|---|
| `test_image_loader.py` | spacing value and **axis order** per format; `is_volumetric` per source, especially 4D NIfTI → `False` and `D == 1` → `False` |
| `test_metrics.py` | `ModeSupport` / `ModeUnsupported` construction; `select()` partitioning; cache keyed by `(name, mode, spacing)`; `get_metric` defaults; `register_metric` with and without a volume factory |
| `test_iqa_evaluator.py` | unchanged behaviour (regression guard) |
| `test_volume_evaluator.py` *(new)* | 5D sample shape; one record per volume; empty slices retained; spacing-mismatch abort; non-volumetric skip; `build_evaluator` dispatch |
| `test_boundary_iou.py` | nD `boundary_region`; anisotropic band equal in mm across axes; spacing-less fallback; 2D results bit-identical to today |
| `test_volume_metrics.py` *(new)* | VS / VS_signed adapters in both modes; spacing invariance; counts |
| `test_evaluation_result.py` | `aggregate_volumes()` matches `aggregate_patient()`; non-reconstructible metrics return no value; missing-counts message |

A property worth asserting directly: for a synthetic pair, volume-mode `dice`
equals `aggregate_volumes()`'s `dice` from a slice-mode run of the same pair.
Two independent paths, one number.

## Known compromises

1. `psnr`, `ssim` and `boundary_iou` use different estimators per mode. Values
   are not comparable across modes.
2. `MetricDirection` gains `not_ranked` to accommodate raw voxel counts.
3. `VolumeEvaluator` depends on two private methods of `IQAEvaluator`
   (`_compute_batch`, `_pick_tensor_batch`). Acceptable only because that class
   is frozen; were it in motion, a sibling class with duplicated logic would be
   the safer trade.
4. `MetricRegistry`'s "built once per run" promise becomes "once per spacing
   value". Harmless in practice — volumetric metrics are weightless MONAI
   functionals, and spacing is usually constant within a dataset — but
   `main.evaluate()`'s docstring must say so.
5. Boundary IoU in volume mode is an extension beyond Cheng et al., which
   defines the band by square erosion. Better justified on anisotropic data,
   but no longer "the" published metric.

## Follow-ups

- **`CLAUDE.md` is stale** and should have the obsolete section removed:
  it documents `mask_writer.py`, `MaskWriter` and `records.best_slice_per_metric()`,
  none of which exist — `src/` has no `mask_writer.py`, `records.py` has no such
  function, and `generate_report()` writes only CSV. Not edited as part of this
  change; recorded here so it is not lost.
- The multi-volume loading class that replaces `main.evaluate()`. In volume mode
  one volume yields one row, which is useless for the notebook dashboard; the
  loading class restores useful row counts by evaluating several volumes per run.
  Resampling belongs there too.
- Reslicing to sagittal/coronal for evaluation — the cheapest way to expose
  through-plane blur, and complementary to volumetric scoring rather than
  replaced by it.
