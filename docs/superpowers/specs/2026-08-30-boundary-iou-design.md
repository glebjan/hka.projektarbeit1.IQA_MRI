# Boundary IoU — Design

Date: 2026-08-30
Branch: `volumetric_similarity`

## Purpose

Add Boundary IoU (Cheng, Girshick, Dollár, Berg, Kirillov — *Boundary IoU:
Improving Object-Centric Image Segmentation Evaluation*, CVPR 2021) to the
`segmentation_metrics` package as a registrable `MetricSpec`.

Mask IoU is biased by object size: the same absolute boundary error looks
progressively better as the object grows. Boundary IoU removes that bias by
computing IoU over a thin band along each mask's contour instead of over the
whole mask region. Measured on this design's implementation (400×400 image,
constant 5-pixel shift, `dilation_ratio=0.02`):

| object size | mask IoU | Boundary IoU |
|---|---|---|
| 40×40   | 0.6203 | 0.4131 |
| 80×80   | 0.7840 | 0.3907 |
| 160×160 | 0.8841 | 0.3822 |
| 320×320 | 0.9399 | 0.3785 |

Mask IoU climbs 0.62 → 0.94 for an unchanged error; Boundary IoU stays flat.

## Definition

For masks `G` (ground truth) and `P` (prediction), and a pixel distance `d`:

```
G_d = G \ erode(G, d)          # boundary band of G
P_d = P \ erode(P, d)          # boundary band of P
Boundary IoU = |G_d ∩ P_d| / |G_d ∪ P_d|
```

`erode` is erosion by a `(2d+1)×(2d+1)` square structuring element — the
reference implementation's `cv2.erode` with a 3×3 kernel applied `d` times.

`d = max(1, round(dilation_ratio * sqrt(H² + W²)))` — proportional to the image
diagonal, so the band width is resolution-independent. Paper default
`dilation_ratio = 0.02`.

The reference implementation zero-pads the mask by one pixel before eroding, so
an object clipped by the image border counts that clipped edge as boundary.
This design reproduces that behaviour exactly.

## Efficient implementation

Rather than iterating a 3×3 erosion `d` times (the reference approach, O(d·H·W)),
compute the chessboard distance transform once and threshold it:

```python
padded = np.pad(mask, 1).astype(np.uint8)
dist   = distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]
band   = mask & (dist <= d)
```

Erosion by a `(2d+1)` square keeps exactly the pixels whose Chebyshev distance
to the nearest background pixel exceeds `d`, so `mask & (dist <= d)` *is* the
boundary band. Padding by one zero pixel supplies the background ring that makes
border-clipped edges count, matching the reference.

This is O(H·W) regardless of `d`, and uses `scipy.ndimage` — already a project
dependency (`scipy==1.13.1`). **No OpenCV dependency is added.**

Verification performed while writing this design:

- Bit-exact agreement with the iterated-erosion reference over 200 randomised
  masks × dilations `d ∈ {1, 2, 3, 5, 9}`, including border-touching objects.
- Runtime on a 512×512 blob mask is flat in `d` (2.9 ms at `d=7` through
  `d=110`), while iterated erosion grows 3.3 ms → 5.6 ms over the same range.

## Non-goals

- No OpenCV dependency.
- No 3D/volumetric boundary band. Framework tensors are `(D, 1, H, W)` and every
  existing metric scores per 2D slice; Boundary IoU does the same.
- No multi-class one-hot averaging. The metric scores channel 0 of each sample,
  matching `MonaiPanopticQualityMetric`'s per-sample loop.
- Not added to `BUILTIN_METRICS` — it joins `SEGMENTATION_METRICS`, so
  `main.py`'s raw-image CLI is unaffected.

## Module layout

New file `src/segmentation_metrics/boundary_iou.py`, self-contained, mirroring
`monai_metrics.py`'s builder-and-constant shape:

| Symbol | Role |
|---|---|
| `DEFAULT_DILATION_RATIO = 0.02` | paper default |
| `dilation_pixels(shape, dilation_ratio)` | band width in pixels, floor 1 |
| `boundary_region(mask, dilation)` | 2D bool mask → 2D bool band |
| `boundary_iou(pred, gt, *, dilation_ratio, label, threshold)` | stateless numpy scalar, NaN when both bands empty |
| `BoundaryIoUMetric` | `Metric`-protocol adapter over an `(N, C, H, W)` torch batch |
| `boundary_iou_metric(*, ...)` | returns `MetricSpec` |
| `BOUNDARY_IOU` | `boundary_iou_metric()` with defaults |

The numpy layer is public (like `volume.py`'s `vs`/`dice`) so it can be used
directly for per-slice DataFrame work; the adapter layer is what the registry
consumes.

## Binarization convention

`boundary_iou` reuses `as_mask` from `segmentation_metrics.volume` — bool passes
through, float in `[0, 1]` thresholds at `threshold` (`>=`), integer label maps
go one-vs-rest via `== label`.

This differs deliberately from `monai_metrics.MonaiSegmentationMetric`, whose
`threshold` defaults to `None` meaning "skip binarization". That option does not
apply here: the band computation needs a boolean array, so a threshold is always
required. Default `threshold=0.5`, consistent with `volume.as_mask`.

## MetricSpec fields

```python
MetricSpec(
    name="boundary_iou",
    direction="higher_is_better",
    reference=True,
    channels="gray",
    factory=lambda: metric,
    builtin=False,
    description="...",
    domain="",
)
```

`domain=""` (domain-agnostic): the band width is a fraction of the image
diagonal, so unlike the MONAI metrics there is no physical-spacing or
class-count parameter to retune per domain. `dilation_ratio` is the single
tunable, and its meaning (fraction of image diagonal) is domain-independent.

## Edge cases

| Case | Result |
|---|---|
| Both masks empty | `NaN` → adapter reports `None` |
| One mask empty | `0.0` (union non-empty, intersection empty) |
| Identical masks | `1.0` |
| Disjoint masks | `0.0` |
| `dilation_ratio` large enough that band = whole mask | equals mask IoU exactly |
| Image so small that `ratio * diag < 0.5` | `d` floors to 1 |
| `pred.shape != gt.shape` | `ValueError` |
| input not 2D | `ValueError` |
| `target is None` in adapter | `ValueError` (full-reference metric) |

## Wire-up

- `metrics.py`: add `from segmentation_metrics.boundary_iou import BOUNDARY_IOU`
  beside the existing late `monai_metrics` import (same circular-import reason —
  `MetricSpec` must already be defined), and append `BOUNDARY_IOU` to the
  `SEGMENTATION_METRICS` tuple.
- `main.py`: add `BOUNDARY_IOU` to the `from metrics import (...)` re-export list.

No change to `IQAEvaluator`, `MetricRegistry`, `ImageEvaluatorRecord`, or the
report writer — `builtin=False` routes the score through `record.extra` and the
existing auto-flattening.

## Testing

New `tests/test_boundary_iou.py`, matching the flat `tests/` layout and the
class-per-component style of `tests/test_segmentation_metrics.py`. Coverage:

1. Band geometry — pixel counts for a known square; border-clipped object.
2. Equivalence with the iterated-erosion reference (`scipy.ndimage.binary_erosion`)
   over randomised masks — pins the implementation to the paper's definition.
3. Scalar edge cases from the table above.
4. Scale-bias property — the four-row table in *Purpose*.
5. `dilation_ratio=1.0` degenerates to mask IoU.
6. Adapter: per-sample scores, `None` for NaN, missing-target error.
7. `MetricSpec` field assertions and registry round-trip.
