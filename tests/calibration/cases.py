"""Hand-derived calibration cases for the segmentation metrics (spec Part 3).

Every case is a fixture small enough for its derivation to fit in a few
lines, plus the value that derivation gives. The value is not the authority
— the derivation is; a reviewer checks the arithmetic, the test checks that
the framework reproduces it through its public path (NIfTI -> load_pair ->
registry -> evaluator -> record).

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
