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

# Note: the both-empty Dice case is a 2-slice volume: in volume mode the policy answers
# None; in slice mode the evaluator would skip both slices and no metric would run (that
# path is covered by test_f3_slice_mode_skips_only_slices_empty_on_both_sides).
