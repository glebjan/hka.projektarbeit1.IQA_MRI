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
