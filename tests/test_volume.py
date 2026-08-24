import numpy as np
import pytest

from segmentation_metrics.volume import as_mask


def test_as_mask_bool_passthrough():
    x = np.array([True, False, True])
    result = as_mask(x)
    assert result is x or np.array_equal(result, x)
    assert result.dtype == bool


def test_as_mask_float_thresholds():
    x = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    result = as_mask(x, threshold=0.5)
    np.testing.assert_array_equal(result, [False, False, True, True, True])


def test_as_mask_float_out_of_range_raises():
    x = np.array([0.2, 1.5, -0.1])
    with pytest.raises(ValueError):
        as_mask(x)


def test_as_mask_int_label_map():
    x = np.array([0, 1, 2, 1, 0])
    result = as_mask(x, label=1)
    np.testing.assert_array_equal(result, [False, True, False, True, False])


def test_as_mask_unsupported_dtype_raises():
    x = np.array(["a", "b"])
    with pytest.raises(TypeError):
        as_mask(x)


from segmentation_metrics.volume import v_pred, v_gt, tp, vs, vs_signed


def _disk_mask(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius**2


def test_identical_masks_perfect_scores():
    mask = _disk_mask((20, 20), (10, 10), 5)
    assert vs(mask, mask) == 1.0
    assert vs_signed(mask, mask) == 0.0
    assert 2 * tp(mask, mask) / (v_pred(mask, mask) + v_gt(mask, mask)) == 1.0


def test_disjoint_equal_size_masks_vs_one_dice_zero():
    pred = np.zeros((20, 20), dtype=bool)
    gt = np.zeros((20, 20), dtype=bool)
    pred[0:5, 0:5] = True   # 25 voxels
    gt[15:20, 15:20] = True  # 25 voxels, disjoint
    assert vs(pred, gt) == 1.0
    dice = 2 * tp(pred, gt) / (v_pred(pred, gt) + v_gt(pred, gt))
    assert dice == 0.0


def test_undersegmentation_negative_signed_vs():
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True  # 100 voxels
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:5, 0:5] = True  # 25 voxels, subset -> smaller than gt
    assert vs_signed(pred, gt) < 0


def test_oversegmentation_positive_signed_vs():
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:10] = True  # 100 voxels
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:5, 0:5] = True  # 25 voxels, subset -> pred bigger than gt
    assert vs_signed(pred, gt) > 0


def test_both_empty_masks_yield_nan():
    pred = np.zeros((20, 20), dtype=bool)
    gt = np.zeros((20, 20), dtype=bool)
    assert np.isnan(vs(pred, gt))
    assert np.isnan(vs_signed(pred, gt))


def test_empty_reference_nonempty_prediction_vs_zero():
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:5, 0:5] = True
    gt = np.zeros((20, 20), dtype=bool)
    assert vs(pred, gt) == 0.0


def test_shape_mismatch_raises_with_both_shapes():
    pred = np.zeros((20, 20), dtype=bool)
    gt = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match=r"\(20, 20\).*\(10, 10\)"):
        vs(pred, gt)


import pandas as pd
from segmentation_metrics.volume import aggregate_patient


def test_aggregate_mean_of_ratios_differs_from_volume_ratio():
    # 3 slices, same patient. Per-slice dice varies a lot; the volume-level
    # dice (computed from summed tp/v_pred/v_gt) must differ from the naive
    # mean of per-slice dice values.
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1"],
            "v_pred": [10, 0, 10],
            "v_gt": [10, 10, 10],
            "tp": [10, 0, 0],
        }
    )
    # per-slice dice: slice0 = 2*10/20=1.0, slice1 = 2*0/10=0.0, slice2 = 2*0/20=0.0
    naive_mean_dice = (1.0 + 0.0 + 0.0) / 3
    result = aggregate_patient(df)
    volume_dice = result.loc["p1", "dice"]
    # volume-level: tp_sum=10, v_pred_sum=20, v_gt_sum=30 -> dice = 2*10/50 = 0.4
    assert volume_dice == pytest.approx(0.4)
    assert volume_dice != pytest.approx(naive_mean_dice)


def test_aggregate_group_col_none_treats_whole_frame_as_one_patient():
    df = pd.DataFrame(
        {
            "v_pred": [10, 20],
            "v_gt": [10, 20],
            "tp": [10, 20],
        }
    )
    result = aggregate_patient(df, group_col=None)
    assert len(result) == 1
    assert result.iloc[0]["dice"] == pytest.approx(1.0)


def test_aggregate_empty_denominator_yields_nan():
    df = pd.DataFrame(
        {"patient_id": ["p1"], "v_pred": [0], "v_gt": [0], "tp": [0]}
    )
    result = aggregate_patient(df)
    assert np.isnan(result.loc["p1", "vs"])
    assert np.isnan(result.loc["p1", "vs_signed"])
    assert np.isnan(result.loc["p1", "dice"])


@pytest.mark.parametrize("seed", range(5))
def test_vs_vs_signed_identity_random_masks(seed):
    rng = np.random.default_rng(seed)
    shape = (30, 30)
    pred = rng.random(shape) > 0.5
    gt = rng.random(shape) > 0.5
    v = vs(pred, gt)
    vsig = vs_signed(pred, gt)
    assert v == pytest.approx(1 - abs(vsig) / 2)
