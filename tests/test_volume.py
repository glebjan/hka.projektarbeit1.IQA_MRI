import numpy as np
import pytest
import torch

from iqaevaluator.segmentation_metrics.volume import (
    as_mask,
    require_binary,
    require_single_channel,
    foreground_counts,
    empty_policy,
    NOT_EMPTY,
)


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


def test_as_mask_int_default_is_any_nonzero():
    x = np.array([0, 1, 2, 255], dtype=np.int16)
    np.testing.assert_array_equal(as_mask(x), [False, True, True, True])


def test_as_mask_integer_valued_float_is_a_label_map():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(as_mask(x), [False, True, True])


def test_as_mask_integer_valued_float_label_selects_one_class():
    x = np.array([0.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(as_mask(x, label=2), [False, False, True, True, False])


def test_as_mask_float_0_255_matches_uint8_0_255():
    values = [0, 255, 255, 0]
    np.testing.assert_array_equal(
        as_mask(np.array(values, dtype=np.float32)),
        as_mask(np.array(values, dtype=np.uint8)),
    )


def test_as_mask_label_on_a_probability_map_raises():
    x = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="label"):
        as_mask(x, label=1)


from iqaevaluator.segmentation_metrics.volume import v_pred, v_gt, tp, vs, vs_signed


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
from iqaevaluator.segmentation_metrics.volume import aggregate_patient


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


class TestRawLabelMapEndToEnd:
    def test_one_vs_rest_on_a_raw_loaded_label_map(self, tmp_path):
        import nibabel as nib
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Raw
        labels = np.zeros((8, 8, 2), dtype=np.int16)
        labels[:4, :, 0] = 1
        labels[4:, :, 0] = 2
        labels[:, :3, 1] = 3
        p = tmp_path / "labels.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        pred = ImageLoader(p, Raw()).tensor[:, 0].numpy()          # (D, H, W) int16
        assert v_pred(pred, pred, label=2) == 32.0
        assert v_pred(pred, pred, label=3) == 24.0
        assert vs(pred, pred, label=2) == 1.0


class TestAdapterHelpers:
    def test_require_binary_accepts_zero_one_floats_and_ints(self):
        require_binary(torch.tensor([[0.0, 1.0]]), metric="dice")
        require_binary(torch.tensor([[0, 1]], dtype=torch.int16), metric="dice")

    def test_require_binary_rejects_anything_else_and_names_mask(self):
        with pytest.raises(ValueError, match=r"dice expects a binary mask.*Mask\(\)"):
            require_binary(torch.tensor([[0.0, 0.5, 1.0]]), metric="dice")
        with pytest.raises(ValueError, match=r"Mask\(\)"):
            require_binary(torch.tensor([[0, 1, 2]], dtype=torch.int16), metric="dice")

    def test_foreground_counts_are_per_sample(self):
        y_pred = torch.zeros(2, 1, 4, 4); y_pred[0, 0, :2] = 1.0
        y = torch.zeros(2, 1, 4, 4); y[1, 0, 1, 1] = 1.0
        n_pred, n_gt = foreground_counts(y_pred, y)
        assert n_pred.tolist() == [8, 0] and n_gt.tolist() == [0, 1]

    def test_foreground_counts_handle_5d_volumes(self):
        vol = torch.ones(1, 1, 2, 3, 3)
        assert foreground_counts(vol, vol)[0].tolist() == [18]

    @pytest.mark.parametrize("n_pred, n_gt, one_sided, expected", [
        (0, 0, 0.0, None), (0, 0, None, None),
        (5, 0, 0.0, 0.0), (0, 5, 0.0, 0.0),
        (5, 0, None, None), (0, 5, None, None),
    ])
    def test_empty_policy_table(self, n_pred, n_gt, one_sided, expected):
        assert empty_policy(n_pred, n_gt, one_sided=one_sided) is expected

    def test_empty_policy_leaves_populated_pairs_alone(self):
        assert empty_policy(3, 4, one_sided=0.0) is NOT_EMPTY

    def test_require_single_channel_accepts_single_channel(self):
        require_single_channel(torch.zeros(2, 1, 4, 4), metric="dice")

    def test_require_single_channel_rejects_multi_channel(self):
        with pytest.raises(ValueError, match=r"one class per run.*Mask\(label=k\)"):
            require_single_channel(torch.zeros(1, 2, 4, 4), metric="dice")
