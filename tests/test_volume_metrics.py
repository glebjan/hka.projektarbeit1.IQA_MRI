"""Tests for the VS / count metric specs."""
import numpy as np
import pytest
import torch

from metrics import MetricRegistry, ModeSupport, SEGMENTATION_METRICS
from segmentation_metrics.volume_metrics import (
    TP, V_GT, V_PRED, VS, VS_SIGNED, VOLUME_METRICS,
)


def _pair_4d():
    """Two slices: slice 0 identical, slice 1 prediction twice the size."""
    gt = torch.zeros(2, 1, 8, 8)
    pred = torch.zeros(2, 1, 8, 8)
    gt[0, 0, 1:3, 1:3] = 1.0
    pred[0, 0, 1:3, 1:3] = 1.0
    gt[1, 0, 1:3, 1:3] = 1.0
    pred[1, 0, 1:5, 1:3] = 1.0
    return pred, gt


def _pair_5d():
    pred, gt = _pair_4d()
    return pred.permute(1, 0, 2, 3).unsqueeze(0), gt.permute(1, 0, 2, 3).unsqueeze(0)


class TestSpecShape:
    def test_all_five_support_both_modes(self):
        for spec in VOLUME_METRICS:
            assert isinstance(spec.slice_mode, ModeSupport), spec.name
            assert isinstance(spec.volume_mode, ModeSupport), spec.name

    def test_counts_are_not_ranked(self):
        assert {s.direction for s in (V_PRED, V_GT, TP)} == {"not_ranked"}

    def test_vs_is_higher_is_better(self):
        assert VS.direction == "higher_is_better"

    def test_included_in_the_segmentation_bundle(self):
        names = {s.name for s in SEGMENTATION_METRICS}
        assert {"vs", "vs_signed", "v_pred", "v_gt", "tp"} <= names


class TestSliceMode:
    def test_vs_per_slice(self):
        metric = MetricRegistry(VS).get_metric("vs")
        pred, gt = _pair_4d()
        scores = metric(pred, gt)
        assert scores[0] == pytest.approx(1.0)          # identical
        assert scores[1] == pytest.approx(1.0 - 4 / 12)  # 8 vs 4 voxels

    def test_counts_per_slice(self):
        pred, gt = _pair_4d()
        assert MetricRegistry(V_PRED).get_metric("v_pred")(pred, gt) == [4.0, 8.0]
        assert MetricRegistry(V_GT).get_metric("v_gt")(pred, gt) == [4.0, 4.0]
        assert MetricRegistry(TP).get_metric("tp")(pred, gt) == [4.0, 4.0]


class TestVolumeMode:
    def test_vs_over_the_whole_volume(self):
        metric = MetricRegistry(VS).get_metric("vs", "volume", (1.0, 1.0, 1.0))
        pred, gt = _pair_5d()
        # totals: pred 12, gt 8  ->  1 - 4/20
        assert metric(pred, gt)[0] == pytest.approx(1.0 - 4 / 20)

    def test_the_volume_factory_hands_back_one_instance_for_every_spacing(self):
        # VS counts voxels and takes no spacing argument, so its volume factory
        # ignores the one it is handed. This asserts that design directly: the
        # registry may key its cache on spacing, but the object behind every key
        # is the same one. (An invariance assertion would be vacuous here —
        # comparing one instance's output to its own.)
        registry = MetricRegistry(VS)
        fine = registry.get_metric("vs", "volume", (1.0, 1.0, 1.0))
        coarse = registry.get_metric("vs", "volume", (3.0, 1.0, 1.0))
        assert fine is coarse

    def test_vs_signed_is_positive_when_oversegmenting(self):
        metric = MetricRegistry(VS_SIGNED).get_metric("vs_signed", "volume", None)
        pred, gt = _pair_5d()
        assert metric(pred, gt)[0] > 0.0

    def test_counts_sum_over_the_volume(self):
        pred, gt = _pair_5d()
        assert MetricRegistry(V_PRED).get_metric("v_pred", "volume", None)(pred, gt) == [12.0]
        assert MetricRegistry(TP).get_metric("tp", "volume", None)(pred, gt) == [8.0]


class TestUndefined:
    def test_two_empty_masks_give_none(self):
        empty = torch.zeros(1, 1, 4, 4, 4)
        assert MetricRegistry(VS).get_metric("vs", "volume", None)(empty, empty) == [None]

    def test_requires_a_target(self):
        pred, _ = _pair_5d()
        with pytest.raises(ValueError, match="target"):
            MetricRegistry(VS).get_metric("vs", "volume", None)(pred)


class TestCrossCheck:
    def test_volume_dice_equals_aggregated_slice_dice(self, tmp_path):
        """Volume-mode dice and aggregate_volumes()'s dice must agree exactly."""
        import nibabel as nib
        from main import evaluate
        from metrics import MetricRegistry
        from segmentation_metrics.monai_metrics import DICE
        from segmentation_metrics.volume_metrics import TP, V_GT, V_PRED

        # Every slice carries foreground on purpose. A blank prediction slice
        # makes aggregate_volumes() return NaN by design — a per-slice run
        # never counts the reference's voxels there, so the volume total is
        # genuinely unknown. That guard has its own coverage in
        # test_evaluation_result.py; here it would only mask the comparison
        # this test exists to make.
        gt = np.zeros((12, 12, 6), dtype="float32")
        gt[3:9, 3:9, :] = 1.0
        pred = np.zeros_like(gt)
        pred[4:10, 3:9, :] = 1.0
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        gt_path = tmp_path / "case_gt.nii.gz"
        pred_path = tmp_path / "case_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred, affine), pred_path)
        nib.save(nib.Nifti1Image(gt, affine), gt_path)

        volume_dice = evaluate(
            pred_path, gt_path, registry=MetricRegistry(DICE), mode="volume"
        ).to_frame().iloc[0]["dice"]

        aggregated = evaluate(
            pred_path, gt_path, registry=MetricRegistry(V_PRED, V_GT, TP), mode="slice"
        ).aggregate_volumes().iloc[0]["dice"]

        assert volume_dice == pytest.approx(aggregated, abs=1e-6)
