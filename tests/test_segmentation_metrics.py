"""Tests for src/segmentation_metrics/monai_metrics.py — MONAI-backed segmentation metrics."""
import numpy as np
import torch
import pytest

from iqaevaluator.metrics import MetricSpec
from iqaevaluator.segmentation_metrics.monai_metrics import (
    MonaiSegmentationMetric,
    dice_metric,
    DICE,
    hausdorff95_metric,
    HAUSDORFF95,
    normalized_surface_dice_metric,
    NSD,
    average_surface_distance_metric,
    ASSD,
    MonaiPanopticQualityMetric,
    panoptic_quality_metric,
    PANOPTIC_QUALITY,
)


def _binary_batch(n=2, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randint(0, 2, (n, 1, h, w), generator=g).float()
    gt = torch.randint(0, 2, (n, 1, h, w), generator=g).float()
    return pred, gt


class TestMonaiSegmentationMetricAdapter:
    def test_call_returns_one_score_per_sample(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True)
        pred, gt = _binary_batch(n=3)
        scores = metric(pred, gt)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_identical_masks_score_perfect_dice(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True)
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)


class TestDiceMetricBuilder:
    def test_returns_metric_spec(self):
        spec = dice_metric()
        assert isinstance(spec, MetricSpec)
        assert spec.name == "dice"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"
        assert "Dice" in spec.description or "dice" in spec.description

    def test_default_constant_matches_builder_defaults(self):
        assert DICE.name == "dice"
        assert DICE.builtin is False

    def test_factory_produces_working_metric(self):
        spec = dice_metric()
        metric = spec.slice_mode.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2

    def test_user_kwargs_pass_through(self):
        spec = dice_metric(include_background=False)
        metric = spec.slice_mode.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2


class TestHausdorff95MetricBuilder:
    def test_returns_metric_spec(self):
        spec = hausdorff95_metric()
        assert spec.name == "hausdorff95"
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_identical_masks_score_zero_distance(self):
        spec = hausdorff95_metric()
        metric = spec.slice_mode.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_default_constant(self):
        assert HAUSDORFF95.name == "hausdorff95"


class TestNormalizedSurfaceDiceMetricBuilder:
    def test_returns_metric_spec(self):
        spec = normalized_surface_dice_metric()
        assert spec.name == "nsd"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_identical_masks_score_perfect(self):
        spec = normalized_surface_dice_metric()
        metric = spec.slice_mode.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)

    def test_custom_class_thresholds_override(self):
        spec = normalized_surface_dice_metric(class_thresholds=[2.0])
        metric = spec.slice_mode.factory()
        pred, gt = _binary_batch(n=1)
        scores = metric(pred, gt)
        assert len(scores) == 1

    def test_default_constant(self):
        assert NSD.name == "nsd"


class TestAverageSurfaceDistanceMetricBuilder:
    def test_returns_metric_spec(self):
        spec = average_surface_distance_metric()
        assert spec.name == "assd"
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_identical_masks_score_zero_distance(self):
        spec = average_surface_distance_metric()
        metric = spec.slice_mode.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_default_constant(self):
        assert ASSD.name == "assd"


class TestMonaiPanopticQualityMetric:
    def test_call_returns_one_score_per_sample(self):
        metric = MonaiPanopticQualityMetric()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_identical_masks_score_perfect_pq(self):
        metric = MonaiPanopticQualityMetric()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)

    def test_rejects_non_integer_values(self):
        metric = MonaiPanopticQualityMetric()
        soft = torch.full((1, 1, 8, 8), 0.5)
        with pytest.raises(ValueError, match=r"Mask\(\).*Raw\(\)"):
            metric(soft, soft)

    def test_accepts_raw_instance_maps(self):
        inst = torch.zeros((1, 1, 8, 8), dtype=torch.int16)
        inst[0, 0, :3, :3] = 1
        inst[0, 0, 5:, 5:] = 2
        assert MonaiPanopticQualityMetric()(inst, inst.clone()) == [pytest.approx(1.0, abs=1e-5)]

    def test_empty_policy_applies(self):
        zeros = torch.zeros(1, 1, 8, 8)
        ones = torch.ones(1, 1, 8, 8)
        assert MonaiPanopticQualityMetric()(zeros, zeros) == [None]
        assert MonaiPanopticQualityMetric()(ones, zeros) == [0.0]

    def test_stale_threshold_raises(self):
        with pytest.raises(TypeError, match="threshold"):
            MonaiPanopticQualityMetric(threshold=0.5)


class TestPanopticQualityMetricBuilder:
    def test_returns_metric_spec(self):
        spec = panoptic_quality_metric()
        assert spec.name == "panoptic_quality"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == "medical (MONAI)"

    def test_default_constant(self):
        assert PANOPTIC_QUALITY.name == "panoptic_quality"

    def test_factory_produces_working_metric(self):
        spec = panoptic_quality_metric()
        metric = spec.slice_mode.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# Volume mode
# ---------------------------------------------------------------------------

import torch
from monai.metrics import compute_dice
from iqaevaluator.metrics import MetricRegistry, ModeSupport
from iqaevaluator.segmentation_metrics.monai_metrics import (
    ASSD, DICE, HAUSDORFF95, NSD, PANOPTIC_QUALITY, hausdorff95_metric,
)


def _volume_pair(shape=(1, 1, 6, 12, 12)):
    """(pred, gt) 5D binary volumes; pred is gt shifted by one voxel along x."""
    gt = torch.zeros(shape)
    gt[..., 2:4, 3:9, 3:9] = 1.0
    pred = torch.zeros(shape)
    pred[..., 2:4, 3:9, 4:10] = 1.0
    return pred, gt


class TestSegmentationVolumeMode:
    def test_all_five_support_volume(self):
        for spec in (DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY):
            assert isinstance(spec.volume_mode, ModeSupport), spec.name

    def test_dice_scores_a_five_dimensional_sample(self):
        metric = MetricRegistry(DICE).get_metric("dice", "volume", (1.0, 1.0, 1.0))
        pred, gt = _volume_pair()
        scores = metric(pred, gt)
        assert len(scores) == 1
        assert 0.0 < scores[0] < 1.0

    def test_panoptic_quality_scores_a_five_dimensional_sample(self):
        metric = MetricRegistry(PANOPTIC_QUALITY).get_metric("panoptic_quality", "volume", None)
        pred, gt = _volume_pair()
        assert len(metric(pred, gt)) == 1

    def test_spacing_changes_hausdorff_distance(self):
        pred, gt = _volume_pair()
        isotropic = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", (1.0, 1.0, 1.0))
        coarse    = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", (1.0, 1.0, 3.0))
        assert coarse(pred, gt)[0] > isotropic(pred, gt)[0]

    def test_missing_spacing_falls_back_to_voxel_units(self):
        pred, gt = _volume_pair()
        metric = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", None)
        assert metric(pred, gt)[0] > 0.0

    def test_explicit_builder_spacing_wins_over_runtime_spacing(self):
        pred, gt = _volume_pair()
        spec = hausdorff95_metric(spacing=(1.0, 1.0, 1.0))
        pinned = MetricRegistry(spec).get_metric("hausdorff95", "volume", (1.0, 1.0, 5.0))
        free   = MetricRegistry(HAUSDORFF95).get_metric("hausdorff95", "volume", (1.0, 1.0, 1.0))
        assert pinned(pred, gt)[0] == pytest.approx(free(pred, gt)[0])

    def test_slice_mode_still_works(self):
        metric = MetricRegistry(DICE).get_metric("dice")
        pred, gt = _volume_pair((4, 1, 12, 12))
        assert len(metric(pred, gt)) == 4


class TestMonaiSegmentationMetricContract:
    def test_non_binary_input_raises_naming_mask(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, name="dice", include_background=True)
        soft = torch.full((1, 1, 8, 8), 0.7)
        with pytest.raises(ValueError, match=r"dice expects a binary mask.*Mask\(\)"):
            metric(soft, torch.ones_like(soft))

    def test_stale_threshold_kwarg_raises_type_error(self):
        from monai.metrics import compute_dice
        with pytest.raises(TypeError, match="threshold"):
            MonaiSegmentationMetric(compute_dice, one_sided=0.0, threshold=0.5)

    def test_missing_target_raises(self):
        from monai.metrics import compute_dice
        with pytest.raises(ValueError, match="target"):
            MonaiSegmentationMetric(compute_dice, one_sided=0.0)(torch.ones(1, 1, 4, 4))

    def test_one_sided_empty_uses_the_policy_value(self):
        from monai.metrics import compute_dice, compute_hausdorff_distance
        pred = torch.zeros(2, 1, 8, 8); pred[0, 0, 2:5, 2:5] = 1.0     # sample 0: gt empty; sample 1: both populated
        gt = torch.zeros(2, 1, 8, 8); gt[1, 0, 2:5, 2:5] = 1.0; pred[1, 0, 2:5, 2:5] = 1.0
        dice = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True, ignore_empty=False)
        hd = MonaiSegmentationMetric(compute_hausdorff_distance, one_sided=None, include_background=True, percentile=95)
        assert dice(pred, gt) == [0.0, pytest.approx(1.0)]
        assert hd(pred, gt) == [None, pytest.approx(0.0)]

    def test_both_empty_is_none_for_every_one_sided_value(self):
        from monai.metrics import compute_dice
        zeros = torch.zeros(1, 1, 8, 8)
        assert MonaiSegmentationMetric(compute_dice, one_sided=0.0, ignore_empty=False)(zeros, zeros) == [None]
        assert MonaiSegmentationMetric(compute_dice, one_sided=None, ignore_empty=False)(zeros, zeros) == [None]

    def test_nonfinite_backend_output_becomes_none_with_a_warning(self, capsys):
        def broken(y_pred, y):
            return torch.full((y_pred.shape[0], 1), float("inf"))
        metric = MonaiSegmentationMetric(broken, one_sided=0.0, name="broken")
        ones = torch.ones(1, 1, 4, 4)
        assert metric(ones, ones) == [None]
        assert "broken" in capsys.readouterr().out

    def test_batch_mixes_policy_and_computed_samples_in_order(self):
        from monai.metrics import compute_dice
        pred = torch.zeros(3, 1, 8, 8); gt = torch.zeros(3, 1, 8, 8)
        pred[0, 0, :4] = 1.0; gt[0, 0, :4] = 1.0        # identical → 1.0
        gt[1, 0, :4] = 1.0                              # pred empty → 0.0
        pred[2, 0, :4] = 1.0; gt[2, 0, 2:6] = 1.0       # half overlap → 0.5
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True, ignore_empty=False)
        assert metric(pred, gt) == [pytest.approx(1.0), 0.0, pytest.approx(0.5)]


@pytest.mark.parametrize("builder", [
    dice_metric, hausdorff95_metric, normalized_surface_dice_metric,
    average_surface_distance_metric, panoptic_quality_metric,
])
@pytest.mark.parametrize("stale", ["threshold", "label"])
def test_builders_reject_the_removed_knobs_loudly(builder, stale):
    with pytest.raises(TypeError, match=stale):
        builder(**{stale: 0.5})


def test_dice_is_built_without_monais_empty_skipping():
    """ignore_empty=False: the policy decides, MONAI must not answer NaN for an empty gt."""
    pred = torch.zeros(1, 1, 8, 8); pred[0, 0, :2] = 1.0
    assert DICE.slice_mode.factory()(pred, torch.zeros_like(pred)) == [0.0]


class TestIntegerMasks:
    def test_identical_integer_masks_score_perfect_dice(self):
        g = torch.Generator().manual_seed(0)
        mask = torch.randint(0, 2, (2, 1, 16, 16), generator=g)          # int64
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, include_background=True)
        assert metric(mask, mask.clone()) == pytest.approx([1.0, 1.0])

    def test_multi_label_integer_map_is_rejected(self):
        labels = torch.zeros((1, 1, 8, 8), dtype=torch.int16)
        labels[0, 0, :4] = 1
        labels[0, 0, 4:] = 3
        metric = MonaiSegmentationMetric(compute_dice, one_sided=0.0, name="dice", include_background=True)
        with pytest.raises(ValueError, match=r"Mask\(\)"):
            metric(labels, torch.ones_like(labels))


class TestMaskLabelMapEndToEnd:
    """A real integer NIfTI label map through Mask() and the adapter."""

    def _label_map(self, tmp_path):
        import nibabel as nib
        labels = np.zeros((10, 4, 1), dtype=np.int16)
        labels[2:4, :, 0] = 1
        labels[4:7, :, 0] = 2
        labels[7:10, :, 0] = 3
        p = tmp_path / "multi_label.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        return p

    def test_any_nonzero_label_is_foreground_under_mask(self, tmp_path):
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Mask
        pred = ImageLoader(self._label_map(tmp_path), Mask()).tensor  # (1, 1, 10, 4) float {0, 1}
        gt = torch.zeros((1, 1, 10, 4)); gt[:, :, 2:10, :] = 1.0
        assert DICE.slice_mode.factory()(pred, gt) == [pytest.approx(1.0)]

    def test_one_label_under_mask_label(self, tmp_path):
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Mask
        pred = ImageLoader(self._label_map(tmp_path), Mask(label=2)).tensor
        gt = torch.zeros((1, 1, 10, 4)); gt[:, :, 4:7, :] = 1.0
        assert DICE.slice_mode.factory()(pred, gt) == [pytest.approx(1.0)]

    def test_raw_label_map_is_rejected_with_a_pointer_to_mask(self, tmp_path):
        from iqaevaluator.image_loader import ImageLoader
        from iqaevaluator.normalization import Raw
        pred = ImageLoader(self._label_map(tmp_path), Raw()).tensor      # int16 {0,1,2,3}
        with pytest.raises(ValueError, match=r"Mask\(\)"):
            DICE.slice_mode.factory()(pred, torch.ones_like(pred))
