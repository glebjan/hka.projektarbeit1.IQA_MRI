"""Tests for src/segmentation_metrics/monai_metrics.py — MONAI-backed segmentation metrics."""
import numpy as np
import torch
import pytest

from metrics import MetricSpec
from segmentation_metrics.monai_metrics import (
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
        metric = MonaiSegmentationMetric(compute_dice, include_background=True)
        pred, gt = _binary_batch(n=3)
        scores = metric(pred, gt)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_identical_masks_score_perfect_dice(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, include_background=True)
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)

    def test_threshold_binarizes_before_computing(self):
        from monai.metrics import compute_dice
        metric = MonaiSegmentationMetric(compute_dice, include_background=True, threshold=0.5)
        pred = torch.full((1, 1, 8, 8), 0.7)
        gt = torch.full((1, 1, 8, 8), 0.9)
        scores = metric(pred, gt)
        assert scores[0] == pytest.approx(1.0)  # both binarize to all-ones


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
from metrics import MetricRegistry, ModeSupport
from segmentation_metrics.monai_metrics import (
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


class TestIntegerMasks:
    def test_identical_integer_masks_score_perfect_dice(self):
        g = torch.Generator().manual_seed(0)
        mask = torch.randint(0, 2, (2, 1, 16, 16), generator=g)          # int64
        metric = MonaiSegmentationMetric(compute_dice, include_background=True)
        assert metric(mask, mask.clone()) == pytest.approx([1.0, 1.0])

    def test_threshold_zero_makes_every_label_foreground(self):
        labels = torch.zeros((1, 1, 8, 8), dtype=torch.int16)
        labels[0, 0, :4] = 1
        labels[0, 0, 4:] = 3
        metric = MonaiSegmentationMetric(compute_dice, include_background=True, threshold=0.0)
        assert metric(labels, torch.ones_like(labels)) == pytest.approx([1.0])


class TestRawLabelMapEndToEnd:
    """End-to-end: a real integer NIfTI label map, loaded via Raw(), through
    MonaiSegmentationMetric. Mirrors TestRawLabelMapEndToEnd in
    tests/test_volume.py, which covers the same namesake bug for the
    numpy-backed volume metrics.
    """

    def test_multi_label_mask_loaded_raw_scores_perfect_dice(self, tmp_path):
        import nibabel as nib
        from image_loader import ImageLoader
        from normalization import Raw

        # Multi-label mask spanning {0, 1, 2, 3} so a min-max scaling has the
        # file's full [0, 3] range to compress into [0, 1]: label 1 -> 0.333
        # (falls under a 0.5 cutoff), labels 2 -> 0.667 and 3 -> 1.0 (both
        # clear it). Label 1's region (rows 2-3) is therefore the sharpest
        # discriminator between Raw() and MinMax() at threshold=0.5.
        labels = np.zeros((10, 4, 1), dtype=np.int16)
        labels[0:2, :, 0] = 0
        labels[2:4, :, 0] = 1
        labels[4:7, :, 0] = 2
        labels[7:10, :, 0] = 3
        p = tmp_path / "multi_label.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))

        pred = ImageLoader(p, Raw()).tensor  # (D, 1, H, W), int16, unscaled
        assert pred.dtype == torch.int16
        assert pred.shape == (1, 1, 10, 4)  # the (10, 4, 1) array's Z axis
        # (size 1) becomes D; X and Y become H and W unchanged -- verified
        # directly, not assumed: pred[0, 0] reproduces the labels array
        # row-for-row (row index == the numpy array's first-axis index).

        # Ground truth is written independently from the known label
        # geometry above -- H rows 0-1 (label 0) are background, H rows 2-9
        # (labels 1, 2, 3) are foreground -- never derived from `pred` by a
        # transformation of it, so the assertion below can actually fail.
        gt = torch.zeros((1, 1, 10, 4), dtype=torch.float32)
        gt[:, :, 2:10, :] = 1.0

        metric = MonaiSegmentationMetric(compute_dice, include_background=True, threshold=0.5)
        scores = metric(pred, gt)

        # threshold=0.5 is the cutoff that caused the original bug. Under
        # Raw() (the fix), label 1 arrives as the raw integer 1, clears 0.5,
        # and pred's foreground exactly matches gt: dice == 1.0 everywhere.
        # Confirmed by direct substitution (see the fix report): loading the
        # same file with MinMax() instead of Raw() scales label 1 to 0.333,
        # which fails the 0.5 threshold and drops rows 2-3 into the
        # background, so those slices score dice < 1.0 under that swap.
        assert scores == pytest.approx([1.0] * pred.shape[0])
