"""Tests for src/segmentation_metrics/monai_metrics.py — MONAI-backed segmentation metrics."""
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
        metric = spec.factory()
        pred, gt = _binary_batch(n=2)
        scores = metric(pred, gt)
        assert len(scores) == 2

    def test_user_kwargs_pass_through(self):
        spec = dice_metric(include_background=False)
        metric = spec.factory()
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
        metric = spec.factory()
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
        metric = spec.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(1.0) for s in scores)

    def test_custom_class_thresholds_override(self):
        spec = normalized_surface_dice_metric(class_thresholds=[2.0])
        metric = spec.factory()
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
        metric = spec.factory()
        pred, _ = _binary_batch(n=2)
        scores = metric(pred, pred.clone())
        assert all(s == pytest.approx(0.0) for s in scores)

    def test_default_constant(self):
        assert ASSD.name == "assd"
