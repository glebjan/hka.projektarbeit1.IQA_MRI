"""Tests for src/segmentation_metrics/monai_metrics.py — MONAI-backed segmentation metrics."""
import torch
import pytest

from metrics import MetricSpec
from segmentation_metrics.monai_metrics import (
    MonaiSegmentationMetric,
    dice_metric,
    DICE,
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
