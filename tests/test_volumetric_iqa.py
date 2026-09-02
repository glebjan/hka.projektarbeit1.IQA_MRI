"""Tests for MONAI-backed volumetric PSNR and SSIM."""
import pytest
import torch

from metrics import Metric, MetricRegistry, ModeSupport, PSNR, SSIM
from volumetric_iqa import MonaiPSNRMetric, MonaiSSIMMetric


def _volume_pair(noise: float = 0.05):
    torch.manual_seed(0)
    gt = torch.rand(1, 1, 6, 16, 16)
    pred = (gt + noise * torch.randn_like(gt)).clamp(0.0, 1.0)
    return pred, gt


class TestMonaiPSNRMetric:
    def test_satisfies_the_metric_protocol(self):
        assert isinstance(MonaiPSNRMetric(), Metric)

    def test_returns_one_score_per_sample(self):
        pred, gt = _volume_pair()
        assert len(MonaiPSNRMetric()(pred, gt)) == 1

    def test_identical_volumes_score_very_high(self):
        gt = torch.rand(1, 1, 4, 8, 8)
        assert MonaiPSNRMetric()(gt, gt)[0] > 60.0

    def test_more_noise_scores_lower(self):
        low, gt = _volume_pair(noise=0.02)
        high, _ = _volume_pair(noise=0.20)
        metric = MonaiPSNRMetric()
        assert metric(high, gt)[0] < metric(low, gt)[0]

    def test_requires_a_target(self):
        pred, _ = _volume_pair()
        with pytest.raises(ValueError, match="target"):
            MonaiPSNRMetric()(pred)


class TestMonaiSSIMMetric:
    def test_returns_one_score_per_sample(self):
        pred, gt = _volume_pair()
        assert len(MonaiSSIMMetric()(pred, gt)) == 1

    def test_identical_volumes_score_one(self):
        gt = torch.rand(1, 1, 6, 16, 16)
        assert MonaiSSIMMetric()(gt, gt)[0] == pytest.approx(1.0, abs=1e-4)

    def test_more_noise_scores_lower(self):
        low, gt = _volume_pair(noise=0.02)
        high, _ = _volume_pair(noise=0.20)
        metric = MonaiSSIMMetric()
        assert metric(high, gt)[0] < metric(low, gt)[0]

    def test_window_shrinks_to_fit_a_thin_volume(self):
        """A 6-slice stack is thinner than MONAI's default 11-voxel window."""
        thin = torch.rand(1, 1, 6, 16, 16)
        assert MonaiSSIMMetric()._window_for(thin.shape) == 5
        assert MonaiSSIMMetric()(thin, thin)[0] == pytest.approx(1.0, abs=1e-4)

    def test_window_is_never_grown_beyond_the_preferred_size(self):
        thick = torch.rand(1, 1, 40, 40, 40)
        assert MonaiSSIMMetric()._window_for(thick.shape) == 11


class TestSpecsWired:
    def test_psnr_and_ssim_support_volume(self):
        assert isinstance(PSNR.volume_mode, ModeSupport)
        assert isinstance(SSIM.volume_mode, ModeSupport)

    def test_registry_builds_the_monai_backends(self):
        registry = MetricRegistry(PSNR, SSIM)
        assert isinstance(registry.get_metric("psnr", "volume", None), MonaiPSNRMetric)
        assert isinstance(registry.get_metric("ssim", "volume", None), MonaiSSIMMetric)
