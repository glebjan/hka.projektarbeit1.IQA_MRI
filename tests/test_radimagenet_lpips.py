"""Tests for src/radimagenet_lpips.py — custom LPIPS backbone with RadImageNet weights."""
import pytest
import torch
from pathlib import Path

# Guard: skip entire module if weights file is absent
from constants import RESNET50
pytestmark = pytest.mark.skipif(
    not Path(RESNET50).exists(),
    reason=f"RadImageNet weights not found at {RESNET50}",
)

from radimagenet_lpips import (
    RadImageNetLPIPS,
    _RadImageNetBackbone,
    _remap_backbone_keys,
)
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.default_model_configs import DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# _remap_backbone_keys
# ---------------------------------------------------------------------------

class TestRemapBackboneKeys:
    def test_stem_remap(self):
        raw = {"backbone.0.weight": torch.tensor(1.0), "backbone.1.bias": torch.tensor(2.0)}
        out = _remap_backbone_keys(raw)
        assert "stem.0.weight" in out
        assert "stem.1.bias" in out
        assert "backbone.0.weight" not in out

    def test_layer_remap(self):
        raw = {
            "backbone.4.0.conv1.weight": torch.tensor(1.0),
            "backbone.7.2.bn3.bias": torch.tensor(2.0),
        }
        out = _remap_backbone_keys(raw)
        assert "layer1.0.conv1.weight" in out
        assert "layer4.2.bn3.bias" in out

    def test_unknown_keys_excluded(self):
        raw = {"totally.unknown.key": torch.tensor(0.0)}
        out = _remap_backbone_keys(raw)
        assert len(out) == 0


# ---------------------------------------------------------------------------
# _RadImageNetBackbone
# ---------------------------------------------------------------------------

class TestRadImageNetBackbone:
    @pytest.fixture(scope="class")
    def backbone(self):
        return _RadImageNetBackbone(str(RESNET50))

    def test_loads_without_error(self, backbone):
        assert backbone is not None

    def test_parameters_frozen(self, backbone):
        for p in backbone.parameters():
            assert not p.requires_grad

    def test_forward_returns_five_feature_maps(self, backbone):
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            feats = backbone(x)
        assert len(feats) == 5

    def test_feature_map_channel_dims(self, backbone):
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            feats = backbone(x)
        expected_channels = _RadImageNetBackbone.CHANNEL_DIMS
        for feat, ch in zip(feats, expected_channels):
            assert feat.shape[1] == ch, f"expected {ch} channels, got {feat.shape[1]}"


# ---------------------------------------------------------------------------
# RadImageNetLPIPS
# ---------------------------------------------------------------------------

class TestRadImageNetLPIPS:
    @pytest.fixture(scope="class")
    def model(self):
        return RadImageNetLPIPS(backbone_path=str(RESNET50)).eval()

    def test_identical_images_score_near_zero(self, model):
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            scores = model(x, x)
        # forward() returns (N, 1) — the squeeze to (N,) happens inside PyIQAMetric
        assert scores.shape[0] == 2
        assert torch.all(scores < 1e-5)

    def test_different_images_positive_score(self, model):
        a = torch.zeros(1, 3, 64, 64)
        b = torch.ones(1, 3, 64, 64)
        with torch.no_grad():
            score = model(a, b)
        assert float(score.sum()) > 0.0

    def test_output_shape(self, model):
        x = torch.rand(3, 3, 64, 64)
        with torch.no_grad():
            out = model(x, x)
        # forward() returns (N, 1)
        assert out.shape[0] == 3

    def test_more_distortion_higher_score(self, model):
        base = torch.rand(1, 3, 64, 64)
        slight = (base + 0.02 * torch.rand_like(base)).clamp(0, 1)
        heavy  = (base + 0.5  * torch.rand_like(base)).clamp(0, 1)
        with torch.no_grad():
            score_slight = model(slight, base).item()
            score_heavy  = model(heavy,  base).item()
        assert score_heavy > score_slight


# ---------------------------------------------------------------------------
# Registration in pyiqa
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_in_default_configs(self):
        assert "radimagenet_lpips" in DEFAULT_CONFIGS
        cfg = DEFAULT_CONFIGS["radimagenet_lpips"]
        assert cfg["lower_better"] is True
        assert cfg["metric_mode"] == "FR"

    def test_in_arch_registry(self):
        # ARCH_REGISTRY stores classes by name; RadImageNetLPIPS should be registered
        assert ARCH_REGISTRY.get("RadImageNetLPIPS") is RadImageNetLPIPS
