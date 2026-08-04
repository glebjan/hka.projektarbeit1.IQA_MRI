"""Tests for src/clip_iqa_medical.py — ClipIQALung, ClipIQABrain."""
import pytest
import torch

from clip_iqa_medical import (
    ClipIQABrain,
    ClipIQALung,
    _BRAIN_PROMPTS,
    _LUNG_PROMPTS,
)
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.default_model_configs import DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Prompt list invariants
# ---------------------------------------------------------------------------

class TestPromptLists:
    def test_lung_prompts_even_count(self):
        assert len(_LUNG_PROMPTS) % 2 == 0, "Lung prompts must be even (pos/neg pairs)"

    def test_brain_prompts_even_count(self):
        assert len(_BRAIN_PROMPTS) % 2 == 0, "Brain prompts must be even (pos/neg pairs)"

    def test_lung_prompts_not_empty(self):
        assert len(_LUNG_PROMPTS) >= 2

    def test_brain_prompts_not_empty(self):
        assert len(_BRAIN_PROMPTS) >= 2

    def test_no_duplicate_prompts_within_lung(self):
        assert len(_LUNG_PROMPTS) == len(set(_LUNG_PROMPTS))

    def test_no_duplicate_prompts_within_brain(self):
        assert len(_BRAIN_PROMPTS) == len(set(_BRAIN_PROMPTS))


# ---------------------------------------------------------------------------
# ClipIQALung
# ---------------------------------------------------------------------------

class TestClipIQALung:
    @pytest.fixture(scope="class")
    def model(self):
        return ClipIQALung().eval()

    def test_output_shape(self, model):
        x = torch.rand(2, 3, 96, 96)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"

    def test_output_in_zero_one(self, model):
        x = torch.rand(3, 3, 96, 96)
        with torch.no_grad():
            out = model(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_single_image(self, model):
        x = torch.rand(1, 3, 96, 96)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_parameters_frozen(self, model):
        for p in model.clip_model[0].parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# ClipIQABrain
# ---------------------------------------------------------------------------

class TestClipIQABrain:
    @pytest.fixture(scope="class")
    def model(self):
        return ClipIQABrain().eval()

    def test_output_shape(self, model):
        x = torch.rand(2, 3, 96, 96)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)

    def test_output_in_zero_one(self, model):
        x = torch.rand(3, 3, 96, 96)
        with torch.no_grad():
            out = model(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_different_prompts_yield_different_scores(self):
        """Lung and Brain models have different prompt sets → different scores for same input."""
        lung_model  = ClipIQALung().eval()
        brain_model = ClipIQABrain().eval()
        x = torch.rand(1, 3, 96, 96)
        with torch.no_grad():
            lung_score  = lung_model(x).item()
            brain_score = brain_model(x).item()
        # The two models use different prompts so scores should differ
        assert lung_score != brain_score


# ---------------------------------------------------------------------------
# Registration in pyiqa
# ---------------------------------------------------------------------------

class TestClipRegistration:
    def test_lung_in_default_configs(self):
        assert "clip_iqa_lung" in DEFAULT_CONFIGS
        cfg = DEFAULT_CONFIGS["clip_iqa_lung"]
        assert cfg["metric_mode"] == "NR"
        assert cfg["lower_better"] is False

    def test_brain_in_default_configs(self):
        assert "clip_iqa_brain" in DEFAULT_CONFIGS
        cfg = DEFAULT_CONFIGS["clip_iqa_brain"]
        assert cfg["metric_mode"] == "NR"
        assert cfg["lower_better"] is False

    def test_lung_in_arch_registry(self):
        assert ARCH_REGISTRY.get("ClipIQALung") is ClipIQALung

    def test_brain_in_arch_registry(self):
        assert ARCH_REGISTRY.get("ClipIQABrain") is ClipIQABrain
