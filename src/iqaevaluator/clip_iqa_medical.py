"""CLIP-IQA variants with domain-specific prompts for lung and brain MRI.

Importing this module registers ClipIQALung and ClipIQABrain in pyiqa's
ARCH_REGISTRY and injects their entries into DEFAULT_CONFIGS so that
pyiqa.create_metric("clip_iqa_lung") and pyiqa.create_metric("clip_iqa_brain")
work out of the box.

Prompt pairs follow the same alternating positive/negative convention as pyiqa's
built-in CLIPIQA: even indices are positive descriptors, odd indices are negative.
The forward() reshapes logits to (-1, 2) before softmax, so pair count must be even.
"""

import clip
import torch
import torch.nn as nn

from pyiqa.archs.clip_model import load
from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY

_LUNG_PROMPTS = [
    "Sharp pulmonary structures",       "Blurry pulmonary structures",
    "Clear lung tissue",                "Noisy lung tissue",
    "Well-defined bronchi",             "Indistinct bronchi",
    "High contrast lung scan",          "Low contrast lung scan",
    "Artifact-free lung MRI",           "Artifact-degraded lung MRI",
]

_BRAIN_PROMPTS = [
    "Sharp brain structures",               "Blurry brain structures",
    "Clear white matter",                   "Noisy white matter",
    "Well-defined cortical boundaries",     "Indistinct cortical boundaries",
    "High contrast brain scan",             "Low contrast brain scan",
    "Artifact-free brain MRI",             "Artifact-degraded brain MRI",
]


class _ClipIQAMedicalBase(nn.Module):
    """Base class for domain-specific CLIP-IQA variants.

    Subclasses supply a list of prompt strings (alternating positive/negative).
    The backbone (CLIP RN50) is shared and frozen — no additional downloads
    beyond the standard CLIP weights.
    """

    _prompts: list[str] = []

    def __init__(self, backbone: str = "RN50", **kwargs):
        super().__init__()
        self.clip_model = [load(backbone, "cpu")]
        self.prompt_pairs = clip.tokenize(self._prompts)
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        for p in self.clip_model[0].parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        clip_model = self.clip_model[0].to(x)
        prompts = self.prompt_pairs.to(x.device)
        logits_per_image, _ = clip_model(x, prompts)
        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        return probs[..., 0].mean(dim=1, keepdim=True)


@ARCH_REGISTRY.register()
class ClipIQALung(_ClipIQAMedicalBase):
    """CLIP-IQA with lung MRI-specific quality prompts."""
    _prompts = _LUNG_PROMPTS


@ARCH_REGISTRY.register()
class ClipIQABrain(_ClipIQAMedicalBase):
    """CLIP-IQA with brain MRI-specific quality prompts."""
    _prompts = _BRAIN_PROMPTS


DEFAULT_CONFIGS["clip_iqa_lung"] = {
    "metric_opts": {"type": "ClipIQALung"},
    "metric_mode": "NR",
    "lower_better": False,
    "score_range": "0, 1",
}

DEFAULT_CONFIGS["clip_iqa_brain"] = {
    "metric_opts": {"type": "ClipIQABrain"},
    "metric_mode": "NR",
    "lower_better": False,
    "score_range": "0, 1",
}
