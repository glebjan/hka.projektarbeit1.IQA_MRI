"""build_evaluator — pick the evaluator for a scoring mode.

Callers select behaviour by mode everywhere else in the framework, so the split
between IQAEvaluator (slices) and VolumeEvaluator (volumes) is kept out of their
way. Both return `list[ImageEvaluatorRecord]` from `run_evaluation()`, so calling
code is unaffected by which one it gets.

This lives in its own module because iqa_evaluator.py cannot import
VolumeEvaluator (which imports IQAEvaluator) without a circular import, and
main.py is scheduled for replacement.
"""

from typing import Optional

from image_loader import ImageLoader
from iqa_evaluator import IQAEvaluator
from metrics import MetricRegistry, ScoringMode
from volume_evaluator import VolumeEvaluator


def build_evaluator(
    input_image:  ImageLoader,
    target_image: Optional[ImageLoader],
    registry:     MetricRegistry,
    mode:         ScoringMode = "slice",
    source_model: Optional[str] = None,
) -> IQAEvaluator:
    """Return the evaluator for `mode`.

    Args:
        mode: "slice" scores every 2D slice separately; "volume" scores the
            whole 3D stack once.

    Raises:
        ValueError: for any other mode.
    """
    if mode == "slice":
        return IQAEvaluator(input_image, target_image, registry, source_model)
    if mode == "volume":
        return VolumeEvaluator(input_image, target_image, registry, source_model)
    raise ValueError(
        f"'{mode}' is not a scoring mode. Use 'slice' to score every 2D slice "
        "separately, or 'volume' to score the whole 3D stack at once."
    )
