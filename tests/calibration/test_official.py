"""Layer 2: framework == official implementation of the same variant (needs the [calibration] extras)."""
import numpy as np
import pytest

from calibration.cases import CASES
from calibration.harness import framework_value
from calibration.official import boundary_iou_official, official_value

HAND = [c for c in CASES if c.kind == "hand"]


def _official_or_skip(case, tmp_path):
    try:
        value = official_value(case, tmp_path)
    except ImportError as exc:
        pytest.skip(f"reference package missing: {exc}")
    if value is None:
        pytest.skip(f"no official implementation of the framework's {case.metric} variant")
    return value


@pytest.mark.parametrize("case", HAND, ids=lambda c: c.name)
def test_framework_matches_the_official_implementation(case, tmp_path):
    official = _official_or_skip(case, tmp_path)
    assert framework_value(case, tmp_path) == pytest.approx(official, abs=case.tolerance)


def test_2d_boundary_iou_anchor_matches_the_vendored_reference():
    """The existing 0.3822 anchor (tests/test_boundary_iou.py) against Cheng et al.'s own code."""
    from iqaevaluator.segmentation_metrics.boundary_iou import boundary_iou

    def square(size, offset):
        m = np.zeros((400, 400), bool); m[offset:offset + size, offset:offset + size] = True
        return m
    a, b = square(160, 20), square(160, 25)
    assert boundary_iou(a, b) == pytest.approx(boundary_iou_official(a, b), abs=1e-9)
    assert boundary_iou(a, b) == pytest.approx(0.3822, abs=1e-4)
