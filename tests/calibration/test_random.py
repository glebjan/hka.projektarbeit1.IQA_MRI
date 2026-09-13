"""Layer 5: seeded random blobs, three spacings — framework == MedPy for Dice, ASSD and HD."""
import pytest

from calibration.cases import CASES
from calibration.harness import framework_value
from calibration.official import official_value

RANDOM = [c for c in CASES if c.kind == "random"]


@pytest.mark.parametrize("case", RANDOM, ids=lambda c: c.name)
def test_framework_agrees_with_medpy_on_random_blobs(case, tmp_path):
    try:
        official = official_value(case, tmp_path)
    except ImportError as exc:
        pytest.skip(f"reference package missing: {exc}")
    assert official is not None
    assert framework_value(case, tmp_path) == pytest.approx(official, abs=case.tolerance)
