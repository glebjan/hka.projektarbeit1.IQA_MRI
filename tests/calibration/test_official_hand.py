"""Layer 2b: official implementation == hand value — pins the reference itself."""
import pytest

from calibration.cases import CASES
from calibration.official import official_value

HAND = [c for c in CASES if c.kind == "hand"]


@pytest.mark.parametrize("case", HAND, ids=lambda c: c.name)
def test_official_implementation_reproduces_the_hand_value(case, tmp_path):
    try:
        official = official_value(case, tmp_path)
    except ImportError as exc:
        pytest.skip(f"reference package missing: {exc}")
    if official is None:
        pytest.skip(f"no official implementation of the framework's {case.metric} variant")
    assert official == pytest.approx(case.expected, abs=case.tolerance), case.derivation
