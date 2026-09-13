"""Layer 3: framework == MONAI's own expected values, through Mask()/Raw() and the adapters."""
import pytest

from calibration.harness import framework_value
from calibration.monai_cases import MONAI_CASES


@pytest.mark.parametrize("case", MONAI_CASES, ids=lambda c: c.name)
def test_framework_reproduces_monais_expected_value(case, tmp_path):
    value = framework_value(case, tmp_path)
    if case.expected is None:
        assert value is None, case.source
    else:
        assert value == pytest.approx(case.expected, abs=case.tolerance), case.source
