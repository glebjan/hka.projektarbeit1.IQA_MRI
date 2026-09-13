"""The report renders (on a two-case subset, to stay fast) and holds no blanks."""
from calibration.cases import CASES
from calibration.monai_cases import MONAI_CASES
from calibration.report import render


def test_render_produces_the_tables():
    subset = [next(c for c in CASES if c.name == "F1_dice"), next(c for c in CASES if c.name == "F3_dice_both_empty")]
    monai = [next(c for c in MONAI_CASES if c.name == "monai_dice_case_1")]
    text = render(subset, monai)
    assert "# Segmentation-metric calibration report" in text
    assert "## dice" in text and "| F1_dice | 0.833333 |" in text
    assert "| F3_dice_both_empty | — | — |" in text
    assert "| monai_dice_case_1 | 0.800000 | 0.800000 |" in text
    assert "TBD" not in text and "| None |" not in text and "| nan |" not in text
