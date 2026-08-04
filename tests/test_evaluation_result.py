"""Tests for src/evaluation_result.py — EvaluationResult.to_frame / generate_report."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation_result import EvaluationResult, _EvaluatedImage
from records import ImageEvaluatorRecord
from metrics import register_metric


def _make_result(n_images: int = 2, slices_each: int = 2) -> EvaluationResult:
    """Build an EvaluationResult with fake ImageLoader paths and records."""
    images = []
    for i in range(n_images):
        records = [
            ImageEvaluatorRecord(
                image_id=f"img{i}_s{j}",
                slice_index=j,
                psnr=30.0 + j,
                ssim=0.9,
            )
            for j in range(slices_each)
        ]
        images.append(_EvaluatedImage(input_path=Path(f"/fake/img{i}.png"), records=records))
    return EvaluationResult(images)


# ---------------------------------------------------------------------------
# to_frame
# ---------------------------------------------------------------------------

class TestToFrame:
    def test_row_count(self):
        res = _make_result(n_images=3, slices_each=4)
        df = res.to_frame()
        assert len(df) == 12

    def test_fixed_columns_present(self):
        res = _make_result()
        df = res.to_frame()
        for col in ("image_id", "psnr", "ssim", "lpips", "is_empty", "mode"):
            assert col in df.columns, f"column '{col}' missing"

    def test_no_extra_column(self):
        # When no custom metrics registered, 'extra' should not appear as column
        res = _make_result()
        df = res.to_frame()
        assert "extra" not in df.columns

    def test_custom_metric_column(self, isolated_registry, fake_metric):
        register_metric("frame_custom", fake_metric, direction="higher_is_better", reference=False)
        images = []
        rec = ImageEvaluatorRecord(image_id="t", slice_index=0)
        rec.extra["frame_custom"] = 0.7
        images.append(_EvaluatedImage(input_path=Path("/fake/t.png"), records=[rec]))
        res = EvaluationResult(images)
        df = res.to_frame()
        assert "frame_custom" in df.columns

    def test_values_correct(self):
        res = _make_result(n_images=1, slices_each=2)
        df = res.to_frame()
        assert list(df["psnr"]) == [30.0, 31.0]


# ---------------------------------------------------------------------------
# generate_report — real I/O into tmp_path
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _build_result_from_real_image(self, tmp_path: Path) -> EvaluationResult:
        """Create a minimal real EvaluationResult from a synthetic PNG."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        from image_loader import ImageLoader
        from iqa_evaluator import IQAEvaluator
        from metrics import registry

        # Keep only fast metrics for report test
        keep = {"psnr", "ssim"}
        for name in [s.name for s in registry.specs]:
            if name not in keep:
                registry._specs.pop(name, None)
                registry._cache.pop(name, None)

        arr = (np.random.default_rng(5).random((96, 96)) * 255).astype("uint8")
        inp_p = tmp_path / "inp.png"; tgt_p = tmp_path / "tgt.png"
        Image.fromarray(arr).save(inp_p)
        noise = np.clip(arr.astype(int) + 5, 0, 255).astype("uint8")
        Image.fromarray(noise).save(tgt_p)

        from image_loader import ImageLoader
        inp = ImageLoader(inp_p); tgt = ImageLoader(tgt_p)
        records = IQAEvaluator(inp, tgt).run_evaluation()
        return EvaluationResult([_EvaluatedImage(input_path=inp_p, records=records)])

    def test_csv_written(self, tmp_path, isolated_registry):
        res = self._build_result_from_real_image(tmp_path)
        csv_path = tmp_path / "report.csv"
        mask_dir = tmp_path / "masks"
        df = res.generate_report(csv_path, mask_dir)
        assert csv_path.exists()

    def test_csv_has_correct_columns(self, tmp_path, isolated_registry):
        res = self._build_result_from_real_image(tmp_path)
        csv_path = tmp_path / "report.csv"
        df = res.generate_report(csv_path, tmp_path / "masks")
        import pandas as pd
        loaded = pd.read_csv(csv_path)
        assert "image_id" in loaded.columns
        assert "psnr" in loaded.columns

    def test_mask_pngs_written(self, tmp_path, isolated_registry):
        res = self._build_result_from_real_image(tmp_path)
        mask_dir = tmp_path / "masks"
        res.generate_report(tmp_path / "rep.csv", mask_dir)
        pngs = list(mask_dir.rglob("*.png"))
        assert len(pngs) > 0

    def test_returns_dataframe_equal_to_to_frame(self, tmp_path, isolated_registry):
        res = self._build_result_from_real_image(tmp_path)
        df_direct = res.to_frame()
        df_report = res.generate_report(tmp_path / "r.csv", tmp_path / "m")
        assert list(df_direct["image_id"]) == list(df_report["image_id"])
