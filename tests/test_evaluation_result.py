"""Tests for src/evaluation_result.py — EvaluationResult.to_frame / generate_report."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation_result import EvaluationResult, _EvaluatedImage
from records import ImageEvaluatorRecord
from metrics import MetricRegistry, PSNR, SSIM


def _make_result(n_images: int = 2, slices_each: int = 2,
                 registry: MetricRegistry | None = None) -> EvaluationResult:
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
    return EvaluationResult(images, registry if registry is not None else MetricRegistry())


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

    def test_custom_metric_column(self, fake_metric):
        reg = MetricRegistry()
        reg.register_metric("frame_custom", fake_metric,
                            direction="higher_is_better", reference=False)
        rec = ImageEvaluatorRecord(image_id="t", slice_index=0)
        rec.extra["frame_custom"] = 0.7
        images = [_EvaluatedImage(input_path=Path("/fake/t.png"), records=[rec])]
        res = EvaluationResult(images, reg)
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
        from image_loader import ImageLoader
        from iqa_evaluator import IQAEvaluator

        reg = MetricRegistry(PSNR, SSIM)   # fast metrics only

        arr = (np.random.default_rng(5).random((96, 96)) * 255).astype("uint8")
        inp_p = tmp_path / "inp.png"; tgt_p = tmp_path / "tgt.png"
        Image.fromarray(arr).save(inp_p)
        noise = np.clip(arr.astype(int) + 5, 0, 255).astype("uint8")
        Image.fromarray(noise).save(tgt_p)

        inp = ImageLoader(inp_p); tgt = ImageLoader(tgt_p)
        records = IQAEvaluator(inp, tgt, reg).run_evaluation()
        return EvaluationResult([_EvaluatedImage(input_path=inp_p, records=records)], reg)

    def test_csv_written(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        csv_path = tmp_path / "report.csv"
        res.generate_report(csv_path)
        assert csv_path.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        csv_path = tmp_path / "report.csv"
        res.generate_report(csv_path)
        import pandas as pd
        loaded = pd.read_csv(csv_path)
        assert "image_id" in loaded.columns
        assert "psnr" in loaded.columns

    def test_returns_dataframe_equal_to_to_frame(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        df_direct = res.to_frame()
        df_report = res.generate_report(tmp_path / "r.csv")
        assert list(df_direct["image_id"]) == list(df_report["image_id"])

    def test_no_mask_dir_created(self, tmp_path):
        res = self._build_result_from_real_image(tmp_path)
        res.generate_report(tmp_path / "r.csv")
        assert not (tmp_path / "masks").exists()
