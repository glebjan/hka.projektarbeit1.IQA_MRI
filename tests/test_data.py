"""Tests for src/data.py — analyze() function."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data import analyze


IMG_SIZE = 96


def _make_png(path: Path, seed: int = 0) -> Path:
    arr = (np.random.default_rng(seed).random((IMG_SIZE, IMG_SIZE)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_nonexistent_path_returns_empty_df(self, tmp_path):
        df = analyze(tmp_path / "does_not_exist", tmp_path / "out.csv")
        assert df.empty
        assert list(df.columns) == ["path", "depth", "channels", "height", "width"]

    def test_single_file_path(self, tmp_path):
        p = _make_png(tmp_path / "img.png")
        df = analyze(p, tmp_path / "out.csv")
        assert len(df) == 1
        assert df.iloc[0]["depth"] == 1
        assert df.iloc[0]["height"] == IMG_SIZE
        assert df.iloc[0]["width"] == IMG_SIZE

    def test_directory_with_multiple_images(self, tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        for i in range(3):
            _make_png(data_dir / f"img_{i}.png", seed=i)
        csv_path = tmp_path / "out.csv"
        df = analyze(data_dir, csv_path)
        assert len(df) == 3

    def test_unsupported_files_skipped(self, tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _make_png(data_dir / "img.png")
        (data_dir / "notes.txt").write_text("not an image")
        (data_dir / "data.csv").write_text("a,b,c")
        df = analyze(data_dir, tmp_path / "out.csv")
        # Only the PNG should appear
        assert len(df) == 1

    def test_csv_written(self, tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _make_png(data_dir / "img.png")
        csv_path = tmp_path / "out.csv"
        analyze(data_dir, csv_path)
        assert csv_path.exists()

    def test_csv_columns(self, tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _make_png(data_dir / "img.png")
        csv_path = tmp_path / "out.csv"
        analyze(data_dir, csv_path)
        import pandas as pd
        loaded = pd.read_csv(csv_path)
        for col in ("path", "depth", "channels", "height", "width"):
            assert col in loaded.columns

    def test_broken_file_skipped_gracefully(self, tmp_path, capsys):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _make_png(data_dir / "good.png", seed=0)
        bad = data_dir / "bad.png"
        bad.write_bytes(b"not a valid image")
        df = analyze(data_dir, tmp_path / "out.csv")
        # good.png should still be recorded
        assert len(df) == 1

    def test_path_column_contains_file_path(self, tmp_path):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        p = _make_png(data_dir / "myimg.png")
        df = analyze(data_dir, tmp_path / "out.csv")
        assert str(p) in list(df["path"])
