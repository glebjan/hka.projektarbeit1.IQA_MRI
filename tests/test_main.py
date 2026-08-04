"""Tests for src/main.py — evaluate() function and CLI entry point."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main as main_module
from evaluation_result import EvaluationResult
from metrics import registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 96


def _make_png(path: Path, seed: int = 0) -> Path:
    arr = (np.random.default_rng(seed).random((IMG_SIZE, IMG_SIZE)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return path


def _restrict_to_fast(reg):
    """Remove all but psnr+ssim to avoid slow/network metrics during main tests."""
    keep = {"psnr", "ssim"}
    for name in [s.name for s in reg.specs]:
        if name not in keep:
            reg._specs.pop(name, None)
            reg._cache.pop(name, None)


# ---------------------------------------------------------------------------
# evaluate() — single-file input
# ---------------------------------------------------------------------------

class TestEvaluateSingleFile:
    def test_returns_evaluation_result(self, tmp_path, isolated_registry):
        _restrict_to_fast(isolated_registry)
        inp = _make_png(tmp_path / "inp.png")
        res = main_module.evaluate(inp)
        assert isinstance(res, EvaluationResult)

    def test_full_reference_single_file(self, tmp_path, isolated_registry):
        _restrict_to_fast(isolated_registry)
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)
        res = main_module.evaluate(inp, tgt)
        df = res.to_frame()
        assert df["psnr"].notna().any()

    def test_no_reference_psnr_none(self, tmp_path, isolated_registry):
        _restrict_to_fast(isolated_registry)
        inp = _make_png(tmp_path / "inp.png")
        res = main_module.evaluate(inp)
        df = res.to_frame()
        assert df["psnr"].isna().all()

    def test_target_is_dir_warns_and_ignores(self, tmp_path, isolated_registry, capsys):
        _restrict_to_fast(isolated_registry)
        inp = _make_png(tmp_path / "inp.png")
        tgt_dir = tmp_path / "targets"
        tgt_dir.mkdir()
        _make_png(tgt_dir / "tgt.png")
        res = main_module.evaluate(inp, tgt_dir)
        out = capsys.readouterr().out
        assert "WARNING" in out
        # Result should still exist (NR mode, target ignored)
        assert isinstance(res, EvaluationResult)


# ---------------------------------------------------------------------------
# evaluate() — directory input
# ---------------------------------------------------------------------------

class TestEvaluateDirectory:
    def test_multiple_images_evaluated(self, tmp_path, isolated_registry):
        _restrict_to_fast(isolated_registry)
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        for i in range(3):
            _make_png(inp_dir / f"case_{i:03d}.png", seed=i)
        res = main_module.evaluate(inp_dir)
        df = res.to_frame()
        assert len(df) == 3

    def test_directory_matching(self, tmp_path, isolated_registry):
        _restrict_to_fast(isolated_registry)
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        tgt_dir = tmp_path / "tgt"; tgt_dir.mkdir()
        for i in range(2):
            _make_png(inp_dir / f"case_{i:03d}.png", seed=i)
            _make_png(tgt_dir / f"case_{i:03d}.png", seed=i + 10)
        res = main_module.evaluate(inp_dir, tgt_dir)
        df = res.to_frame()
        assert df["psnr"].notna().any()

    def test_empty_directory_warns(self, tmp_path, isolated_registry, capsys):
        _restrict_to_fast(isolated_registry)
        empty = tmp_path / "empty"; empty.mkdir()
        main_module.evaluate(empty)
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_unmatched_targets_warns(self, tmp_path, isolated_registry, capsys):
        _restrict_to_fast(isolated_registry)
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        tgt_dir = tmp_path / "tgt"; tgt_dir.mkdir()
        _make_png(inp_dir / "caseA_inp.png", seed=0)
        # Target with entirely different name → no match
        _make_png(tgt_dir / "zzzz_ref.png", seed=1)
        main_module.evaluate(inp_dir, tgt_dir)
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_exception_in_one_image_continues(self, tmp_path, isolated_registry, capsys):
        """A broken image should not abort evaluation of remaining images."""
        _restrict_to_fast(isolated_registry)
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        _make_png(inp_dir / "good.png", seed=0)
        # Create a file with PNG extension but invalid content
        bad = inp_dir / "bad.png"
        bad.write_bytes(b"not an image at all")
        res = main_module.evaluate(inp_dir)
        # At least the good image should produce a record
        df = res.to_frame()
        assert len(df) >= 1

    def test_nonexistent_path_returns_empty_result(self, tmp_path, isolated_registry, capsys):
        _restrict_to_fast(isolated_registry)
        res = main_module.evaluate(tmp_path / "does_not_exist")
        assert isinstance(res, EvaluationResult)
        out = capsys.readouterr().out
        assert "No input" in out


# ---------------------------------------------------------------------------
# CLI main() — via monkeypatch
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_cli_writes_report(self, tmp_path, isolated_registry, monkeypatch, capsys):
        _restrict_to_fast(isolated_registry)
        inp = _make_png(tmp_path / "inp.png")
        report = tmp_path / "report.csv"

        # Redirect REPORT constant and sys.argv
        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp)])

        main_module.main()

        assert report.exists()

    def test_cli_with_target(self, tmp_path, isolated_registry, monkeypatch, capsys):
        _restrict_to_fast(isolated_registry)
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)
        report = tmp_path / "report.csv"

        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp), str(tgt)])

        main_module.main()

        assert report.exists()
