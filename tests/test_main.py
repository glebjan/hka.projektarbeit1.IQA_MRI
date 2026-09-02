"""Tests for src/main.py — evaluate() function and CLI entry point."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main as main_module
from evaluation_result import EvaluationResult
from iqa_evaluator import IQAEvaluator
from metrics import MetricRegistry, PSNR, SSIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 96


def _make_png(path: Path, seed: int = 0) -> Path:
    arr = (np.random.default_rng(seed).random((IMG_SIZE, IMG_SIZE)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return path


def _fast_registry() -> MetricRegistry:
    """Only psnr+ssim — avoids slow/network metrics during main tests."""
    return MetricRegistry(PSNR, SSIM)


# ---------------------------------------------------------------------------
# evaluate() — single-file input
# ---------------------------------------------------------------------------

class TestEvaluateSingleFile:
    def test_returns_evaluation_result(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png")
        res = main_module.evaluate(inp, registry=_fast_registry())
        assert isinstance(res, EvaluationResult)

    def test_full_reference_single_file(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)
        res = main_module.evaluate(inp, tgt, registry=_fast_registry())
        df = res.to_frame()
        assert df["psnr"].notna().any()

    def test_no_reference_psnr_none(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png")
        res = main_module.evaluate(inp, registry=_fast_registry())
        df = res.to_frame()
        assert df["psnr"].isna().all()

    def test_target_is_dir_warns_and_ignores(self, tmp_path, capsys):
        inp = _make_png(tmp_path / "inp.png")
        tgt_dir = tmp_path / "targets"
        tgt_dir.mkdir()
        _make_png(tgt_dir / "tgt.png")
        res = main_module.evaluate(inp, tgt_dir, registry=_fast_registry())
        out = capsys.readouterr().out
        assert "WARNING" in out
        # Result should still exist (NR mode, target ignored)
        assert isinstance(res, EvaluationResult)


# ---------------------------------------------------------------------------
# evaluate() — directory input
# ---------------------------------------------------------------------------

class TestEvaluateDirectory:
    def test_multiple_images_evaluated(self, tmp_path):
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        for i in range(3):
            _make_png(inp_dir / f"case_{i:03d}.png", seed=i)
        res = main_module.evaluate(inp_dir, registry=_fast_registry())
        df = res.to_frame()
        assert len(df) == 3

    def test_directory_matching(self, tmp_path):
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        tgt_dir = tmp_path / "tgt"; tgt_dir.mkdir()
        for i in range(2):
            _make_png(inp_dir / f"case_{i:03d}.png", seed=i)
            _make_png(tgt_dir / f"case_{i:03d}.png", seed=i + 10)
        res = main_module.evaluate(inp_dir, tgt_dir, registry=_fast_registry())
        df = res.to_frame()
        assert df["psnr"].notna().any()

    def test_empty_directory_warns(self, tmp_path, capsys):
        empty = tmp_path / "empty"; empty.mkdir()
        main_module.evaluate(empty, registry=_fast_registry())
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_unmatched_targets_warns(self, tmp_path, capsys):
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        tgt_dir = tmp_path / "tgt"; tgt_dir.mkdir()
        _make_png(inp_dir / "caseA_inp.png", seed=0)
        # Target with entirely different name → no match
        _make_png(tgt_dir / "zzzz_ref.png", seed=1)
        main_module.evaluate(inp_dir, tgt_dir, registry=_fast_registry())
        out = capsys.readouterr().out
        assert "WARNING" in out

    def test_exception_in_one_image_continues(self, tmp_path, capsys):
        """A broken image should not abort evaluation of remaining images."""
        inp_dir = tmp_path / "inp"; inp_dir.mkdir()
        _make_png(inp_dir / "good.png", seed=0)
        # Create a file with PNG extension but invalid content
        bad = inp_dir / "bad.png"
        bad.write_bytes(b"not an image at all")
        res = main_module.evaluate(inp_dir, registry=_fast_registry())
        # At least the good image should produce a record
        df = res.to_frame()
        assert len(df) >= 1

    def test_nonexistent_path_returns_empty_result(self, tmp_path, capsys):
        res = main_module.evaluate(tmp_path / "does_not_exist", registry=_fast_registry())
        assert isinstance(res, EvaluationResult)
        out = capsys.readouterr().out
        assert "No input" in out


# ---------------------------------------------------------------------------
# CLI main() — via monkeypatch
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_cli_writes_report(self, tmp_path, monkeypatch, capsys):
        # main() builds its registry from main.BUILTIN_METRICS — patch that
        # bundle down to psnr+ssim to keep the CLI test fast.
        monkeypatch.setattr("main.BUILTIN_METRICS", (PSNR, SSIM))
        inp = _make_png(tmp_path / "inp.png")
        report = tmp_path / "report.csv"

        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp)])

        main_module.main()

        assert report.exists()

    def test_cli_with_target(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("main.BUILTIN_METRICS", (PSNR, SSIM))
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)
        report = tmp_path / "report.csv"

        monkeypatch.setattr("main.REPORT", report)
        monkeypatch.setattr(sys, "argv", ["main.py", str(inp), str(tgt)])

        main_module.main()

        assert report.exists()


# ---------------------------------------------------------------------------
# evaluate() — registry threading
# ---------------------------------------------------------------------------

class TestEvaluateRegistryThreading:
    def test_registry_is_required(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png")
        with pytest.raises(TypeError):
            main_module.evaluate(inp)

    def test_same_registry_instance_shared_across_images(self, tmp_path, monkeypatch):
        seen = []
        real_init = IQAEvaluator.__init__

        def spy_init(self, input_image, target_image, registry, source_model=None):
            seen.append(registry)
            real_init(self, input_image, target_image, registry, source_model)

        monkeypatch.setattr("iqa_evaluator.IQAEvaluator.__init__", spy_init)

        _make_png(tmp_path / "a.png", seed=0)
        _make_png(tmp_path / "b.png", seed=1)
        reg = _fast_registry()
        main_module.evaluate(tmp_path, registry=reg)

        assert len(seen) == 2, "expected one evaluator per image"
        assert all(r is reg for r in seen), "every image must share one registry instance"

    def test_two_runs_use_different_metrics(self, tmp_path):
        inp = _make_png(tmp_path / "inp.png", seed=0)
        tgt = _make_png(tmp_path / "tgt.png", seed=1)

        run_psnr = main_module.evaluate(inp, tgt, registry=MetricRegistry(PSNR))
        run_ssim = main_module.evaluate(inp, tgt, registry=MetricRegistry(SSIM))

        df_psnr = run_psnr.to_frame()
        df_ssim = run_ssim.to_frame()
        assert df_psnr["psnr"].notna().any()
        assert df_psnr["ssim"].isna().all()
        assert df_ssim["ssim"].notna().any()
        assert df_ssim["psnr"].isna().all()


# ---------------------------------------------------------------------------
# Scoring mode
# ---------------------------------------------------------------------------

import pytest

from main import evaluate, report_skipped_metrics
from metrics import MetricRegistry, MetricSpec, ModeSupport, ModeUnsupported, SkippedMetric


def _spec(name, *, volume, reason="only reads flat pictures"):
    def make(*_args):
        return lambda inp, tgt=None: [float(inp[i].mean()) for i in range(inp.shape[0])]

    return MetricSpec(
        name=name, direction="higher_is_better", reference=False, channels="gray",
        slice_mode=ModeSupport(make),
        volume_mode=ModeSupport(make) if volume else ModeUnsupported(reason),
        builtin=False,
    )


class TestSkipMessage:
    def test_names_every_skipped_metric(self, capsys):
        report_skipped_metrics(
            [SkippedMetric("lpips", "reads flat pictures"),
             SkippedMetric("niqe", "reads flat pictures")],
            "volume",
        )
        out = capsys.readouterr().out
        assert "lpips" in out and "niqe" in out

    def test_states_the_reason(self, capsys):
        report_skipped_metrics([SkippedMetric("lpips", "reads flat pictures")], "volume")
        assert "reads flat pictures" in capsys.readouterr().out

    def test_names_the_other_mode_as_the_way_out(self, capsys):
        report_skipped_metrics([SkippedMetric("lpips", "r")], "volume")
        assert 'mode="slice"' in capsys.readouterr().out

    def test_silent_when_nothing_is_skipped(self, capsys):
        report_skipped_metrics([], "volume")
        assert capsys.readouterr().out == ""


class TestEvaluateMode:
    def test_default_is_slice(self, nifti_volume):
        result = evaluate(nifti_volume, None, registry=MetricRegistry(_spec("m", volume=True)))
        assert len(result.to_frame()) == 6

    def test_volume_mode_yields_one_row(self, nifti_volume):
        result = evaluate(nifti_volume, None,
                          registry=MetricRegistry(_spec("m", volume=True)), mode="volume")
        df = result.to_frame()
        assert len(df) == 1
        assert df.iloc[0]["scoring"] == "volume"

    def test_skip_message_appears_once_per_run(self, tmp_path, nifti_volume, capsys):
        import shutil

        directory = tmp_path / "many"
        directory.mkdir()
        for i in range(3):
            shutil.copy(nifti_volume, directory / f"vol{i}.nii.gz")
        # reason deliberately avoids the substring "flat" so the count below
        # isolates how many times the metric is *reported*, not how many
        # times the word happens to occur in the explanation text.
        registry = MetricRegistry(_spec("ok", volume=True), _spec("flat", volume=False, reason="only reads pictures"))
        evaluate(directory, None, registry=registry, mode="volume")
        assert capsys.readouterr().out.count("flat") == 1

    def test_hard_error_when_no_metric_can_serve_the_mode(self, nifti_volume):
        registry = MetricRegistry(_spec("flat", volume=False))
        with pytest.raises(ValueError, match="none of"):
            evaluate(nifti_volume, None, registry=registry, mode="volume")

    def test_non_volumetric_files_are_skipped_not_fatal(self, tmp_path, nifti_volume, synthetic_png, capsys):
        import shutil

        directory = tmp_path / "mixed"
        directory.mkdir()
        shutil.copy(nifti_volume, directory / "vol.nii.gz")
        shutil.copy(synthetic_png, directory / "flat.png")
        registry = MetricRegistry(_spec("m", volume=True))
        result = evaluate(directory, None, registry=registry, mode="volume")
        assert len(result.to_frame()) == 1
        assert "flat.png" in capsys.readouterr().out
