"""Tests for src/iqa_evaluator.py — IQAEvaluator, batching, slice skipping."""
import numpy as np
import torch
import pytest

from image_loader import ImageLoader, LoadedImage
from iqa_evaluator import IQAEvaluator, BATCH_SIZE
from metrics import (MetricRegistry, MetricSpec, ModeSupport, PSNR, SSIM,
                     FSIM, GMSD, VSI, MUSIQ, MANIQA, PAQ2PIQ, PIQE, ILNIQE)
from normalization import MinMax
from records import ImageEvaluatorRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePath:
    """Minimal path-like object for ImageLoader construction bypass."""
    def __init__(self, name="fake.png"):
        self.name = name
    def __str__(self):
        return self.name


def _make_loader(n_slices: int = 3, h: int = 64, w: int = 64) -> ImageLoader:
    """Build an ImageLoader without touching the filesystem."""
    loader = object.__new__(ImageLoader)
    loader.path = _FakePath()
    loader.suffix = ".png"
    loader.normalizer = MinMax()
    loader._loaded = LoadedImage(np.random.default_rng(0).random((n_slices, h, w)).astype("float32"))
    loader._tensor = None
    loader._intensity_range = None
    loader._intensity_range_known = False
    return loader


# ---------------------------------------------------------------------------
# __init__ — shape mismatch
# ---------------------------------------------------------------------------

class TestIQAEvaluatorInit:
    def test_shape_mismatch_raises(self):
        inp = _make_loader(2, 64, 64)
        tgt = _make_loader(2, 32, 32)   # Different spatial size
        with pytest.raises(ValueError, match="shape mismatch"):
            IQAEvaluator(inp, tgt, MetricRegistry())

    def test_matching_shapes_ok(self):
        inp = _make_loader(2, 64, 64)
        tgt = _make_loader(2, 64, 64)
        ev = IQAEvaluator(inp, tgt, MetricRegistry())
        assert ev.input is inp and ev.target is tgt

    def test_no_target_ok(self):
        inp = _make_loader(3)
        ev = IQAEvaluator(inp, None, MetricRegistry())
        assert ev.target is None

    def test_registry_is_stored(self):
        inp = _make_loader(2)
        reg = MetricRegistry(PSNR)
        ev = IQAEvaluator(inp, None, reg)
        assert ev.registry is reg


# ---------------------------------------------------------------------------
# _pick_tensor_batch
# ---------------------------------------------------------------------------

class TestPickTensorBatch:
    def test_gray_channel(self):
        loader = _make_loader(5)
        ev = IQAEvaluator(loader, None, MetricRegistry())
        batch = ev._pick_tensor_batch(loader, "gray", [0, 2, 4])
        assert batch.shape == (3, 1, 64, 64)
        assert torch.equal(batch[0], loader.tensor[0])
        assert torch.equal(batch[2], loader.tensor[4])

    def test_rgb_channel(self):
        loader = _make_loader(3)
        ev = IQAEvaluator(loader, None, MetricRegistry())
        batch = ev._pick_tensor_batch(loader, "rgb", [1, 2])
        # rgb_tensor expands to 3 channels
        assert batch.shape == (2, 3, 64, 64)


# ---------------------------------------------------------------------------
# _format_slice_id
# ---------------------------------------------------------------------------

class TestFormatSliceId:
    def test_without_model(self):
        loader = _make_loader()
        loader.path = _FakePath("patient_001.png")
        ev = IQAEvaluator(loader, None, MetricRegistry())
        assert ev._format_slice_id(0)  == "patient_001_s000"
        assert ev._format_slice_id(42) == "patient_001_s042"

    def test_with_model_prefix(self):
        loader = _make_loader()
        loader.path = _FakePath("patient_001.png")
        ev = IQAEvaluator(loader, None, MetricRegistry(), source_model="UNet")
        assert ev._format_slice_id(3) == "UNet/patient_001_s003"


# ---------------------------------------------------------------------------
# _compute_batch — error handling
# ---------------------------------------------------------------------------

class TestComputeBatch:
    def test_exception_in_metric_returns_nones(self, capsys):
        def bad_metric(inp, tgt=None):
            raise RuntimeError("simulated failure")

        spec = MetricSpec("bad", "higher_is_better", False, "gray",
                          ModeSupport(lambda: bad_metric), builtin=False)
        reg = MetricRegistry()
        reg.register(spec)

        loader = _make_loader(3)
        ev = IQAEvaluator(loader, None, reg)
        result = ev._compute_batch(spec, [0, 1, 2])
        assert result == [None, None, None]

    def test_full_reference_batch_uses_target(self):
        received = {}

        def capturing_metric(inp, tgt=None):
            received["tgt"] = tgt
            return [0.5] * inp.shape[0]

        spec = MetricSpec("capture", "higher_is_better", True, "gray",
                          ModeSupport(lambda: capturing_metric), builtin=False)
        reg = MetricRegistry()
        reg.register(spec)

        inp = _make_loader(2)
        tgt = _make_loader(2)
        ev = IQAEvaluator(inp, tgt, reg)
        ev._compute_batch(spec, [0, 1])
        assert received["tgt"] is not None

    def test_no_reference_batch_no_target_passed(self):
        received = {}

        def capturing_metric(inp, tgt=None):
            received["tgt"] = tgt
            return [0.5] * inp.shape[0]

        spec = MetricSpec("capture_nr", "higher_is_better", False, "gray",
                          ModeSupport(lambda: capturing_metric), builtin=False)
        reg = MetricRegistry()
        reg.register(spec)

        loader = _make_loader(2)
        ev = IQAEvaluator(loader, None, reg)
        ev._compute_batch(spec, [0, 1])
        assert received["tgt"] is None


# ---------------------------------------------------------------------------
# run_evaluation — full pipeline
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def _psnr_ssim_registry(self) -> MetricRegistry:
        """Registry with psnr + ssim for fast real-metric tests."""
        return MetricRegistry(PSNR, SSIM)

    def test_record_count_equals_slices(self, synthetic_png):
        reg = self._psnr_ssim_registry()
        loader = ImageLoader(synthetic_png)
        ev = IQAEvaluator(loader, None, reg)
        records = ev.run_evaluation()
        assert len(records) == loader.tensor.shape[0]

    def test_fr_metrics_none_when_no_target(self, synthetic_png):
        reg = self._psnr_ssim_registry()
        loader = ImageLoader(synthetic_png)
        ev = IQAEvaluator(loader, None, reg)
        records = ev.run_evaluation()
        for rec in records:
            assert rec.psnr is None
            assert rec.ssim is None

    def test_fr_metrics_filled_with_target(self, input_target_pair):
        reg = self._psnr_ssim_registry()
        inp_path, tgt_path = input_target_pair
        inp = ImageLoader(inp_path)
        tgt = ImageLoader(tgt_path)
        ev = IQAEvaluator(inp, tgt, reg)
        records = ev.run_evaluation()
        active = [r for r in records if not r.is_empty]
        assert all(r.psnr is not None for r in active)
        assert all(r.ssim is not None for r in active)

    def test_empty_slices_have_no_metric_values(self):
        reg = self._psnr_ssim_registry()
        loader = _make_loader(3)
        # Force first slice to be empty (constant zero tensor)
        loader._loaded.raw[0] = 0.0
        # Verify our setup: the empty_slice_mask should flag slice 0
        assert loader.empty_slice_mask[0].item()

        ev = IQAEvaluator(loader, None, reg)
        records = ev.run_evaluation()
        assert records[0].is_empty
        assert records[0].psnr is None

    def test_custom_metric_stored_in_extra(self, fake_metric, synthetic_png):
        # Only NR custom metric so we don't need a target
        reg = MetricRegistry()
        reg.register_metric("custom_eval", fake_metric,
                            direction="higher_is_better", reference=False)

        loader = ImageLoader(synthetic_png)
        ev = IQAEvaluator(loader, None, reg)
        records = ev.run_evaluation()
        active = [r for r in records if not r.is_empty]
        assert all("custom_eval" in r.extra for r in active)

    def test_batching_covers_all_slices(self, tmp_path):
        """With n_slices > BATCH_SIZE every active slice should be evaluated."""
        reg = self._psnr_ssim_registry()
        n = BATCH_SIZE + 5
        # Create a multi-slice PNG by stacking frames into a NIfTI file
        import numpy as np, nibabel as nib
        arr = np.random.default_rng(42).random((n, 64, 64))
        # Shape for nibabel: (X, Y, Z) → (64, 64, n)
        img = nib.Nifti1Image(arr.transpose(1, 2, 0), np.eye(4))
        p = tmp_path / "vol.nii"
        nib.save(img, str(p))

        from image_loader import _load_nifti
        loader = object.__new__(ImageLoader)
        loader.path = p
        loader.suffix = ".nii"
        loader.normalizer = MinMax()
        loader._loaded = _load_nifti(p)
        loader._tensor = None
        loader._intensity_range = None
        loader._intensity_range_known = False
        assert loader._loaded.raw.shape[0] == n

        ev = IQAEvaluator(loader, loader, reg)  # self-comparison
        records = ev.run_evaluation()
        active = [r for r in records if not r.is_empty]
        # All active slices must have psnr filled
        assert all(r.psnr is not None for r in active)


# ---------------------------------------------------------------------------
# Per-run metric selection — the point of this refactor
# ---------------------------------------------------------------------------

class TestPerRunMetricSelection:
    def test_two_evaluators_compute_different_metrics(self, fake_metric):
        inp = _make_loader(2)

        reg_a = MetricRegistry()
        reg_a.register_metric("metric_a", fake_metric,
                              direction="higher_is_better", reference=False)
        reg_b = MetricRegistry()
        reg_b.register_metric("metric_b", fake_metric,
                              direction="higher_is_better", reference=False)

        recs_a = IQAEvaluator(inp, None, reg_a).run_evaluation()
        recs_b = IQAEvaluator(inp, None, reg_b).run_evaluation()

        assert "metric_a" in recs_a[0].extra
        assert "metric_b" not in recs_a[0].extra
        assert "metric_b" in recs_b[0].extra
        assert "metric_a" not in recs_b[0].extra


class TestStructuralFRMetricsEndToEnd:
    """A registered fsim/gmsd/vsi run must land in the record's own fields."""

    def test_scores_land_in_dedicated_fields(self):
        inp = _make_loader(2, 96, 96)
        tgt = _make_loader(2, 96, 96)
        registry = MetricRegistry(FSIM, GMSD, VSI)
        records = IQAEvaluator(inp, tgt, registry).run_evaluation()

        assert len(records) == 2
        for record in records:
            assert isinstance(record.fsim, float)
            assert isinstance(record.gmsd, float)
            assert isinstance(record.vsi, float)
            assert record.extra == {}


class TestStructuralNRMetricsEndToEnd:
    """A registered no-reference run needs no target and fills its own fields."""

    def test_scores_land_in_dedicated_fields(self):
        inp = _make_loader(2, 96, 96)
        registry = MetricRegistry(MUSIQ, MANIQA, PAQ2PIQ, PIQE, ILNIQE)
        records = IQAEvaluator(inp, None, registry).run_evaluation()

        assert len(records) == 2
        for record in records:
            assert isinstance(record.musiq, float)
            assert isinstance(record.maniqa, float)
            assert isinstance(record.paq2piq, float)
            assert isinstance(record.piqe, float)
            assert isinstance(record.ilniqe, float)
            assert record.extra == {}


class TestScaleFieldsOnRecords:
    def _registry(self):
        return MetricRegistry(PSNR)

    def test_minmax_run_reports_its_own_range(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        rec = IQAEvaluator(loader, None, self._registry()).run_evaluation()[0]
        assert rec.normalization == "minmax"
        assert (rec.scale_lo, rec.scale_hi) == (loader.raw_range.lo, loader.raw_range.hi)
        assert (rec.input_min, rec.input_max) == (loader.raw_range.lo, loader.raw_range.hi)

    def test_full_reference_run_reports_the_targets_scale(self, tmp_path):
        import nibabel as nib
        from image_loader import load_pair
        target = np.random.default_rng(0).random((16, 16, 3)) * 800
        for name, arr in (("inp.nii", 2 * target + 500), ("tgt.nii", target)):
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(tmp_path / name))
        inp, tgt = load_pair(tmp_path / "inp.nii", tmp_path / "tgt.nii")
        rec = IQAEvaluator(inp, tgt, self._registry()).run_evaluation()[0]
        assert (rec.scale_lo, rec.scale_hi) == (tgt.raw_range.lo, tgt.raw_range.hi)
        assert rec.input_max > rec.scale_hi          # the overshoot is visible
        assert rec.input_min > rec.scale_lo

    def test_raw_run_has_no_scale(self, tmp_path):
        import nibabel as nib
        from normalization import Raw
        labels = np.zeros((8, 8, 2), dtype=np.int16); labels[:4] = 1
        p = tmp_path / "m.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        rec = IQAEvaluator(ImageLoader(p, Raw()), None, MetricRegistry()).run_evaluation()[0]
        assert rec.normalization == "raw"
        assert rec.scale_lo is None and rec.scale_hi is None
        assert (rec.input_min, rec.input_max) == (0.0, 1.0)
