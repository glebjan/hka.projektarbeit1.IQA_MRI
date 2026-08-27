"""Tests for src/records.py — ImageEvaluatorRecord."""
import pytest

from records import ImageEvaluatorRecord


# ---------------------------------------------------------------------------
# ImageEvaluatorRecord.to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_extra_merged_flat(self):
        rec = ImageEvaluatorRecord(image_id="x")
        rec.extra["my_custom"] = 3.14
        d = rec.to_dict()
        assert "my_custom" in d
        assert "extra" not in d
        assert d["my_custom"] == 3.14

    def test_builtin_fields_present(self):
        rec = ImageEvaluatorRecord(image_id="img_001", psnr=42.0, ssim=0.99)
        d = rec.to_dict()
        assert d["image_id"] == "img_001"
        assert d["psnr"] == 42.0
        assert d["ssim"] == 0.99

    def test_none_values_kept(self):
        rec = ImageEvaluatorRecord(image_id="y", psnr=None)
        d = rec.to_dict()
        assert d["psnr"] is None

    def test_multiple_extra_keys(self):
        rec = ImageEvaluatorRecord(image_id="z")
        rec.extra["a"] = 1.0
        rec.extra["b"] = 2.0
        d = rec.to_dict()
        assert d["a"] == 1.0 and d["b"] == 2.0


# ---------------------------------------------------------------------------
# records.py must not depend on metrics.py
# ---------------------------------------------------------------------------

class TestRecordsHasNoMetricsDependency:
    def test_best_slice_per_metric_is_gone(self):
        import records
        assert not hasattr(records, "best_slice_per_metric")
        assert not hasattr(records, "_record_metric_value")

    def test_mask_writer_module_is_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import mask_writer  # noqa: F401
