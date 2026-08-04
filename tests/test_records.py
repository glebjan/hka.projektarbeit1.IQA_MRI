"""Tests for src/records.py — ImageEvaluatorRecord, best_slice_per_metric."""
import pytest
import torch

from records import ImageEvaluatorRecord, _record_metric_value, best_slice_per_metric
from metrics import MetricSpec, MetricRegistry


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
# _record_metric_value
# ---------------------------------------------------------------------------

class TestRecordMetricValue:
    def test_reads_builtin_attribute(self):
        rec = ImageEvaluatorRecord(image_id="a", psnr=35.0)
        assert _record_metric_value(rec, "psnr") == 35.0

    def test_reads_from_extra(self):
        rec = ImageEvaluatorRecord(image_id="b")
        rec.extra["custom"] = 7.5
        assert _record_metric_value(rec, "custom") == 7.5

    def test_extra_takes_priority_for_overlapping_name(self):
        rec = ImageEvaluatorRecord(image_id="c", psnr=10.0)
        rec.extra["psnr"] = 99.0  # Unusual but possible
        assert _record_metric_value(rec, "psnr") == 99.0

    def test_missing_returns_none(self):
        rec = ImageEvaluatorRecord(image_id="d")
        assert _record_metric_value(rec, "nonexistent") is None


# ---------------------------------------------------------------------------
# best_slice_per_metric
# ---------------------------------------------------------------------------

def _make_records(values: dict[str, list]) -> list[ImageEvaluatorRecord]:
    """Create one record per slice index from {metric_name: [values]} dict."""
    n = len(next(iter(values.values())))
    records = [ImageEvaluatorRecord(image_id=f"s{i}", slice_index=i) for i in range(n)]
    for metric, vals in values.items():
        for rec, v in zip(records, vals):
            setattr(rec, metric, v)
    return records


class TestBestSlicePerMetric:
    def test_higher_is_better_selects_max(self, isolated_registry):
        records = _make_records({"psnr": [20.0, 35.0, 28.0]})
        best = best_slice_per_metric(records)
        assert best["psnr"] == 1  # index with value 35.0

    def test_lower_is_better_selects_min(self, isolated_registry):
        records = _make_records({"lpips": [0.8, 0.1, 0.5]})
        best = best_slice_per_metric(records)
        assert best["lpips"] == 1  # index with value 0.1

    def test_empty_slices_skipped(self, isolated_registry):
        records = _make_records({"psnr": [10.0, 40.0, 30.0]})
        records[1].is_empty = True  # best would be index 1 but it's empty
        best = best_slice_per_metric(records)
        assert best["psnr"] == 2  # next best

    def test_none_values_ignored(self, isolated_registry):
        records = _make_records({"psnr": [None, 35.0, None]})
        best = best_slice_per_metric(records)
        assert best["psnr"] == 1

    def test_all_none_metric_excluded(self, isolated_registry):
        records = _make_records({"psnr": [None, None]})
        best = best_slice_per_metric(records)
        assert "psnr" not in best

    def test_all_empty_excluded(self, isolated_registry):
        records = _make_records({"psnr": [30.0, 40.0]})
        for r in records:
            r.is_empty = True
        best = best_slice_per_metric(records)
        assert "psnr" not in best

    def test_custom_metric_via_extra(self, isolated_registry, fake_metric):
        from metrics import register_metric
        register_metric("my_nr", fake_metric, direction="higher_is_better", reference=False)
        records = [ImageEvaluatorRecord(image_id=f"s{i}", slice_index=i) for i in range(3)]
        records[0].extra["my_nr"] = 0.1
        records[1].extra["my_nr"] = 0.9
        records[2].extra["my_nr"] = 0.5
        best = best_slice_per_metric(records)
        assert best["my_nr"] == 1
