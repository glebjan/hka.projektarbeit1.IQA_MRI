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


class TestStructuralFRFields:
    """fsim / gmsd / vsi are builtin metrics, so they need dedicated fields.

    Builtin scores are written with setattr(record, spec.name, value); on a
    dataclass without the field that write is silently dropped by asdict().
    """

    NAMES = ["fsim", "gmsd", "vsi"]

    @pytest.mark.parametrize("name", NAMES)
    def test_field_defaults_to_none(self, name):
        record = ImageEvaluatorRecord(image_id="img")
        assert getattr(record, name) is None

    @pytest.mark.parametrize("name", NAMES)
    def test_field_is_declared_not_ad_hoc(self, name):
        assert name in ImageEvaluatorRecord.__annotations__

    def test_values_survive_to_dict(self):
        record = ImageEvaluatorRecord(image_id="img", fsim=0.9, gmsd=0.05, vsi=0.97)
        d = record.to_dict()
        assert d["fsim"] == 0.9
        assert d["gmsd"] == 0.05
        assert d["vsi"] == 0.97

    def test_not_stored_in_extra(self):
        record = ImageEvaluatorRecord(image_id="img", fsim=0.9)
        assert "fsim" not in record.extra


class TestStructuralNRFields:
    """musiq / maniqa / paq2piq / piqe / ilniqe are builtin metrics too.

    Same silent-drop hazard as the full-reference block: setattr on a
    dataclass without the field never reaches asdict().
    """

    NAMES = ["musiq", "maniqa", "paq2piq", "piqe", "ilniqe"]

    @pytest.mark.parametrize("name", NAMES)
    def test_field_defaults_to_none(self, name):
        record = ImageEvaluatorRecord(image_id="img")
        assert getattr(record, name) is None

    @pytest.mark.parametrize("name", NAMES)
    def test_field_is_declared_not_ad_hoc(self, name):
        assert name in ImageEvaluatorRecord.__annotations__

    def test_values_survive_to_dict(self):
        record = ImageEvaluatorRecord(image_id="img", musiq=52.1, maniqa=0.41,
                                      paq2piq=71.3, piqe=38.0, ilniqe=24.5)
        d = record.to_dict()
        assert d["musiq"]   == 52.1
        assert d["maniqa"]  == 0.41
        assert d["paq2piq"] == 71.3
        assert d["piqe"]    == 38.0
        assert d["ilniqe"]  == 24.5

    def test_not_stored_in_extra(self):
        record = ImageEvaluatorRecord(image_id="img", musiq=52.1)
        assert "musiq" not in record.extra


class TestDreamSimField:
    """dreamsim's default spec is builtin=True, so it needs a dedicated field.

    Builtin scores are written with setattr(record, spec.name, value); on a
    dataclass without the field, that write is silently dropped by asdict().
    """

    def test_field_defaults_to_none(self):
        assert ImageEvaluatorRecord(image_id="img").dreamsim is None

    def test_field_is_declared_not_ad_hoc(self):
        assert "dreamsim" in ImageEvaluatorRecord.__annotations__

    def test_value_survives_to_dict(self):
        record = ImageEvaluatorRecord(image_id="img", dreamsim=0.12)
        assert record.to_dict()["dreamsim"] == 0.12

    def test_not_stored_in_extra(self):
        record = ImageEvaluatorRecord(image_id="img", dreamsim=0.12)
        assert "dreamsim" not in record.extra
