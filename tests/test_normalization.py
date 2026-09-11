"""Tests for src/normalization.py — strategies and the scale() mapping."""
import numpy as np
import pytest
import torch

from normalization import (
    NORMALIZER_NAMES, FixedRange, IntensityRange, MinMax, Normalizer,
    Percentile, Raw, normalizer_from_name, scale,
)


class TestScale:
    def test_maps_range_onto_exactly_zero_and_one(self):
        raw = np.array([[[0.0, 1e-8]]], dtype=np.float32)
        t = scale(raw, IntensityRange(0.0, 1e-8))
        assert float(t.min()) == 0.0
        assert float(t.max()) == 1.0

    def test_clips_values_outside_the_range(self):
        raw = np.array([[[-5.0, 5.0, 15.0]]], dtype=np.float32)
        t = scale(raw, IntensityRange(0.0, 10.0))
        assert t.tolist() == [[[0.0, 0.5, 1.0]]]

    def test_output_is_float32_without_channel_axis(self):
        raw = np.random.default_rng(0).integers(0, 4000, (3, 4, 5), dtype=np.uint16)
        t = scale(raw, IntensityRange(0.0, 4000.0))
        assert t.dtype == torch.float32
        assert t.shape == (3, 4, 5)

    def test_constant_image_keeps_its_clipped_value_and_warns(self, capsys):
        raw = np.full((2, 3, 3), 42.0, dtype=np.float32)
        t = scale(raw, IntensityRange(42.0, 42.0), label="flat.png")
        assert torch.all(t == 1.0)
        out = capsys.readouterr().out
        assert "flat.png" in out and "constant" in out

    def test_constant_zero_image_stays_zero(self):
        raw = np.zeros((1, 3, 3), dtype=np.float32)
        assert torch.all(scale(raw, IntensityRange(0.0, 0.0)) == 0.0)

    def test_none_range_preserves_dtype_and_values(self):
        raw = np.array([[[0, 1, 2, 3]]], dtype=np.int16)
        t = scale(raw, None)
        assert t.dtype == torch.int16
        assert t.tolist() == [[[0, 1, 2, 3]]]

    def test_none_range_accepts_non_contiguous_input(self):
        raw = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
        t = scale(raw, None)
        assert t.shape == (4, 2, 3)


class TestMinMax:
    def test_uses_the_extremes(self):
        raw = np.array([[[100.0, 110.0, 4000.0]]])
        assert MinMax().range_of(raw) == IntensityRange(100.0, 4000.0)

    def test_name(self):
        assert MinMax().name == "minmax"


class TestPercentile:
    def test_ignores_a_single_spike(self):
        raw = np.full((1, 100, 100), 120.0)
        raw[0, :50] = 100.0
        raw[0, 0, 0] = 4000.0
        rng = Percentile().range_of(raw)
        assert rng.lo == pytest.approx(100.0)
        assert rng.hi == pytest.approx(120.0)

    def test_scaled_bulk_spans_the_unit_interval_and_spike_clips(self):
        raw = np.linspace(100.0, 140.0, 10_000, dtype=np.float32).reshape(1, 100, 100)
        raw[0, 0, 0] = 4000.0
        t = scale(raw, Percentile().range_of(raw))
        assert float(t[0, 0, 0]) == 1.0
        assert float(t[0, 50, 50]) == pytest.approx(0.5, abs=0.02)

    def test_name_encodes_the_percentiles(self):
        assert Percentile().name == "percentile_0.5_99.5"
        assert Percentile(1, 99).name == "percentile_1_99"

    @pytest.mark.parametrize("lower,upper", [(99.5, 0.5), (50, 50), (-1, 99), (0, 101)])
    def test_invalid_bounds_raise(self, lower, upper):
        with pytest.raises(ValueError):
            Percentile(lower, upper)


class TestFixedRange:
    def test_ignores_the_data(self):
        rng = IntensityRange(0.0, 800.0)
        assert FixedRange(rng, "minmax").range_of(np.array([[[5.0]]])) is rng

    def test_carries_the_name_it_was_given(self):
        assert FixedRange(IntensityRange(0.0, 1.0), "minmax").name == "minmax"

    def test_none_range_means_raw(self):
        assert FixedRange(None, "raw").range_of(np.array([[[5.0]]])) is None

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError):
            FixedRange(IntensityRange(10.0, 0.0), "x")


class TestRaw:
    def test_returns_no_range(self):
        assert Raw().range_of(np.array([[[1, 2]]])) is None

    def test_name(self):
        assert Raw().name == "raw"


class TestProtocol:
    @pytest.mark.parametrize("strategy", [MinMax(), Percentile(), FixedRange(None, "raw"), Raw()])
    def test_every_strategy_satisfies_normalizer(self, strategy):
        assert isinstance(strategy, Normalizer)


class TestFromName:
    def test_known_names(self):
        assert isinstance(normalizer_from_name("minmax"), MinMax)
        assert isinstance(normalizer_from_name("percentile"), Percentile)
        assert isinstance(normalizer_from_name("raw"), Raw)

    def test_names_match_the_registry(self):
        assert set(NORMALIZER_NAMES) == {"minmax", "percentile", "raw"}

    def test_unknown_name_lists_the_choices(self):
        with pytest.raises(ValueError, match="minmax"):
            normalizer_from_name("zscore")
