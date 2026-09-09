"""Tests for src/metrics.py — MetricRegistry, MetricSpec, PyIQAMetric, DEVICE."""
import torch
import pytest

from metrics import (
    DEVICE,
    MetricRegistry,
    MetricSpec,
    PyIQAMetric,
    register_metric,
    registry,
    _pyiqa_factory,
)


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

class TestDevice:
    def test_is_torch_device(self):
        assert isinstance(DEVICE, torch.device)

    def test_consistent_with_cuda_availability(self):
        if torch.cuda.is_available():
            assert DEVICE.type == "cuda"
        else:
            assert DEVICE.type == "cpu"


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_register_and_specs(self, fake_metric):
        reg = MetricRegistry()
        spec = MetricSpec("dummy", "higher_is_better", False, "gray", lambda: fake_metric, builtin=False)
        reg.register(spec)
        assert any(s.name == "dummy" for s in reg.specs)

    def test_get_metric_lazy_init(self, fake_metric):
        reg = MetricRegistry()
        calls = []
        def factory():
            calls.append(1)
            return fake_metric
        spec = MetricSpec("lazy", "higher_is_better", False, "gray", factory, builtin=False)
        reg.register(spec)
        assert len(calls) == 0
        reg.get_metric("lazy")
        assert len(calls) == 1
        # Second call returns cached object — factory not called again
        reg.get_metric("lazy")
        assert len(calls) == 1

    def test_get_metric_same_object_on_second_call(self, fake_metric):
        reg = MetricRegistry()
        spec = MetricSpec("id_test", "higher_is_better", False, "gray", lambda: fake_metric, builtin=False)
        reg.register(spec)
        m1 = reg.get_metric("id_test")
        m2 = reg.get_metric("id_test")
        assert m1 is m2

    def test_re_register_clears_cache(self, fake_metric):
        reg = MetricRegistry()
        spec1 = MetricSpec("dup", "higher_is_better", False, "gray", lambda: fake_metric, builtin=False)
        reg.register(spec1)
        _ = reg.get_metric("dup")
        assert "dup" in reg._cache
        # Re-register should evict cache
        spec2 = MetricSpec("dup", "lower_is_better", False, "gray", lambda: fake_metric, builtin=False)
        reg.register(spec2)
        assert "dup" not in reg._cache

    def test_direction_property(self, fake_metric):
        reg = MetricRegistry()
        reg.register(MetricSpec("a", "higher_is_better", False, "gray", lambda: fake_metric))
        reg.register(MetricSpec("b", "lower_is_better",  False, "gray", lambda: fake_metric))
        d = reg.direction
        assert d["a"] == "higher_is_better"
        assert d["b"] == "lower_is_better"

    def test_unknown_metric_raises(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError):
            reg.get_metric("does_not_exist")


# ---------------------------------------------------------------------------
# register_metric (public API)
# ---------------------------------------------------------------------------

class TestRegisterMetric:
    def test_custom_metric_accessible_via_registry(self, fake_metric, isolated_registry):
        register_metric("custom_test", fake_metric, direction="higher_is_better", reference=False)
        spec = next((s for s in isolated_registry.specs if s.name == "custom_test"), None)
        assert spec is not None
        assert spec.builtin is False
        assert spec.direction == "higher_is_better"

    def test_custom_metric_returns_scores(self, fake_metric, isolated_registry):
        register_metric("custom_scores", fake_metric, direction="higher_is_better", reference=False)
        m = isolated_registry.get_metric("custom_scores")
        inp = torch.rand(3, 1, 32, 32)
        out = m(inp)
        assert len(out) == 3
        assert all(isinstance(v, float) for v in out)


# ---------------------------------------------------------------------------
# _pyiqa_factory
# ---------------------------------------------------------------------------

class TestPyiqaFactory:
    def test_returns_callable(self):
        factory = _pyiqa_factory("psnr")
        m = factory()
        assert callable(m)

    def test_kwargs_passed_through(self):
        factory = _pyiqa_factory("psnr", test_y_channel=True)
        m = factory()
        assert isinstance(m, PyIQAMetric)
        assert m._kwargs.get("test_y_channel") is True


# ---------------------------------------------------------------------------
# PyIQAMetric (real, with locally cached weights)
# ---------------------------------------------------------------------------

class TestPyIQAMetricPSNR:
    """PSNR: full-reference, gray channel, higher-is-better."""

    def test_lazy_init(self):
        m = PyIQAMetric("psnr")
        assert m._impl is None
        inp = torch.rand(1, 1, 64, 64)
        m(inp, inp)
        assert m._impl is not None

    def test_identical_images_high_score(self):
        m = PyIQAMetric("psnr")
        inp = torch.rand(1, 1, 64, 64)
        scores = m(inp, inp)
        assert len(scores) == 1
        # Identical images → perfect score (very high PSNR, often 100 or inf from pyiqa)
        assert scores[0] > 40.0

    def test_noisy_image_lower_score(self):
        m = PyIQAMetric("psnr")
        inp = torch.rand(2, 1, 64, 64)
        noisy = (inp + 0.2 * torch.rand_like(inp)).clamp(0, 1)
        score_perfect = m(inp, inp)[0]
        score_noisy = m(noisy, inp)[0]
        assert score_perfect > score_noisy

    def test_batch_output_length(self):
        m = PyIQAMetric("psnr")
        inp = torch.rand(4, 1, 64, 64)
        scores = m(inp, inp)
        assert len(scores) == 4

    def test_returns_list_of_floats(self):
        m = PyIQAMetric("psnr")
        inp = torch.rand(2, 1, 64, 64)
        scores = m(inp, inp)
        assert all(isinstance(s, float) for s in scores)


class TestPyIQAMetricLPIPS:
    """LPIPS: full-reference, RGB channel, lower-is-better (real weights)."""

    def test_identical_images_near_zero(self):
        m = PyIQAMetric("lpips")
        inp = torch.rand(1, 3, 64, 64)
        scores = m(inp, inp)
        assert scores[0] < 0.01

    def test_different_images_higher_score(self):
        m = PyIQAMetric("lpips")
        a = torch.zeros(1, 3, 64, 64)
        b = torch.ones(1, 3, 64, 64)
        scores = m(a, b)
        assert scores[0] > 0.0

    def test_batch_output_length(self):
        m = PyIQAMetric("lpips")
        inp = torch.rand(3, 3, 64, 64)
        scores = m(inp, inp)
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# FSIM / GMSD / VSI — classical 2D full-reference metrics on greyscale input
# ---------------------------------------------------------------------------

class TestStructuralFRMetrics:
    """These three are 2D-only and were designed for RGB photographs.

    ImageLoader feeds them replicated greyscale via rgb_tensor, so the tests
    below pin the greyscale behaviour the framework actually relies on.
    """

    @staticmethod
    def _pair(n: int = 2, size: int = 96):
        """(distorted, reference) as 3-channel replicated greyscale in [0, 1]."""
        ref = torch.rand(n, 1, size, size)
        distorted = (ref + 0.15 * torch.rand_like(ref)).clamp(0, 1)
        return distorted.expand(-1, 3, -1, -1), ref.expand(-1, 3, -1, -1)

    @pytest.mark.parametrize("name", ["fsim", "gmsd", "vsi"])
    def test_batch_output_length(self, name):
        m = PyIQAMetric(name)
        distorted, ref = self._pair(n=3)
        assert len(m(distorted, ref)) == 3

    @pytest.mark.parametrize("name", ["fsim", "gmsd", "vsi"])
    def test_returns_floats(self, name):
        m = PyIQAMetric(name)
        distorted, ref = self._pair()
        assert all(isinstance(s, float) for s in m(distorted, ref))

    @pytest.mark.parametrize("name,perfect", [("fsim", 1.0), ("gmsd", 0.0), ("vsi", 1.0)])
    def test_identical_images_hit_perfect_score(self, name, perfect):
        m = PyIQAMetric(name)
        _, ref = self._pair(n=1)
        assert m(ref, ref)[0] == pytest.approx(perfect, abs=1e-4)

    @pytest.mark.parametrize("name", ["fsim", "vsi"])
    def test_higher_is_better_degrades_with_noise(self, name):
        m = PyIQAMetric(name)
        distorted, ref = self._pair(n=1)
        assert m(distorted, ref)[0] < m(ref, ref)[0]

    def test_gmsd_lower_is_better_grows_with_noise(self):
        m = PyIQAMetric("gmsd")
        distorted, ref = self._pair(n=1)
        assert m(distorted, ref)[0] > m(ref, ref)[0]

    @pytest.mark.parametrize("name", ["fsim", "gmsd"])
    def test_rejects_single_channel_input(self, name):
        """channels="rgb" in the MetricSpec is mandatory, not a preference.

        fsim asserts 3 channels for its chromatic term, gmsd's to_y_channel
        asserts (N, 3, H, W).
        """
        m = PyIQAMetric(name)
        gray = torch.rand(1, 1, 96, 96)
        with pytest.raises(Exception):
            m(gray, gray)

    def test_vsi_tolerates_grey_but_warns(self):
        """vsi replicates 1 channel itself — same score, one warning per call.

        We still register it as "rgb" so ImageLoader.rgb_tensor does the
        replication once instead of pyiqa warning on every batch.
        """
        m = PyIQAMetric("vsi")
        gray = torch.rand(2, 1, 96, 96)
        with pytest.warns(UserWarning):
            from_gray = m(gray, gray)
        assert from_gray == pytest.approx(m(gray.expand(-1, 3, -1, -1),
                                            gray.expand(-1, 3, -1, -1)), abs=1e-9)

    def test_fsim_chromatic_term_is_neutral_on_greyscale(self):
        """FSIMc == FSIM here: replicated grey has I = Q = 0, so S_C == 1."""
        distorted, ref = self._pair(n=2)
        chromatic = PyIQAMetric("fsim")                    # pyiqa default: chromatic=True
        achromatic = PyIQAMetric("fsim", chromatic=False)
        assert chromatic(distorted, ref) == pytest.approx(achromatic(distorted, ref), abs=1e-6)


# ---------------------------------------------------------------------------
# Built-in registry: all 13 expected metrics registered
# ---------------------------------------------------------------------------

class TestBuiltinRegistry:
    EXPECTED = {
        "psnr":               ("higher_is_better", True,  "gray"),
        "ssim":               ("higher_is_better", True,  "gray"),
        "fsim":               ("higher_is_better", True,  "rgb"),
        "gmsd":               ("lower_is_better",  True,  "rgb"),
        "vsi":                ("higher_is_better", True,  "rgb"),
        "lpips":              ("lower_is_better",  True,  "rgb"),
        "dists":              ("lower_is_better",  True,  "rgb"),
        "radimagenet_lpips":  ("lower_is_better",  True,  "rgb"),
        "clipiqa":            ("higher_is_better", False, "rgb"),
        "clip_iqa_lung":      ("higher_is_better", False, "rgb"),
        "clip_iqa_brain":     ("higher_is_better", False, "rgb"),
        "brisque":            ("lower_is_better",  False, "rgb"),
        "niqe":               ("lower_is_better",  False, "rgb"),
    }

    def test_all_names_registered(self):
        names = {s.name for s in registry.specs}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing from registry"

    @pytest.mark.parametrize("name,attrs", EXPECTED.items())
    def test_spec_attributes(self, name, attrs):
        direction, reference, channels = attrs
        spec = next(s for s in registry.specs if s.name == name)
        assert spec.direction == direction,  f"{name}: direction mismatch"
        assert spec.reference == reference,  f"{name}: reference mismatch"
        assert spec.channels  == channels,   f"{name}: channels mismatch"
        assert spec.builtin   is True,       f"{name}: should be builtin"
