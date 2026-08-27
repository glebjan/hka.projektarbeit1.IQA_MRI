"""Tests for src/metrics.py — MetricRegistry, MetricSpec, PyIQAMetric, DEVICE."""
import torch
import pytest

from metrics import (
    DEVICE,
    MetricRegistry,
    MetricSpec,
    PyIQAMetric,
    _pyiqa_factory,
    BUILTIN_METRICS,
    SEGMENTATION_METRICS,
    PSNR,
    SSIM,
)

from segmentation_metrics.monai_metrics import DICE as _DICE  # sanity: same objects re-exported


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


class TestMetricRegistryInstances:
    def test_constructor_accepts_specs(self):
        reg = MetricRegistry(PSNR, SSIM)
        assert {s.name for s in reg.specs} == {"psnr", "ssim"}

    def test_constructor_empty_by_default(self):
        reg = MetricRegistry()
        assert reg.specs == []

    def test_instances_are_independent(self):
        reg_a = MetricRegistry(PSNR)
        reg_b = MetricRegistry(SSIM)
        assert {s.name for s in reg_a.specs} == {"psnr"}
        assert {s.name for s in reg_b.specs} == {"ssim"}

    def test_caches_are_independent(self, fake_metric):
        reg_a = MetricRegistry()
        reg_b = MetricRegistry()
        reg_a.register_metric("shared_name", fake_metric,
                              direction="higher_is_better", reference=False)
        assert reg_a.get_metric("shared_name") is fake_metric
        with pytest.raises(KeyError):
            reg_b.get_metric("shared_name")

    def test_register_metric_is_a_method(self, fake_metric):
        reg = MetricRegistry()
        reg.register_metric("custom", fake_metric,
                            direction="lower_is_better", reference=True, channels="gray")
        spec = next(s for s in reg.specs if s.name == "custom")
        assert spec.builtin is False
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert reg.get_metric("custom") is fake_metric

    def test_no_module_level_singleton(self):
        import metrics
        assert not hasattr(metrics, "registry"), "global registry singleton must be gone"
        assert not callable(getattr(metrics, "register_metric", None)), \
            "free register_metric function must be gone"


# ---------------------------------------------------------------------------
# register_metric (public API)
# ---------------------------------------------------------------------------

class TestRegisterMetric:
    def test_custom_metric_accessible_via_registry(self, fake_metric):
        reg = MetricRegistry()
        reg.register_metric("custom_test", fake_metric, direction="higher_is_better", reference=False)
        spec = next((s for s in reg.specs if s.name == "custom_test"), None)
        assert spec is not None
        assert spec.builtin is False
        assert spec.direction == "higher_is_better"

    def test_custom_metric_returns_scores(self, fake_metric):
        reg = MetricRegistry()
        reg.register_metric("custom_scores", fake_metric, direction="higher_is_better", reference=False)
        m = reg.get_metric("custom_scores")
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
# Built-in metric specs: exposed as constants, not auto-registered
# ---------------------------------------------------------------------------

class TestBuiltinMetrics:
    EXPECTED = {
        "psnr":               ("higher_is_better", True,  "gray"),
        "ssim":               ("higher_is_better", True,  "gray"),
        "lpips":              ("lower_is_better",  True,  "rgb"),
        "dists":              ("lower_is_better",  True,  "rgb"),
        "radimagenet_lpips":  ("lower_is_better",  True,  "rgb"),
        "clipiqa":            ("higher_is_better", False, "rgb"),
        "clip_iqa_lung":      ("higher_is_better", False, "rgb"),
        "clip_iqa_brain":     ("higher_is_better", False, "rgb"),
        "brisque":            ("lower_is_better",  False, "rgb"),
        "niqe":               ("lower_is_better",  False, "rgb"),
    }

    def test_not_registered_by_default(self):
        # A fresh registry starts empty — nothing self-registers at import time.
        assert MetricRegistry().specs == []

    def test_all_names_present_in_bundle(self):
        names = {s.name for s in BUILTIN_METRICS}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing from BUILTIN_METRICS"

    @pytest.mark.parametrize("name,attrs", EXPECTED.items())
    def test_spec_attributes(self, name, attrs):
        direction, reference, channels = attrs
        spec = next(s for s in BUILTIN_METRICS if s.name == name)
        assert spec.direction == direction,  f"{name}: direction mismatch"
        assert spec.reference == reference,  f"{name}: reference mismatch"
        assert spec.channels  == channels,   f"{name}: channels mismatch"
        assert spec.builtin   is True,       f"{name}: should be builtin"

    def test_register_opts_in(self):
        reg = MetricRegistry()
        reg.register(*BUILTIN_METRICS)
        names = {s.name for s in reg.specs}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing after explicit registration"


# ---------------------------------------------------------------------------
# Segmentation metrics (MONAI-backed)
# ---------------------------------------------------------------------------

class TestSegmentationMetrics:
    EXPECTED = {
        "dice":              ("higher_is_better", True, "gray"),
        "hausdorff95":       ("lower_is_better",  True, "gray"),
        "nsd":               ("higher_is_better", True, "gray"),
        "assd":              ("lower_is_better",  True, "gray"),
        "panoptic_quality":  ("higher_is_better", True, "gray"),
    }

    def test_not_registered_by_default(self):
        # A fresh registry starts empty — nothing self-registers at import time.
        assert MetricRegistry().specs == []

    def test_all_names_present_in_bundle(self):
        names = {s.name for s in SEGMENTATION_METRICS}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing from SEGMENTATION_METRICS"

    def test_kept_out_of_builtin_metrics(self):
        names = {s.name for s in BUILTIN_METRICS}
        assert not (set(self.EXPECTED) & names)

    @pytest.mark.parametrize("name,attrs", EXPECTED.items())
    def test_spec_attributes(self, name, attrs):
        direction, reference, channels = attrs
        spec = next(s for s in SEGMENTATION_METRICS if s.name == name)
        assert spec.direction == direction, f"{name}: direction mismatch"
        assert spec.reference == reference, f"{name}: reference mismatch"
        assert spec.channels  == channels,  f"{name}: channels mismatch"
        assert spec.builtin   is False,     f"{name}: should not be builtin"
        assert spec.domain    == "medical (MONAI)"
        assert spec.description  # non-empty

    def test_register_opts_in(self):
        reg = MetricRegistry()
        reg.register(*SEGMENTATION_METRICS)
        names = {s.name for s in reg.specs}
        for name in self.EXPECTED:
            assert name in names, f"'{name}' missing after explicit registration"

    def test_metrics_module_reexports_same_objects(self):
        from metrics import DICE
        assert DICE is _DICE


# ---------------------------------------------------------------------------
# MetricSpec description and domain fields
# ---------------------------------------------------------------------------

class TestMetricSpecDescriptionFields:
    def test_defaults_are_empty_strings(self):
        spec = MetricSpec("dummy", "higher_is_better", False, "gray", lambda: None)
        assert spec.description == ""
        assert spec.domain == ""

    def test_accepts_explicit_values(self):
        spec = MetricSpec(
            "dummy", "higher_is_better", False, "gray", lambda: None,
            description="measures X", domain="medical (MONAI)",
        )
        assert spec.description == "measures X"
        assert spec.domain == "medical (MONAI)"
