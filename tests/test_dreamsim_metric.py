"""Tests for src/dreamsim_metric.py — the DreamSim adapter and its spec factory.

Everything here runs offline against a fake loader: no dreamsim import, no
weight download. The real model is exercised in test_dreamsim_integration.py.
"""
import sys

import pytest
import torch

from constants import DREAMSIM_CACHE
# metrics must be imported before dreamsim_metric: dreamsim_metric imports
# metrics at its top and metrics imports it back at the bottom, the same
# deliberate cycle the segmentation_metrics modules use. Importing the metric
# module first hits it mid-initialisation. Same order as
# tests/test_segmentation_metrics.py.
from metrics import DEVICE
from dreamsim_metric import (
    INPUT_SIZE,
    PATCH_CAPABLE,
    VALID_TYPES,
    DreamSimMetric,
    _import_dreamsim,
)


class FakeModel:
    """Stands in for the DreamSim model: records its inputs, returns distances."""

    def __init__(self, *, scalar: bool = False):
        self.scalar = scalar
        self.calls: list[tuple[torch.Size, torch.Size]] = []

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        self.calls.append((img1.shape, img2.shape))
        if self.scalar:
            return torch.tensor(0.5)
        return torch.arange(img1.shape[0], dtype=torch.float32) / 10.0


def make_loader(*, scalar: bool = False):
    """Return a callable mimicking dreamsim.dreamsim(), with call recording."""
    model = FakeModel(scalar=scalar)
    kwargs_seen: list[dict] = []

    def loader(**kwargs):
        kwargs_seen.append(kwargs)
        return model, (lambda img: img)

    loader.model = model
    loader.kwargs_seen = kwargs_seen
    return loader


def pair(n: int = 2, c: int = 3, h: int = 96, w: int = 96):
    """An input/target batch pair in the framework's (N, C, H, W) [0,1] format."""
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(n, c, h, w, generator=generator),
            torch.rand(n, c, h, w, generator=generator))


class TestLazyLoading:
    def test_constructor_does_not_load(self):
        loader = make_loader()
        DreamSimMetric(loader=loader)
        assert loader.kwargs_seen == []

    def test_first_call_loads_once(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair()
        metric(inp, ref)
        metric(inp, ref)
        assert len(loader.kwargs_seen) == 1


class TestLoaderArguments:
    def test_all_six_parameters_reach_the_loader(self):
        loader = make_loader()
        metric = DreamSimMetric(
            dreamsim_type="dino_vitb16",
            pretrained=False,
            device="cpu",
            cache_dir=DREAMSIM_CACHE,
            normalize_embeds=False,
            use_patch_model=True,
            loader=loader,
        )
        inp, ref = pair()
        metric(inp, ref)
        kwargs = loader.kwargs_seen[0]
        assert kwargs["dreamsim_type"]    == "dino_vitb16"
        assert kwargs["pretrained"]       is False
        assert kwargs["device"]           == "cpu"
        assert kwargs["cache_dir"]        == str(DREAMSIM_CACHE)
        assert kwargs["normalize_embeds"] is False
        assert kwargs["use_patch_model"]  is True

    def test_device_defaults_to_framework_device(self):
        metric = DreamSimMetric(loader=make_loader())
        assert metric.device == DEVICE

    def test_explicit_device_overrides(self):
        metric = DreamSimMetric(device="cpu", loader=make_loader())
        assert metric.device == torch.device("cpu")


class TestPreparation:
    def test_resizes_both_tensors_to_224(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=2, h=96, w=64)
        metric(inp, ref)
        shape_in, shape_ref = loader.model.calls[0]
        assert tuple(shape_in)  == (2, 3, INPUT_SIZE, INPUT_SIZE)
        assert tuple(shape_ref) == (2, 3, INPUT_SIZE, INPUT_SIZE)

    def test_expands_single_channel_to_rgb(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=3, c=1)
        metric(inp, ref)
        shape_in, _ = loader.model.calls[0]
        assert tuple(shape_in) == (3, 3, INPUT_SIZE, INPUT_SIZE)

    def test_already_correct_size_is_untouched(self):
        loader = make_loader()
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=1, h=INPUT_SIZE, w=INPUT_SIZE)
        metric(inp, ref)
        shape_in, _ = loader.model.calls[0]
        assert tuple(shape_in) == (1, 3, INPUT_SIZE, INPUT_SIZE)


class TestScores:
    def test_returns_one_float_per_slice(self):
        metric = DreamSimMetric(loader=make_loader())
        inp, ref = pair(n=4)
        scores = metric(inp, ref)
        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)

    def test_scalar_return_falls_back_to_per_pair_scoring(self):
        loader = make_loader(scalar=True)
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=3)
        scores = metric(inp, ref)
        assert scores == [0.5, 0.5, 0.5]
        # One batched attempt, then one call per pair.
        assert len(loader.model.calls) == 4
        assert all(shape[0] == 1 for shape, _ in loader.model.calls[1:])

    def test_single_slice_batch_needs_no_fallback(self):
        loader = make_loader(scalar=True)
        metric = DreamSimMetric(loader=loader)
        inp, ref = pair(n=1)
        assert metric(inp, ref) == [0.5]
        assert len(loader.model.calls) == 1


class TestErrors:
    def test_missing_target_raises(self):
        metric = DreamSimMetric(loader=make_loader())
        inp, _ = pair()
        with pytest.raises(ValueError, match="full-reference"):
            metric(inp)

    def test_absent_package_message_is_actionable(self, monkeypatch):
        # A None entry in sys.modules makes `from dreamsim import ...` raise,
        # which is how the real "package not installed" path behaves.
        monkeypatch.setitem(sys.modules, "dreamsim", None)
        with pytest.raises(ImportError, match="pip install dreamsim"):
            _import_dreamsim()

    def test_loader_failure_names_the_cache_dir(self):
        def broken_loader(**kwargs):
            raise RuntimeError("connection reset")

        metric = DreamSimMetric(loader=broken_loader)
        inp, ref = pair()
        with pytest.raises(RuntimeError, match=str(DREAMSIM_CACHE)):
            metric(inp, ref)


from dreamsim_metric import DREAMSIM, dreamsim_spec
from metrics import (
    REASON_DEEP_2D,
    MetricRegistry,
    ModeSupport,
    ModeUnsupported,
)


class TestSpecNaming:
    CASES = [
        ({},                                                    "dreamsim"),
        ({"dreamsim_type": "dino_vitb16"},                       "dreamsim_dino_vitb16"),
        ({"dreamsim_type": "clip_vitb32"},                       "dreamsim_clip_vitb32"),
        ({"dreamsim_type": "dino_vitb16", "use_patch_model": True},
                                                                 "dreamsim_dino_vitb16_patch"),
        ({"pretrained": False},                                  "dreamsim_scratch"),
        ({"normalize_embeds": False},                            "dreamsim_rawembeds"),
        ({"dreamsim_type": "dinov2_vitb14", "pretrained": False},
                                                                 "dreamsim_dinov2_vitb14_scratch"),
    ]

    @pytest.mark.parametrize("kwargs,expected", CASES)
    def test_name_is_derived(self, kwargs, expected):
        assert dreamsim_spec(**kwargs).name == expected

    def test_device_does_not_change_the_name(self):
        assert dreamsim_spec(device="cpu").name == "dreamsim"

    def test_cache_dir_does_not_change_the_name(self, tmp_path):
        assert dreamsim_spec(cache_dir=tmp_path).name == "dreamsim"

    def test_explicit_name_wins(self):
        assert dreamsim_spec(name="dreamsim_mps").name == "dreamsim_mps"

    def test_distinct_configurations_get_distinct_names(self):
        names = {dreamsim_spec(**kwargs).name for kwargs, _ in self.CASES}
        assert len(names) == len(self.CASES)


class TestSpecAttributes:
    def test_default_spec_attributes(self):
        spec = dreamsim_spec()
        assert spec.direction == "lower_is_better"
        assert spec.reference is True
        assert spec.channels  == "rgb"
        assert spec.builtin   is True
        assert spec.domain    == ""

    def test_variants_are_not_builtin(self):
        assert dreamsim_spec(dreamsim_type="dino_vitb16").builtin is False

    def test_slice_mode_is_supported(self):
        assert isinstance(dreamsim_spec().slice_mode, ModeSupport)

    def test_volume_mode_is_unsupported_with_the_shared_reason(self):
        volume_mode = dreamsim_spec().volume_mode
        assert isinstance(volume_mode, ModeUnsupported)
        assert volume_mode.reason == REASON_DEEP_2D

    def test_module_constant_is_the_default_spec(self):
        assert DREAMSIM.name == "dreamsim"
        assert DREAMSIM.builtin is True


class TestSpecValidation:
    def test_unknown_type_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="ensemble"):
            dreamsim_spec(dreamsim_type="vitb99")

    def test_every_valid_type_is_accepted(self):
        for name in VALID_TYPES:
            assert dreamsim_spec(dreamsim_type=name).name.startswith("dreamsim")

    def test_patch_model_rejected_for_incapable_type(self):
        with pytest.raises(ValueError, match="use_patch_model"):
            dreamsim_spec(dreamsim_type="clip_vitb32", use_patch_model=True)

    def test_patch_model_accepted_for_capable_types(self):
        for name in PATCH_CAPABLE:
            assert dreamsim_spec(dreamsim_type=name, use_patch_model=True).builtin is False


class TestSpecFactoryBuildsTheMetric:
    def test_factory_returns_a_dreamsim_metric(self):
        metric = dreamsim_spec(device="cpu").slice_mode.factory()
        assert isinstance(metric, DreamSimMetric)
        assert metric.device == torch.device("cpu")

    def test_configuration_reaches_the_metric(self):
        metric = dreamsim_spec(dreamsim_type="dino_vitb16",
                               use_patch_model=True).slice_mode.factory()
        assert metric._kwargs["dreamsim_type"]   == "dino_vitb16"
        assert metric._kwargs["use_patch_model"] is True

    def test_registry_can_hold_two_variants_at_once(self):
        registry = MetricRegistry(DREAMSIM,
                                  dreamsim_spec(dreamsim_type="dino_vitb16"))
        assert sorted(s.name for s in registry.specs) == [
            "dreamsim", "dreamsim_dino_vitb16",
        ]

    def test_registry_skips_it_in_volume_mode(self):
        applicable, skipped = MetricRegistry(DREAMSIM).select("volume")
        assert applicable == []
        assert [(s.name, s.reason) for s in skipped] == [("dreamsim", REASON_DEEP_2D)]

    def test_registry_keeps_it_in_slice_mode(self):
        applicable, skipped = MetricRegistry(DREAMSIM).select("slice")
        assert [s.name for s in applicable] == ["dreamsim"]
        assert skipped == []

    def test_registry_caches_the_instance(self):
        registry = MetricRegistry(dreamsim_spec(device="cpu"))
        assert registry.get_metric("dreamsim") is registry.get_metric("dreamsim")
