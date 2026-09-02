"""Tests for VolumeEvaluator and build_evaluator."""
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluator_factory import build_evaluator
from image_loader import ImageLoader
from iqa_evaluator import IQAEvaluator
from metrics import MetricRegistry, MetricSpec, ModeSupport, ModeUnsupported
from volume_evaluator import VolumeEvaluator


class _ShapeSpy:
    """A Metric that records the shape it was handed and returns one score per sample."""

    def __init__(self, spacing=None):
        self.seen_shapes = []
        self.spacing = spacing

    def __call__(self, input, target=None):
        self.seen_shapes.append(tuple(input.shape))
        return [float(input[i].mean()) for i in range(input.shape[0])]


@pytest.fixture()
def spy_registry():
    """A registry with one metric that works in both modes, plus the spy instances."""
    slice_spy = _ShapeSpy()
    volume_spies = {}

    def volume_factory(spacing):
        volume_spies[spacing] = _ShapeSpy(spacing)
        return volume_spies[spacing]

    spec = MetricSpec(
        name="spy", direction="higher_is_better", reference=False, channels="gray",
        slice_mode=ModeSupport(lambda: slice_spy),
        volume_mode=ModeSupport(volume_factory),
        builtin=False,
    )
    return MetricRegistry(spec), slice_spy, volume_spies


class TestVolumeSample:
    def test_metric_receives_one_five_dimensional_sample(self, nifti_volume: Path, spy_registry):
        registry, _, volume_spies = spy_registry
        loader = ImageLoader(nifti_volume)          # (6, 1, 8, 10)
        VolumeEvaluator(loader, None, registry).run_evaluation()
        spy = next(iter(volume_spies.values()))
        assert spy.seen_shapes == [(1, 1, 6, 8, 10)]

    def test_spacing_reaches_the_volume_factory(self, nifti_volume: Path, spy_registry):
        registry, _, volume_spies = spy_registry
        VolumeEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()
        # nifti_volume's spacing is (1.2, 1.0, 1.5) — see conftest.py's nifti_volume
        # fixture (shape X=8,Y=10,Z=6, voxel size dx=1.0, dy=1.5, dz=1.2). Compared
        # with approx because the NIfTI header stores zooms as float32 (see
        # test_image_geometry.py for the same pattern).
        assert list(volume_spies) == [pytest.approx((1.2, 1.0, 1.5))]


class TestVolumeRecords:
    def test_one_record_per_volume(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        records = VolumeEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()
        assert len(records) == 1

    def test_record_fields(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        record = VolumeEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()[0]
        assert record.scoring == "volume"
        assert record.slice_index is None
        assert record.image_id == "vol"          # no _sNNN suffix
        assert record.mode == "no_reference"
        assert record.extra["spy"] is not None

    def test_source_model_prefixes_the_id(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        record = VolumeEvaluator(
            ImageLoader(nifti_volume), None, registry, source_model="smore"
        ).run_evaluation()[0]
        assert record.image_id == "smore/vol"


class TestVolumeGuards:
    def test_rejects_a_non_volumetric_image(self, synthetic_png: Path, spy_registry):
        registry, _, _ = spy_registry
        with pytest.raises(ValueError, match="3D"):
            VolumeEvaluator(ImageLoader(synthetic_png), None, registry)

    def test_rejects_a_spacing_mismatch(self, nifti_volume: Path, tmp_path: Path, spy_registry):
        import nibabel as nib

        registry, _, _ = spy_registry
        data = np.asarray(nib.load(str(nifti_volume)).get_fdata())
        other = tmp_path / "other.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), other)
        with pytest.raises(ValueError, match="voxel"):
            VolumeEvaluator(ImageLoader(nifti_volume), ImageLoader(other), registry)

    def test_skips_metrics_that_cannot_do_volume(self, nifti_volume: Path):
        spec = MetricSpec(
            name="flat", direction="higher_is_better", reference=False, channels="gray",
            slice_mode=ModeSupport(lambda: (lambda inp, tgt=None: [0.0])),
            volume_mode=ModeUnsupported("only reads flat pictures"),
            builtin=False,
        )
        record = VolumeEvaluator(ImageLoader(nifti_volume), None, MetricRegistry(spec)).run_evaluation()[0]
        assert "flat" not in record.extra

    def test_empty_slices_are_not_dropped(self, tmp_path: Path, spy_registry):
        import nibabel as nib

        registry, _, volume_spies = spy_registry
        data = np.random.default_rng(7).random((8, 10, 6)).astype("float32")
        data[:, :, :2] = 0.0                     # two blank slices
        p = tmp_path / "gappy.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.2, 1.0])), p)
        VolumeEvaluator(ImageLoader(p), None, registry).run_evaluation()
        assert next(iter(volume_spies.values())).seen_shapes == [(1, 1, 6, 8, 10)]


class TestBuildEvaluator:
    def test_slice_mode_returns_the_frozen_evaluator(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        ev = build_evaluator(ImageLoader(nifti_volume), None, registry, mode="slice")
        assert type(ev) is IQAEvaluator

    def test_volume_mode_returns_the_subclass(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        ev = build_evaluator(ImageLoader(nifti_volume), None, registry, mode="volume")
        assert isinstance(ev, VolumeEvaluator)

    def test_default_is_slice(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        assert type(build_evaluator(ImageLoader(nifti_volume), None, registry)) is IQAEvaluator

    def test_rejects_an_unknown_mode(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        with pytest.raises(ValueError, match="slice"):
            build_evaluator(ImageLoader(nifti_volume), None, registry, mode="3d")


class TestSliceRecordsUnchanged:
    def test_slice_records_say_slice(self, nifti_volume: Path, spy_registry):
        registry, _, _ = spy_registry
        records = IQAEvaluator(ImageLoader(nifti_volume), None, registry).run_evaluation()
        assert len(records) == 6
        assert all(r.scoring == "slice" for r in records)
        assert records[0].slice_index == 0
