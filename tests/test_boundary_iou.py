"""Tests for src/segmentation_metrics/boundary_iou.py — Boundary IoU (Cheng et al., CVPR 2021)."""
import numpy as np
import pytest
import torch
from scipy.ndimage import binary_erosion

from metrics import MetricSpec
from segmentation_metrics.boundary_iou import (
    DEFAULT_DILATION_RATIO,
    boundary_iou,
    boundary_region,
    dilation_pixels,
    BoundaryIoUMetric,
    boundary_iou_metric,
    BOUNDARY_IOU,
)


def _reference_boundary(mask: np.ndarray, dilation: int) -> np.ndarray:
    """The paper's reference recipe: zero-pad by 1, erode with a 3x3 kernel
    `dilation` times, crop back, subtract. Used to pin our fast distance-
    transform implementation to the published definition."""
    h, w = mask.shape
    padded = np.pad(mask, 1)
    eroded = binary_erosion(
        padded, structure=np.ones((3, 3), bool), iterations=dilation, border_value=0
    )
    return mask & ~eroded[1 : h + 1, 1 : w + 1]


class TestDilationPixels:
    def test_default_ratio_is_paper_value(self):
        assert DEFAULT_DILATION_RATIO == 0.02

    @pytest.mark.parametrize("shape,expected", [
        ((512, 512), 14),   # diag 724.08 * 0.02 = 14.48 -> 14
        ((400, 400), 11),   # diag 565.69 * 0.02 = 11.31 -> 11
        ((200, 100), 4),    # diag 223.61 * 0.02 =  4.47 ->  4
    ])
    def test_scales_with_image_diagonal(self, shape, expected):
        assert dilation_pixels(shape) == expected

    def test_floors_at_one_pixel(self):
        # 10x10 diagonal is 14.14; 0.02 * 14.14 rounds to 0, must clamp to 1.
        assert dilation_pixels((10, 10)) == 1

    def test_explicit_ratio_overrides_default(self):
        assert dilation_pixels((400, 400), 0.1) == 57  # 565.69 * 0.1 = 56.57

    def test_exact_half_rounds_to_even(self):
        # 3-4-5 triangle: diag = 5.0 exactly, * 0.5 = 2.5 exactly representable.
        # Python's round()/int(round()) is round-half-to-even (banker's
        # rounding), so 2.5 -> 2 (nearest even), not 3.
        assert dilation_pixels((3, 4), 0.5) == 2


class TestBoundaryRegion:
    def test_band_of_a_square_is_the_shell(self):
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True          # 20x20 square
        # d=1 leaves an 18x18 core; d=2 leaves 16x16.
        assert boundary_region(mask, 1).sum() == 400 - 324
        assert boundary_region(mask, 2).sum() == 400 - 256

    def test_band_saturates_at_the_whole_mask(self):
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        assert boundary_region(mask, 10).sum() == 400

    def test_border_clipped_object_counts_its_clipped_edge(self):
        """An object flush against the image border has no pixels outside it,
        but the reference implementation's zero pad still treats that edge as
        boundary — so a corner square bands identically to a free-floating one."""
        corner = np.zeros((40, 40), bool)
        corner[0:20, 0:20] = True
        assert boundary_region(corner, 2).sum() == 144

    def test_full_mask_bands_from_the_image_border_inward(self):
        full = np.ones((10, 10), bool)
        assert boundary_region(full, 1).sum() == 36   # 100 - 8*8
        assert boundary_region(full, 2).sum() == 64   # 100 - 6*6

    def test_empty_mask_has_empty_band(self):
        assert boundary_region(np.zeros((10, 10), bool), 3).sum() == 0

    def test_result_is_a_subset_of_the_mask(self):
        rng = np.random.default_rng(3)
        mask = rng.random((30, 30)) > 0.4
        band = boundary_region(mask, 2)
        assert np.array_equal(band & mask, band)

    def test_accepts_non_2d_input(self):
        """boundary_region's ndim guard was removed in Task 6 — nD masks are
        now handled generically (see TestBoundaryRegion3D for 3D coverage);
        the shape restriction now lives in boundary_iou instead."""
        band = boundary_region(np.ones((2, 8, 8), bool), 1)
        assert band.shape == (2, 8, 8)

    @pytest.mark.parametrize("dilation", [1, 2, 3, 5, 9])
    def test_matches_iterated_erosion_reference(self, dilation):
        """Bit-exact agreement with the published iterated-erosion recipe
        (`cv2.erode` in the reference release, `scipy.ndimage.binary_erosion`
        here — equivalent because the zero pad makes the two border
        conventions agree), including masks that touch the image border."""
        rng = np.random.default_rng(dilation)
        for _ in range(20):
            h, w = rng.integers(6, 40, 2)
            mask = rng.random((h, w)) > rng.uniform(0.2, 0.8)
            mask[:2, :] |= rng.random((2, w)) > 0.5   # force border contact
            np.testing.assert_array_equal(
                boundary_region(mask, dilation), _reference_boundary(mask, dilation)
            )


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float("nan") if union == 0 else np.count_nonzero(a & b) / union


def _square(size: int, offset: int, image: int = 400) -> np.ndarray:
    mask = np.zeros((image, image), bool)
    mask[offset : offset + size, offset : offset + size] = True
    return mask


class TestBoundaryIoU:
    def test_identical_masks_score_one(self):
        mask = _square(160, 120)
        assert boundary_iou(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks_score_zero(self):
        a = np.zeros((64, 64), bool); a[2:10, 2:10] = True
        b = np.zeros((64, 64), bool); b[50:60, 50:60] = True
        assert boundary_iou(a, b) == 0.0

    def test_both_empty_is_nan(self):
        empty = np.zeros((64, 64), bool)
        assert np.isnan(boundary_iou(empty, empty))

    def test_one_empty_scores_zero(self):
        mask = _square(160, 120)
        assert boundary_iou(mask, np.zeros_like(mask)) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            boundary_iou(np.zeros((8, 8), bool), np.zeros((8, 9), bool))

    def test_3d_input_is_now_supported(self):
        """boundary_iou's guard widened from 2D-only to 2D-or-3D in Task 6
        (see TestBoundaryIoU3D for full 3D coverage); only ranks outside
        {2, 3} are still rejected."""
        mask = np.zeros((2, 8, 8), bool)
        mask[0, 2:6, 2:6] = True
        assert boundary_iou(mask, mask) == pytest.approx(1.0)

    def test_non_2d_or_3d_input_raises(self):
        with pytest.raises(ValueError):
            boundary_iou(np.zeros((1, 2, 8, 8), bool), np.zeros((1, 2, 8, 8), bool))

    def test_full_ratio_degenerates_to_mask_iou(self):
        """dilation_ratio=1.0 makes the band the entire mask, so the metric
        must reduce exactly to plain mask IoU."""
        a, b = _square(160, 20), _square(160, 25)
        assert boundary_iou(a, b, dilation_ratio=1.0) == pytest.approx(_mask_iou(a, b))

    def test_penalises_a_shift_more_than_mask_iou(self):
        a, b = _square(160, 20), _square(160, 25)   # 5-pixel diagonal shift
        assert _mask_iou(a, b) == pytest.approx(0.8841, abs=1e-4)
        assert boundary_iou(a, b) == pytest.approx(0.3822, abs=1e-4)

    def test_is_symmetric(self):
        a, b = _square(160, 20), _square(160, 25)
        assert boundary_iou(a, b) == pytest.approx(boundary_iou(b, a))

    def test_is_insensitive_to_object_scale(self):
        """The paper's central claim. For a fixed 5-pixel shift, mask IoU
        improves steeply as the object grows, while Boundary IoU stays flat."""
        sizes = (40, 80, 160, 320)
        mask_scores, boundary_scores = [], []
        for size in sizes:
            offset = (400 - size) // 2
            a, b = _square(size, offset), _square(size, offset + 5)
            mask_scores.append(_mask_iou(a, b))
            boundary_scores.append(boundary_iou(a, b))

        assert mask_scores == sorted(mask_scores)          # rises with size
        assert mask_scores[0] < 0.65 and mask_scores[-1] > 0.93
        assert max(boundary_scores) - min(boundary_scores) < 0.05
        assert all(0.37 < s < 0.42 for s in boundary_scores)

    def test_thresholds_float_probability_masks(self):
        soft_pred = np.where(_square(160, 120), 0.9, 0.1)
        soft_gt = np.where(_square(160, 120), 0.7, 0.2)
        assert boundary_iou(soft_pred, soft_gt, threshold=0.5) == pytest.approx(1.0)

    def test_selects_one_class_from_an_integer_label_map(self):
        labels = np.zeros((64, 64), np.int32)
        labels[10:30, 10:30] = 1
        labels[40:60, 40:60] = 2
        assert boundary_iou(labels, labels, label=2) == pytest.approx(1.0)
        assert boundary_iou(labels, np.zeros_like(labels), label=2) == 0.0


def _batch(masks: list[np.ndarray]) -> torch.Tensor:
    """Stack 2D masks into the framework's (N, 1, H, W) float32 tensor."""
    return torch.from_numpy(np.stack(masks)[:, None].astype(np.float32))


class TestBoundaryIoUMetricAdapter:
    def test_scores_each_sample_independently(self):
        """Three samples with three distinct expected scores — a per-sample bug
        (e.g. scoring the whole batch at once) cannot pass this."""
        metric = BoundaryIoUMetric()
        a, b = _square(160, 20), _square(160, 25)
        empty = np.zeros((400, 400), bool)
        scores = metric(_batch([a, a, a]), _batch([a, b, empty]))
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.3822, abs=1e-4)
        assert scores[2] == 0.0

    def test_reports_none_for_undefined_scores(self):
        metric = BoundaryIoUMetric()
        empty = np.zeros((400, 400), bool)
        solid = _square(160, 120)
        scores = metric(_batch([empty, solid]), _batch([empty, solid]))
        assert scores[0] is None
        assert scores[1] == pytest.approx(1.0)

    def test_missing_target_raises(self):
        metric = BoundaryIoUMetric()
        with pytest.raises(ValueError):
            metric(_batch([_square(160, 120)]))

    def test_dilation_ratio_is_forwarded(self):
        a, b = _square(160, 20), _square(160, 25)
        wide = BoundaryIoUMetric(dilation_ratio=1.0)(_batch([a]), _batch([b]))
        assert wide[0] == pytest.approx(_mask_iou(a, b))

    def test_threshold_binarizes_soft_masks(self):
        soft_pred = np.where(_square(160, 120), 0.9, 0.1)
        soft_gt = np.where(_square(160, 120), 0.7, 0.2)
        scores = BoundaryIoUMetric(threshold=0.5)(_batch([soft_pred]), _batch([soft_gt]))
        assert scores[0] == pytest.approx(1.0)


class TestBoundaryIoUMetricBuilder:
    def test_returns_metric_spec(self):
        spec = boundary_iou_metric()
        assert isinstance(spec, MetricSpec)
        assert spec.name == "boundary_iou"
        assert spec.direction == "higher_is_better"
        assert spec.reference is True
        assert spec.channels == "gray"
        assert spec.builtin is False
        assert spec.domain == ""
        assert "Boundary IoU" in spec.description

    def test_default_constant_matches_builder_defaults(self):
        assert BOUNDARY_IOU.name == "boundary_iou"
        assert BOUNDARY_IOU.builtin is False

    def test_factory_produces_a_working_metric(self):
        metric = BOUNDARY_IOU.slice_mode.factory()
        mask = _square(160, 120)
        assert metric(_batch([mask]), _batch([mask]))[0] == pytest.approx(1.0)

    def test_builder_overrides_reach_the_metric(self):
        a, b = _square(160, 20), _square(160, 25)
        metric = boundary_iou_metric(dilation_ratio=1.0).slice_mode.factory()
        assert metric(_batch([a]), _batch([b]))[0] == pytest.approx(_mask_iou(a, b))


class TestFrameworkRegistration:
    def test_exported_from_metrics(self):
        import metrics
        assert metrics.BOUNDARY_IOU is BOUNDARY_IOU

    def test_included_in_segmentation_bundle(self):
        from metrics import SEGMENTATION_METRICS
        assert BOUNDARY_IOU in SEGMENTATION_METRICS

    def test_not_in_builtin_bundle(self):
        """main.py's raw-image CLI must stay unaffected by segmentation metrics."""
        from metrics import BUILTIN_METRICS
        assert BOUNDARY_IOU not in BUILTIN_METRICS

    def test_reexported_from_main(self):
        import main
        assert main.BOUNDARY_IOU is BOUNDARY_IOU

    def test_registry_round_trip(self):
        from metrics import MetricRegistry
        registry = MetricRegistry(BOUNDARY_IOU)
        assert "boundary_iou" in registry.direction
        assert registry.direction["boundary_iou"] == "higher_is_better"
        metric = registry.get_metric("boundary_iou")
        mask = _square(160, 120)
        assert metric(_batch([mask]), _batch([mask]))[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# nD boundary bands and volume mode
# ---------------------------------------------------------------------------

from metrics import MetricRegistry, ModeSupport
from segmentation_metrics.boundary_iou import band_width


def _cube(shape=(10, 20, 20), lo=(2, 5, 5), hi=(8, 15, 15)):
    m = np.zeros(shape, dtype=bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


class TestBoundaryRegion3D:
    def test_accepts_a_3d_mask(self):
        band = boundary_region(_cube(), 1)
        assert band.shape == (10, 20, 20)
        assert band.dtype == bool

    def test_band_is_a_subset_of_the_mask(self):
        mask = _cube()
        assert np.all(boundary_region(mask, 2) <= mask)

    def test_band_includes_the_caps(self):
        """The first and last occupied slice are surface in 3D, unlike in 2D."""
        mask = _cube()
        band = boundary_region(mask, 1)
        assert band[2].any() and band[7].any()

    def test_a_solid_interior_is_excluded(self):
        mask = np.ones((9, 9, 9), dtype=bool)
        band = boundary_region(mask, 1)
        assert not band[4, 4, 4]

    def test_anisotropic_sampling_thins_the_band_along_the_coarse_axis(self):
        mask = _cube()
        isotropic = boundary_region(mask, 2, sampling=(1.0, 1.0, 1.0))
        coarse = boundary_region(mask, 2, sampling=(3.0, 1.0, 1.0))
        assert coarse.sum() < isotropic.sum()


class TestBandWidth:
    def test_without_spacing_it_matches_the_pixel_diagonal(self):
        # Voxel/chessboard path rounds to the nearest integer (see
        # dilation_pixels); the physical path below does not.
        assert band_width((256, 256), 0.02) == pytest.approx(round(0.02 * np.hypot(256, 256)))

    def test_with_spacing_it_uses_physical_extent(self):
        # extent = (130*1.2, 256*1.0, 256*1.0) mm
        expected = 0.02 * float(np.linalg.norm([130 * 1.2, 256.0, 256.0]))
        assert band_width((130, 256, 256), 0.02, (1.2, 1.0, 1.0)) == pytest.approx(expected)

    def test_never_below_one(self):
        assert band_width((2, 2), 0.001) >= 1.0

    def test_voxel_path_rounds_rather_than_truncates(self):
        """0.1 * hypot(400, 400) = 56.5685 -- round gives 57, truncation gives 56."""
        assert band_width((400, 400), 0.1) == pytest.approx(57.0)
        assert band_width((400, 400), 0.1) == pytest.approx(float(dilation_pixels((400, 400), 0.1)))


class TestBoundaryIoU3D:
    def test_identical_volumes_score_one(self):
        mask = _cube()
        assert boundary_iou(mask, mask) == pytest.approx(1.0)

    def test_shifted_volume_scores_below_one(self):
        gt = _cube()
        pred = _cube(lo=(2, 6, 5), hi=(8, 16, 15))
        assert boundary_iou(pred, gt) < 1.0

    def test_both_empty_is_nan(self):
        empty = np.zeros((6, 8, 8), dtype=bool)
        assert np.isnan(boundary_iou(empty, empty))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes"):
            boundary_iou(_cube(), _cube(shape=(10, 20, 21)))


class TestTwoDimensionalPathUnchanged:
    def test_chessboard_band_is_used_without_spacing(self):
        from scipy.ndimage import distance_transform_cdt

        mask = np.zeros((40, 40), dtype=bool)
        mask[8:32, 8:32] = True
        padded = np.pad(mask, 1).astype(np.uint8)
        expected = mask & (distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1] <= 3)
        assert np.array_equal(boundary_region(mask, 3), expected)


class TestBoundaryIoUVolumeMode:
    def test_spec_supports_volume(self):
        assert isinstance(BOUNDARY_IOU.volume_mode, ModeSupport)

    def test_registry_passes_spacing_into_the_adapter(self):
        metric = MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", (1.2, 1.0, 1.0))
        assert isinstance(metric, BoundaryIoUMetric)
        assert metric._spacing == (1.2, 1.0, 1.0)

    def test_scores_a_five_dimensional_sample(self):
        gt = torch.from_numpy(_cube().astype("float32")).unsqueeze(0).unsqueeze(0)
        pred = torch.from_numpy(_cube(lo=(2, 6, 5), hi=(8, 16, 15)).astype("float32")).unsqueeze(0).unsqueeze(0)
        metric = MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", (1.0, 1.0, 1.0))
        scores = metric(pred, gt)
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_falls_back_without_spacing(self):
        gt = torch.from_numpy(_cube().astype("float32")).unsqueeze(0).unsqueeze(0)
        metric = MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", None)
        assert metric(gt, gt)[0] == pytest.approx(1.0)

    def test_the_fallback_says_so(self, capsys):
        MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", None)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "voxels" in out

    def test_no_warning_when_spacing_is_known(self, capsys):
        MetricRegistry(BOUNDARY_IOU).get_metric("boundary_iou", "volume", (1.0, 1.0, 1.0))
        assert capsys.readouterr().out == ""
