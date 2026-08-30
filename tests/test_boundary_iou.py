"""Tests for src/segmentation_metrics/boundary_iou.py — Boundary IoU (Cheng et al., CVPR 2021)."""
import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from segmentation_metrics.boundary_iou import (
    DEFAULT_DILATION_RATIO,
    boundary_region,
    dilation_pixels,
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

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            boundary_region(np.ones((2, 8, 8), bool), 1)

    @pytest.mark.parametrize("dilation", [1, 2, 3, 5, 9])
    def test_matches_iterated_erosion_reference(self, dilation):
        """Bit-exact agreement with the published cv2-based recipe, including
        masks that touch the image border."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            h, w = rng.integers(6, 40, 2)
            mask = rng.random((h, w)) > rng.uniform(0.2, 0.8)
            mask[:2, :] |= rng.random((2, w)) > 0.5   # force border contact
            np.testing.assert_array_equal(
                boundary_region(mask, dilation), _reference_boundary(mask, dilation)
            )
