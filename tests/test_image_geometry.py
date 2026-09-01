"""Tests for voxel geometry: LoadedImage, spacing, is_volumetric."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from image_loader import (
    ImageLoader,
    LoadedImage,
    _load_nifti,
    _load_pil,
    _load_sitk,
)


class TestLoadedImage:
    def test_is_a_frozen_dataclass_with_three_fields(self):
        li = LoadedImage(tensor=torch.zeros(1, 1, 2, 2), spacing=(1.0, 2.0, 3.0), is_volumetric=True)
        assert li.tensor.shape == (1, 1, 2, 2)
        assert li.spacing == (1.0, 2.0, 3.0)
        assert li.is_volumetric is True
        with pytest.raises(Exception):
            li.spacing = (1.0, 1.0, 1.0)


class TestPilGeometry:
    def test_png_has_no_spacing_and_is_not_volumetric(self, synthetic_png: Path):
        loaded = _load_pil(synthetic_png)
        assert isinstance(loaded, LoadedImage)
        assert loaded.tensor.shape[0] == 1
        assert loaded.spacing is None
        assert loaded.is_volumetric is False


class TestNiftiGeometry:
    def test_3d_spacing_is_reordered_to_dz_dy_dx(self, nifti_volume: Path):
        loaded = _load_nifti(nifti_volume)
        # header zooms are (dx, dy, dz) = (1.0, 1.0, 1.2); tensor axes are (Z, X, Y)
        assert loaded.tensor.shape == (6, 1, 8, 10)
        assert loaded.spacing == pytest.approx((1.2, 1.0, 1.0))

    def test_3d_is_volumetric(self, nifti_volume: Path):
        assert _load_nifti(nifti_volume).is_volumetric is True

    def test_4d_is_not_volumetric_and_has_no_spacing(self, nifti_4d: Path):
        loaded = _load_nifti(nifti_4d)
        assert loaded.tensor.shape[0] == 6 * 3  # depth and time flattened together
        assert loaded.is_volumetric is False
        assert loaded.spacing is None


class TestSitkGeometry:
    def test_spacing_axis_order_is_reversed(self, sitk_volume: Path):
        loaded = _load_sitk(sitk_volume)
        # GetSpacing() is (x, y, z) = (0.5, 0.5, 2.0); the array is (z, y, x)
        assert loaded.tensor.shape == (6, 1, 10, 8)
        assert loaded.spacing == pytest.approx((2.0, 0.5, 0.5))

    def test_is_volumetric(self, sitk_volume: Path):
        assert _load_sitk(sitk_volume).is_volumetric is True


class TestSingleSliceIsNotVolumetric:
    def test_depth_one_nifti(self, tmp_path: Path):
        import nibabel as nib

        data = np.random.default_rng(6).random((8, 10, 1)).astype("float32")
        p = tmp_path / "flat.nii.gz"
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), p)
        assert _load_nifti(p).is_volumetric is False


class TestImageLoaderProperties:
    def test_exposes_spacing_and_is_volumetric(self, nifti_volume: Path):
        loader = ImageLoader(nifti_volume)
        assert loader.spacing == pytest.approx((1.2, 1.0, 1.0))
        assert loader.is_volumetric is True
        assert loader.tensor.shape == (6, 1, 8, 10)

    def test_png_through_loader(self, synthetic_png: Path):
        loader = ImageLoader(synthetic_png)
        assert loader.spacing is None
        assert loader.is_volumetric is False

    def test_loads_only_once(self, nifti_volume: Path):
        loader = ImageLoader(nifti_volume)
        first = loader.tensor
        assert loader.spacing is not None
        assert loader.tensor is first
