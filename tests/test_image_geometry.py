"""Tests for voxel geometry: LoadedImage, spacing, is_volumetric."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from image_loader import (
    ImageLoader,
    LoadedImage,
    _load_dicom,
    _load_nifti,
    _load_pil,
    _load_sitk,
)


def _make_dicom(
    path: Path,
    arr: np.ndarray,
    *,
    photometric: str = "MONOCHROME2",
    pixel_spacing=None,
    slice_thickness=None,
    frame_time=None,
    cine_rate=None,
) -> Path:
    """Build a minimal DICOM file from scratch (no pydicom test data required).

    Same construction as `TestLoadDicom._make_dicom` in test_image_loader.py,
    extended with the geometry tags (`PixelSpacing`, `SliceThickness`,
    `FrameTime`, `CineRate`) and multi-frame support that this file's tests need.
    """
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    ds.SOPClassUID    = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality       = "CT"

    h, w = arr.shape[-2], arr.shape[-1]
    ds.Rows, ds.Columns = h, w
    ds.BitsAllocated    = 16
    ds.BitsStored       = 16
    ds.HighBit          = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel  = 1
    ds.PhotometricInterpretation = photometric
    ds.RescaleSlope     = 1.0
    ds.RescaleIntercept = 0.0
    if arr.ndim == 3:
        ds.NumberOfFrames = arr.shape[0]
    if pixel_spacing is not None:
        ds.PixelSpacing = list(pixel_spacing)
    if slice_thickness is not None:
        ds.SliceThickness = slice_thickness
    if frame_time is not None:
        ds.FrameTime = frame_time
    if cine_rate is not None:
        ds.CineRate = cine_rate
    ds.PixelData = arr.astype(np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)
    return path


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
        # header zooms are (dx, dy, dz) = (1.0, 1.5, 1.2); tensor axes are (Z, X, Y),
        # so the expected spacing is (dz, dx, dy) = (1.2, 1.0, 1.5). All three values
        # are distinct, so a transposition of any two would be caught here.
        assert loaded.tensor.shape == (6, 1, 8, 10)
        assert loaded.spacing == pytest.approx((1.2, 1.0, 1.5))

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
        # GetSpacing() is (x, y, z) = (0.5, 0.75, 2.0); the array is (z, y, x), so the
        # expected spacing is (2.0, 0.75, 0.5). All three values are distinct, so a
        # transposition of any two would be caught here.
        assert loaded.tensor.shape == (6, 1, 10, 8)
        assert loaded.spacing == pytest.approx((2.0, 0.75, 0.5))

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
        assert loader.spacing == pytest.approx((1.2, 1.0, 1.5))
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


class TestDicomGeometry:
    def test_multi_slice_spacing_is_dz_dy_dx(self, tmp_path: Path):
        arr = np.random.default_rng(7).integers(0, 4000, (5, 64, 64), dtype=np.uint16)
        p = _make_dicom(tmp_path / "vol.dcm", arr, pixel_spacing=[0.8, 1.1], slice_thickness=2.5)
        loaded = _load_dicom(p)
        # PixelSpacing = [dy, dx] = [0.8, 1.1], SliceThickness = dz = 2.5 — all
        # three distinct, so a transposition of any two would be caught here.
        assert loaded.spacing == pytest.approx((2.5, 0.8, 1.1))
        assert loaded.is_volumetric is True

    def test_single_frame_is_not_volumetric(self, tmp_path: Path):
        arr = np.random.default_rng(8).integers(0, 4000, (64, 64), dtype=np.uint16)
        p = _make_dicom(tmp_path / "slice.dcm", arr, pixel_spacing=[0.8, 1.1], slice_thickness=2.5)
        loaded = _load_dicom(p)
        assert loaded.is_volumetric is False

    def test_missing_geometry_tags_yield_no_spacing(self, tmp_path: Path):
        arr = np.random.default_rng(9).integers(0, 4000, (5, 64, 64), dtype=np.uint16)
        p = _make_dicom(tmp_path / "nospacing.dcm", arr)
        loaded = _load_dicom(p)
        assert loaded.spacing is None
        assert loaded.is_volumetric is False

    def test_cine_series_is_not_volumetric(self, tmp_path: Path):
        arr = np.random.default_rng(10).integers(0, 4000, (5, 64, 64), dtype=np.uint16)
        p = _make_dicom(
            tmp_path / "cine.dcm", arr,
            pixel_spacing=[0.8, 1.1], slice_thickness=2.5, frame_time=33.3,
        )
        loaded = _load_dicom(p)
        # Spacing is still reported (the tags are present) but a cine series is
        # not treated as spatially volumetric.
        assert loaded.spacing == pytest.approx((2.5, 0.8, 1.1))
        assert loaded.is_volumetric is False
