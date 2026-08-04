"""Tests for src/image_loader.py — all pure-logic, no network."""
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import torch
from PIL import Image

from image_loader import (
    ImageLoader,
    _dicom_array_to_depth_first,
    _load_nifti,
    _load_pil,
    _load_sitk,
    _to_normalized_channel_tensor,
    canonical_suffix,
    find_matching_target,
    is_supported,
    list_images,
    strip_all_extensions,
    _shared_prefix_length,
)


# ---------------------------------------------------------------------------
# _to_normalized_channel_tensor
# ---------------------------------------------------------------------------

class TestToNormalizedChannelTensor:
    def test_shape(self):
        arr = np.random.default_rng(0).random((5, 64, 64))
        t = _to_normalized_channel_tensor(arr)
        assert t.shape == (5, 1, 64, 64)

    def test_values_in_range(self):
        arr = np.random.default_rng(1).random((3, 32, 32)) * 1000
        t = _to_normalized_channel_tensor(arr)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0 + 1e-6

    def test_constant_array_becomes_zeros(self):
        arr = np.full((2, 16, 16), 42.0)
        t = _to_normalized_channel_tensor(arr)
        assert torch.all(t == 0)

    def test_dtype_float32(self):
        arr = np.random.default_rng(2).random((2, 16, 16))
        t = _to_normalized_channel_tensor(arr)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# _load_pil
# ---------------------------------------------------------------------------

class TestLoadPil:
    def test_shape_and_range(self, tmp_path):
        arr = np.random.default_rng(0).integers(0, 256, (64, 64), dtype="uint8")
        p = tmp_path / "img.png"
        Image.fromarray(arr).save(p)
        t = _load_pil(p)
        assert t.shape == (1, 1, 64, 64)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0 + 1e-6

    def test_rgb_png_converted_to_grayscale(self, tmp_path):
        arr = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype="uint8")
        p = tmp_path / "rgb.png"
        Image.fromarray(arr, mode="RGB").save(p)
        t = _load_pil(p)
        # Should be (1, 1, H, W) — grayscale
        assert t.shape == (1, 1, 32, 32)


# ---------------------------------------------------------------------------
# _dicom_array_to_depth_first
# ---------------------------------------------------------------------------

class TestDicomArrayToDepthFirst:
    def test_2d_becomes_1_h_w(self):
        arr = np.zeros((64, 64))
        out = _dicom_array_to_depth_first(arr, "MONOCHROME2")
        assert out.shape == (1, 64, 64)

    def test_3d_depth_first_passthrough(self):
        arr = np.zeros((10, 64, 64))
        out = _dicom_array_to_depth_first(arr, "MONOCHROME2")
        assert out.shape == (10, 64, 64)

    def test_rgb_converts_to_luminance(self):
        arr = np.ones((64, 64, 3), dtype=np.float32)
        arr[..., 0] = 100; arr[..., 1] = 150; arr[..., 2] = 50
        out = _dicom_array_to_depth_first(arr, "RGB")
        assert out.shape == (1, 64, 64)
        expected = 0.2989 * 100 + 0.5870 * 150 + 0.1140 * 50
        assert abs(float(out[0, 0, 0]) - expected) < 0.5

    def test_unsupported_ndim_raises(self):
        arr = np.zeros((2, 3, 4, 5))
        with pytest.raises(ValueError, match="Unsupported DICOM"):
            _dicom_array_to_depth_first(arr, "MONOCHROME2")


# ---------------------------------------------------------------------------
# _load_dicom
# ---------------------------------------------------------------------------

class TestLoadDicom:
    def _make_dicom(self, path: Path, arr: np.ndarray,
                    slope: float = 1.0, intercept: float = 0.0,
                    photometric: str = "MONOCHROME2") -> Path:
        """Build a minimal DICOM file from scratch (no pydicom test data required)."""
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.sequence import Sequence
        from pydicom.uid import (
            ExplicitVRLittleEndian,
            generate_uid,
            UID,
        )

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_implicit_VR  = False
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
        ds.RescaleSlope     = slope
        ds.RescaleIntercept = intercept
        ds.PixelData        = arr.astype(np.uint16).tobytes()
        ds.save_as(str(path), write_like_original=False)
        return path

    def test_loads_and_normalised(self, tmp_path):
        arr = np.random.default_rng(0).integers(100, 2000, (64, 64), dtype=np.uint16)
        p = self._make_dicom(tmp_path / "test.dcm", arr)
        from image_loader import _load_dicom
        t = _load_dicom(p)
        assert t.shape[1] == 1  # channel dim
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0 + 1e-6

    def test_monochrome1_inverted(self, tmp_path):
        arr = np.zeros((64, 64), dtype=np.uint16)
        arr[:32, :] = 1000
        p = self._make_dicom(tmp_path / "m1.dcm", arr, photometric="MONOCHROME1")
        from image_loader import _load_dicom
        t = _load_dicom(p)
        # After MONOCHROME1 inversion the originally-bright top half should now
        # have lower normalised values than the originally-dark bottom half.
        top_mean = float(t[0, 0, :32, :].mean())
        bot_mean = float(t[0, 0, 32:, :].mean())
        assert bot_mean > top_mean


# ---------------------------------------------------------------------------
# _load_nifti
# ---------------------------------------------------------------------------

class TestLoadNifti:
    def test_3d_volume(self, tmp_path):
        arr = np.random.default_rng(0).random((30, 64, 64))
        img = nib.Nifti1Image(arr, np.eye(4))
        p = tmp_path / "vol.nii"
        nib.save(img, str(p))
        t = _load_nifti(p)
        # 3D: (H, W, D) → depth_first (D, H, W) → (D, 1, H, W)
        assert t.ndim == 4
        assert t.shape[1] == 1

    def test_4d_volume(self, tmp_path):
        arr = np.random.default_rng(1).random((16, 16, 5, 3))  # (X,Y,Z,T)
        img = nib.Nifti1Image(arr, np.eye(4))
        p = tmp_path / "vol4d.nii"
        nib.save(img, str(p))
        t = _load_nifti(p)
        assert t.ndim == 4
        assert t.shape[1] == 1

    def test_unsupported_ndim_raises(self, tmp_path):
        arr = np.zeros((4, 4))  # 2D: not valid for NIfTI loader path
        img = nib.Nifti1Image(arr, np.eye(4))
        p = tmp_path / "bad.nii"
        nib.save(img, str(p))
        with pytest.raises(ValueError, match="Unsupported NIfTI ndim"):
            _load_nifti(p)


# ---------------------------------------------------------------------------
# _load_sitk
# ---------------------------------------------------------------------------

class TestLoadSitk:
    def test_3d_mha(self, tmp_path):
        arr = np.random.default_rng(0).random((10, 64, 64)).astype(np.float32)
        itk_img = sitk.GetImageFromArray(arr)
        p = tmp_path / "vol.mha"
        sitk.WriteImage(itk_img, str(p))
        t = _load_sitk(p)
        assert t.ndim == 4 and t.shape[1] == 1

    def test_2d_mha_gets_depth_dim(self, tmp_path):
        arr = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        itk_img = sitk.GetImageFromArray(arr)
        p = tmp_path / "slice.mha"
        sitk.WriteImage(itk_img, str(p))
        t = _load_sitk(p)
        assert t.shape == (1, 1, 64, 64)


# ---------------------------------------------------------------------------
# canonical_suffix / is_supported
# ---------------------------------------------------------------------------

class TestCanonicalSuffix:
    def test_nii_gz(self):
        assert canonical_suffix(Path("brain.nii.gz")) == ".nii"

    def test_uppercase_dcm(self):
        assert canonical_suffix(Path("SCAN.DCM")) == ".dcm"

    def test_regular_png(self):
        assert canonical_suffix(Path("image.png")) == ".png"


class TestIsSupported:
    @pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.jpeg", "a.dcm", "a.nii", "a.nrrd", "a.mha", "a.mhd"])
    def test_supported(self, name):
        assert is_supported(Path(name))

    def test_nii_gz_supported(self):
        assert is_supported(Path("vol.nii.gz"))

    @pytest.mark.parametrize("name", ["a.txt", "a.csv", "a.tiff", "a.bmp"])
    def test_unsupported(self, name):
        assert not is_supported(Path(name))


# ---------------------------------------------------------------------------
# strip_all_extensions / _shared_prefix_length
# ---------------------------------------------------------------------------

class TestStringHelpers:
    def test_strip_single_extension(self):
        assert strip_all_extensions(Path("brain.nii")) == "brain"

    def test_strip_double_extension(self):
        assert strip_all_extensions(Path("brain.nii.gz")) == "brain"

    def test_shared_prefix_full(self):
        assert _shared_prefix_length("hello", "hello") == 5

    def test_shared_prefix_partial(self):
        assert _shared_prefix_length("abcde", "abcXY") == 3

    def test_shared_prefix_none(self):
        assert _shared_prefix_length("abc", "xyz") == 0


# ---------------------------------------------------------------------------
# list_images
# ---------------------------------------------------------------------------

class TestListImages:
    def test_returns_sorted_supported_files(self, tmp_path):
        (tmp_path / "c.png").touch()
        (tmp_path / "a.png").touch()
        (tmp_path / "b.txt").touch()   # unsupported
        sub = tmp_path / "sub"; sub.mkdir()
        (sub / "d.dcm").touch()
        result = list_images(tmp_path)
        names = [p.name for p in result]
        assert "b.txt" not in names
        assert names == sorted(names)
        assert "a.png" in names and "c.png" in names and "d.dcm" in names


# ---------------------------------------------------------------------------
# find_matching_target
# ---------------------------------------------------------------------------

class TestFindMatchingTarget:
    def _paths(self, *names):
        return [Path(n) for n in names]

    def test_best_prefix_match(self):
        targets = self._paths("case_001_ref.png", "case_002_ref.png", "other.png")
        inp = Path("case_001_input.png")
        result = find_matching_target(inp, targets)
        assert result is not None and result.name == "case_001_ref.png"

    def test_no_match_below_threshold(self):
        targets = self._paths("abc.png")
        inp = Path("xyz.png")
        assert find_matching_target(inp, targets) is None

    def test_exact_threshold_boundary(self):
        # 4-char prefix exactly meets _MIN_MATCH_PREFIX_LENGTH
        targets = self._paths("abcd_ref.png")
        inp = Path("abcd_inp.png")
        result = find_matching_target(inp, targets)
        assert result is not None

    def test_empty_targets(self):
        assert find_matching_target(Path("img.png"), []) is None


# ---------------------------------------------------------------------------
# ImageLoader
# ---------------------------------------------------------------------------

class TestImageLoader:
    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            ImageLoader(p)

    def test_tensor_cached(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        t1 = loader.tensor
        t2 = loader.tensor
        assert t1 is t2

    def test_tensor_shape(self, synthetic_png):
        t = ImageLoader(synthetic_png).tensor
        # PNG is treated as single slice: (1, 1, H, W)
        assert t.ndim == 4
        assert t.shape[1] == 1

    def test_rgb_tensor_expands_channels(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        rgb = loader.rgb_tensor
        assert rgb.shape[1] == 3
        # All three channels should be identical (expanded from grayscale)
        assert torch.equal(rgb[:, 0], rgb[:, 1])
        assert torch.equal(rgb[:, 0], rgb[:, 2])

    def test_empty_slice_mask_flat_image(self, tmp_path):
        # Constant array → all values < 1e-3 after normalisation → empty
        arr = np.full((64, 64), 42, dtype="uint8")
        p = tmp_path / "flat.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p)
        mask = loader.empty_slice_mask
        assert bool(mask[0].item())

    def test_empty_slice_mask_structured_image(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        mask = loader.empty_slice_mask
        # A random image should NOT be marked empty
        assert not bool(mask[0].item())

    def test_log_tensor_shape_returns_size(self, synthetic_png, capsys):
        loader = ImageLoader(synthetic_png)
        size = loader.log_tensor_shape()
        assert list(size) == list(loader.tensor.shape)
        captured = capsys.readouterr()
        assert "tensor size" in captured.out
