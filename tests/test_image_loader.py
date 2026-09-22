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

from iqaevaluator.image_loader import (
    ImageLoader,
    _dicom_array_to_depth_first,
    _load_nifti,
    _load_pil,
    _load_sitk,
    canonical_suffix,
    find_matching_target,
    is_supported,
    list_images,
    strip_all_extensions,
    _shared_prefix_length,
    load_pair,
)
from iqaevaluator.normalization import IntensityRange, Mask, MinMax, Percentile, Raw

IMG_SIZE = 96  # must match tests/conftest.py


# ---------------------------------------------------------------------------
# _load_pil
# ---------------------------------------------------------------------------

class TestLoadPil:
    def test_shape_and_range(self, tmp_path):
        arr = np.random.default_rng(0).integers(0, 256, (64, 64), dtype="uint8")
        p = tmp_path / "img.png"
        Image.fromarray(arr).save(p)
        t = ImageLoader(p).tensor
        assert t.shape == (1, 1, 64, 64)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0 + 1e-6

    def test_rgb_png_keeps_three_channels(self, tmp_path):
        arr = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype="uint8")
        p = tmp_path / "rgb.png"
        Image.fromarray(arr, mode="RGB").save(p)
        t = ImageLoader(p).tensor
        # Colour PNGs now carry their channel axis through — see TestColourDecoding.
        assert t.shape == (1, 3, 32, 32)


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
        from iqaevaluator.image_loader import _load_dicom
        t = ImageLoader(p).tensor
        assert t.shape[1] == 1  # channel dim
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0 + 1e-6

    def test_monochrome1_inverted(self, tmp_path):
        arr = np.zeros((64, 64), dtype=np.uint16)
        arr[:32, :] = 1000
        p = self._make_dicom(tmp_path / "m1.dcm", arr, photometric="MONOCHROME1")
        from iqaevaluator.image_loader import _load_dicom
        t = ImageLoader(p).tensor
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
        t = ImageLoader(p).tensor
        # 3D: (H, W, D) → depth_first (D, H, W) → (D, 1, H, W)
        assert t.ndim == 4
        assert t.shape[1] == 1

    def test_4d_volume(self, tmp_path):
        arr = np.random.default_rng(1).random((16, 16, 5, 3))  # (X,Y,Z,T)
        img = nib.Nifti1Image(arr, np.eye(4))
        p = tmp_path / "vol4d.nii"
        nib.save(img, str(p))
        t = ImageLoader(p).tensor
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
        t = ImageLoader(p).tensor
        assert t.ndim == 4 and t.shape[1] == 1

    def test_2d_mha_gets_depth_dim(self, tmp_path):
        arr = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        itk_img = sitk.GetImageFromArray(arr)
        p = tmp_path / "slice.mha"
        sitk.WriteImage(itk_img, str(p))
        t = ImageLoader(p).tensor
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
        # Constant array → zero spread → every slice empty
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

    def test_empty_slice_mask_survives_a_spike_voxel(self, tmp_path):
        # One corrupt voxel used to compress the whole volume into a sliver of
        # [0, 1], pushing every slice's std under the threshold.
        vol = np.random.default_rng(0).random((32, 32, 6)).astype(np.float32) * 100
        vol[0, 0, 0] = 1e6
        p = tmp_path / "spike.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        mask = ImageLoader(p).empty_slice_mask
        assert not mask.any()

    def test_empty_slice_mask_flags_the_blank_slice_only(self, tmp_path):
        vol = np.random.default_rng(0).random((32, 32, 4)).astype(np.float32) * 100
        vol[:, :, 2] = 0.0
        p = tmp_path / "hole.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        assert ImageLoader(p).empty_slice_mask.tolist() == [False, False, True, False]

    def test_empty_slice_mask_is_independent_of_the_strategy(self, tmp_path):
        vol = np.random.default_rng(0).random((32, 32, 4)).astype(np.float32) * 100
        vol[:, :, 2] = 0.0
        p = tmp_path / "hole.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        assert torch.equal(ImageLoader(p, Raw()).empty_slice_mask, ImageLoader(p).empty_slice_mask)

    def test_empty_slice_mask_keeps_a_sparse_mask_slice(self, tmp_path):
        # A label map whose foreground is below 0.5 % of the volume: the
        # percentile span collapses to 0 and must fall back to the extremes.
        labels = np.zeros((64, 64, 4), dtype=np.int16)
        labels[10:13, 10, 1] = 1
        p = tmp_path / "sparse.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        assert ImageLoader(p, Raw()).empty_slice_mask.tolist() == [True, False, True, True]

    def test_log_tensor_shape_returns_size(self, synthetic_png, capsys):
        loader = ImageLoader(synthetic_png)
        size = loader.log_tensor_shape()
        assert list(size) == list(loader.tensor.shape)
        captured = capsys.readouterr()
        assert "tensor size" in captured.out

    def test_flat_image_keeps_its_value_and_warns(self, tmp_path, capsys):
        arr = np.full((64, 64), 42, dtype="uint8")
        p = tmp_path / "flat.png"
        Image.fromarray(arr).save(p)
        t = ImageLoader(p).tensor
        assert torch.all(t == 1.0)
        assert "flat.png" in capsys.readouterr().out

    def test_raw_is_the_decoded_array(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        assert loader.raw.dtype == np.uint8
        assert loader.raw.shape == (1, IMG_SIZE, IMG_SIZE)

    def test_raw_range_and_intensity_range_under_minmax(self, synthetic_png):
        loader = ImageLoader(synthetic_png)
        assert loader.raw_range.lo == float(loader.raw.min())
        assert loader.raw_range.hi == float(loader.raw.max())
        assert loader.intensity_range == loader.raw_range

    def test_percentile_strategy_is_applied(self, tmp_path):
        arr = np.linspace(100, 140, 64 * 64, dtype=np.float32).reshape(64, 64)
        arr[0, 0] = 4000.0
        p = tmp_path / "spike.nii"
        nib.save(nib.Nifti1Image(arr[..., np.newaxis], np.eye(4)), str(p))
        t = ImageLoader(p, Percentile()).tensor
        assert float(t.max()) == 1.0
        assert float(t[0, 0, 32, 32]) == pytest.approx(0.5, abs=0.05)

    def test_raw_strategy_preserves_integer_labels(self, tmp_path):
        labels = np.zeros((8, 8, 3), dtype=np.int16)
        labels[:4, :, 0] = 1
        labels[4:, :, 1] = 2
        labels[:, :4, 2] = 3
        p = tmp_path / "labels.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        loader = ImageLoader(p, Raw())
        assert loader.tensor.dtype == torch.int16
        assert loader.intensity_range is None
        assert sorted(torch.unique(loader.tensor).tolist()) == [0, 1, 2, 3]

    def test_default_normalizer_is_minmax(self, synthetic_png):
        assert ImageLoader(synthetic_png).normalizer == MinMax()


class TestMaskLoading:
    def test_png_0_255_and_nifti_0_1_give_identical_tensors(self, tmp_path):
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[4:12, 4:12] = 1
        png = tmp_path / "m.png"
        Image.fromarray(arr * 255).save(png)
        nii = tmp_path / "m.nii"
        nib.save(nib.Nifti1Image(arr[:, :, None], np.eye(4)), str(nii))
        a, b = ImageLoader(png, Mask()).tensor, ImageLoader(nii, Mask()).tensor
        assert a.shape == b.shape == (1, 1, 16, 16)
        assert torch.equal(a, b)
        assert a.dtype == torch.float32 and float(a.sum()) == 64.0

    def test_label_selects_one_class_through_the_loader(self, tmp_path):
        labels = np.zeros((8, 8, 2), dtype=np.int16)
        labels[:4, :, 0] = 1
        labels[4:, :, 1] = 2
        p = tmp_path / "labels.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p))
        assert float(ImageLoader(p, Mask(label=2)).tensor.sum()) == 32.0
        assert float(ImageLoader(p, Mask()).tensor.sum()) == 64.0

    def test_float_map_outside_unit_interval_names_the_file(self, tmp_path):
        p = tmp_path / "logits.nii"
        logits = np.full((4, 4, 1), 4.0, dtype=np.float32)
        logits[:2] = -3.0                               # a negative logit: not a label map
        nib.save(nib.Nifti1Image(logits, np.eye(4)), str(p))
        with pytest.raises(ValueError, match="logits.nii"):
            ImageLoader(p, Mask()).tensor

    def test_empty_slice_mask_is_exact_under_mask(self, tmp_path):
        vol = np.zeros((32, 32, 3), dtype=np.uint8)
        vol[5, 5, 1] = 1
        p = tmp_path / "one_voxel.nii"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
        assert ImageLoader(p, Mask()).empty_slice_mask.tolist() == [True, False, True]
        assert ImageLoader(p, Mask(label=2)).empty_slice_mask.tolist() == [True, True, True]

    def test_intensity_range_is_none_and_raw_range_survives(self, tmp_path):
        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[2:4, 2:4] = 255
        p = tmp_path / "m.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p, Mask())
        assert loader.intensity_range is None
        assert loader.raw_range == IntensityRange(0.0, 255.0)


# ---------------------------------------------------------------------------
# load_pair
# ---------------------------------------------------------------------------

def _save_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(path))
    return path


class TestLoadPair:
    def test_input_is_scaled_on_the_targets_range(self, tmp_path):
        target = np.random.default_rng(0).random((16, 16, 4)) * 800
        inp_p = _save_nifti(tmp_path / "inp.nii", 2 * target + 500)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        inp, tgt = load_pair(inp_p, tgt_p)
        assert inp.intensity_range == tgt.intensity_range
        assert inp.normalizer.name == "minmax"

    def test_affine_bias_is_no_longer_invisible(self, tmp_path):
        # The regression this design fixes: under independent MinMax the two
        # tensors were identical, so every full-reference metric saw a perfect
        # match. On the target's scale the bias shows up as clipping.
        target = np.random.default_rng(0).random((16, 16, 4)) * 800
        inp_p = _save_nifti(tmp_path / "inp.nii", 2 * target + 500)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        independent = torch.allclose(ImageLoader(inp_p).tensor, ImageLoader(tgt_p).tensor, atol=1e-6)
        inp, tgt = load_pair(inp_p, tgt_p)
        assert independent
        assert not torch.allclose(inp.tensor, tgt.tensor, atol=1e-6)
        assert float(inp.tensor.max()) == 1.0

    def test_strategy_is_taken_from_the_target(self, tmp_path):
        target = np.random.default_rng(1).random((16, 16, 4)) * 100
        target[0, 0, 0] = 5000.0
        inp_p = _save_nifti(tmp_path / "inp.nii", target)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        inp, tgt = load_pair(inp_p, tgt_p, Percentile())
        assert inp.normalizer.name == "percentile_0.5_99.5"
        assert inp.intensity_range.hi < 5000.0

    def test_raw_pair_stays_raw(self, tmp_path):
        labels = np.zeros((8, 8, 2), dtype=np.int16); labels[:4] = 1
        p1 = tmp_path / "a.nii"; p2 = tmp_path / "b.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p1))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p2))
        inp, tgt = load_pair(p1, p2, Raw())
        assert inp.intensity_range is None and tgt.intensity_range is None
        assert inp.tensor.dtype == torch.int16
        assert inp.normalizer.name == "raw"

    def test_constant_target_does_not_mislabel_a_non_constant_input(self, tmp_path, capsys):
        # A constant target under MinMax hands the input a degenerate
        # FixedRange(lo == hi). The input itself is not constant, so it must
        # not be reported as one — even though it still has no meaningful
        # scale to be measured against.
        target = np.full((8, 8, 2), 500.0, dtype=np.float32)
        inp = np.linspace(0.0, 100.0, 8 * 8 * 2, dtype=np.float32).reshape(8, 8, 2)
        inp_p = _save_nifti(tmp_path / "inp.nii", inp)
        tgt_p = _save_nifti(tmp_path / "tgt.nii", target)
        loaded_inp, loaded_tgt = load_pair(inp_p, tgt_p)
        loaded_inp.tensor  # trigger scale() and its warning
        out = capsys.readouterr().out
        assert "inp.nii" in out
        assert "constant image" not in out   # the false claim from the bug
        assert "not constant" in out          # the message is explicit instead
        assert "degenerate" in out
        assert loaded_tgt.intensity_range == IntensityRange(500.0, 500.0)

    def test_mask_pair_binarizes_each_side_independently(self, tmp_path):
        # Review 3.1: a 0/1 NIfTI prediction against a 0/255 PNG reference.
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[4:12, 4:12] = 1
        pred = tmp_path / "case_pred.nii"
        nib.save(nib.Nifti1Image(arr[:, :, None], np.eye(4)), str(pred))
        gt = tmp_path / "case_gt.png"
        Image.fromarray(arr * 255).save(gt)
        inp, tgt = load_pair(pred, gt, Mask())
        assert inp.normalizer == Mask() and tgt.normalizer == Mask()
        assert torch.equal(inp.tensor, tgt.tensor)

    def test_raw_pair_is_not_wrapped_in_a_fixed_range(self, tmp_path):
        labels = np.zeros((8, 8, 2), dtype=np.int16); labels[:4] = 1
        p1 = tmp_path / "a.nii"; p2 = tmp_path / "b.nii"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p1))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(p2))
        inp, _ = load_pair(p1, p2, Raw())
        assert inp.normalizer == Raw()


class TestColourDecoding:
    def _rgb_array(self, h: int = 32, w: int = 32) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.integers(0, 256, (h, w, 3), dtype="uint8")

    def test_colour_png_keeps_three_channels(self, tmp_path):
        p = tmp_path / "colour.png"
        Image.fromarray(self._rgb_array()).save(p)
        loader = ImageLoader(p)
        assert loader.channels == 3
        assert loader.tensor.shape == (1, 3, 32, 32)

    def test_grayscale_png_stays_single_channel(self, tmp_path):
        arr = np.random.default_rng(8).integers(0, 256, (32, 32), dtype="uint8")
        p = tmp_path / "gray.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p)
        assert loader.channels == 1
        assert loader.tensor.shape == (1, 1, 32, 32)

    def test_rgba_and_palette_become_three_channels(self, tmp_path):
        rgba = np.concatenate(
            [self._rgb_array(), np.full((32, 32, 1), 255, dtype="uint8")], axis=-1
        )
        p_rgba = tmp_path / "rgba.png"
        Image.fromarray(rgba, mode="RGBA").save(p_rgba)
        p_pal = tmp_path / "palette.png"
        Image.fromarray(self._rgb_array()).convert("P").save(p_pal)
        assert ImageLoader(p_rgba).channels == 3
        assert ImageLoader(p_pal).channels == 3

    def test_to_luma_matches_pil_grayscale(self, tmp_path):
        arr = self._rgb_array()
        p = tmp_path / "colour.png"
        Image.fromarray(arr).save(p)
        from iqaevaluator.image_loader import to_luma

        ours = to_luma(np.asarray(Image.open(p).convert("RGB"))[np.newaxis])
        pil = np.asarray(Image.open(p).convert("L"), dtype=np.float32)[np.newaxis]
        # PIL truncates its fixed-point result, so allow one grey level.
        assert np.abs(ours - pil).max() <= 1.0

    def test_to_luma_passes_single_channel_through(self):
        from iqaevaluator.image_loader import to_luma

        arr = np.arange(2 * 3 * 4, dtype="uint8").reshape(2, 3, 4)
        assert to_luma(arr) is arr

    def test_colour_is_scaled_on_one_shared_range(self, tmp_path):
        # A blue-tinted image: the blue channel never reaches the image maximum.
        arr = np.zeros((16, 16, 3), dtype="uint8")
        arr[..., 0] = 200   # red
        arr[..., 2] = 100   # blue
        p = tmp_path / "tint.png"
        Image.fromarray(arr).save(p)
        tensor = ImageLoader(p, MinMax()).tensor
        # One range over the whole image: red hits 1.0, blue stays below it.
        assert float(tensor[0, 0].max()) == pytest.approx(1.0)
        assert float(tensor[0, 2].max()) == pytest.approx(0.5, abs=0.01)

    def test_multichannel_is_rejected_for_medical_formats(self, tmp_path, monkeypatch):
        import iqaevaluator.image_loader as il

        p = tmp_path / "vol.nrrd"
        p.write_bytes(b"")  # never decoded — the decoder is patched below
        fake = il.LoadedImage(np.zeros((2, 4, 4, 3), dtype="float32"))
        monkeypatch.setitem(il._LOADERS, ".nrrd", lambda _path: fake)
        with pytest.raises(ValueError, match="single channel"):
            _ = ImageLoader(p).raw


class TestChannelRequests:
    def test_colour_image_serves_both_shapes(self, tmp_path):
        arr = np.random.default_rng(11).integers(0, 256, (16, 16, 3), dtype="uint8")
        p = tmp_path / "colour.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p)
        assert loader.rgb_tensor.shape == (1, 3, 16, 16)
        assert loader.gray_tensor.shape == (1, 1, 16, 16)
        # gray_tensor really is a weighted mix, not just the first channel.
        assert not torch.allclose(loader.gray_tensor[:, 0], loader.tensor[:, 0])

    def test_grayscale_image_serves_both_shapes(self, tmp_path):
        arr = np.random.default_rng(12).integers(0, 256, (16, 16), dtype="uint8")
        p = tmp_path / "gray.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p)
        assert loader.rgb_tensor.shape == (1, 3, 16, 16)
        assert loader.gray_tensor.shape == (1, 1, 16, 16)
        # Replication, so all three channels are identical.
        assert torch.equal(loader.rgb_tensor[:, 0], loader.rgb_tensor[:, 2])
        assert torch.equal(loader.gray_tensor, loader.tensor)


class TestColourMasks:
    def _save(self, tmp_path, arr, name="mask.png"):
        p = tmp_path / name
        Image.fromarray(arr).save(p)
        return p

    def test_grey_rgb_mask_is_accepted(self, tmp_path):
        flat = np.zeros((16, 16), dtype="uint8")
        flat[4:8, 4:8] = 255
        rgb = np.stack([flat] * 3, axis=-1)
        loader = ImageLoader(self._save(tmp_path, rgb), Mask())
        assert loader.channels == 1
        assert loader.tensor.shape == (1, 1, 16, 16)
        assert set(loader.tensor.unique().tolist()) <= {0.0, 1.0}

    def test_real_colour_mask_is_rejected(self, tmp_path):
        arr = np.zeros((16, 16, 3), dtype="uint8")
        arr[4:8, 4:8, 0] = 255   # red label
        arr[9:12, 9:12, 1] = 255  # green label
        loader = ImageLoader(self._save(tmp_path, arr, "labels.png"), Mask())
        with pytest.raises(ValueError, match="colour"):
            _ = loader.tensor

    def test_raw_strategy_rejects_colour_too(self, tmp_path):
        arr = np.zeros((16, 16, 3), dtype="uint8")
        arr[..., 1] = 7
        loader = ImageLoader(self._save(tmp_path, arr, "instances.png"), Raw())
        with pytest.raises(ValueError, match="colour"):
            _ = loader.tensor

    def test_empty_slice_detection_uses_luma(self, tmp_path):
        arr = np.random.default_rng(14).integers(0, 256, (32, 32, 3), dtype="uint8")
        loader = ImageLoader(self._save(tmp_path, arr, "busy.png"))
        mask = loader.empty_slice_mask
        assert mask.shape == (1,)
        assert not bool(mask[0])


class TestPairChannelHarmonisation:
    def _pair(self, tmp_path):
        rng = np.random.default_rng(15)
        colour = rng.integers(0, 256, (32, 32, 3), dtype="uint8")
        gray = rng.integers(0, 256, (32, 32), dtype="uint8")
        inp = tmp_path / "inp.png"
        tgt = tmp_path / "tgt.png"
        Image.fromarray(colour).save(inp)
        Image.fromarray(gray).save(tgt)
        return inp, tgt

    def test_mixed_pair_is_compared_on_luma(self, tmp_path, capsys):
        inp, tgt = self._pair(tmp_path)
        loaded_input, loaded_target = load_pair(inp, tgt, MinMax())
        assert loaded_input.channels == 1
        assert loaded_target.channels == 1
        assert loaded_input.tensor.shape == loaded_target.tensor.shape
        printed = capsys.readouterr().out
        assert "inp.png" in printed and "tgt.png" in printed

    def test_matching_colour_pair_keeps_colour(self, tmp_path):
        rng = np.random.default_rng(16)
        inp = tmp_path / "a.png"
        tgt = tmp_path / "b.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype="uint8")).save(inp)
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype="uint8")).save(tgt)
        loaded_input, loaded_target = load_pair(inp, tgt, MinMax())
        assert loaded_input.channels == 3 and loaded_target.channels == 3

    def test_mixed_pair_scales_on_the_targets_luma_range(self, tmp_path):
        # Target is a greyscale ramp from 10 to 200; input is colour.
        gray = np.linspace(10, 200, 32 * 32, dtype="uint8").reshape(32, 32)
        colour = np.zeros((32, 32, 3), dtype="uint8")
        colour[..., 0] = 255
        inp = tmp_path / "inp.png"
        tgt = tmp_path / "tgt.png"
        Image.fromarray(colour).save(inp)
        Image.fromarray(gray).save(tgt)
        loaded_input, loaded_target = load_pair(inp, tgt, MinMax())
        assert loaded_target.intensity_range == IntensityRange(10.0, 200.0)
        assert loaded_input.intensity_range == IntensityRange(10.0, 200.0)

    def test_force_gray_after_the_tensor_exists_is_refused(self, tmp_path):
        arr = np.random.default_rng(17).integers(0, 256, (16, 16, 3), dtype="uint8")
        p = tmp_path / "c.png"
        Image.fromarray(arr).save(p)
        loader = ImageLoader(p)
        _ = loader.tensor
        with pytest.raises(RuntimeError, match="before"):
            loader.force_gray()
