"""Tests for src/mask_writer.py — segmentation helpers + MaskWriter."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mask_writer import (
    MaskWriter,
    _apply_color_overlay,
    _mask_to_uint8,
    _segment_otsu,
    _slice_to_uint8,
)
from records import ImageEvaluatorRecord
from image_loader import ImageLoader
from metrics import PSNR


# ---------------------------------------------------------------------------
# _segment_otsu
# ---------------------------------------------------------------------------

class TestSegmentOtsu:
    def test_flat_image_returns_all_false(self):
        flat = np.full((64, 64), 0.5)
        mask = _segment_otsu(flat)
        assert not mask.any()

    def test_bimodal_image_has_foreground(self):
        img = np.zeros((64, 64))
        img[:32, :] = 0.8   # bright upper half
        mask = _segment_otsu(img)
        # Foreground should cover (roughly) the bright region
        assert mask[:32, :].sum() > mask[32:, :].sum()

    def test_output_dtype_bool(self):
        img = np.random.default_rng(0).random((32, 32))
        mask = _segment_otsu(img)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# _slice_to_uint8
# ---------------------------------------------------------------------------

class TestSliceToUint8:
    def test_range_0_255(self):
        arr = np.array([[0.0, 0.5, 1.0]])
        out = _slice_to_uint8(arr)
        assert out.dtype == np.uint8
        assert int(out[0, 0]) == 0
        assert int(out[0, 2]) == 255

    def test_clipping_above_1(self):
        arr = np.array([[1.5]])
        out = _slice_to_uint8(arr)
        assert int(out[0, 0]) == 255

    def test_clipping_below_0(self):
        arr = np.array([[-0.5]])
        out = _slice_to_uint8(arr)
        assert int(out[0, 0]) == 0


# ---------------------------------------------------------------------------
# _mask_to_uint8
# ---------------------------------------------------------------------------

class TestMaskToUint8:
    def test_true_becomes_255(self):
        mask = np.array([[True, False]])
        out = _mask_to_uint8(mask)
        assert out.dtype == np.uint8
        assert int(out[0, 0]) == 255
        assert int(out[0, 1]) == 0


# ---------------------------------------------------------------------------
# _apply_color_overlay
# ---------------------------------------------------------------------------

class TestApplyColorOverlay:
    def test_output_shape_rgb(self):
        gray = np.ones((32, 32)) * 0.5
        mask = np.zeros((32, 32), dtype=bool)
        out = _apply_color_overlay(gray, mask)
        assert out.shape == (32, 32, 3)
        assert out.dtype == np.uint8

    def test_tint_applied_only_to_foreground(self):
        gray = np.zeros((4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True   # only top-left pixel is foreground
        out = _apply_color_overlay(gray, mask, tint_color=(255, 0, 0), tint_strength=1.0)
        # Foreground pixel: red channel should be 255 (full tint, gray was 0)
        assert int(out[0, 0, 0]) == 255
        # Background pixel: all channels should be 0 (gray=0, no tint)
        assert int(out[1, 0, 0]) == 0

    def test_all_values_in_valid_range(self):
        gray = np.random.default_rng(0).random((32, 32))
        mask = np.random.default_rng(1).random((32, 32)) > 0.5
        out = _apply_color_overlay(gray, mask)
        assert out.min() >= 0 and out.max() <= 255


# ---------------------------------------------------------------------------
# MaskWriter.write
# ---------------------------------------------------------------------------

class TestMaskWriter:
    def _make_loader_from_array(self, arr: np.ndarray, tmp_path: Path) -> ImageLoader:
        """Build an ImageLoader from a grayscale PNG (single slice)."""
        import torch
        p = tmp_path / "img.png"
        Image.fromarray(arr.astype("uint8")).save(p)
        return ImageLoader(p)

    def _make_loader_multislice(self, n: int, tmp_path: Path) -> ImageLoader:
        """Build an ImageLoader with n slices via a NIfTI file."""
        import nibabel as nib
        arr = (np.random.default_rng(77).random((n, 96, 96)) * 200)
        # nibabel expects (X, Y, Z) = (96, 96, n)
        img = nib.Nifti1Image(arr.transpose(1, 2, 0), np.eye(4))
        p = tmp_path / "vol.nii"
        nib.save(img, str(p))
        return ImageLoader(p)

    def _make_records_with_scores(self, n: int = 1) -> list[ImageEvaluatorRecord]:
        records = []
        for i in range(n):
            rec = ImageEvaluatorRecord(image_id=f"img_s{i:03d}", slice_index=i)
            rec.psnr = float(30 + i)
            rec.ssim = 0.9
            records.append(rec)
        return records

    def test_pngs_created(self, tmp_path, isolated_registry):
        isolated_registry.register(PSNR)
        n = 3
        loader = self._make_loader_multislice(n, tmp_path)
        records = self._make_records_with_scores(n=n)
        writer = MaskWriter(tmp_path / "masks")
        saved = writer.write(loader, records)
        assert len(saved) > 0
        for p in saved:
            assert p.exists()

    def test_naming_scheme(self, tmp_path, isolated_registry):
        isolated_registry.register(PSNR)
        arr = (np.random.default_rng(0).random((96, 96)) * 200).astype("uint8")
        loader = self._make_loader_from_array(arr, tmp_path)
        records = self._make_records_with_scores(n=1)
        records[0].psnr = 35.0
        mask_dir = tmp_path / "masks"
        writer = MaskWriter(mask_dir)
        saved = writer.write(loader, records)
        names = [p.name for p in saved]
        # Every file must match pattern: stem_metric_sNNN_{slice|mask|overlay}.png
        for name in names:
            assert name.endswith(".png")
            assert "_s0" in name   # slice index part

    def test_three_files_per_metric(self, tmp_path, isolated_registry):
        isolated_registry.register(PSNR)
        arr = (np.random.default_rng(0).random((96, 96)) * 200).astype("uint8")
        loader = self._make_loader_from_array(arr, tmp_path)
        # Single record with one builtin metric filled
        rec = ImageEvaluatorRecord(image_id="x_s000", slice_index=0, psnr=35.0)
        writer = MaskWriter(tmp_path / "masks")
        saved = writer.write(loader, [rec])
        psnr_files = [p for p in saved if "psnr" in p.name]
        assert len(psnr_files) == 3  # slice, mask, overlay

    def test_output_dir_created(self, tmp_path):
        arr = (np.random.default_rng(0).random((96, 96)) * 200).astype("uint8")
        loader = self._make_loader_from_array(arr, tmp_path)
        rec = ImageEvaluatorRecord(image_id="x_s000", slice_index=0, psnr=35.0)
        new_dir = tmp_path / "brand" / "new"
        assert not new_dir.exists()
        writer = MaskWriter(new_dir)
        writer.write(loader, [rec])
        assert new_dir.exists()
