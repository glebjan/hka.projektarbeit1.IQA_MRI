"""Shared fixtures for all tests.

sys.path is patched so that flat imports like `from metrics import ...`
work the same way as `PYTHONPATH=src python ...`.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make src/ importable with bare module names (mirrors PYTHONPATH=src).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 96  # Minimum for niqe (which needs ≥ ~96×96 internally)
RNG = np.random.default_rng(0)


def _make_gray_array(h: int = IMG_SIZE, w: int = IMG_SIZE, *, seed: int = 0) -> np.ndarray:
    """Return a uint8 grayscale array with deterministic random content."""
    return (np.random.default_rng(seed).random((h, w)) * 255).astype("uint8")


@pytest.fixture()
def synthetic_png(tmp_path: Path) -> Path:
    """A single grayscale PNG with structured content (not flat)."""
    arr = _make_gray_array()
    p = tmp_path / "image.png"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture()
def input_target_pair(tmp_path: Path):
    """(inp_path, tgt_path) — same size, target slightly noisy version of input."""
    arr = _make_gray_array()
    noise = np.clip(arr.astype(int) + np.random.default_rng(99).integers(-10, 10, arr.shape), 0, 255).astype("uint8")

    inp = tmp_path / "inp.png"
    tgt = tmp_path / "tgt.png"
    Image.fromarray(arr).save(inp)
    Image.fromarray(noise).save(tgt)
    return inp, tgt


@pytest.fixture()
def fake_metric():
    """A minimal Metric-protocol-compatible callable (no network)."""
    import torch

    def _metric(inp: "torch.Tensor", tgt=None):
        return [float(inp[i].mean()) for i in range(inp.shape[0])]

    return _metric


@pytest.fixture()
def nifti_volume(tmp_path: Path) -> Path:
    """A 3D NIfTI with fully anisotropic voxels: 1.0 x 1.5 x 1.2 mm, shape (X=8, Y=10, Z=6).

    All three voxel sizes are distinct so a transposed axis order in the decoder's
    spacing tuple is detectable, not just a depth-vs-in-plane mixup.
    """
    import nibabel as nib

    data = np.random.default_rng(3).random((8, 10, 6)).astype("float32")
    affine = np.diag([1.0, 1.5, 1.2, 1.0])
    p = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), p)
    return p


@pytest.fixture()
def nifti_4d(tmp_path: Path) -> Path:
    """A 4D NIfTI (X=8, Y=10, Z=6, T=3) — depth and time get flattened on load."""
    import nibabel as nib

    data = np.random.default_rng(4).random((8, 10, 6, 3)).astype("float32")
    affine = np.diag([1.0, 1.0, 1.2, 1.0])
    p = tmp_path / "vol4d.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), p)
    return p


@pytest.fixture()
def sitk_volume(tmp_path: Path) -> Path:
    """A 3D volume written with SimpleITK, fully anisotropic spacing (x=0.5, y=0.75, z=2.0) mm.

    All three voxel sizes are distinct so a transposed axis order in the decoder's
    spacing tuple is detectable, not just a depth-vs-in-plane mixup.
    """
    import SimpleITK as sitk

    arr = np.random.default_rng(5).random((6, 10, 8)).astype("float32")  # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.75, 2.0))
    p = tmp_path / "vol.nrrd"
    sitk.WriteImage(img, str(p))
    return p
