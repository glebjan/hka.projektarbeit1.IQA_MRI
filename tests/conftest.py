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
