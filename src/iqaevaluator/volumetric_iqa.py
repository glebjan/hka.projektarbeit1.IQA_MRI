"""MONAI-backed PSNR and SSIM for whole volumes.

pyiqa's metrics take 4D tensors only, so they cannot score a `(1, C, D, H, W)`
sample. MONAI provides both measures with 3D support, and MONAI is already a
dependency of this project.

`SSIMMetric(spatial_dims=3)` uses a 3D gaussian window, so unlike a mean over
per-slice SSIM it actually sees structure across slices.

Note: these are different estimators from the pyiqa ones used in slice mode
(different window and data-range conventions). A `psnr` or `ssim` column from a
slice run and from a volume run are therefore not directly comparable.
"""

from typing import Optional

import torch
from monai.metrics import PSNRMetric, SSIMMetric


class MonaiPSNRMetric:
    """Peak signal-to-noise ratio over a whole volume.

    `max_val` is the maximum possible intensity: ImageLoader normalises every
    image to [0, 1], so the default of 1.0 is correct for this framework.
    """

    def __init__(self, *, max_val: float = 1.0):
        self._impl = PSNRMetric(max_val=max_val)

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[float]:
        if target is None:
            raise ValueError("psnr is a full-reference metric and requires a target image")
        scores = self._impl(y_pred=input, y=target)
        return [float(s) for s in scores.flatten()]


class MonaiSSIMMetric:
    """Structural similarity with a 3D window over a whole volume.

    The window is built at call time, because it cannot be larger than the volume
    it slides through: MRI stacks are routinely thinner than 11 slices, and
    MONAI's default window of 11 would fail on them. The window is shrunk to the
    largest odd size that fits every spatial axis, never grown beyond `win_size`
    — but never below 3 either, since a smaller window measures no structure at
    all. A volume thinner than 3 slices therefore keeps a window that does not
    fit it, MONAI rejects the call, and `VolumeEvaluator` records no score for
    that volume. Score such a stack with mode="slice" instead.

    Args:
        spatial_dims: 3 for volumes. Kept configurable because MONAI supports 2
            as well, which is useful when comparing against the slice backend.
        data_range: maximum intensity range; 1.0 for this framework's [0, 1] tensors.
        win_size: the preferred gaussian window side length, in voxels.
    """

    def __init__(self, *, spatial_dims: int = 3, data_range: float = 1.0, win_size: int = 11):
        self._spatial_dims = spatial_dims
        self._data_range   = data_range
        self._win_size     = win_size

    def _window_for(self, shape: torch.Size) -> int:
        smallest = min(int(s) for s in shape[-self._spatial_dims:])
        fitted = min(self._win_size, smallest)
        if fitted % 2 == 0:
            fitted -= 1
        return max(3, fitted)

    def __call__(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> list[float]:
        if target is None:
            raise ValueError("ssim is a full-reference metric and requires a target image")
        impl = SSIMMetric(
            spatial_dims=self._spatial_dims,
            data_range=self._data_range,
            win_size=self._window_for(input.shape),
        )
        scores = impl(y_pred=input, y=target)
        return [float(s) for s in scores.flatten()]
