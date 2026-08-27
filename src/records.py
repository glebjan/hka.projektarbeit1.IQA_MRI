"""ImageEvaluatorRecord — per-slice metric results."""

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ImageEvaluatorRecord:
    image_id:            str
    source_model:        Optional[str]   = None
    mode:                str             = "no_reference"
    slice_index:         int             = 0
    is_empty:            bool            = False
    # Full-reference metrics (None when no target is available)
    psnr:                Optional[float] = None
    ssim:                Optional[float] = None
    lpips:               Optional[float] = None
    dists:               Optional[float] = None
    radimagenet_lpips:   Optional[float] = None
    # No-reference metrics
    clipiqa:             Optional[float] = None
    clip_iqa_lung:       Optional[float] = None
    clip_iqa_brain:      Optional[float] = None
    brisque:             Optional[float] = None
    niqe:                Optional[float] = None
    # User-registered metrics (see metrics.register_metric) — flattened into to_dict()
    extra:               dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("extra"))
        return d
