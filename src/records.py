"""ImageEvaluatorRecord — per-slice metric results, plus best-slice lookup."""

from dataclasses import dataclass, asdict, field
from typing import Optional

from metrics import registry


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

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d.update(d.pop("extra"))
        return d


def _record_metric_value(record: ImageEvaluatorRecord, metric: str) -> Optional[float]:
    return record.extra.get(metric) if metric in record.extra else getattr(record, metric, None)


def best_slice_per_metric(records: list[ImageEvaluatorRecord]) -> dict[str, int]:
    """Return the slice index that achieves the best score for each metric."""
    direction = registry.direction
    value_index_pairs: dict[str, list[tuple[float, int]]] = {
        metric: [] for metric in direction
    }
    for record in records:
        if record.is_empty:
            continue
        for metric in direction:
            value = _record_metric_value(record, metric)
            if value is not None:
                value_index_pairs[metric].append((value, record.slice_index))

    best: dict[str, int] = {}
    for metric, pairs in value_index_pairs.items():
        if not pairs:
            continue
        if direction[metric] == "higher_is_better":
            _, idx = max(pairs, key=lambda p: p[0])
        else:
            _, idx = min(pairs, key=lambda p: p[0])
        best[metric] = idx
    return best
