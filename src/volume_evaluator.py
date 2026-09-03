"""VolumeEvaluator — scores a whole 3D volume as a single sample.

Extends IQAEvaluator without modifying it. Where IQAEvaluator hands each metric
a `(D, C, H, W)` batch of slices, this hands it one `(1, C, D, H, W)` sample, so
metrics that understand 3D geometry (surface distance, panoptic quality,
boundary bands) measure across slices instead of within them.

Two deliberate differences from the slice path:

- Empty slices are NOT filtered. Removing slices from a volume is not filtering,
  it is a change of geometry: the body gains holes and surface distances start
  measuring edges that do not exist.
- Metrics are built per image geometry, because HD95, ASSD, NSD and Boundary IoU
  need the voxel spacing to return physical distances.

None of IQAEvaluator's computation is reused. `run_evaluation` is replaced
outright, and the work it calls — `_compute_volume`, `_pick_volume` — is defined
here beside the inherited `_compute_batch` and `_pick_tensor_batch` rather than
overriding them: the inherited pair asks the registry for the slice-mode
instance and indexes a 4D batch, both wrong for a whole volume. What is actually
inherited is `__init__`: the loaders, the registry, the source model and the
input/target shape check. Subclassing for that alone is a compromise, and it is
only safe because IQAEvaluator is frozen by requirement — it cannot be edited to
take a mode.
"""

from typing import Optional

import torch

from image_loader import ImageLoader, strip_all_extensions
from iqa_evaluator import IQAEvaluator
from metrics import DEVICE, MetricChannels, MetricRegistry, MetricSpec, ModeUnsupported
from records import ImageEvaluatorRecord


class VolumeEvaluator(IQAEvaluator):
    """Computes one registry's metrics over one input/target volume pair."""

    def __init__(
        self,
        input_image:  ImageLoader,
        target_image: Optional[ImageLoader],
        registry:     MetricRegistry,
        source_model: Optional[str] = None,
    ):
        super().__init__(input_image, target_image, registry, source_model)

        if not input_image.is_volumetric:
            raise ValueError(
                f"'{input_image.path.name}' is not a 3D volume, so it cannot be "
                "scored in volume mode. Its slices are not stacked along a "
                "spatial axis — this is the case for PNG and JPEG images, for a "
                "single slice, and for 4D scans whose frames are time steps. "
                "Score it with mode='slice' instead."
            )
        if target_image is not None and target_image.spacing != input_image.spacing:
            raise ValueError(
                f"'{input_image.path.name}' and '{target_image.path.name}' are "
                f"recorded at different voxel sizes ({input_image.spacing} vs "
                f"{target_image.spacing}, in millimetres as depth/height/width). "
                "Comparing them directly would measure that size difference "
                "rather than image quality. Resample one onto the other's grid "
                "first — and note that which one you resample changes the scores."
            )
        self.spacing = input_image.spacing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_volume(self, img: ImageLoader, channels: MetricChannels) -> torch.Tensor:
        """(D, C, H, W) -> (1, C, D, H, W): the whole volume as a single sample."""
        base = img.tensor if channels == "gray" else img.rgb_tensor
        return base.permute(1, 0, 2, 3).unsqueeze(0)

    def _compute_volume(self, spec: MetricSpec) -> Optional[float]:
        metric = self.registry.get_metric(spec.name, "volume", self.spacing)
        inp = self._pick_volume(self.input, spec.channels).to(DEVICE)
        try:
            if spec.reference:
                ref = self._pick_volume(self.target, spec.channels).to(DEVICE)
                scores = list(metric(inp, ref))
            else:
                scores = list(metric(inp))
        except Exception as exc:
            print(f"[{self.input.path}] metric '{spec.name}' failed: {exc}")
            return None
        return scores[0] if scores else None

    def _format_volume_id(self) -> str:
        base = strip_all_extensions(self.input.path)
        return f"{self.source_model}/{base}" if self.source_model else base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation(self) -> list[ImageEvaluatorRecord]:
        """Evaluate every applicable metric once over the whole volume."""
        has_target = self.target is not None
        record = ImageEvaluatorRecord(
            image_id=self._format_volume_id(),
            source_model=self.source_model,
            mode="full_reference" if has_target else "no_reference",
            scoring="volume",
            slice_index=None,
            is_empty=bool(self.input.empty_slice_mask.all().item()),
        )

        for spec in self.registry.specs:
            if spec.reference and not has_target:
                continue
            if isinstance(spec.volume_mode, ModeUnsupported):
                continue
            value = self._compute_volume(spec)
            if spec.builtin:
                setattr(record, spec.name, value)
            else:
                record.extra[spec.name] = value

        return [record]
