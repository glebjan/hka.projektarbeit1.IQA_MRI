"""IQAEvaluator — pure metric computation, no file I/O."""

from typing import Optional

import torch

from image_loader import ImageLoader, strip_all_extensions
from metrics import DEVICE, MetricChannels, MetricSpec, registry
from records import ImageEvaluatorRecord

BATCH_SIZE = 32  # Slices pro Batch-Call. Bei OOM reduzieren.


class IQAEvaluator:
    """Computes all registered IQA metrics for one input/target image pair.

    The evaluator is intentionally free of file I/O: it only returns
    ImageEvaluatorRecord objects.  Writing results to disk is handled by
    MaskWriter (segmentation images) and EvaluationResult (CSV).
    """

    def __init__(
        self,
        input_image:  ImageLoader,
        target_image: Optional[ImageLoader],
        source_model: Optional[str] = None,
    ) -> None:
        self.input        = input_image
        self.target       = target_image
        self.source_model = source_model

        if self.target is not None and self.input.tensor.shape != self.target.tensor.shape:
            raise ValueError(
                f"shape mismatch: input {tuple(self.input.tensor.shape)} "
                f"vs target {tuple(self.target.tensor.shape)}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_tensor_batch(self, img: ImageLoader, channels: MetricChannels, indices: list[int]) -> torch.Tensor:
        base = img.tensor if channels == "gray" else img.rgb_tensor
        return base[indices]  # (len(indices), C, H, W)

    def _compute_batch(self, spec: MetricSpec, indices: list[int]) -> list[Optional[float]]:
        metric = registry.get_metric(spec.name)
        inp = self._pick_tensor_batch(self.input, spec.channels, indices).to(DEVICE)
        try:
            if spec.reference:
                ref = self._pick_tensor_batch(self.target, spec.channels, indices).to(DEVICE)
                return list(metric(inp, ref))
            return list(metric(inp))
        except Exception as exc:
            print(f"[{self.input.path}] metric '{spec.name}' batch failed: {exc}")
            return [None] * len(indices)

    def _format_slice_id(self, slice_index: int) -> str:
        base = f"{strip_all_extensions(self.input.path)}_s{slice_index:03d}"
        return f"{self.source_model}/{base}" if self.source_model else base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation(self) -> list[ImageEvaluatorRecord]:
        """Evaluate all metrics for every slice.  No files are written."""
        D          = self.input.tensor.shape[0]
        empty_mask = self.input.empty_slice_mask
        has_target = self.target is not None
        mode       = "full_reference" if has_target else "no_reference"

        records = [
            ImageEvaluatorRecord(
                image_id=self._format_slice_id(i),
                source_model=self.source_model,
                mode=mode,
                slice_index=i,
                is_empty=bool(empty_mask[i].item()),
            )
            for i in range(D)
        ]

        active = [i for i in range(D) if not records[i].is_empty]

        for spec in registry.specs:
            if spec.reference and not has_target:
                continue
            for chunk_start in range(0, len(active), BATCH_SIZE):
                chunk = active[chunk_start : chunk_start + BATCH_SIZE]
                values = self._compute_batch(spec, chunk)
                for idx, value in zip(chunk, values):
                    if spec.builtin:
                        setattr(records[idx], spec.name, value)
                    else:
                        records[idx].extra[spec.name] = value

        return records
