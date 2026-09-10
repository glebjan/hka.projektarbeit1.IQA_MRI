"""DreamSim as a full-reference metric.

DreamSim (Fu et al., NeurIPS 2023) is a LoRA-finetuned ViT feature extractor
whose distance is the cosine distance between image embeddings. It was trained
on human two-alternative similarity judgements (the NIGHTS dataset), so it
weighs mid-level structure — layout, shape, arrangement — more than pixel-local
texture. That makes it useful for ranking reconstructions of the same anatomy.

This module is the framework's first non-pyiqa `Metric` implementation.
DreamSim brings its own loader, weight management and device handling, so
wrapping it in pyiqa would only add indirection: `DreamSimMetric` implements the
`metrics.Metric` protocol directly, and `IQAEvaluator` never notices the
difference.

Domain caveat, stated once and meant: both the backbones' pretraining and the
human-judgement finetuning happened on natural photographs. MRI contrast, noise
characteristics and the meaning of a grey value are all different. Treat a
dreamsim score as a *relative ranking within one run*, never as an absolute
quality figure. `radimagenet_lpips` is the medically pretrained perceptual
alternative already in the framework.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from constants import DREAMSIM_CACHE
from metrics import DEVICE, MetricSpec, ModeSupport, ModeUnsupported, REASON_DEEP_2D

# DreamSim's ViTs have fixed positional embeddings — 224x224 is not a
# suggestion, it is the only accepted input size.
INPUT_SIZE = 224

VALID_TYPES = (
    "ensemble",
    "dino_vitb16",
    "clip_vitb32",
    "open_clip_vitb32",
    "dinov2_vitb14",
    "synclr_vitb16",
)

# Only these two upstream types ship a patch-feature variant.
PATCH_CAPABLE = ("dino_vitb16", "dinov2_vitb14")


def _import_dreamsim() -> Callable:
    """Return `dreamsim.dreamsim`, or raise with an actionable message.

    Imported lazily and in one place: `metrics.py` must stay importable on a
    machine without the package, and `IQAEvaluator` swallows metric exceptions
    into a printed line, so this message is what the user actually sees.
    """
    try:
        from dreamsim import dreamsim
    except ImportError as exc:
        raise ImportError(
            "dreamsim is not installed — run: pip install dreamsim"
        ) from exc
    return dreamsim


class DreamSimMetric:
    """Adapter that makes DreamSim satisfy the `metrics.Metric` protocol.

    The model is built on the first `__call__`, not in `__init__`, so
    registering the spec costs nothing — no import, no weight download.

    Measured cost (2026-09-10, Apple M-series, warm model, 8 slice pairs of
    96x96): ensemble ~213 ms per slice pair on cpu and ~115 ms on mps;
    dino_vitb16 ~70 ms on cpu and ~36 ms on mps. The ensemble is the most
    expensive metric in the framework — roughly 20 s of compute for a
    100-slice volume on cpu, three times cheaper on a single backbone — and
    the first call additionally pays a one-off weight load of tens of seconds.

    Args:
        dreamsim_type: which backbones produce the embedding. See
            `dreamsim_spec` for what each choice costs and measures.
        pretrained: True loads DreamSim's LoRA weights (finetuned on human
            judgements over natural images); False leaves the backbones raw.
        device: None resolves to the framework's global `DEVICE`. The adapter
            moves its own tensors, because `IQAEvaluator` has already moved them
            to `DEVICE` and an override would otherwise mismatch.
        cache_dir: where weights are downloaded on first use.
        normalize_embeds: L2-normalize embeddings before taking the distance.
        use_patch_model: use dense patch features alongside the CLS token.
            Only available for the types in `PATCH_CAPABLE`.
        loader: injection seam for tests — a callable with
            `dreamsim.dreamsim`'s signature. None means import the real one.
    """

    def __init__(
        self,
        *,
        dreamsim_type:    str = "ensemble",
        pretrained:       bool = True,
        device:           Optional[str | torch.device] = None,
        cache_dir:        Path = DREAMSIM_CACHE,
        normalize_embeds: bool = True,
        use_patch_model:  bool = False,
        loader:           Optional[Callable] = None,
    ):
        self.device    = torch.device(device) if device is not None else DEVICE
        self._cache_dir = Path(cache_dir)
        self._loader   = loader
        self._kwargs   = {
            "dreamsim_type":    dreamsim_type,
            "pretrained":       pretrained,
            "device":           str(self.device),
            "cache_dir":        str(self._cache_dir),
            "normalize_embeds": normalize_embeds,
            "use_patch_model":  use_patch_model,
        }
        self._impl: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Callable:
        """Build the model, annotating any failure with the cache location."""
        loader = self._loader if self._loader is not None else _import_dreamsim()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            model, _preprocess = loader(**self._kwargs)
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"dreamsim failed to load (weight cache: {self._cache_dir}): {exc}"
            ) from exc
        return model

    def _prepare(self, batch: torch.Tensor) -> torch.Tensor:
        """Move to the metric's device, replicate to RGB, resize to 224x224.

        The resize is the same squashing `Resize((224, 224))` upstream's own
        `preprocess` performs, so scores stay comparable to published DreamSim
        numbers and no anatomy is cropped away. The cost is real: high-frequency
        MRI detail — noise texture, thin edges — does not survive the
        downsample, and that detail is often what separates two
        reconstructions.
        """
        batch = batch.to(self.device)
        if batch.shape[1] == 1:
            batch = batch.expand(-1, 3, -1, -1)
        if batch.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            batch = F.interpolate(
                batch, size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear", align_corners=False, antialias=True,
            )
        return batch

    @staticmethod
    def _as_scores(raw, expected: int) -> Optional[list[float]]:
        """Return `expected` floats, or None if the model did not produce them."""
        flat = torch.as_tensor(raw).detach().reshape(-1)
        return [float(v) for v in flat] if flat.numel() == expected else None

    # ------------------------------------------------------------------
    # Metric protocol
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        input:  torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> list[float]:
        if target is None:
            raise ValueError(
                "dreamsim is a full-reference metric — it needs a target image"
            )
        if self._impl is None:
            self._impl = self._load()

        inp = self._prepare(input)
        ref = self._prepare(target)
        n   = inp.shape[0]

        # Upstream documents model(img1, img2) without committing to batch
        # behaviour, so trust the shape rather than the docs: if the call does
        # not return one score per pair, score the pairs one at a time.
        scores = self._as_scores(self._impl(inp, ref), n)
        if scores is None:
            scores = [
                self._as_scores(self._impl(inp[i:i + 1], ref[i:i + 1]), 1)[0]
                for i in range(n)
            ]
        return scores


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def _derive_name(
    dreamsim_type:    str,
    pretrained:       bool,
    normalize_embeds: bool,
    use_patch_model:  bool,
) -> str:
    """Build a metric name that is unique per scoring-relevant configuration.

    Only what changes the score enters the name; `device` and `cache_dir` do
    not. Suffix order is fixed — type, patch, scratch, rawembeds — so the same
    configuration always yields the same column in the report.
    """
    parts = ["dreamsim"]
    if dreamsim_type != "ensemble":
        parts.append(dreamsim_type)
    if use_patch_model:
        parts.append("patch")
    if not pretrained:
        parts.append("scratch")
    if not normalize_embeds:
        parts.append("rawembeds")
    return "_".join(parts)


def dreamsim_spec(
    *,
    dreamsim_type:    str = "ensemble",
    pretrained:       bool = True,
    device:           Optional[str | torch.device] = None,
    cache_dir:        Path = DREAMSIM_CACHE,
    normalize_embeds: bool = True,
    use_patch_model:  bool = False,
    name:             Optional[str] = None,
) -> MetricSpec:
    """Build a configured DreamSim `MetricSpec`.

    `MetricSpec` is a frozen, static description; configuration lives in the
    factory closure, the same seam `_pyiqa_factory("radimagenet_lpips",
    backbone_path=...)` uses in metrics.py.

        registry = MetricRegistry(PSNR, SSIM, DREAMSIM)
        registry = MetricRegistry(dreamsim_spec(dreamsim_type="dino_vitb16",
                                               device="mps"))

    The scores are a distance: lower is more similar, roughly within [0, 1]
    when `normalize_embeds` is True.

    Args:
        dreamsim_type: which backbones produce the embedding. "ensemble"
            (default) concatenates DINO ViT-B/16 + CLIP ViT-B/32 + OpenCLIP
            ViT-B/32, each LoRA-finetuned; it agrees best with human judgement
            and costs three forward passes per image — six per slice pair —
            plus roughly three times the weights and memory. A single backbone
            ("dino_vitb16", "clip_vitb32", "open_clip_vitb32", "dinov2_vitb14",
            "synclr_vitb16") is about a third of the cost and slightly weaker.
            They differ in kind, not just in size: DINO/DINOv2 are
            self-supervised and carry more structural and textural
            information, CLIP/OpenCLIP are language-supervised and carry more
            semantics. For MRI, structure is usually the relevant axis.
            Worth changing: iterate on a single backbone, produce final numbers
            with "ensemble", and never mix the two within one comparison —
            they are different scales.
        pretrained: True loads DreamSim's LoRA weights, trained on human
            similarity judgements over natural images (NIGHTS). False yields
            the raw, unfinetuned backbones. Both the finetuning and the
            backbones' own pretraining are generic-domain, which is why a
            dreamsim score ranks MRI reconstructions but does not measure
            diagnostic quality. Worth changing: running the same evaluation
            with False is the cheapest available ablation for how much the
            natural-image prior contributes here — if the ranking barely
            moves, the finetuning is not transferring. For a medically
            pretrained perceptual metric, see `radimagenet_lpips`.
        device: None resolves to the framework's global `DEVICE` (cuda if
            available, else cpu). Worth changing: "mps" on Apple Silicon,
            "cpu" when a backend lacks an op, "cuda:N" to pick a GPU, or when
            memory is tight — the ensemble holds three ViTs at once.
        cache_dir: where the weights are downloaded on first use. Defaults to
            `constants.DREAMSIM_CACHE`. Worth changing to share one cache
            across projects.
        normalize_embeds: L2-normalizes embeddings before the distance is
            taken. True is the reference configuration and keeps scores in
            roughly [0, 1]. False lets embedding magnitude enter the distance,
            so values stop being comparable across images — experiments only.
        use_patch_model: use dense patch features alongside the CLS token,
            making the metric more sensitive to local fine-structure
            differences (plausibly better for MRI artefacts) at a higher
            compute cost. Only available for the types in `PATCH_CAPABLE`.
        name: overrides the derived metric name. Use it when two specs differ
            only in a parameter the name ignores (`device`, `cache_dir`) and
            both should appear in one report.

    Returns:
        A `MetricSpec` whose `volume_mode` is unsupported: DreamSim is a 2D
        ViT and has no way to see a slice stack as one body.

    Raises:
        ValueError: on an unknown `dreamsim_type`, or `use_patch_model=True`
            for a type without a patch variant.
    """
    if dreamsim_type not in VALID_TYPES:
        raise ValueError(
            f"unknown dreamsim_type '{dreamsim_type}' — valid types are: "
            + ", ".join(VALID_TYPES)
        )
    if use_patch_model and dreamsim_type not in PATCH_CAPABLE:
        raise ValueError(
            f"use_patch_model is only available for {' and '.join(PATCH_CAPABLE)}, "
            f"not for '{dreamsim_type}'"
        )

    metric_name = name or _derive_name(
        dreamsim_type, pretrained, normalize_embeds, use_patch_model
    )

    def build() -> DreamSimMetric:
        return DreamSimMetric(
            dreamsim_type=dreamsim_type,
            pretrained=pretrained,
            device=device,
            cache_dir=cache_dir,
            normalize_embeds=normalize_embeds,
            use_patch_model=use_patch_model,
        )

    return MetricSpec(
        metric_name,
        "lower_is_better",
        True,
        "rgb",
        ModeSupport(build),
        ModeUnsupported(REASON_DEEP_2D),
        # Only the default configuration owns a dedicated
        # ImageEvaluatorRecord field; variants land in record.extra, which the
        # report flattens automatically.
        builtin=(metric_name == "dreamsim"),
    )


# The default ensemble configuration. Deliberately NOT part of
# BUILTIN_METRICS: registering it downloads about a gigabyte of weights and
# spends six ViT forward passes per slice pair, which no default run should do
# behind the user's back. Opt in explicitly:
#
#     registry = MetricRegistry(*BUILTIN_METRICS, DREAMSIM)
DREAMSIM = dreamsim_spec()
