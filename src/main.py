import argparse
from pathlib import Path
from typing import Optional

from constants import REPORT
from evaluation_result import EvaluationResult, _EvaluatedImage
from evaluator_factory import build_evaluator
from image_loader import ImageLoader, find_matching_target, list_images
from metrics import (  # noqa: F401 — re-exported for users
    DEVICE, Metric, MetricSpec, MetricRegistry, ScoringMode, SkippedMetric,
    PSNR, SSIM, LPIPS, DISTS, RADIMAGENET_LPIPS,
    CLIPIQA, CLIP_IQA_LUNG, CLIP_IQA_BRAIN, BRISQUE, NIQE, BUILTIN_METRICS,
    DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU,
    VS, VS_SIGNED, V_PRED, V_GT, TP, SEGMENTATION_METRICS,
)

_OTHER_MODE = {"slice": "volume", "volume": "slice"}


def report_skipped_metrics(skipped: list[SkippedMetric], mode: ScoringMode) -> None:
    """Tell the user, once per run, which metrics will not be computed and why.

    Metrics are grouped by reason so a shared explanation is printed once. Prints
    nothing when nothing is skipped.
    """
    if not skipped:
        return
    by_reason: dict[str, list[str]] = {}
    for item in skipped:
        by_reason.setdefault(item.reason, []).append(item.name)

    print(f"\nSome metrics cannot be scored in {mode} mode and will be skipped.")
    for reason, names in by_reason.items():
        print(f"\n  {', '.join(sorted(names))}")
        print(f"  This metric {reason}.")
    print(f'\nTo score them, run again with mode="{_OTHER_MODE[mode]}".\n')


# ---------------------------------------------------------------------------
# Top-level evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    input_path: Path,
    target_path: Optional[Path] = None,
    *,
    registry: MetricRegistry,
    mode: ScoringMode = "slice",
) -> EvaluationResult:
    """Discover input/target images and compute one registry's metrics.

    Args:
        input_path:  Path to an input image file or a directory of images.
        target_path: Optional path to a reference image file or directory.
                     Pass None for a no-reference (NR-only) evaluation.
        registry:    The metrics to compute. One instance is shared across
                     every image in the run, so network-backed metrics are
                     built once rather than once per image.
        mode:        "slice" scores every 2D slice separately and produces one
                     row per slice; "volume" scores each 3D stack once and
                     produces one row per volume. Metrics that cannot serve the
                     chosen mode are skipped with a message; if none can, this
                     raises.

    No files are written; use EvaluationResult.generate_report() for output.
    """
    applicable, skipped = registry.select(mode)
    if not applicable:
        raise ValueError(
            f"none of the selected metrics can be scored in {mode} mode, so this "
            f'run would compute nothing. Run again with mode="{_OTHER_MODE[mode]}", '
            "or pick metrics that work on whole volumes (dice, hausdorff95, nsd, "
            "assd, panoptic_quality, boundary_iou, vs, psnr, ssim)."
        )
    report_skipped_metrics(skipped, mode)

    evaluated: list[_EvaluatedImage] = []

    def _run_one(inp: Path, tgt: Optional[Path]) -> None:
        try:
            input_loader = ImageLoader(inp)
            input_loader.log_tensor_shape()
            target_loader: Optional[ImageLoader] = None
            if tgt is not None:
                target_loader = ImageLoader(tgt)
                target_loader.log_tensor_shape()
            if mode == "volume" and not input_loader.is_volumetric:
                print(
                    f"[{inp.name}] skipped: this is not a 3D volume. Its slices "
                    "are not stacked along a spatial axis, which is the case for "
                    "PNG and JPEG images, for a single slice, and for 4D scans "
                    "whose frames are time steps."
                )
                return
            records = build_evaluator(
                input_loader, target_loader, registry, mode
            ).run_evaluation()
            evaluated.append(_EvaluatedImage(input_path=inp, records=records))
        except Exception as exc:
            print(f"[{inp}] evaluation failed: {exc}")

    if input_path.is_file():
        if target_path is not None and not target_path.is_file():
            print(f"[WARNING] input is a file but target '{target_path}' is not — target ignored.")
        tgt = target_path if (target_path is not None and target_path.is_file()) else None
        _run_one(input_path, tgt)
    elif input_path.is_dir():
        available_targets: list[Path] = []
        if target_path is not None and target_path.is_dir():
            available_targets = list_images(target_path)
        elif target_path is not None and target_path.is_file():
            available_targets = [target_path]
        elif target_path is not None:
            print(f"[WARNING] target '{target_path}' is neither a file nor a directory — target ignored.")
        input_files = list_images(input_path)
        if not input_files:
            print(f"[WARNING] No supported image files found in '{input_path}'.")
        matched_targets: set[Path] = set()
        for inp in input_files:
            tgt = find_matching_target(inp, available_targets)
            if tgt is not None and tgt in matched_targets:
                print(f"[WARNING] '{tgt.name}' matched multiple inputs — last match: '{inp.name}'.")
            if tgt is not None:
                matched_targets.add(tgt)
            _run_one(inp, tgt)
        unmatched = [t for t in available_targets if t not in matched_targets]
        if unmatched:
            print(f"[WARNING] {len(unmatched)} target file(s) had no matching input: "
                  + ", ".join(t.name for t in unmatched))
    else:
        print(f"No input file or directory at {input_path}")

    return EvaluationResult(evaluated, registry)


# ---------------------------------------------------------------------------
# CLI entry point — evaluate and write report (backward-compatible behaviour)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IQA metrics and write a report.")
    parser.add_argument("input", type=Path, help="Input image file or directory.")
    parser.add_argument(
        "target", type=Path, nargs="?", default=None,
        help="Optional reference image file or directory (omit for NR-only evaluation).",
    )
    parser.add_argument(
        "--mode", choices=["slice", "volume"], default="slice",
        help="Score every 2D slice separately (default) or each 3D volume once.",
    )
    args = parser.parse_args()

    # Reference usage: pick the metrics for this run. Swap BUILTIN_METRICS for
    # SEGMENTATION_METRICS (or any subset, e.g. MetricRegistry(PSNR, SSIM)) to
    # evaluate something else — each run owns its own registry.
    registry = MetricRegistry(*BUILTIN_METRICS)

    result = evaluate(args.input, args.target, registry=registry, mode=args.mode)
    report = result.generate_report(REPORT)
    print(report.describe())
    print(f"Report written: {REPORT}")


if "__main__" == __name__:
    main()
