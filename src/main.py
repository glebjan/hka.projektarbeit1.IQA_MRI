import argparse
from pathlib import Path
from typing import Optional

from constants import REPORT
from evaluation_result import EvaluationResult, _EvaluatedImage
from image_loader import ImageLoader, find_matching_target, list_images
from iqa_evaluator import IQAEvaluator
from metrics import DEVICE, Metric, MetricSpec, registry, register_metric  # noqa: F401 — re-exported for users

# ---------------------------------------------------------------------------
# Top-level evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    input_path: Path,
    target_path: Optional[Path] = None,
) -> EvaluationResult:
    """Discover input/target images and compute all IQA metrics.

    Args:
        input_path:  Path to an input image file or a directory of images.
        target_path: Optional path to a reference image file or directory.
                     Pass None for a no-reference (NR-only) evaluation.

    No files are written; use EvaluationResult.generate_report() for output.
    """
    evaluated: list[_EvaluatedImage] = []

    def _run_one(inp: Path, tgt: Optional[Path]) -> None:
        try:
            input_loader = ImageLoader(inp)
            input_loader.log_tensor_shape()
            target_loader: Optional[ImageLoader] = None
            if tgt is not None:
                target_loader = ImageLoader(tgt)
                target_loader.log_tensor_shape()
            records = IQAEvaluator(input_loader, target_loader).run_evaluation()
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

    return EvaluationResult(evaluated)


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
    args = parser.parse_args()
    result = evaluate(args.input, args.target)
    report = result.generate_report(REPORT)
    print(report.describe())
    print(f"Report written: {REPORT}")


if "__main__" == __name__:
    main()
