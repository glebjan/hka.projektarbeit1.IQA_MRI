"""Render docs/calibration/segmentation.md from the calibration cases.

Run with `hatch run calibration:report`. Needs the [calibration] extras for
the official columns; a missing package shows up as "n/a", never as a
silent blank. One table per metric: case | hand value | derivation |
official (package@version, variant) | framework | Δ | divergent variants.
"""
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))          # `calibration.*` when run as a script

from calibration.cases import CASES, CalibrationCase                      # noqa: E402
from calibration.harness import framework_value                          # noqa: E402
from calibration.monai_cases import MONAI_CASES, MONAI_VERSION           # noqa: E402
from calibration.official import divergent_values, official_value, reference_versions  # noqa: E402

OUT = ROOT / "docs" / "calibration" / "segmentation.md"
ORDER = ["dice", "hausdorff95", "assd", "nsd", "panoptic_quality", "boundary_iou", "vs"]
VARIANT = {
    "dice": "MedPy `dc` — voxel Dice",
    "hausdorff95": "MedPy `hd` — max over directed maxima (only at percentile=None)",
    "assd": "MedPy `assd`/`asd` — voxel surfaces, cross-structure erosion",
    "nsd": "— (author implementation `surface-distance` is area-weighted, see ¹)",
    "panoptic_quality": "panopticapi `pq_compute_single_core` — background as stuff, things-only PQ",
    "boundary_iou": "boundary-iou-api `mask_to_boundary` (vendored) — 2D only",
    "vs": "— (no reference implementation; counts-based)",
}


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.6f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _framework_versions() -> dict[str, str]:
    import monai, numpy, pyiqa, scipy, torch
    return {"monai": monai.__version__, "pyiqa": pyiqa.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "numpy": numpy.__version__}


def _official_text(case: CalibrationCase, tmp: Path) -> str:
    try:
        value = official_value(case, tmp)
    except ImportError as exc:
        return f"n/a ({exc.name} not installed)"
    if value is None:
        return "— (no same-variant implementation)"
    return f"{_fmt(value)} — {VARIANT.get(case.metric, '')}"


def _divergent_text(case: CalibrationCase, tmp: Path) -> str:
    try:
        values = divergent_values(case, tmp)
    except ImportError as exc:
        return f"n/a ({exc.name} not installed)"
    return "<br>".join(f"{k}: {_fmt(v)}" for k, v in values.items()) or "—"


def render(cases: list[CalibrationCase] = CASES, monai_cases: list[CalibrationCase] = MONAI_CASES) -> str:
    versions = ", ".join(f"{k} {v}" for k, v in _framework_versions().items())
    refs = ", ".join(f"{k} {v}" for k, v in reference_versions().items())
    lines = [
        "# Segmentation-metric calibration report",
        "",
        f"Generated {date.today().isoformat()} by `hatch run calibration:report` at framework commit `{_git_sha()}`.",
        f"Framework stack: {versions}. Reference packages: {refs}.",
        "",
        "Authority, in order: (1) the metric's definition applied by hand to a fixture small enough for the derivation "
        "to fit in a few lines — the derivation is what a reviewer checks; (2) the implementation the community cites, "
        "run on the same fixture with the same definitional variant; (3) MONAI's own test cases through this framework's "
        "loader and adapters. Δ = framework − hand value. Every framework value went through the public path "
        "(NIfTI → `load_pair` → `MetricRegistry` → evaluator → record).",
        "",
    ]
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for metric in ORDER:
            hand = [c for c in cases if c.metric == metric and c.kind == "hand"]
            policy = [c for c in cases if c.metric == metric and c.kind == "policy"]
            random_cases = [c for c in cases if c.metric == metric and c.kind == "random"]
            monai = [c for c in monai_cases if c.metric == metric]
            if not (hand or policy or random_cases or monai):
                continue
            lines += [f"## {metric}", ""]
            if hand:
                lines += ["| case | hand value | derivation | official (package@version, variant) | framework | Δ | divergent variants¹ |",
                          "|---|---|---|---|---|---|---|"]
                for c in hand:
                    fw = framework_value(c, tmp)
                    delta = "—" if fw is None else f"{fw - c.expected:+.2e}"
                    lines.append(f"| {c.name} | {_fmt(c.expected)} | {_cell(c.derivation)}<br>*{_cell(c.source)}* | "
                                 f"{_official_text(c, tmp)} | {_fmt(fw)} | {delta} | {_divergent_text(c, tmp)} |")
                lines.append("")
            if policy:
                lines += ["Empty-mask policy (spec D3):", "", "| case | expected | framework |", "|---|---|---|"]
                for c in policy:
                    lines.append(f"| {c.name} | {_fmt(c.expected)} | {_fmt(framework_value(c, tmp))} |")
                lines.append("")
            if random_cases:
                lines += ["Seeded random blobs — framework vs MedPy (same variant):", "",
                          "| case | spacing | framework | MedPy | Δ |", "|---|---|---|---|---|"]
                for c in random_cases:
                    fw = framework_value(c, tmp)
                    try:
                        off = official_value(c, tmp)
                    except ImportError:
                        off = None
                    delta = "—" if fw is None or off is None else f"{fw - off:+.2e}"
                    lines.append(f"| {c.name} | {c.spacing} | {_fmt(fw)} | {_fmt(off)} | {delta} |")
                lines.append("")
            if monai:
                lines += [f"MONAI {MONAI_VERSION} test cases through `Mask()`/`Raw()` and the adapters:", "",
                          "| case | MONAI expected | framework | source |", "|---|---|---|---|"]
                for c in monai:
                    lines.append(f"| {c.name} | {_fmt(c.expected)} | {_fmt(framework_value(c, tmp))} | {_cell(c.source)} |")
                lines.append("")
    lines += [
        "¹ The same metric under a different definitional variant (spec D10), reported for orientation and never "
        "used in an equality assertion: MedPy `hd95` takes the 95th percentile over the *concatenated* directed "
        "distances (MONAI/this framework: max of the two directed percentiles); DeepMind's `surface-distance` weights "
        "surface elements by area (MONAI/this framework: one surface voxel = one element). Boundary IoU's 3D physical "
        "band is this framework's extension of Cheng et al. and has a hand value only.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
