"""EvaluationResult — holds computed records, owns all output operations."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from constants import REPORT
from metrics import MetricRegistry
from records import ImageEvaluatorRecord

_SLICE_SUFFIX = re.compile(r"_s\d+$")


@dataclass
class _EvaluatedImage:
    input_path: Path
    records:    list[ImageEvaluatorRecord]


class EvaluationResult:
    """Container for the results of one evaluation run.

    Normal use (notebook, no file I/O):
        result = evaluate(INPUT, TARGET, registry=registry)
        df     = result.to_frame()

    Optional report output:
        result.generate_report(Path("report/my_output.csv"))

    Volume-level numbers from a per-slice run:
        result.aggregate_volumes()
    """

    def __init__(self, images: list[_EvaluatedImage], registry: MetricRegistry):
        self._images   = images
        self._registry = registry

    # ------------------------------------------------------------------
    # Pure data access — no file I/O
    # ------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return all records as a DataFrame.  Nothing is written to disk."""
        rows = [record.to_dict() for img in self._images for record in img.records]
        fixed_columns = [k for k in ImageEvaluatorRecord.__annotations__ if k != "extra"]
        extra_columns = [spec.name for spec in self._registry.specs if not spec.builtin]
        return pd.DataFrame(rows, columns=fixed_columns + extra_columns)

    def aggregate_volumes(self) -> pd.DataFrame:
        """Volume-level numbers from a per-slice run, computed correctly.

        Averaging per-slice scores is wrong for every ratio metric: the mean of
        per-slice ratios is not the ratio of the summed volume, and a slice
        holding three voxels would count as much as one holding three thousand.
        This method sums the voxel counts per volume first and divides once.

        Only metrics that can be rebuilt from per-slice counts appear: `dice`,
        `vs` and `vs_signed`. Distance and perceptual metrics (hausdorff95,
        assd, nsd, panoptic_quality, boundary_iou, lpips, ...) cannot be
        reconstructed from per-slice values at all and are deliberately left
        out rather than averaged — run again with mode="volume" for those.

        Returns:
            DataFrame indexed by volume id (the slice id without its `_sNNN`
            suffix) with columns v_pred, v_gt, tp, dice, vs, vs_signed. Ratio
            columns are NaN where the denominator is 0.

        Raises:
            ValueError: if the run was already scored per volume, or if the
                voxel-count metrics were not registered.
        """
        df = self.to_frame()

        if not df.empty and (df["scoring"] == "volume").any():
            raise ValueError(
                "This report was already scored one row per volume, so there is "
                "nothing left to aggregate. Use to_frame() to read it."
            )

        missing = [c for c in ("v_pred", "v_gt", "tp") if c not in df.columns]
        if missing:
            raise ValueError(
                "Volume-level numbers are built from the voxel counts v_pred, "
                f"v_gt and tp, and this report has no {', '.join(missing)} "
                "column. Add the counting metrics to the registry and run "
                "again: MetricRegistry(..., V_PRED, V_GT, TP)."
            )

        counts = df[["image_id", "v_pred", "v_gt", "tp"]].copy()
        counts["image_id"] = counts["image_id"].str.replace(_SLICE_SUFFIX, "", regex=True)
        grouped = counts.groupby("image_id")[["v_pred", "v_gt", "tp"]].sum(min_count=1)

        v_pred_sum = grouped["v_pred"].astype(float)
        v_gt_sum   = grouped["v_gt"].astype(float)
        tp_sum     = grouped["tp"].astype(float)
        denom      = v_pred_sum + v_gt_sum

        grouped["dice"]      = np.where(denom == 0, np.nan, 2.0 * tp_sum / denom)
        grouped["vs"]        = np.where(denom == 0, np.nan,
                                        1.0 - (v_pred_sum - v_gt_sum).abs() / denom)
        grouped["vs_signed"] = np.where(denom == 0, np.nan,
                                        2.0 * (v_pred_sum - v_gt_sum) / denom)
        return grouped

    # ------------------------------------------------------------------
    # Optional output
    # ------------------------------------------------------------------

    def generate_report(self, report_path: Path = REPORT) -> pd.DataFrame:
        """Write the CSV report, then return the DataFrame.

        Args:
            report_path: Destination for the CSV file.

        Returns:
            The same DataFrame that to_frame() would return.
        """
        df = self.to_frame()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
        print(f"CSV written: {report_path}")
        return df
