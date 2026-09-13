"""Spec D3 — the empty-mask policy table, every cell, every scored metric, both modes.

Fixture F3: volume (4, 8, 8); the populated side is the block [1:3, 2:5, 2:5]
(18 voxels). In slice mode slices 0 and 3 are empty on both sides at the
metric level; the evaluator would skip them, the metric itself answers None.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from iqaevaluator.metrics import MetricRegistry, SEGMENTATION_METRICS
from iqaevaluator.segmentation_metrics.boundary_iou import BOUNDARY_IOU
from iqaevaluator.segmentation_metrics.monai_metrics import ASSD, DICE, HAUSDORFF95, NSD, PANOPTIC_QUALITY
from iqaevaluator.segmentation_metrics.volume_metrics import TP, V_GT, V_PRED, VS, VS_SIGNED

SCORED = [DICE, HAUSDORFF95, NSD, ASSD, PANOPTIC_QUALITY, BOUNDARY_IOU, VS]
ONE_SIDED = {
    "dice": 0.0, "vs": 0.0, "nsd": 0.0, "boundary_iou": 0.0, "panoptic_quality": 0.0,
    "hausdorff95": None, "assd": None,
}
ISO = (1.0, 1.0, 1.0)


def _f3(case: str) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.zeros(1, 1, 4, 8, 8)
    block[..., 1:3, 2:5, 2:5] = 1.0
    empty = torch.zeros_like(block)
    return {"gt_empty": (block, empty), "pred_empty": (empty, block), "both_empty": (empty, empty)}[case]


@pytest.mark.parametrize("case", ["gt_empty", "pred_empty", "both_empty"])
@pytest.mark.parametrize("spec", SCORED, ids=lambda s: s.name)
def test_policy_table_in_volume_mode(spec, case):
    pred, gt = _f3(case)
    metric = MetricRegistry(spec).get_metric(spec.name, "volume", ISO)
    expected = None if case == "both_empty" else ONE_SIDED[spec.name]
    assert metric(pred, gt) == [expected]


@pytest.mark.parametrize("case", ["gt_empty", "pred_empty", "both_empty"])
@pytest.mark.parametrize("spec", SCORED, ids=lambda s: s.name)
def test_policy_table_in_slice_mode(spec, case):
    pred5, gt5 = _f3(case)
    pred, gt = pred5[0].permute(1, 0, 2, 3), gt5[0].permute(1, 0, 2, 3)     # (4, 1, 8, 8)
    metric = MetricRegistry(spec).get_metric(spec.name)
    one = None if case == "both_empty" else ONE_SIDED[spec.name]
    assert metric(pred, gt) == [None, one, one, None]


def test_counts_and_vs_signed_have_no_policy():
    pred, gt = _f3("gt_empty")
    registry = MetricRegistry(V_PRED, V_GT, TP, VS_SIGNED)
    value = lambda name, p, g: registry.get_metric(name, "volume", ISO)(p, g)[0]
    assert (value("v_pred", pred, gt), value("v_gt", pred, gt), value("tp", pred, gt)) == (18.0, 0.0, 0.0)
    assert value("vs_signed", pred, gt) == pytest.approx(2.0)
    empty, _ = _f3("both_empty")
    assert value("v_pred", empty, empty) == 0.0
    assert value("vs_signed", empty, empty) is None


@pytest.mark.parametrize("spec", SEGMENTATION_METRICS, ids=lambda s: s.name)
def test_every_adapter_rejects_a_non_binary_tensor_naming_mask(spec):
    half = torch.full((1, 1, 8, 8), 0.5)
    with pytest.raises(ValueError, match=r"Mask\(\)"):
        MetricRegistry(spec).get_metric(spec.name)(half, half)


def test_a_report_with_one_sided_empty_volumes_holds_no_inf(tmp_path):
    import nibabel as nib
    from main import evaluate
    from iqaevaluator.normalization import Mask

    gt = np.zeros((16, 16, 6), dtype=np.uint8)
    gt[4:12, 4:12, 1:5] = 1
    pred_path, gt_path = tmp_path / "case_pred.nii.gz", tmp_path / "case_gt.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros_like(gt), np.eye(4)), pred_path)
    nib.save(nib.Nifti1Image(gt, np.eye(4)), gt_path)

    df = evaluate(pred_path, gt_path, registry=MetricRegistry(*SEGMENTATION_METRICS),
                  mode="volume", normalization=Mask()).to_frame()
    numeric = df.select_dtypes(include="number").to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    row = df.iloc[0]
    assert (row["dice"], row["nsd"], row["boundary_iou"], row["vs"], row["panoptic_quality"]) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert pd.isna(row["hausdorff95"]) and pd.isna(row["assd"])
    assert (row["v_pred"], row["v_gt"], row["tp"]) == (0.0, 256.0, 0.0)
    assert row["vs_signed"] == pytest.approx(-2.0)
