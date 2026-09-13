"""Every metric module must be importable first, in a fresh interpreter.

Before metric_spec.py existed, metrics.py imported the segmentation modules
and dreamsim_metric at its bottom while they imported MetricSpec back from
metrics — whichever side was imported first hit the other mid-initialisation
(review 1.2). A fresh subprocess is the only honest test: the pytest process
has already imported everything.
"""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "iqaevaluator.segmentation_metrics.boundary_iou",
    "iqaevaluator.segmentation_metrics.monai_metrics",
    "iqaevaluator.segmentation_metrics.volume_metrics",
    "iqaevaluator.dreamsim_metric",
])
def test_module_imports_first_in_a_fresh_interpreter(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
