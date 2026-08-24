import numpy as np
import pytest

from segmentation_metrics.volume import as_mask


def test_as_mask_bool_passthrough():
    x = np.array([True, False, True])
    result = as_mask(x)
    assert result is x or np.array_equal(result, x)
    assert result.dtype == bool


def test_as_mask_float_thresholds():
    x = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    result = as_mask(x, threshold=0.5)
    np.testing.assert_array_equal(result, [False, False, True, True, True])


def test_as_mask_float_out_of_range_raises():
    x = np.array([0.2, 1.5, -0.1])
    with pytest.raises(ValueError):
        as_mask(x)


def test_as_mask_int_label_map():
    x = np.array([0, 1, 2, 1, 0])
    result = as_mask(x, label=1)
    np.testing.assert_array_equal(result, [False, True, False, True, False])


def test_as_mask_unsupported_dtype_raises():
    x = np.array(["a", "b"])
    with pytest.raises(TypeError):
        as_mask(x)
