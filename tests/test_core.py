import numpy as np
from synthetic_dataset_utils import (
    generate_linear_regression,
    summary_statistics,
    mse,
    mae,
    ks_statistic,
)

def test_linear_regression_shape():
    df = generate_linear_regression(n=100)
    assert len(df) == 100
    assert "x" in df and "y" in df

def test_summary_statistics():
    stats = summary_statistics([1, 2, 3, 4])
    assert stats["mean"] == 2.5
    assert stats["count"] == 4

def test_mse_mae():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])
    assert mse(y_true, y_pred) > 0
    assert mae(y_true, y_pred) > 0

def test_ks_statistic_zero():
    a = np.random.normal(0, 1, 1000)
    assert ks_statistic(a, a) == 0.0
