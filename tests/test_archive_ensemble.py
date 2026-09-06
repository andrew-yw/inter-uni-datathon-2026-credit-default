"""Focused safety tests for the clean archive-ensemble reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_archive_ensemble import fit_convex_weights, load_saved_folds
from src.archive_ensemble.data import FEATURES
from src.archive_ensemble.features import build_features, safe_ratio


def example_rows() -> pd.DataFrame:
    row = {name: 0 for name in FEATURES}
    row.update(
        {
            "LIMIT_BAL": 100_000,
            "SEX": 2,
            "EDUCATION": 3,
            "MARRIAGE": 1,
            "AGE": 42,
            "PAY_0": 2,
            "PAY_2": 1,
            "BILL_AMT1": 80_000,
            "BILL_AMT2": 75_000,
            "PAY_AMT1": 2_000,
        }
    )
    return pd.DataFrame([row, {**row, "PAY_0": 0, "LIMIT_BAL": 0}])


def test_safe_ratio_handles_zero_and_negative_denominators() -> None:
    value = safe_ratio(np.array([10.0, 10.0, 10.0]), np.array([0.0, -5.0, 5.0]))
    np.testing.assert_allclose(value, [0.1, -0.1, 0.1])
    assert np.isfinite(value).all()


def test_archive_feature_mode_is_row_local_and_finite() -> None:
    raw = example_rows()
    result = build_features(raw, "v2:engineered:status,repayment,cashflow,utilization")
    assert len(result) == len(raw)
    assert "v2_late1_recent_run" in result
    assert "v2_paid_under_5pct_count_5" in result
    assert "v2_utilization_recent2_mean" in result
    assert "default" not in result and "client_id" not in result
    assert np.isfinite(result.to_numpy()).all()


def test_convex_weight_fit_returns_probability_mixture() -> None:
    matrix = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6], [0.2, 0.3, 0.4], [0.9, 0.8, 0.7]])
    weights = fit_convex_weights(matrix, np.array([0, 1, 0, 1]))
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all((matrix @ weights > 0) & (matrix @ weights < 1))


def test_saved_folds_keep_duplicate_profiles_together(tmp_path) -> None:
    raw = example_rows()
    train = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    train.insert(0, "client_id", ["a", "b", "c"])
    saved = pd.DataFrame({"client_id": ["a", "b", "c"], "fold": [0, 1, 0]})
    path = tmp_path / "folds.csv"
    saved.to_csv(path, index=False)
    folds = load_saved_folds(path, train, 2)
    assert folds.tolist() == [0, 1, 0]
