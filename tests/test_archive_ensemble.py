"""Focused safety tests for the clean archive-ensemble reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_disagreement_submission import validate_retrained_oof
from scripts.train_archive_ensemble import fit_convex_weights, load_saved_folds, path_for_manifest
from src.archive_ensemble.blend import disagreement_adjustment, fixed_probability_blend
from src.archive_ensemble.data import FEATURES
from src.archive_ensemble.features import build_features, safe_ratio
from src.pipeline import sha256_file


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


def test_disagreement_blend_returns_finite_probabilities() -> None:
    matrix = np.array([[0.1, 0.15, 0.2], [0.7, 0.8, 0.6]])
    weights = [0.5, 0.3, 0.2]
    base = fixed_probability_blend(matrix, weights)
    adjusted = disagreement_adjustment(matrix, weights, np.zeros(8))
    np.testing.assert_allclose(adjusted, base)
    assert np.all((adjusted > 0) & (adjusted < 1))


def test_manifest_path_supports_isolated_output(tmp_path) -> None:
    external = tmp_path / "submission.csv"
    assert path_for_manifest(external) == str(external.resolve())


def test_retrained_oof_allows_only_documented_numeric_replay_noise(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    reference = pd.DataFrame(
        {
            "client_id": ["a", "b"],
            "target": [0, 1],
            "fold": [0, 1],
            "model": [0.2, 0.8],
        }
    )
    candidate = reference.copy()
    candidate.loc[0, "model"] += 1e-13
    reference.to_csv(reference_path, index=False)
    candidate.to_csv(candidate_path, index=False)

    maximum = validate_retrained_oof(
        candidate_path,
        reference_path,
        sha256_file(reference_path),
        tolerance=2e-12,
    )

    assert 0 < maximum <= 2e-12

    candidate.loc[0, "model"] += 1e-3
    candidate.to_csv(candidate_path, index=False)
    with pytest.raises(ValueError, match="drifted"):
        validate_retrained_oof(
            candidate_path,
            reference_path,
            sha256_file(reference_path),
            tolerance=2e-12,
        )
