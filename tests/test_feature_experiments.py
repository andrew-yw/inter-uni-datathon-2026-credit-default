"""Safety and row-locality tests for experimental feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive_ensemble.data import FEATURES
from src.feature_experiments import BLOCKS, build_experiment_features


def example_frame() -> pd.DataFrame:
    rows = []
    for offset in (0, 1):
        row = {name: 0 for name in FEATURES}
        row.update(
            {
                "LIMIT_BAL": 100_000 + offset * 50_000,
                "SEX": 1 + offset,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 34 + offset * 20,
                "PAY_0": 2 - offset,
                "PAY_2": 1,
                "BILL_AMT1": 90_000,
                "BILL_AMT2": 80_000,
                "PAY_AMT1": 2_000,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_all_experimental_blocks_are_finite_and_unique() -> None:
    result = build_experiment_features(example_frame(), BLOCKS)
    assert result.columns.is_unique
    assert np.isfinite(result.to_numpy()).all()
    assert "exp_late2_longest_run" in result
    assert "cat__exp_status_util_band" in result


def test_features_are_row_local() -> None:
    frame = example_frame()
    together = build_experiment_features(frame, ("trajectory", "coverage", "trends", "bands"))
    alone = build_experiment_features(frame.iloc[[0]].copy(), ("trajectory", "coverage", "trends", "bands"))
    np.testing.assert_allclose(together.iloc[0].to_numpy(), alone.iloc[0].to_numpy())


def test_unknown_block_is_rejected() -> None:
    try:
        build_experiment_features(example_frame(), ("unsupported",))
    except ValueError as error:
        assert "Unknown" in str(error)
    else:
        raise AssertionError("Unknown feature blocks must not be accepted")
