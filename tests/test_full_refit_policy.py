"""Unit tests for the isolated full-refit iteration-policy experiment."""

import pandas as pd

from scripts.research_full_refit_policy import load_train_only, policy_iterations
from src.pipeline import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN, sha256_file


def test_policy_iterations_uses_documented_aggregations() -> None:
    selected = policy_iterations([100, 200, 300, 400])

    assert selected == {
        "inner_median": 250,
        "inner_mean": 250,
        "inner_p75": 325,
        "archive_fixed_437": 437,
    }


def test_policy_iterations_never_returns_zero_trees() -> None:
    selected = policy_iterations([0, 0, 0, 0])

    assert selected["inner_median"] == 1
    assert selected["inner_mean"] == 1
    assert selected["inner_p75"] == 1


def test_train_only_loader_validates_the_labelled_file(tmp_path) -> None:
    rows = []
    for index, target in enumerate((0, 1)):
        row = {ID_COLUMN: f"client-{index}"}
        row.update({column: index for column in FEATURE_COLUMNS})
        row[TARGET_COLUMN] = target
        rows.append(row)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    train, target = load_train_only(path, sha256_file(path))

    assert list(train.columns) == [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    assert target.tolist() == [0, 1]
