#!/usr/bin/env python3
"""Nested experiment for CV-selected iterations followed by full refitting.

No new folds are created. Within each outer fold, the other four immutable fold
IDs are rotated as inner validation folds. After selecting a tree count, the
model is refitted on all four outer-training folds before outer evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from scripts.train_archive_ensemble import load_saved_folds
from src.archive_ensemble.blend import disagreement_adjustment
from src.archive_ensemble.features import build_features
from src.archive_ensemble.models import Recipe, fit_model
from src.pipeline import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN, sha256_file


def load_train_only(path: Path, expected_hash: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and validate only labelled training data for this OOF-only experiment."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing organizer training file: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Unexpected content for train.csv: {actual_hash}")
    train = pd.read_csv(path, dtype={ID_COLUMN: "string"})
    expected_columns = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    if list(train.columns) != expected_columns:
        raise ValueError(f"Unexpected train schema: {list(train.columns)}")
    if train[ID_COLUMN].isna().any() or train[ID_COLUMN].duplicated().any():
        raise ValueError("Missing or duplicate training identifiers")
    if train[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("This version expects a complete organizer training matrix")
    if set(train[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training target must contain exactly classes 0 and 1")
    return train, train[TARGET_COLUMN].to_numpy(dtype=np.int8)


def policy_iterations(best: list[int]) -> dict[str, int]:
    values = np.asarray(best, dtype=float)
    return {
        "inner_median": max(1, int(round(np.median(values)))),
        "inner_mean": max(1, int(round(values.mean()))),
        "inner_p75": max(1, int(round(np.quantile(values, 0.75)))),
        "archive_fixed_437": 437,
    }


def main() -> int:
    experiment = json.loads((ROOT / "full_refit_experiment_config.json").read_text())
    ensemble = json.loads((ROOT / "ensemble_config.json").read_text())
    disagreement = json.loads((ROOT / "disagreement_config.json").read_text())
    submission_path = ROOT / "submission.csv"
    submission_hash_before = sha256_file(submission_path)
    train, y = load_train_only(
        ROOT / "data" / "raw" / "train.csv", ensemble["input_sha256"]["train.csv"]
    )
    fold_path = ROOT / ensemble["validation"]["folds_path"]
    if sha256_file(fold_path) != ensemble["validation"]["folds_sha256"]:
        raise ValueError("Saved folds changed")
    fold_id = load_saved_folds(fold_path, train, ensemble["validation"]["n_splits"])
    features = build_features(train, experiment["feature_mode"])
    spec = ensemble["models"][0]
    recipe = Recipe(spec["name"], spec["family"], spec["features"], dict(spec["params"]), False)
    policies = experiment["iteration_policies"]
    oof = {policy: np.full(len(y), np.nan) for policy in policies}
    inner_rows, outer_rows = [], []

    with threadpool_limits(limits=6):
        for outer_fold in range(ensemble["validation"]["n_splits"]):
            outer_train = fold_id != outer_fold
            outer_valid = fold_id == outer_fold
            inner_best = []
            for inner_fold in sorted(set(fold_id) - {outer_fold}):
                inner_train = outer_train & (fold_id != inner_fold)
                inner_valid = fold_id == inner_fold
                started = time.perf_counter()
                model, best_iteration = fit_model(
                    recipe,
                    features.loc[inner_train],
                    y[inner_train],
                    features.loc[inner_valid],
                    y[inner_valid],
                    seed=experiment["seed"],
                    threads=6,
                )
                inner_probability = model.predict(features.loc[inner_valid])
                inner_best.append(best_iteration)
                inner_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": int(inner_fold),
                        "training_rows": int(inner_train.sum()),
                        "validation_rows": int(inner_valid.sum()),
                        "best_iteration": best_iteration,
                        "inner_log_loss": log_loss(y[inner_valid], inner_probability),
                        "runtime_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    f"outer {outer_fold} inner {inner_fold}: best_iteration={best_iteration}",
                    flush=True,
                )

            selected = policy_iterations(inner_best)
            for policy in policies:
                started = time.perf_counter()
                model, fitted_iterations = fit_model(
                    recipe,
                    features.loc[outer_train],
                    y[outer_train],
                    seed=experiment["seed"],
                    threads=6,
                    iterations=selected[policy],
                )
                probability = model.predict(features.loc[outer_valid])
                oof[policy][outer_valid] = probability
                outer_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "policy": policy,
                        "inner_best_iterations": ",".join(map(str, inner_best)),
                        "selected_iterations": selected[policy],
                        "fitted_iterations": fitted_iterations,
                        "training_rows": int(outer_train.sum()),
                        "validation_rows": int(outer_valid.sum()),
                        "cat_log_loss": log_loss(y[outer_valid], probability),
                        "runtime_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    f"outer {outer_fold} {policy}: iterations={selected[policy]}, "
                    f"log_loss={outer_rows[-1]['cat_log_loss']:.9f}",
                    flush=True,
                )

    base = pd.read_csv(ROOT / "artifacts" / "archive_ensemble_v1" / "oof_predictions.csv")
    base_matrix = base[disagreement["model_order"]].to_numpy()
    current_final = disagreement_adjustment(
        base_matrix, disagreement["base_weights"], disagreement["theta"]
    )
    current_final_log_loss = log_loss(y, current_final)
    summary = []
    final_columns = {}
    for policy in policies:
        matrix = base_matrix.copy()
        matrix[:, 0] = oof[policy]
        final = disagreement_adjustment(matrix, disagreement["base_weights"], disagreement["theta"])
        final_columns[f"final__{policy}"] = final
        policy_outer = pd.DataFrame(outer_rows).query("policy == @policy")
        summary.append(
            {
                "policy": policy,
                "cat_oof_log_loss": log_loss(y, oof[policy]),
                "cat_oof_roc_auc": roc_auc_score(y, oof[policy]),
                "final_oof_log_loss": log_loss(y, final),
                "final_oof_roc_auc": roc_auc_score(y, final),
                "gain_vs_current_final": current_final_log_loss - log_loss(y, final),
                "fold_log_loss_std": policy_outer.cat_log_loss.std(ddof=1),
                "mean_selected_iterations": policy_outer.selected_iterations.mean(),
            }
        )

    # Translate the winning aggregation rule to the five original outer-fold
    # best iterations. This is the tree count a later full-data fit would use.
    original_metrics = pd.read_csv(ROOT / "artifacts" / "archive_ensemble_v1" / "fold_metrics.csv")
    original_best = original_metrics.query("candidate == 'cat_behavior_d5'").best_iteration.astype(int).tolist()
    full_data_iterations = policy_iterations(original_best)
    for row in summary:
        row["proposed_full_data_iterations"] = full_data_iterations[row["policy"]]

    output = ROOT / "artifacts" / "full_refit_policy_v1"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inner_rows).to_csv(output / "inner_fold_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(outer_rows).to_csv(output / "outer_fold_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(summary).sort_values("final_oof_log_loss").to_csv(
        output / "summary.csv", index=False, float_format="%.12g"
    )
    pd.DataFrame(
        {ID_COLUMN: train[ID_COLUMN], "target": y, "fold": fold_id, **oof, **final_columns}
    ).to_csv(output / "oof_predictions.csv", index=False, float_format="%.12g")
    submission_hash_after = sha256_file(submission_path)
    if submission_hash_after != submission_hash_before:
        raise RuntimeError("submission.csv changed during the isolated experiment")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "current_final_oof_log_loss": current_final_log_loss,
                "current_submission_sha256": submission_hash_after,
                "version": experiment["version"],
                "protocol": experiment["protocol"],
                "original_outer_best_iterations": original_best,
                "proposed_full_data_iterations": full_data_iterations,
                "saved_folds_sha256": sha256_file(fold_path),
                "test_data_accessed": False,
                "submission_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
