#!/usr/bin/env python3
"""Evaluate additional clean boosting candidates on the immutable saved folds.

This is a development script: it writes OOF predictions and fold metrics but
does not replace the final submission or fit on test data. Candidate comparison
therefore remains isolated from test inference and leaderboard feedback.
"""

from __future__ import annotations

import argparse
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
from src.archive_ensemble.features import build_features
from src.archive_ensemble.models import Recipe, fit_model
from src.pipeline import ID_COLUMN, load_competition_data, sha256_file


BEHAVIOR = "v2:engineered:status,repayment,cashflow,utilization"


def candidates() -> list[Recipe]:
    """Small, hypothesis-driven search rather than a large tuning sweep."""

    lgb_base = {
        "learning_rate": 0.025,
        "max_depth": -1,
        "reg_lambda": 10.0,
        "reg_alpha": 0.2,
        "colsample_bytree": 0.85,
        "subsample": 0.85,
        "subsample_freq": 1,
        "max_bin": 127,
        "n_estimators": 4000,
    }
    xgb_base = {
        "learning_rate": 0.03,
        "min_child_weight": 15,
        "reg_lambda": 15.0,
        "reg_alpha": 0.2,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "max_bin": 128,
        "n_estimators": 4000,
    }
    cat_base = {
        "learning_rate": 0.035,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
        "iterations": 3500,
    }
    return [
        Recipe("lgb_compact_l15", "lightgbm", "compact", {**lgb_base, "num_leaves": 15, "min_child_samples": 100}),
        Recipe("lgb_engineered_l15", "lightgbm", "engineered", {**lgb_base, "num_leaves": 15, "min_child_samples": 100}),
        Recipe("lgb_behavior_l15", "lightgbm", BEHAVIOR, {**lgb_base, "num_leaves": 15, "min_child_samples": 100}),
        Recipe("xgb_compact_d3", "xgboost", "compact", {**xgb_base, "max_depth": 3}),
        Recipe("xgb_engineered_d3", "xgboost", "engineered", {**xgb_base, "max_depth": 3}),
        Recipe("xgb_behavior_d3", "xgboost", BEHAVIOR, {**xgb_base, "max_depth": 3}),
        Recipe("cat_behavior_d4", "catboost", BEHAVIOR, {**cat_base, "depth": 4}),
        Recipe("cat_behavior_d6", "catboost", BEHAVIOR, {**cat_base, "depth": 6}),
        Recipe(
            "cat_compact_ordered",
            "catboost",
            "compact",
            {
                **cat_base,
                "depth": 4,
                "learning_rate": 0.045,
                "l2_leaf_reg": 8.0,
                "boosting_type": "Ordered",
                "bagging_temperature": 0.4,
            },
        ),
        Recipe(
            "cat_compact_status_categories",
            "catboost",
            "compact",
            {**cat_base, "depth": 5, "l2_leaf_reg": 8.0, "random_strength": 0.7},
            status_categories=True,
        ),
        Recipe(
            "cat_behavior_status_categories",
            "catboost",
            BEHAVIOR,
            {**cat_base, "depth": 5, "l2_leaf_reg": 8.0, "random_strength": 0.7},
            status_categories=True,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "boosting_research_v1")
    parser.add_argument("--only", help="Comma-separated candidate names; existing results are retained")
    args = parser.parse_args()

    config = json.loads((ROOT / "ensemble_config.json").read_text())
    data = load_competition_data(ROOT / "data" / "raw", config["input_sha256"])
    fold_path = ROOT / config["validation"]["folds_path"]
    if sha256_file(fold_path) != config["validation"]["folds_sha256"]:
        raise ValueError("Saved fold hash differs from ensemble_config.json")
    fold_id = load_saved_folds(fold_path, data.train, config["validation"]["n_splits"])
    all_recipes = candidates()
    requested = set(args.only.split(",")) if args.only else {r.name for r in all_recipes}
    unknown = requested - {r.name for r in all_recipes}
    if unknown:
        raise ValueError(f"Unknown candidates: {sorted(unknown)}")
    recipes = [recipe for recipe in all_recipes if recipe.name in requested]
    cache = {mode: build_features(data.train, mode) for mode in sorted({r.features for r in recipes})}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "oof_predictions.csv"
    metrics_path = args.output_dir / "fold_metrics.csv"
    prediction_columns = {}
    if prediction_path.exists():
        previous = pd.read_csv(prediction_path)
        prediction_columns = {
            name: previous[name].to_numpy()
            for name in previous.columns
            if name not in (ID_COLUMN, "target", "fold") and name not in requested
        }
    rows = []
    if metrics_path.exists():
        previous_metrics = pd.read_csv(metrics_path)
        rows = previous_metrics.loc[~previous_metrics.candidate.isin(requested)].to_dict("records")

    with threadpool_limits(limits=args.threads):
        for recipe in recipes:
            feature = cache[recipe.features]
            oof = np.full(len(data.y), np.nan)
            for fold in range(config["validation"]["n_splits"]):
                train_rows, valid_rows = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
                started = time.perf_counter()
                model, iteration = fit_model(
                    recipe,
                    feature.iloc[train_rows],
                    data.y[train_rows],
                    feature.iloc[valid_rows],
                    data.y[valid_rows],
                    seed=config["validation"]["seed"],
                    threads=args.threads,
                )
                probability = model.predict(feature.iloc[valid_rows])
                oof[valid_rows] = probability
                rows.append(
                    {
                        "candidate": recipe.name,
                        "fold": fold,
                        "log_loss": log_loss(data.y[valid_rows], probability),
                        "roc_auc": roc_auc_score(data.y[valid_rows], probability),
                        "best_iteration": iteration,
                        "runtime_seconds": time.perf_counter() - started,
                    }
                )
                print(f"{recipe.name} fold {fold}: {rows[-1]['log_loss']:.9f}", flush=True)
            prediction_columns[recipe.name] = oof
            print(f"{recipe.name} overall: {log_loss(data.y, oof):.9f}", flush=True)

    predictions = pd.DataFrame(
        {ID_COLUMN: data.train[ID_COLUMN], "target": data.y, "fold": fold_id, **prediction_columns}
    )
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False, float_format="%.12g")
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False, float_format="%.12g")
    overall = []
    for name, probability in prediction_columns.items():
        subset = fold_metrics[fold_metrics.candidate == name]
        overall.append(
            {
                "candidate": name,
                "log_loss": log_loss(data.y, probability),
                "roc_auc": roc_auc_score(data.y, probability),
                "fold_log_loss_std": subset.log_loss.std(ddof=1),
                "runtime_seconds": subset.runtime_seconds.sum(),
            }
        )
    pd.DataFrame(overall).sort_values("log_loss").to_csv(
        args.output_dir / "overall_metrics.csv", index=False, float_format="%.12g"
    )
    (args.output_dir / "recipes.json").write_text(
        json.dumps([recipe.to_dict() for recipe in all_recipes], indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
