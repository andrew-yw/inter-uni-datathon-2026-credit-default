#!/usr/bin/env python3
"""Measure whether seed averaging stabilizes the tree ensemble on saved folds."""

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


def recipes() -> dict[str, Recipe]:
    cat = {
        "iterations": 3500,
        "depth": 5,
        "learning_rate": 0.035,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
    }
    status_cat = {**cat, "l2_leaf_reg": 8.0, "random_strength": 0.7}
    rf = {
        "n_estimators": 400,
        "criterion": "gini",
        "min_samples_leaf": 30,
        "max_features": 0.5,
        "max_depth": None,
        "bootstrap": True,
        "max_samples": 0.85,
        "class_weight": None,
    }
    return {
        "cat_behavior_d5": Recipe("cat_behavior_d5", "catboost", BEHAVIOR, cat),
        "cat_behavior_status_categories": Recipe(
            "cat_behavior_status_categories", "catboost", BEHAVIOR, status_cat, status_categories=True
        ),
        "rf_behavior_leaf30": Recipe("rf_behavior_leaf30", "random_forest", BEHAVIOR, rf),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="cat_behavior_d5,cat_behavior_status_categories")
    parser.add_argument("--seeds", default="137,4099,7919")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "seed_bagging_v1")
    args = parser.parse_args()

    config = json.loads((ROOT / "ensemble_config.json").read_text())
    data = load_competition_data(ROOT / "data" / "raw", config["input_sha256"])
    fold_path = ROOT / config["validation"]["folds_path"]
    if sha256_file(fold_path) != config["validation"]["folds_sha256"]:
        raise ValueError("Saved fold hash differs from ensemble_config.json")
    fold_id = load_saved_folds(fold_path, data.train, config["validation"]["n_splits"])
    available = recipes()
    model_names, seeds = args.models.split(","), [int(value) for value in args.seeds.split(",")]
    unknown = set(model_names) - set(available)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, metric_path = args.output_dir / "oof_predictions.csv", args.output_dir / "fold_metrics.csv"
    prior_predictions = pd.read_csv(prediction_path) if prediction_path.exists() else None
    columns = {
        name: prior_predictions[name].to_numpy()
        for name in ([] if prior_predictions is None else prior_predictions.columns)
        if name not in (ID_COLUMN, "target", "fold")
    }
    rows = pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else []
    requested_columns = {f"{name}__seed{seed}" for name in model_names for seed in seeds}
    rows = [row for row in rows if row["candidate"] not in requested_columns]
    feature = build_features(data.train, BEHAVIOR)

    with threadpool_limits(limits=args.threads):
        for model_name in model_names:
            recipe = available[model_name]
            for seed in seeds:
                candidate = f"{model_name}__seed{seed}"
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
                        seed=seed,
                        threads=args.threads,
                    )
                    probability = model.predict(feature.iloc[valid_rows])
                    oof[valid_rows] = probability
                    rows.append(
                        {
                            "candidate": candidate,
                            "fold": fold,
                            "log_loss": log_loss(data.y[valid_rows], probability),
                            "roc_auc": roc_auc_score(data.y[valid_rows], probability),
                            "best_iteration": iteration,
                            "runtime_seconds": time.perf_counter() - started,
                        }
                    )
                    print(f"{candidate} fold {fold}: {rows[-1]['log_loss']:.9f}", flush=True)
                columns[candidate] = oof
                print(f"{candidate} overall: {log_loss(data.y, oof):.9f}", flush=True)

    pd.DataFrame({ID_COLUMN: data.train[ID_COLUMN], "target": data.y, "fold": fold_id, **columns}).to_csv(
        prediction_path, index=False, float_format="%.12g"
    )
    pd.DataFrame(rows).to_csv(metric_path, index=False, float_format="%.12g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
