#!/usr/bin/env python3
"""Run controlled feature-block ablations on immutable saved folds."""

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
from src.archive_ensemble.blend import disagreement_adjustment
from src.archive_ensemble.models import Recipe, fit_model
from src.feature_experiments import BASE_MODE, build_experiment_features
from src.pipeline import ID_COLUMN, load_competition_data, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "feature_engineering_v1")
    args = parser.parse_args()

    experiment = json.loads((ROOT / "feature_experiment_config.json").read_text())
    ensemble = json.loads((ROOT / "ensemble_config.json").read_text())
    disagreement = json.loads((ROOT / "disagreement_config.json").read_text())
    data = load_competition_data(ROOT / "data" / "raw", ensemble["input_sha256"])
    fold_path = ROOT / ensemble["validation"]["folds_path"]
    if sha256_file(fold_path) != ensemble["validation"]["folds_sha256"]:
        raise ValueError("Saved folds changed")
    fold_id = load_saved_folds(fold_path, data.train, ensemble["validation"]["n_splits"])
    base_oof = pd.read_csv(ROOT / "artifacts" / "archive_ensemble_v1" / "oof_predictions.csv")
    if base_oof[ID_COLUMN].tolist() != data.train[ID_COLUMN].tolist():
        raise ValueError("Base OOF rows do not align with training data")
    base_matrix = base_oof[disagreement["model_order"]].to_numpy()
    current_final = disagreement_adjustment(
        base_matrix, disagreement["base_weights"], disagreement["theta"], experiment["probability_clip"]
    )

    cat_params = dict(ensemble["models"][0]["params"])
    recipe = Recipe("feature_experiment_cat", "catboost", BASE_MODE, cat_params, status_categories=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_columns, fold_rows, overall_rows = {}, [], []

    with threadpool_limits(limits=args.threads):
        for candidate, blocks in experiment["candidates"].items():
            features = build_experiment_features(data.train, tuple(blocks))
            oof = np.full(len(data.y), np.nan)
            candidate_started = time.perf_counter()
            for fold in range(ensemble["validation"]["n_splits"]):
                train_rows, valid_rows = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
                fold_started = time.perf_counter()
                model, iteration = fit_model(
                    recipe,
                    features.iloc[train_rows],
                    data.y[train_rows],
                    features.iloc[valid_rows],
                    data.y[valid_rows],
                    seed=experiment["seed"],
                    threads=args.threads,
                )
                probability = model.predict(features.iloc[valid_rows])
                oof[valid_rows] = probability
                fold_rows.append(
                    {
                        "candidate": candidate,
                        "fold": fold,
                        "feature_count": features.shape[1],
                        "best_iteration": iteration,
                        "cat_log_loss": log_loss(data.y[valid_rows], probability),
                        "cat_roc_auc": roc_auc_score(data.y[valid_rows], probability),
                        "runtime_seconds": time.perf_counter() - fold_started,
                    }
                )
                print(f"{candidate} fold {fold}: {fold_rows[-1]['cat_log_loss']:.9f}", flush=True)

            candidate_matrix = base_matrix.copy()
            candidate_matrix[:, 0] = oof
            final_probability = disagreement_adjustment(
                candidate_matrix,
                disagreement["base_weights"],
                disagreement["theta"],
                experiment["probability_clip"],
            )
            oof_columns[f"cat__{candidate}"] = oof
            oof_columns[f"final__{candidate}"] = final_probability
            overall_rows.append(
                {
                    "candidate": candidate,
                    "blocks": ",".join(blocks),
                    "feature_count": features.shape[1],
                    "cat_log_loss": log_loss(data.y, oof),
                    "cat_roc_auc": roc_auc_score(data.y, oof),
                    "final_log_loss": log_loss(data.y, final_probability),
                    "final_roc_auc": roc_auc_score(data.y, final_probability),
                    "gain_vs_current_final": log_loss(data.y, current_final) - log_loss(data.y, final_probability),
                    "runtime_seconds": time.perf_counter() - candidate_started,
                }
            )
            print(f"{candidate} final: {overall_rows[-1]['final_log_loss']:.9f}", flush=True)

    pd.DataFrame({ID_COLUMN: data.train[ID_COLUMN], "target": data.y, "fold": fold_id, **oof_columns}).to_csv(
        args.output_dir / "oof_predictions.csv", index=False, float_format="%.12g"
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(overall_rows).sort_values("final_log_loss").to_csv(
        args.output_dir / "overall_metrics.csv", index=False, float_format="%.12g"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": experiment["version"],
                "saved_folds_sha256": sha256_file(fold_path),
                "candidate_count": len(overall_rows),
                "current_final_log_loss": log_loss(data.y, current_final),
                "external_training_data_used": False,
                "source_label_matching_used": False,
                "test_data_accessed": False,
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
