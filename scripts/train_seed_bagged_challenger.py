#!/usr/bin/env python3
"""Fit the stable CatBoost seed bag and build the disagreement challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from src.archive_ensemble.blend import disagreement_adjustment
from src.archive_ensemble.features import build_features
from src.archive_ensemble.models import Recipe, fit_model
from src.pipeline import ID_COLUMN, PREDICTION_COLUMN, load_competition_data, sha256_file


BEHAVIOR = "v2:engineered:status,repayment,cashflow,utilization"


def cat_recipe(status_categories: bool) -> Recipe:
    params = {
        "iterations": 3500,
        "depth": 5,
        "learning_rate": 0.035,
        "l2_leaf_reg": 8.0 if status_categories else 6.0,
        "random_strength": 0.7 if status_categories else 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
    }
    name = "cat_behavior_status_categories" if status_categories else "cat_behavior_d5"
    return Recipe(name, "catboost", BEHAVIOR, params, status_categories=status_categories)


def checked_frame(spec: dict) -> pd.DataFrame:
    path = ROOT / spec["path"]
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"Research artifact changed: {path}: {actual}")
    return pd.read_csv(path, dtype={ID_COLUMN: "string"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "seed_bagging_config.json")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "seed_bagged_challenger_v1")
    args = parser.parse_args()

    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    ensemble = json.loads((ROOT / "ensemble_config.json").read_text())
    disagreement = json.loads((ROOT / "disagreement_config.json").read_text())
    data = load_competition_data(ROOT / "data" / "raw", ensemble["input_sha256"])
    sources = {name: checked_frame(spec) for name, spec in config["input_artifacts"].items()}
    base_oof, base_test = sources["base_oof"], sources["base_test"]
    for frame in sources.values():
        expected_ids = data.test[ID_COLUMN] if len(frame) == len(data.test) else data.train[ID_COLUMN]
        if frame[ID_COLUMN].tolist() != expected_ids.tolist():
            raise ValueError("A component artifact is not aligned by client_id")

    # Assemble the development Cat branch from predictions produced on the same
    # immutable folds. Seeds and the 50/50 view split were frozen before full fit.
    numeric_oof = [base_oof["cat_behavior_d5"].to_numpy()]
    categorical_oof = [sources["booster_oof"]["cat_behavior_status_categories"].to_numpy()]
    for seed in config["seeds"]:
        if seed == 2026:
            continue
        numeric_oof.append(sources["seed_oof"][f"cat_behavior_d5__seed{seed}"].to_numpy())
        categorical_oof.append(
            sources["seed_oof"][f"cat_behavior_status_categories__seed{seed}"].to_numpy()
        )
    cat_oof = np.mean(numeric_oof + categorical_oof, axis=0)
    current_matrix = base_oof[disagreement["model_order"]].to_numpy()
    challenger_matrix = current_matrix.copy()
    challenger_matrix[:, 0] = cat_oof
    current_oof = disagreement_adjustment(
        current_matrix, disagreement["base_weights"], disagreement["theta"], config["probability_clip"]
    )
    challenger_oof = disagreement_adjustment(
        challenger_matrix, disagreement["base_weights"], disagreement["theta"], config["probability_clip"]
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    test_features = build_features(data.test, BEHAVIOR)
    train_features = build_features(data.train, BEHAVIOR)
    member_test, member_rows = [], []
    with threadpool_limits(limits=args.threads):
        for status_categories in (False, True):
            view = "categorical_status" if status_categories else "numeric_status"
            recipe = cat_recipe(status_categories)
            for seed in config["seeds"]:
                member_started = time.perf_counter()
                model, iteration = fit_model(
                    recipe,
                    train_features,
                    data.y,
                    seed=seed,
                    threads=args.threads,
                    iterations=config["full_fit_iterations_per_member"],
                )
                probability = model.predict(test_features)
                member_test.append(probability)
                path = model_dir / f"{view}_seed{seed}.joblib"
                joblib.dump(model, path, compress=3)
                member_rows.append(
                    {
                        "view": view,
                        "seed": seed,
                        "iterations": iteration,
                        "runtime_seconds": time.perf_counter() - member_started,
                        "model_sha256": sha256_file(path),
                    }
                )
                print(f"fitted {view} seed {seed}", flush=True)

    cat_test = np.mean(member_test, axis=0)
    challenger_test_matrix = base_test[disagreement["model_order"]].to_numpy().copy()
    challenger_test_matrix[:, 0] = cat_test
    current_test = disagreement_adjustment(
        base_test[disagreement["model_order"]].to_numpy(),
        disagreement["base_weights"],
        disagreement["theta"],
        config["probability_clip"],
    )
    challenger_test = disagreement_adjustment(
        challenger_test_matrix,
        disagreement["base_weights"],
        disagreement["theta"],
        config["probability_clip"],
    )

    fold_id = base_oof["fold"].to_numpy(dtype=int)
    fold_rows = []
    for name, prediction in (("current", current_oof), ("seed_bagged_challenger", challenger_oof)):
        for fold in sorted(np.unique(fold_id)):
            rows = fold_id == fold
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": int(fold),
                    "log_loss": log_loss(data.y[rows], prediction[rows]),
                    "roc_auc": roc_auc_score(data.y[rows], prediction[rows]),
                }
            )
    pd.DataFrame(fold_rows).to_csv(args.artifacts_dir / "fold_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(
        {
            ID_COLUMN: data.train[ID_COLUMN],
            "target": data.y,
            "fold": fold_id,
            "current_probability": current_oof,
            "seed_bagged_probability": challenger_oof,
        }
    ).to_csv(args.artifacts_dir / "oof_predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame({ID_COLUMN: data.test[ID_COLUMN], "seed_bagged_cat_probability": cat_test, "current_probability": current_test, "final_probability": challenger_test}).to_csv(
        args.artifacts_dir / "test_predictions.csv", index=False, float_format="%.12g"
    )
    pd.DataFrame(member_rows).to_csv(args.artifacts_dir / "members.csv", index=False, float_format="%.12g")

    keyed = pd.Series(challenger_test, index=data.test[ID_COLUMN])
    submission = data.sample[[ID_COLUMN]].copy()
    submission[PREDICTION_COLUMN] = keyed.reindex(submission[ID_COLUMN]).to_numpy()
    submission.to_csv(args.output, index=False, float_format="%.10f", lineterminator="\n")
    digest = sha256_file(args.output)
    expected = config.get("expected_submission_sha256", "")
    if expected not in ("", "TO_BE_FILLED_AFTER_VERIFIED_RUN") and digest != expected:
        raise RuntimeError(f"Submission bytes drifted: expected {expected}, got {digest}")

    current_loss, challenger_loss = log_loss(data.y, current_oof), log_loss(data.y, challenger_oof)
    manifest = {
        "version": config["version"],
        "cat_members": member_rows,
        "oof": {
            "current_log_loss": current_loss,
            "challenger_log_loss": challenger_loss,
            "absolute_gain": current_loss - challenger_loss,
            "challenger_roc_auc": roc_auc_score(data.y, challenger_oof),
            "fold_log_loss_std": float(
                pd.DataFrame(fold_rows).query("candidate == 'seed_bagged_challenger'").log_loss.std(ddof=1)
            ),
        },
        "submission": {
            "path": str(args.output.relative_to(ROOT)),
            "sha256": digest,
            "rows": len(submission),
            "minimum_probability": float(challenger_test.min()),
            "maximum_probability": float(challenger_test.max()),
            "mean_probability": float(challenger_test.mean()),
            "mean_absolute_change_vs_current": float(
                np.mean(np.abs(challenger_test - current_test))
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "external_training_data_used": False,
        "source_label_matching_used": False,
        "test_labels_accessed": False,
    }
    (args.artifacts_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
