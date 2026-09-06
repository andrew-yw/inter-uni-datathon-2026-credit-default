#!/usr/bin/env python3
"""Validate, train, explain, and reproduce the committed final submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.pipeline import (
    ID_COLUMN,
    PREDICTION_COLUMN,
    build_model,
    coefficient_table,
    load_competition_data,
    make_grouped_folds,
    positive_probability,
    sha256_file,
)


def metric_row(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Official log loss plus ranking and calibration diagnostics."""

    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def csv_bytes(frame: pd.DataFrame, probability_digits: int) -> bytes:
    """Use stable formatting so the same model run creates the same file bytes."""

    return frame.to_csv(index=False, lineterminator="\n", float_format=f"%.{probability_digits}f").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submissions" / "submission_logistic_interpretable_v1.csv",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation = config["validation"]
    model_config = config["model"]
    epsilon = float(config["probability_clip"])
    data = load_competition_data(args.data_dir, config["input_sha256"])
    X, y = data.X, data.y
    folds = make_grouped_folds(X, y, validation["n_splits"], validation["seed"])

    # Each validation prediction is produced by preprocessing and a model fitted
    # only on that fold's training rows.
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    for fold, (training, validation_rows) in enumerate(folds):
        model = build_model(model_config["C"], validation["seed"], model_config["max_iter"])
        model.fit(X.iloc[training], y[training])
        probability = positive_probability(model, X.iloc[validation_rows], epsilon)
        oof[validation_rows] = probability
        fold_rows.append(
            {
                "fold": fold,
                "rows": len(validation_rows),
                "positives": int(y[validation_rows].sum()),
                **metric_row(y[validation_rows], probability),
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions are incomplete")

    # Refit the unchanged pipeline on all allowed training rows for final test inference.
    final_model = build_model(model_config["C"], validation["seed"], model_config["max_iter"])
    final_model.fit(X, y)
    test_probability = positive_probability(final_model, data.test[X.columns], epsilon)
    submission = data.sample[[ID_COLUMN]].copy()
    submission[PREDICTION_COLUMN] = test_probability
    content = csv_bytes(submission, config["serialization"]["probability_digits"])
    digest = hashlib.sha256(content).hexdigest()

    # Refuse silent output drift once config.json has frozen the reviewed file hash.
    expected_digest = config.get("expected_submission_sha256", "")
    if expected_digest not in ("", "TO_BE_FILLED_AFTER_VERIFIED_RUN") and digest != expected_digest:
        raise RuntimeError(f"Submission bytes differ: expected {expected_digest}, got {digest}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    reread = pd.read_csv(args.output, dtype={ID_COLUMN: "string"})
    if reread[ID_COLUMN].tolist() != data.sample[ID_COLUMN].tolist():
        raise RuntimeError("Written submission does not preserve official row order")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.artifacts_dir / "fold_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(
        {ID_COLUMN: data.train[ID_COLUMN], "target": y, "oof_probability": oof}
    ).to_csv(args.artifacts_dir / "oof_predictions.csv", index=False, float_format="%.12g")
    coefficient_table(final_model).to_csv(
        args.artifacts_dir / "coefficients.csv", index=False, float_format="%.12g"
    )
    joblib.dump(final_model, args.artifacts_dir / "final_model.joblib", compress=3)

    overall = metric_row(y, oof)
    metrics = ["log_loss", "roc_auc", "pr_auc", "brier"]
    manifest = {
        "model": model_config,
        "validation": validation,
        "probability_clip": epsilon,
        "fold_mean": {name: float(fold_metrics[name].mean()) for name in metrics},
        "fold_std": {name: float(fold_metrics[name].std(ddof=1)) for name in metrics},
        "overall_oof": overall,
        "submission": {
            "path": str(args.output.relative_to(ROOT)),
            "sha256": digest,
            "rows": len(submission),
            "minimum_probability": float(test_probability.min()),
            "maximum_probability": float(test_probability.max()),
            "mean_probability": float(test_probability.mean()),
        },
        "input_sha256": {name: sha256_file(args.data_dir / name) for name in config["input_sha256"]},
        "external_training_data_used": False,
        "source_label_matching_used": False,
        "manual_prediction_postprocessing": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (args.artifacts_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
