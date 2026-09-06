#!/usr/bin/env python3
"""Train and blend the archive's honest CatBoost/RF/Bayesian fallback.

Only organizer-provided train labels are used.  In particular, this program
does not read the archive ZIP, the external UCI table, or any test labels.
Every OOF prediction is made by a model that did not fit its validation fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import catboost
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from src.archive_ensemble.features import build_features, model_view
from src.archive_ensemble.models import Recipe, fit_model
from src.pipeline import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    PREDICTION_COLUMN,
    load_competition_data,
    sha256_file,
)


def path_for_manifest(path: Path) -> str:
    """Prefer repository-relative paths while supporting isolated audit runs."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Official log loss plus useful ranking and calibration diagnostics."""

    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def load_saved_folds(path: Path, train: pd.DataFrame, n_splits: int) -> np.ndarray:
    """Load row assignments by ID; never create or silently repair them."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing saved folds: {path}. Training refuses to recreate validation partitions."
        )
    saved = pd.read_csv(path, dtype={ID_COLUMN: "string"})
    if list(saved.columns) != [ID_COLUMN, "fold"] or saved[ID_COLUMN].duplicated().any():
        raise ValueError("Saved folds must contain unique client_id,fold rows")
    keyed = saved.set_index(ID_COLUMN)["fold"]
    if set(keyed.index) != set(train[ID_COLUMN]):
        raise ValueError("Saved-fold IDs do not exactly match training IDs")
    fold = keyed.reindex(train[ID_COLUMN]).to_numpy(dtype=int)
    if set(fold) != set(range(n_splits)):
        raise ValueError(f"Expected folds 0..{n_splits - 1}; found {sorted(set(fold))}")

    # The original grouping rule keeps identical feature vectors together.
    groups = pd.util.hash_pandas_object(train[FEATURE_COLUMNS], index=False)
    check = pd.DataFrame({"group": groups, "fold": fold}).groupby("group")["fold"].nunique()
    if (check != 1).any():
        raise ValueError("An identical feature profile crosses saved folds")
    return fold


def make_recipe(spec: dict) -> Recipe:
    """Translate the reviewed JSON recipe into the archived estimator contract."""

    return Recipe(
        name=spec["name"],
        family=spec["family"],
        features=spec["features"],
        params=spec["params"],
        status_categories=bool(spec.get("status_categories", False)),
    )


def fit_convex_weights(matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit non-negative weights summing to one on development OOF predictions."""

    start = np.full(matrix.shape[1], 1.0 / matrix.shape[1])

    def objective(weight: np.ndarray) -> tuple[float, np.ndarray]:
        probability = np.clip(matrix @ weight, 1e-7, 1 - 1e-7)
        value = -np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))
        gradient = matrix.T @ ((probability - y) / (probability * (1 - probability))) / len(y)
        return float(value), gradient

    result = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"maxiter": 500, "ftol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(f"Blend optimization failed: {result.message}")
    weight = np.clip(result.x, 0, 1)
    return weight / weight.sum()


def feature_importance_rows(bundle, feature_frame: pd.DataFrame, model_name: str) -> list[dict]:
    """Expose tree importance with transformed feature names where available."""

    if model_name.startswith("cat_"):
        values = np.asarray(bundle.model.feature_importances_, dtype=float)
        names = list(model_view(feature_frame, "catboost", False)[0].columns)
    elif model_name.startswith("rf_"):
        transform = bundle.model.named_steps["transform"]
        values = np.asarray(bundle.model.named_steps["classifier"].feature_importances_, dtype=float)
        names = list(transform.get_feature_names_out())
    else:
        return []
    total = max(float(values.sum()), 1e-12)
    return [
        {"model": model_name, "feature": name, "importance": float(value / total)}
        for name, value in zip(names, values)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--config", type=Path, default=ROOT / "ensemble_config.json")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "archive_ensemble_v1")
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submissions" / "submission_archive_blend_v1.csv",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    epsilon = float(config["probability_clip"])
    data = load_competition_data(args.data_dir, config["input_sha256"])
    fold_path = ROOT / config["validation"]["folds_path"]
    expected_fold_hash = config["validation"]["folds_sha256"]
    actual_fold_hash = sha256_file(fold_path)
    if expected_fold_hash != actual_fold_hash:
        raise ValueError(f"Saved-fold hash mismatch: expected {expected_fold_hash}, got {actual_fold_hash}")
    fold_id = load_saved_folds(fold_path, data.train, config["validation"]["n_splits"])
    y = data.y
    model_specs = config["models"]
    model_names = [spec["name"] for spec in model_specs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Row-local feature tables may be safely cached across folds because they
    # contain no labels and fit no population statistics.
    train_features: dict[str, pd.DataFrame] = {}
    test_features: dict[str, pd.DataFrame] = {}
    for mode in sorted({spec["features"] for spec in model_specs}):
        train_features[mode] = build_features(data.train, mode)
        test_features[mode] = build_features(data.test, mode)

    oof_columns: dict[str, np.ndarray] = {}
    test_columns: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    importance_rows: list[dict] = []

    with threadpool_limits(limits=args.threads):
        for spec in model_specs:
            recipe = make_recipe(spec)
            features = train_features[recipe.features]
            test_frame = test_features[recipe.features]
            oof = np.full(len(y), np.nan, dtype=float)
            best_iterations: list[int] = []
            model_started = time.perf_counter()

            for fold in range(config["validation"]["n_splits"]):
                validation = np.flatnonzero(fold_id == fold)
                training = np.flatnonzero(fold_id != fold)
                fold_started = time.perf_counter()
                bundle, best_iteration = fit_model(
                    recipe,
                    features.iloc[training],
                    y[training],
                    features.iloc[validation],
                    y[validation],
                    seed=config["validation"]["seed"],
                    threads=args.threads,
                )
                probability = np.clip(bundle.predict(features.iloc[validation]), epsilon, 1 - epsilon)
                oof[validation] = probability
                best_iterations.append(best_iteration)
                fold_rows.append(
                    {
                        "candidate": recipe.name,
                        "fold": fold,
                        "rows": len(validation),
                        "positives": int(y[validation].sum()),
                        "best_iteration": best_iteration,
                        "runtime_seconds": time.perf_counter() - fold_started,
                        **metrics(y[validation], probability),
                    }
                )
                print(f"completed {recipe.name} fold {fold}: log_loss={fold_rows[-1]['log_loss']:.9f}", flush=True)

            if not np.isfinite(oof).all():
                raise RuntimeError(f"Incomplete OOF predictions for {recipe.name}")

            # The archive recorded 437 trees for its final CatBoost fit. Other
            # families either have their tree count in params or converge fully.
            final_iterations = spec.get("full_fit_iterations")
            final_bundle, final_iteration = fit_model(
                recipe,
                features,
                y,
                seed=config["validation"]["seed"],
                threads=args.threads,
                iterations=final_iterations,
            )
            test_probability = np.clip(final_bundle.predict(test_frame), epsilon, 1 - epsilon)
            joblib.dump(final_bundle, model_dir / f"{recipe.name}.joblib", compress=3)
            importance_rows.extend(feature_importance_rows(final_bundle, features, recipe.name))
            oof_columns[recipe.name] = oof
            test_columns[recipe.name] = test_probability
            np.savez_compressed(
                args.output_dir / f"{recipe.name}_predictions.npz",
                oof=oof,
                test=test_probability,
                fold_id=fold_id,
                best_iterations=np.asarray(best_iterations),
                final_iteration=final_iteration,
            )
            print(
                f"completed {recipe.name}: overall_oof={metrics(y, oof)['log_loss']:.9f}, "
                f"runtime={time.perf_counter() - model_started:.1f}s",
                flush=True,
            )

    oof_matrix = np.column_stack([oof_columns[name] for name in model_names])
    test_matrix = np.column_stack([test_columns[name] for name in model_names])
    fixed_weights = np.asarray([config["archive_fixed_blend"][name] for name in model_names], dtype=float)
    if not np.isclose(fixed_weights.sum(), 1.0) or (fixed_weights < 0).any():
        raise ValueError("Archive blend weights must be non-negative and sum to one")
    refit_weights = fit_convex_weights(oof_matrix, y)

    fixed_oof = np.clip(oof_matrix @ fixed_weights, epsilon, 1 - epsilon)
    fixed_test = np.clip(test_matrix @ fixed_weights, epsilon, 1 - epsilon)
    refit_oof = np.clip(oof_matrix @ refit_weights, epsilon, 1 - epsilon)
    refit_test = np.clip(test_matrix @ refit_weights, epsilon, 1 - epsilon)
    oof_columns["archive_fixed_blend"] = fixed_oof
    oof_columns["oof_refit_blend_development_only"] = refit_oof
    test_columns["archive_fixed_blend"] = fixed_test
    test_columns["oof_refit_blend_development_only"] = refit_test

    for candidate in ("archive_fixed_blend", "oof_refit_blend_development_only"):
        for fold in range(config["validation"]["n_splits"]):
            rows = fold_id == fold
            fold_rows.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "rows": int(rows.sum()),
                    "positives": int(y[rows].sum()),
                    "best_iteration": np.nan,
                    "runtime_seconds": np.nan,
                    **metrics(y[rows], oof_columns[candidate][rows]),
                }
            )

    oof_frame = pd.DataFrame({ID_COLUMN: data.train[ID_COLUMN], "target": y, "fold": fold_id, **oof_columns})
    test_frame = pd.DataFrame({ID_COLUMN: data.test[ID_COLUMN], **test_columns})
    oof_frame.to_csv(args.output_dir / "oof_predictions.csv", index=False, float_format="%.12g")
    test_frame.to_csv(args.output_dir / "test_predictions.csv", index=False, float_format="%.12g")
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False, float_format="%.12g")

    overall_rows = []
    for name, probability in oof_columns.items():
        subset = fold_metrics[fold_metrics["candidate"] == name]
        overall_rows.append(
            {
                "candidate": name,
                **metrics(y, probability),
                "fold_log_loss_mean": float(subset["log_loss"].mean()),
                "fold_log_loss_std": float(subset["log_loss"].std(ddof=1)),
            }
        )
    pd.DataFrame(overall_rows).to_csv(args.output_dir / "overall_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(np.corrcoef(oof_matrix, rowvar=False), index=model_names, columns=model_names).to_csv(
        args.output_dir / "oof_correlations.csv", float_format="%.12g"
    )
    if importance_rows:
        pd.DataFrame(importance_rows).sort_values(["model", "importance"], ascending=[True, False]).to_csv(
            args.output_dir / "feature_importance.csv", index=False, float_format="%.12g"
        )

    weights_record = {
        "model_order": model_names,
        "archive_fixed_weights": fixed_weights.tolist(),
        "oof_refit_weights_development_only": refit_weights.tolist(),
        "selection": "archive_fixed_weights",
        "warning": "Refit weights optimize the same OOF matrix and are not a nested validation estimate.",
    }
    (args.output_dir / "blend_weights.json").write_text(
        json.dumps(weights_record, indent=2) + "\n", encoding="utf-8"
    )

    # The selected file uses the fixed historical blend, not the optimistically
    # refitted weights, and follows the same clipping policy as OOF evaluation.
    keyed = pd.Series(fixed_test, index=data.test[ID_COLUMN])
    submission = data.sample[[ID_COLUMN]].copy()
    submission[PREDICTION_COLUMN] = keyed.reindex(submission[ID_COLUMN]).to_numpy()
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission, index=False, float_format="%.10f", lineterminator="\n")
    submission_hash = sha256_file(args.submission)
    expected_submission_hash = config.get("expected_submission_sha256")
    if expected_submission_hash and submission_hash != expected_submission_hash:
        raise RuntimeError(
            f"Submission bytes drifted: expected {expected_submission_hash}, got {submission_hash}"
        )

    manifest = {
        "version": config["version"],
        "models": model_specs,
        "blend": weights_record,
        "overall_oof": {row["candidate"]: row for row in overall_rows},
        "saved_folds_sha256": actual_fold_hash,
        "input_sha256": {name: sha256_file(args.data_dir / name) for name in config["input_sha256"]},
        "submission": {
            "path": path_for_manifest(args.submission),
            "sha256": submission_hash,
            "rows": len(submission),
            "minimum_probability": float(fixed_test.min()),
            "maximum_probability": float(fixed_test.max()),
            "mean_probability": float(fixed_test.mean()),
        },
        "runtime_seconds": time.perf_counter() - started,
        "external_training_data_used": False,
        "source_label_matching_used": False,
        "test_labels_accessed": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
