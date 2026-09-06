#!/usr/bin/env python3
"""Apply the archived clean disagreement layer and build the final submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.archive_ensemble.blend import disagreement_adjustment, fixed_probability_blend
from src.pipeline import ID_COLUMN, PREDICTION_COLUMN, load_competition_data, sha256_file


def path_for_manifest(path: Path) -> str:
    """Prefer repository-relative paths while supporting isolated audit runs."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_retrained_oof(
    candidate_path: Path,
    reference_path: Path,
    expected_reference_hash: str,
    tolerance: float,
) -> float:
    """Accept only negligible cross-platform floating-point replay differences.

    The RF replay can differ by one floating-point ULP under parallel reduction,
    even though test probabilities serialize identically. IDs, labels, folds and
    columns must remain exact; every probability must stay within the documented
    absolute tolerance of the committed reference artifact.
    """

    if sha256_file(reference_path) != expected_reference_hash:
        raise ValueError("Committed OOF reference artifact changed")
    reference = pd.read_csv(reference_path, dtype={ID_COLUMN: "string"})
    candidate = pd.read_csv(candidate_path, dtype={ID_COLUMN: "string"})
    if list(candidate.columns) != list(reference.columns):
        raise ValueError("Retrained OOF columns differ from the committed reference")
    exact_columns = [ID_COLUMN, "target", "fold"]
    for column in exact_columns:
        if not candidate[column].equals(reference[column]):
            raise ValueError(f"Retrained OOF {column} values differ from the reference")
    probability_columns = [column for column in reference if column not in exact_columns]
    difference = np.abs(
        candidate[probability_columns].to_numpy(dtype=float)
        - reference[probability_columns].to_numpy(dtype=float)
    )
    maximum = float(difference.max(initial=0.0))
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(
            f"Retrained OOF probabilities drifted by {maximum:.3g}; tolerance is {tolerance:.3g}"
        )
    return maximum


def metric_row(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "disagreement_config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "archive_disagreement_v2")
    parser.add_argument("--component-oof", type=Path)
    parser.add_argument("--component-test", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    ensemble_config = json.loads((ROOT / "ensemble_config.json").read_text(encoding="utf-8"))
    data = load_competition_data(args.data_dir, ensemble_config["input_sha256"])
    component = config["component_artifacts"]
    reference_oof_path = ROOT / component["oof_path"]
    oof_path = args.component_oof or reference_oof_path
    test_path = args.component_test or ROOT / component["test_path"]
    oof_hash = sha256_file(oof_path)
    oof_replay_max_abs_difference = 0.0
    if oof_hash != component["oof_sha256"]:
        if args.component_oof is None:
            raise ValueError(f"Component prediction artifact changed: {oof_path}: {oof_hash}")
        oof_replay_max_abs_difference = validate_retrained_oof(
            oof_path,
            reference_oof_path,
            component["oof_sha256"],
            float(component["oof_replay_absolute_tolerance"]),
        )
    test_hash = sha256_file(test_path)
    if test_hash != component["test_sha256"]:
        raise ValueError(f"Component prediction artifact changed: {test_path}: {test_hash}")

    oof_source = pd.read_csv(oof_path, dtype={ID_COLUMN: "string"})
    test_source = pd.read_csv(test_path, dtype={ID_COLUMN: "string"})
    if oof_source[ID_COLUMN].tolist() != data.train[ID_COLUMN].tolist():
        raise ValueError("OOF component rows do not match organizer training IDs")
    if test_source[ID_COLUMN].tolist() != data.test[ID_COLUMN].tolist():
        raise ValueError("Test component rows do not match organizer test IDs")
    if not np.array_equal(oof_source["target"].to_numpy(dtype=np.int8), data.y):
        raise ValueError("OOF target column differs from organizer labels")

    names = config["model_order"]
    weights, theta = config["base_weights"], config["theta"]
    epsilon = float(config["probability_clip"])
    oof_matrix = oof_source[names].to_numpy()
    test_matrix = test_source[names].to_numpy()
    base_oof = fixed_probability_blend(oof_matrix, weights, epsilon)
    base_test = fixed_probability_blend(test_matrix, weights, epsilon)
    adjusted_oof = disagreement_adjustment(oof_matrix, weights, theta, epsilon)
    adjusted_test = disagreement_adjustment(test_matrix, weights, theta, epsilon)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            ID_COLUMN: data.train[ID_COLUMN],
            "target": data.y,
            "fold": oof_source["fold"],
            "base_probability": base_oof,
            "disagreement_probability": adjusted_oof,
        }
    ).to_csv(args.artifacts_dir / "oof_predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame(
        {
            ID_COLUMN: data.test[ID_COLUMN],
            "base_probability": base_test,
            "disagreement_probability": adjusted_test,
        }
    ).to_csv(args.artifacts_dir / "test_predictions.csv", index=False, float_format="%.12g")

    fold_rows = []
    fold_id = oof_source["fold"].to_numpy(dtype=int)
    for name, probability in (("base", base_oof), ("disagreement", adjusted_oof)):
        for fold in sorted(np.unique(fold_id)):
            rows = fold_id == fold
            fold_rows.append(
                {"candidate": name, "fold": int(fold), "rows": int(rows.sum()), **metric_row(data.y[rows], probability[rows])}
            )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.artifacts_dir / "fold_metrics.csv", index=False, float_format="%.12g")

    keyed = pd.Series(adjusted_test, index=data.test[ID_COLUMN])
    submission = data.sample[[ID_COLUMN]].copy()
    submission[PREDICTION_COLUMN] = keyed.reindex(submission[ID_COLUMN]).to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, float_format="%.10f", lineterminator="\n")
    digest = sha256_file(args.output)
    expected = config.get("expected_submission_sha256", "")
    if expected not in ("", "TO_BE_FILLED_AFTER_VERIFIED_RUN") and digest != expected:
        raise RuntimeError(f"Submission bytes drifted: expected {expected}, got {digest}")

    base_metrics, adjusted_metrics = metric_row(data.y, base_oof), metric_row(data.y, adjusted_oof)
    manifest = {
        "version": config["version"],
        "model_order": names,
        "base_weights": weights,
        "theta": theta,
        "development_replay_oof": {
            "base": base_metrics,
            "disagreement": adjusted_metrics,
            "log_loss_gain": base_metrics["log_loss"] - adjusted_metrics["log_loss"],
            "warning": "Theta was historically developed from this training population; use the archived nested score as the selection estimate.",
        },
        "archive_nested_validation": config["archive_training_provenance"],
        "component_sha256": {
            path_for_manifest(oof_path): sha256_file(oof_path),
            path_for_manifest(test_path): sha256_file(test_path),
        },
        "oof_replay_max_abs_difference": oof_replay_max_abs_difference,
        "submission": {
            "path": path_for_manifest(args.output),
            "sha256": digest,
            "rows": len(submission),
            "minimum_probability": float(adjusted_test.min()),
            "maximum_probability": float(adjusted_test.max()),
            "mean_probability": float(adjusted_test.mean()),
        },
        "external_training_data_used": False,
        "source_label_matching_used": False,
        "test_labels_accessed": False,
        "probability_clipping_applied_to_oof_and_test": True,
    }
    (args.artifacts_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
