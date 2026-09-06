#!/usr/bin/env python3
"""Quantify whether screened boosters add enough diversity to the final blend."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import log_loss


def main() -> int:
    base = pd.read_csv(ROOT / "artifacts" / "archive_ensemble_v1" / "oof_predictions.csv")
    research = pd.read_csv(ROOT / "artifacts" / "boosting_research_v1" / "oof_predictions.csv")
    keys = ["client_id", "target", "fold"]
    if not base[keys].equals(research[keys]):
        raise ValueError("Research predictions do not align with the saved-fold base OOF rows")

    y = base["target"].to_numpy()
    selected = base["archive_fixed_blend"].to_numpy()
    new_names = [name for name in research.columns if name not in keys]
    pair_rows = []
    for name in new_names:
        candidate = research[name].to_numpy()

        def pair_loss(weight: float) -> float:
            return float(log_loss(y, (1 - weight) * selected + weight * candidate))

        fit = minimize_scalar(pair_loss, bounds=(0, 1), method="bounded")
        pair_rows.append(
            {
                "candidate": name,
                "candidate_weight": float(fit.x),
                "development_oof_log_loss": float(fit.fun),
                "gain_vs_selected": float(log_loss(y, selected) - fit.fun),
                "correlation_with_selected": float(np.corrcoef(selected, candidate)[0, 1]),
            }
        )

    existing_names = ["cat_behavior_d5", "rf_behavior_leaf30", "bayes_laplace_curved"]
    names = existing_names + new_names
    matrix = np.column_stack([base[existing_names].to_numpy(), research[new_names].to_numpy()])

    def objective(weights: np.ndarray) -> float:
        return float(log_loss(y, np.clip(matrix @ weights, 1e-7, 1 - 1e-7)))

    fit = minimize(
        objective,
        np.full(len(names), 1 / len(names)),
        method="SLSQP",
        bounds=[(0, 1)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not fit.success:
        raise RuntimeError(fit.message)

    selected_loss = float(log_loss(y, selected))
    requested_loss = selected_loss - 0.02
    global_loss = objective(fit.x)
    summary = {
        "selected_fixed_blend_log_loss": selected_loss,
        "requested_absolute_gain": 0.02,
        "requested_log_loss": requested_loss,
        "best_global_refit_log_loss_development_only": global_loss,
        "observed_gain_development_only": selected_loss - global_loss,
        "remaining_gap_to_requested_loss": global_loss - requested_loss,
        "global_refit_weights": {name: float(weight) for name, weight in zip(names, fit.x)},
        "candidate_count": len(new_names),
        "decision": "Do not promote: gain is tiny, selected on the same OOF matrix, and lacks nested validation.",
    }
    output = ROOT / "artifacts" / "boosting_research_v1"
    pd.DataFrame(pair_rows).sort_values("development_oof_log_loss").to_csv(
        output / "pair_blend_screen.csv", index=False, float_format="%.12g"
    )
    (output / "blend_screen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
