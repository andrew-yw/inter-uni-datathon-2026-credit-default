"""Row-local feature engineering: no labels, identifiers or fitted population statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, DEMOGRAPHICS, PAY_STATUS, BILLS, PAYMENTS


def safe_ratio(numerator, denominator, floor=100.0):
    """Stabilize zero/near-zero denominators while retaining their sign."""
    denominator = np.asarray(denominator, dtype=float)
    denominator = np.where(denominator < 0, -1.0, 1.0) * np.maximum(np.abs(denominator), floor)
    return np.clip(np.asarray(numerator, dtype=float) / denominator, -1000.0, 1000.0)


def build_features(raw: pd.DataFrame, mode: str = "engineered") -> pd.DataFrame:
    if mode.startswith("v2:"):
        from .features_v2 import build_v2_features
        return build_v2_features(raw, mode)
    # Explicit allowlist prevents accidental target / identifier leakage.
    result = raw.loc[:, FEATURES].astype(np.float64).copy()
    if mode == "raw":
        return result
    if mode not in ("compact", "engineered"):
        raise ValueError(f"Unknown feature mode {mode}")
    status = result[PAY_STATUS].to_numpy()
    bills = result[BILLS].to_numpy()
    payments = result[PAYMENTS].to_numpy()
    limit = result["LIMIT_BAL"].to_numpy()
    engineered = {}

    for width in (3, 6):
        history = status[:, :width]
        overdue = np.maximum(history, 0)
        engineered[f"late_count_{width}"] = (history > 0).sum(axis=1)
        engineered[f"severe_late_count_{width}"] = (history >= 2).sum(axis=1)
        engineered[f"late_max_{width}"] = overdue.max(axis=1)
        engineered[f"late_sum_{width}"] = overdue.sum(axis=1)
        for code in (-2, -1, 0):
            engineered[f"status_{code}_count_{width}"] = (history == code).sum(axis=1)
    weights = np.arange(6, 0, -1)
    engineered["late_recency_weighted"] = (np.maximum(status, 0) * weights).sum(axis=1) / weights.sum()
    engineered["late_run_recent"] = np.cumprod(status > 0, axis=1).sum(axis=1)
    engineered["status_latest_minus_oldest"] = status[:, 0] - status[:, -1]
    engineered["status_worsening_count"] = (status[:, :-1] > status[:, 1:]).sum(axis=1)
    engineered["status_improving_count"] = (status[:, :-1] < status[:, 1:]).sum(axis=1)

    for name, values in (("bill", bills), ("payment", payments)):
        engineered[f"{name}_mean"] = values.mean(axis=1)
        engineered[f"{name}_std"] = values.std(axis=1)
        engineered[f"{name}_min"] = values.min(axis=1)
        engineered[f"{name}_max"] = values.max(axis=1)
        engineered[f"{name}_zero_count"] = (values == 0).sum(axis=1)
        engineered[f"{name}_negative_count"] = (values < 0).sum(axis=1)
        engineered[f"{name}_recent3_mean"] = values[:, :3].mean(axis=1)
        engineered[f"{name}_recent_minus_old3"] = values[:, :3].mean(axis=1) - values[:, 3:].mean(axis=1)
        engineered[f"{name}_mean_to_limit"] = safe_ratio(values.mean(axis=1), limit)
        engineered[f"{name}_max_to_limit"] = safe_ratio(values.max(axis=1), limit)
        engineered[f"{name}_latest_minus_oldest"] = values[:, 0] - values[:, -1]
    engineered["payment_total_to_bill_total"] = safe_ratio(payments.sum(axis=1), bills.sum(axis=1))
    engineered["negative_bill_flag"] = (bills < 0).any(axis=1).astype(int)
    engineered["zero_payment_with_positive_bill"] = ((payments == 0) & (bills > 0)).sum(axis=1)
    engineered["bills_above_limit_count"] = (bills > limit[:, None]).sum(axis=1)
    engineered["education_unmapped"] = np.isin(result["EDUCATION"], [0, 5, 6]).astype(int)
    engineered["marriage_unmapped"] = (result["MARRIAGE"] == 0).astype(int)

    for month in range(6):
        engineered[f"bill_to_limit_{month+1}"] = safe_ratio(bills[:, month], limit)
        engineered[f"payment_to_limit_{month+1}"] = safe_ratio(payments[:, month], limit)
        engineered[f"payment_bill_same_index_ratio_{month+1}"] = safe_ratio(payments[:, month], bills[:, month])
    if mode == "engineered":
        for month in range(5):
            engineered[f"bill_delta_{month+1}_{month+2}"] = bills[:, month] - bills[:, month+1]
            engineered[f"payment_delta_{month+1}_{month+2}"] = payments[:, month] - payments[:, month+1]
            engineered[f"status_delta_{month+1}_{month+2}"] = status[:, month] - status[:, month+1]
            engineered[f"payment_older_bill_ratio_{month+1}"] = safe_ratio(payments[:, month], bills[:, month+1])
            # An arithmetic proxy, not an asserted accounting identity.
            engineered[f"bill_change_plus_payment_proxy_{month+1}"] = bills[:, month] - bills[:, month+1] + payments[:, month]
        for name, values in (("bill", bills), ("payment", payments)):
            engineered[f"{name}_relative_variation"] = safe_ratio(values.std(axis=1), np.abs(values.mean(axis=1)))
            engineered[f"{name}_linear_trend"] = values @ np.array([2.5, 1.5, .5, -.5, -1.5, -2.5]) / 17.5
        for col in ["LIMIT_BAL", *BILLS, *PAYMENTS]:
            values = result[col].to_numpy()
            engineered[f"signed_log_{col}"] = np.sign(values) * np.log1p(np.abs(values))
    result = pd.concat([result, pd.DataFrame(engineered, index=result.index)], axis=1)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Feature engineering produced nonfinite values")
    return result


def model_view(features: pd.DataFrame, family: str, status_categories: bool = False):
    view = features.copy()
    categories = DEMOGRAPHICS + (PAY_STATUS if status_categories else [])
    categories += [c for c in view if c.startswith("cat__")]
    if family == "catboost":
        for col in categories:
            view[col] = view[col].astype(int).astype(str)
    elif family == "lightgbm":
        for col in categories:
            # Explicit nonnegative codes; LightGBM treats negative category codes as missing.
            view[col] = view[col].astype(int) + (2 if col in PAY_STATUS else 0)
    return view, categories
