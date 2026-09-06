"""Ablatable row-local behaviour features, with no fitted encoders or target access.

Mode syntax: ``v2:<raw|compact|engineered>:<blocks separated by comma>:<drops>``.
The final field is optional. Month 1 is the most recent supplied record.
Adjacent-bill calculations are arithmetic proxies, not accounting identities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, PAY_STATUS, BILLS, PAYMENTS

BLOCKS = ("status", "repayment", "cashflow", "utilization", "patterns", "demography")
DROPS = ("logs", "same_bill_ratios", "raw_status_deltas")


def build_v2_features(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    from .features import build_features, safe_ratio

    parts = mode.split(":")
    if len(parts) not in (3, 4) or parts[0] != "v2":
        raise ValueError(f"Invalid v2 feature mode: {mode}")
    _, base, blocks, *drop_part = parts
    if base not in ("raw", "compact", "engineered"):
        raise ValueError(f"Unknown base feature set: {base}")
    requested = blocks.split(",") if blocks else []
    drops = drop_part[0].split(",") if drop_part and drop_part[0] else []
    if len(set(requested)) != len(requested) or set(requested) - set(BLOCKS):
        raise ValueError(f"Unknown or duplicated feature block: {blocks}")
    if set(drops) - set(DROPS):
        raise ValueError(f"Unknown feature removal: {drops}")
    result = build_features(raw, base)
    remove = []
    for col in result:
        if (("logs" in drops and col.startswith("signed_log_"))
            or ("same_bill_ratios" in drops and col.startswith("payment_bill_same_index_ratio_"))
            or ("raw_status_deltas" in drops and col.startswith("status_delta_"))):
            remove.append(col)
    result = result.drop(columns=remove)

    # Explicit raw-column allowlist; each operation depends on this row alone.
    values = raw.loc[:, FEATURES].astype(float)
    status = values[PAY_STATUS].to_numpy()
    overdue = np.maximum(status, 0)
    bills = values[BILLS].to_numpy()
    positive_bills = np.maximum(bills, 0)
    payments = values[PAYMENTS].to_numpy()
    limit = values["LIMIT_BAL"].to_numpy()
    amount_floor = np.maximum(100.0, limit * .001)
    extra = {}

    if "status" in requested:
        for threshold in (1, 2):
            late = status >= threshold
            label = f"v2_late{threshold}"
            extra[f"{label}_recent2_count"] = late[:, :2].sum(1)
            extra[f"{label}_recent_run"] = np.cumprod(late, axis=1).sum(1)
            extra[f"{label}_months_since_latest"] = np.where(late.any(1), late.argmax(1), 6)
            extra[f"{label}_new_recent"] = late[:, 0] & ~late[:, 1]
            extra[f"{label}_resolved_recent"] = ~late[:, 0] & late[:, 1]
            extra[f"{label}_new_episode_count"] = (late[:, :-1] & ~late[:, 1:]).sum(1)
            extra[f"{label}_resolved_episode_count"] = (~late[:, :-1] & late[:, 1:]).sum(1)
            extra[f"{label}_recent_minus_old3"] = late[:, :3].sum(1) - late[:, 3:].sum(1)
        extra["v2_overdue_recent2_mean"] = overdue[:, :2].mean(1)
        extra["v2_overdue_recent_minus_old3"] = overdue[:, :3].mean(1) - overdue[:, 3:].mean(1)
        extra["v2_overdue_recent_minus_previous_max"] = overdue[:, 0] - overdue[:, 1:].max(1)
        extra["v2_overdue_std"] = overdue.std(1)
        extra["v2_status_unchanged_count"] = (status[:, :-1] == status[:, 1:]).sum(1)

    if "repayment" in requested:
        # Compare repayment at index t with the preceding supplied bill at t+1.
        older = positive_bills[:, 1:]
        paid = payments[:, :5]
        active = older > 0
        ratio = paid / np.maximum(older, amount_floor[:, None])
        share = paid / (paid + older + amount_floor[:, None])
        uncovered = np.maximum(older - paid, 0)
        for month in range(5):
            extra[f"v2_paid_share_older_bill_{month+1}"] = share[:, month]
            extra[f"v2_uncovered_older_bill_to_limit_{month+1}"] = safe_ratio(uncovered[:, month], limit)
        for width in (2, 5):
            valid_count = np.maximum(active[:, :width].sum(1), 1)
            extra[f"v2_active_older_bill_count_{width}"] = active[:, :width].sum(1)
            extra[f"v2_repayment_ratio_active_mean_{width}"] = (
                (np.minimum(ratio[:, :width], 2) * active[:, :width]).sum(1) / valid_count)
            extra[f"v2_paid_share_mean_{width}"] = share[:, :width].mean(1)
            extra[f"v2_uncovered_older_bill_mean_to_limit_{width}"] = safe_ratio(uncovered[:, :width].mean(1), limit)
            for threshold in (.02, .05, .1):
                # Exploratory thresholds, not an assumed contractual minimum payment.
                extra[f"v2_paid_under_{int(threshold*100)}pct_count_{width}"] = (
                    (ratio[:, :width] < threshold) & active[:, :width]).sum(1)
            extra[f"v2_paid_covers_older_bill_count_{width}"] = ((paid[:, :width] >= older[:, :width]) & active[:, :width]).sum(1)
        extra["v2_paid_share_recent_minus_old"] = share[:, :2].mean(1) - share[:, 2:].mean(1)
        extra["v2_zero_payment_positive_older_bill_count"] = ((paid == 0) & active).sum(1)
        extra["v2_zero_payment_recent_run"] = np.cumprod(payments == 0, axis=1).sum(1)
        extra["v2_payment_multiple_1000_count"] = ((payments > 0) & (payments % 1000 == 0)).sum(1)

    if "cashflow" in requested:
        change = bills[:, :5] - bills[:, 1:]
        proxy = change + payments[:, :5]
        normalized = safe_ratio(proxy, limit[:, None])
        for month in range(5):
            extra[f"v2_bill_change_plus_payment_to_limit_{month+1}"] = normalized[:, month]
        for width in (2, 5):
            history = proxy[:, :width]
            extra[f"v2_cashflow_proxy_mean_{width}"] = history.mean(1)
            extra[f"v2_cashflow_proxy_std_{width}"] = history.std(1)
            extra[f"v2_cashflow_proxy_mean_to_limit_{width}"] = normalized[:, :width].mean(1)
            extra[f"v2_cashflow_proxy_negative_count_{width}"] = (history < 0).sum(1)
            extra[f"v2_bill_growing_count_{width}"] = (change[:, :width] > 0).sum(1)
        extra["v2_cashflow_proxy_recent_minus_old"] = proxy[:, :2].mean(1) - proxy[:, 2:].mean(1)
        extra["v2_cashflow_proxy_relative_variation"] = safe_ratio(proxy.std(1), np.abs(proxy.mean(1)))
        extra["v2_cashflow_proxy_to_payment_mean"] = safe_ratio(proxy.mean(1), payments[:, :5].mean(1))
        extra["v2_bill_growing_recent_run"] = np.cumprod(change > 0, axis=1).sum(1)

    if "utilization" in requested:
        utilization = safe_ratio(bills, limit[:, None])
        extra["v2_utilization_std"] = utilization.std(1)
        extra["v2_utilization_recent2_mean"] = utilization[:, :2].mean(1)
        extra["v2_utilization_recent_minus_old3"] = utilization[:, :3].mean(1) - utilization[:, 3:].mean(1)
        extra["v2_utilization_latest_minus_oldest"] = utilization[:, 0] - utilization[:, -1]
        extra["v2_bill_range_to_limit"] = safe_ratio(bills.max(1) - bills.min(1), limit)
        for threshold in (.5, .8, 1.):
            extra[f"v2_utilization_over_{int(threshold*100)}pct_count"] = (utilization > threshold).sum(1)
        extra["v2_over_limit_amount_mean"] = np.maximum(bills - limit[:, None], 0).mean(1)
        extra["v2_credit_headroom_latest"] = limit - bills[:, 0]
        extra["v2_credit_headroom_min"] = limit - bills.max(1)
        extra["v2_recent_late_utilization"] = overdue[:, 0] * utilization[:, 0]
        extra["v2_payment_recent2_mean"] = payments[:, :2].mean(1)
        extra["v2_bill_recent2_mean"] = bills[:, :2].mean(1)
        extra["v2_bill_nonpositive_recent_run"] = np.cumprod(bills <= 0, axis=1).sum(1)

    if "patterns" in requested:
        # Nonnegative, collision-free mixed-radix codes within the documented range.
        # A cat__ prefix tells CatBoost/LightGBM to treat these codes as categories.
        code = status.astype(np.int64) + 2
        if ((code < 0) | (code >= 12)).any():
            raise ValueError("Repayment status outside pattern encoder range [-2, 9]")
        extra["cat__status_recent2"] = code[:, 0] * 12 + code[:, 1]
        extra["cat__status_recent3"] = code[:, :3] @ np.array([144, 12, 1])
        for threshold in (1, 2):
            extra[f"cat__late{threshold}_six_month_pattern"] = (status >= threshold) @ (2 ** np.arange(6))
        extra["cat__status_latest"] = code[:, 0]

    if "demography" in requested:
        age = values["AGE"].to_numpy()
        sex = values["SEX"].to_numpy()
        education = values["EDUCATION"].to_numpy()
        marriage = values["MARRIAGE"].to_numpy()
        extra["v2_limit_per_age"] = limit / np.maximum(age, 1)
        extra["cat__age_decade"] = np.floor(age / 10)
        extra["cat__sex_marriage"] = sex * 10 + marriage
        extra["cat__education_marriage"] = education * 10 + marriage
        extra["cat__education_sex"] = education * 10 + sex

    result = pd.concat([result, pd.DataFrame(extra, index=result.index)], axis=1).astype(np.float64)
    if result.columns.duplicated().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Invalid v2 feature columns or nonfinite values")
    return result
