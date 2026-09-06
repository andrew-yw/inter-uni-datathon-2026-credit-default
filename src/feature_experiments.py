"""Independent, row-local feature blocks for controlled feature experiments.

The archive reproduction code remains untouched.  Every feature below depends
only on one competition row and uses fixed domain thresholds, so no target or
test-population statistic can enter preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive_ensemble.data import BILLS, FEATURES, PAYMENTS, PAY_STATUS
from src.archive_ensemble.features import build_features, safe_ratio


BASE_MODE = "v2:engineered:status,repayment,cashflow,utilization"
BLOCKS = ("trajectory", "coverage", "trends", "bands", "interactions")


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest consecutive true run across the supplied six-period order."""

    longest = np.zeros(len(mask), dtype=float)
    current = np.zeros(len(mask), dtype=float)
    for column in mask.T:
        current = np.where(column, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _fixed_band(values: np.ndarray, boundaries: list[float]) -> np.ndarray:
    """Create stable nonnegative codes without learning quantiles from data."""

    return np.digitize(np.asarray(values, dtype=float), boundaries).astype(float)


def build_experiment_features(raw: pd.DataFrame, blocks: tuple[str, ...]) -> pd.DataFrame:
    requested = tuple(blocks)
    if len(set(requested)) != len(requested) or set(requested) - set(BLOCKS):
        raise ValueError(f"Unknown or repeated experimental feature blocks: {requested}")

    # Preserve the archived feature set as a common base for fair ablations.
    result = build_features(raw, BASE_MODE)
    x = raw.loc[:, FEATURES].astype(float)
    status = x[PAY_STATUS].to_numpy()
    overdue = np.maximum(status, 0)
    bills = x[BILLS].to_numpy()
    payments = x[PAYMENTS].to_numpy()
    limit = np.maximum(x["LIMIT_BAL"].to_numpy(), 100.0)
    floor = np.maximum(100.0, limit * 0.001)
    positive_older_bills = np.maximum(bills[:, 1:], 0)
    paid = payments[:, :5]
    repayment_ratio = np.clip(paid / np.maximum(positive_older_bills, floor[:, None]), 0, 5)
    paid_share = paid / (paid + positive_older_bills + floor[:, None])
    utilization = np.clip(bills / limit[:, None], -2, 5)
    bill_change = bills[:, :5] - bills[:, 1:]
    cashflow_proxy = (bill_change + paid) / limit[:, None]
    extra: dict[str, np.ndarray] = {}

    if "trajectory" in requested:
        # Repayment codes have ordered severity semantics; undocumented
        # demographic codes remain nominal elsewhere in the pipeline.
        weights = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
        for threshold in (1, 2, 3):
            late = status >= threshold
            extra[f"exp_late{threshold}_longest_run"] = _longest_run(late)
            extra[f"exp_late{threshold}_weighted_share"] = late @ weights
        differences = status[:, :-1] - status[:, 1:]
        extra["exp_overdue_weighted_severity"] = overdue @ weights
        extra["exp_overdue_recent_peak_gap"] = overdue[:, :2].max(1) - overdue[:, 2:].max(1)
        extra["exp_status_escalation_magnitude"] = np.maximum(differences, 0).sum(1)
        extra["exp_status_resolution_magnitude"] = np.maximum(-differences, 0).sum(1)
        extra["exp_status_transition_count"] = (differences != 0).sum(1)
        extra["exp_status_unique_count"] = np.apply_along_axis(lambda row: len(np.unique(row)), 1, status)
        extra["exp_nonlate_recent_run"] = _longest_run(status[:, :3] <= 0)

    if "coverage" in requested:
        active = positive_older_bills > 0
        for width in (2, 5):
            ratios = repayment_ratio[:, :width]
            shares = paid_share[:, :width]
            valid = np.maximum(active[:, :width].sum(1), 1)
            extra[f"exp_repayment_ratio_median_{width}"] = np.median(ratios, axis=1)
            extra[f"exp_repayment_ratio_min_active_{width}"] = np.where(
                active[:, :width], ratios, np.inf
            ).min(1)
            extra[f"exp_repayment_ratio_min_active_{width}"] = np.where(
                np.isfinite(extra[f"exp_repayment_ratio_min_active_{width}"]),
                extra[f"exp_repayment_ratio_min_active_{width}"],
                0,
            )
            extra[f"exp_paid_share_std_{width}"] = shares.std(1)
            extra[f"exp_underpaid_active_fraction_{width}"] = (
                ((ratios < 0.1) & active[:, :width]).sum(1) / valid
            )
        extra["exp_underpay_10pct_longest_run"] = _longest_run((repayment_ratio < 0.1) & active)
        extra["exp_underpay_50pct_longest_run"] = _longest_run((repayment_ratio < 0.5) & active)
        extra["exp_paid_share_linear_trend"] = paid_share @ np.array([2, 1, 0, -1, -2]) / 10
        extra["exp_payment_nonzero_fraction"] = (payments > 0).mean(1)
        extra["exp_payment_to_positive_bill_total"] = safe_ratio(
            payments.sum(1), np.maximum(bills, 0).sum(1), floor=100.0
        )

    if "trends" in requested:
        six_slope = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5]) / 17.5
        five_slope = np.array([2, 1, 0, -1, -2]) / 10
        extra["exp_utilization_linear_trend"] = utilization @ six_slope
        extra["exp_utilization_median"] = np.median(utilization, axis=1)
        extra["exp_utilization_range"] = utilization.max(1) - utilization.min(1)
        extra["exp_high_util_80pct_longest_run"] = _longest_run(utilization > 0.8)
        extra["exp_overlimit_longest_run"] = _longest_run(utilization > 1.0)
        extra["exp_overlimit_max_to_limit"] = np.maximum(utilization - 1, 0).max(1)
        extra["exp_cashflow_proxy_median"] = np.median(cashflow_proxy, axis=1)
        extra["exp_cashflow_proxy_min"] = cashflow_proxy.min(1)
        extra["exp_cashflow_proxy_max"] = cashflow_proxy.max(1)
        extra["exp_cashflow_proxy_linear_trend"] = cashflow_proxy @ five_slope
        extra["exp_payment_change_sign_count"] = (np.diff(payments, axis=1) != 0).sum(1)
        extra["exp_bill_change_sign_count"] = (np.diff(bills, axis=1) != 0).sum(1)

    if "bands" in requested:
        late_count = (status > 0).sum(1)
        severe_count = (status >= 2).sum(1)
        recent_coverage = paid_share[:, 0]
        latest_util = utilization[:, 0]
        extra["cat__exp_limit_band"] = _fixed_band(limit, [50_000, 100_000, 200_000, 300_000, 500_000])
        extra["cat__exp_age_band"] = _fixed_band(x["AGE"].to_numpy(), [25, 35, 45, 55, 65])
        extra["cat__exp_latest_status_band"] = _fixed_band(status[:, 0], [-1, 0, 1, 2, 3])
        extra["cat__exp_late_count_band"] = late_count.astype(float)
        extra["cat__exp_severe_count_band"] = severe_count.astype(float)
        extra["cat__exp_latest_util_band"] = _fixed_band(latest_util, [0, 0.25, 0.5, 0.75, 1, 1.25])
        extra["cat__exp_recent_coverage_band"] = _fixed_band(recent_coverage, [0.02, 0.05, 0.1, 0.25, 0.5, 1])
        # Mixed-radix combinations are collision-free for the fixed band ranges.
        extra["cat__exp_status_util_band"] = extra["cat__exp_latest_status_band"] * 8 + extra["cat__exp_latest_util_band"]
        extra["cat__exp_late_underpay_band"] = late_count * 8 + _fixed_band(
            (repayment_ratio < 0.1).sum(1), [1, 2, 3, 4, 5]
        )

    if "interactions" in requested:
        late_count = (status > 0).sum(1)
        severe_count = (status >= 2).sum(1)
        underpay_fraction = ((repayment_ratio < 0.1) & (positive_older_bills > 0)).mean(1)
        extra["exp_recent_severity_x_utilization"] = overdue[:, 0] * np.maximum(utilization[:, 0], 0)
        extra["exp_late_count_x_underpay_fraction"] = late_count * underpay_fraction
        extra["exp_severe_count_x_overlimit"] = severe_count * np.maximum(utilization.max(1) - 1, 0)
        extra["exp_utilization_x_uncovered_share"] = np.maximum(utilization[:, 0], 0) * (1 - paid_share[:, 0])
        extra["exp_zero_payments_x_late_count"] = (payments == 0).sum(1) * late_count

    additions = pd.DataFrame(extra, index=result.index).astype(float)
    result = pd.concat([result, additions], axis=1)
    if result.columns.duplicated().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Experimental features contain duplicates or nonfinite values")
    return result
