"""Probability blending used by the archive's clean fallback model."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def fixed_probability_blend(matrix, weights, epsilon: float = 1e-7) -> np.ndarray:
    """Combine component probabilities with reviewed non-negative weights."""

    values = np.asarray(matrix, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weight.shape != (values.shape[1],):
        raise ValueError("One blend weight is required per prediction column")
    if (weight < 0).any() or not np.isclose(weight.sum(), 1.0):
        raise ValueError("Blend weights must be non-negative and sum to one")
    return np.clip(values @ weight, epsilon, 1 - epsilon)


def disagreement_adjustment(matrix, weights, theta, epsilon: float = 1e-7) -> np.ndarray:
    """Adjust a base blend using calibration, model deviations and uncertainty.

    The design follows the archived fallback exactly: two beta-calibration
    terms, an intercept, each model's logit deviation from the base blend,
    model-logit spread, and a spread-by-base-logit interaction.
    """

    values = np.clip(np.asarray(matrix, dtype=float), epsilon, 1 - epsilon)
    base = fixed_probability_blend(values, weights, epsilon)
    offset = logit(base)
    component_logits = logit(values)
    spread = component_logits.std(axis=1)
    design = np.column_stack(
        [
            np.log(base),
            -np.log1p(-base),
            np.ones(len(base)),
            *(component_logits - offset[:, None]).T,
            spread,
            offset * spread,
        ]
    )
    coefficient = np.asarray(theta, dtype=float)
    if coefficient.shape != (design.shape[1],):
        raise ValueError(f"Expected {design.shape[1]} disagreement coefficients")
    return np.clip(expit(offset + design @ coefficient), epsilon, 1 - epsilon)
