"""Fold-fitted Bayesian classifiers with their own compact feature representations.

``histogram_nb`` integrates a symmetric Dirichlet categorical likelihood and a
Beta(1, 1) class prior. It deliberately excludes all six parallel monthly vectors.
``laplace_logistic`` places a Gaussian prior on spline/logistic coefficients,
approximates their posterior by a full-covariance Gaussian at its MAP, and
integrates predictive probabilities with 20-point Gauss-Hermite quadrature.
The latter is approximate Bayesian prediction, not just a renamed MAP estimate.
Neither fitting method reads outer validation rows or labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from threadpoolctl import threadpool_limits

from .data import BILLS, DEMOGRAPHICS, FEATURES, PAYMENTS, PAY_STATUS


def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def bayesian_features(raw: pd.DataFrame, representation: str = "behavior") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed row-local features; IDs and targets are excluded explicitly."""
    x = raw.loc[:, FEATURES].astype(float)
    status = x[PAY_STATUS].to_numpy()
    bills = x[BILLS].to_numpy()
    payments = x[PAYMENTS].to_numpy()
    limit = np.maximum(x.LIMIT_BAL.to_numpy(), 100.)
    positive_bills = np.maximum(bills, 0)
    floor = np.maximum(limit * .001, 100.)
    coverage = payments[:, :5] / (positive_bills[:, 1:] + payments[:, :5] + floor[:, None])
    cats = x[DEMOGRAPHICS].astype(int).copy()
    for i, name in enumerate(PAY_STATUS):
        # Rare positive repayment codes share a regularized >=3 category.
        cats[name] = np.minimum(status[:, i], 3).astype(int)
    cont = {
        "log_limit": np.log1p(limit), "age": x.AGE.to_numpy(),
        "bill_recent_log": _signed_log(bills[:, 0]),
        "bill_history_log": _signed_log(bills[:, 1:].mean(1)),
        "pay_recent_log": _signed_log(payments[:, 0]),
        "pay_history_log": _signed_log(payments[:, 1:].mean(1)),
        "util_recent": np.clip(bills[:, 0] / limit, -2, 3),
        "util_history": np.clip(bills[:, 1:].mean(1) / limit, -2, 3),
        "cover_recent": coverage[:, 0], "cover_history": coverage[:, 1:].mean(1),
        "late_older_count": (status[:, 1:] > 0).sum(1),
        "bill_change_to_limit": np.clip((bills[:, 0] - bills[:, 1]) / limit, -3, 3),
        "cashflow_proxy_to_limit": np.clip((bills[:, 0] - bills[:, 1] + payments[:, 0]) / limit, -3, 3),
        "pay_zero_count": (payments == 0).sum(1),
        "negative_bill_count": (bills < 0).sum(1),
    }
    if representation == "behavior":
        for i in range(1, 6):
            cont[f"pay_month_{i+1}_log"] = _signed_log(payments[:, i])
        for i in (1, 2, 5):
            cont[f"bill_month_{i+1}_log"] = _signed_log(bills[:, i])
        cont["pay_recent_trend"] = _signed_log(payments[:, :2].mean(1) - payments[:, 2:].mean(1))
        cont["bill_relative_variation"] = np.clip(bills.std(1) / limit, 0, 3)
        cont["late_recent_run"] = np.cumprod(status > 0, axis=1).sum(1)
    elif representation != "compact":
        raise ValueError(f"Unknown Bayesian representation: {representation}")
    continuous = pd.DataFrame(cont, index=raw.index, dtype=float)
    if not np.isfinite(continuous.to_numpy()).all():
        raise ValueError("Nonfinite Bayesian features")
    return cats, continuous


class BayesianDesign:
    """Train-only clipping, quantile spline knots, category encoding and scaling."""

    def __init__(self, representation="behavior", knots=4, interactions=True, interaction_splines=False):
        self.representation, self.knots, self.interactions = representation, knots, interactions
        self.interaction_splines = interaction_splines

    def _categories(self, cats):
        selected = cats.copy()
        if self.interactions:
            selected["status_recent_pair"] = (cats.PAY_0 + 2) * 8 + cats.PAY_2 + 2
        return selected

    def _unscaled(self, cats, continuous):
        clipped = np.clip(continuous.to_numpy(), self.low_, self.high_)
        spline = self.spline_.transform(clipped[:, self.spline_columns_])
        linear = clipped[:, ~self.spline_columns_]
        encoded = self.encoder_.transform(self._categories(cats))
        pieces = [encoded, spline, linear]
        if self.interactions:
            # Economic effects may differ with latest repayment status. Use a
            # small fixed interaction set, rather than every pair of features.
            economic = (clipped[:, self.interaction_indices_] - self.interaction_mean_) / self.interaction_scale_
            groups = np.column_stack([cats.PAY_0.to_numpy() == value for value in (-2, -1, 1, 2, 3)])
            pieces.append((groups[:, :, None] * economic[:, None, :]).reshape(len(cats), -1))
            if self.interaction_splines:
                curved = self.interaction_spline_.transform(clipped[:, self.interaction_indices_])
                pieces.append((groups[:, :, None] * curved[:, None, :]).reshape(len(cats), -1))
        return np.column_stack(pieces)

    def fit_transform(self, raw):
        cats, continuous = bayesian_features(raw, self.representation)
        values = continuous.to_numpy()
        self.continuous_columns_ = list(continuous.columns)
        self.low_, self.high_ = np.quantile(values, [.0025, .9975], axis=0)
        clipped = np.clip(values, self.low_, self.high_)
        self.spline_columns_ = np.array([len(np.unique(clipped[:, j])) >= 6 for j in range(values.shape[1])])
        self.spline_ = SplineTransformer(n_knots=self.knots, degree=3, knots="quantile", include_bias=False, extrapolation="constant")
        self.spline_.fit(clipped[:, self.spline_columns_])
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        self.encoder_.fit(self._categories(cats))
        economic_names = ["log_limit", "age", "pay_recent_log", "pay_history_log", "util_recent", "cover_recent", "cover_history"]
        self.interaction_indices_ = [continuous.columns.get_loc(col) for col in economic_names]
        economic = clipped[:, self.interaction_indices_]
        self.interaction_mean_ = economic.mean(0)
        self.interaction_scale_ = np.maximum(economic.std(0), .05)
        if self.interaction_splines:
            self.interaction_spline_ = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False, extrapolation="constant")
            self.interaction_spline_.fit(economic)
        basis = self._unscaled(cats, continuous)
        self.mean_ = basis.mean(0)
        self.scale_ = np.maximum(basis.std(0), .1)
        return np.column_stack([np.ones(len(raw)), (basis - self.mean_) / self.scale_])

    def transform(self, raw):
        cats, continuous = bayesian_features(raw, self.representation)
        basis = self._unscaled(cats, continuous)
        return np.column_stack([np.ones(len(raw)), (basis - self.mean_) / self.scale_])


@dataclass
class LaplaceLogistic:
    design: BayesianDesign
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    metadata: dict
    threads: int = 2

    def predict(self, raw):
        design = self.design.transform(raw)
        with threadpool_limits(limits=self.threads):
            mean = design @ self.posterior_mean
            variance = np.maximum(np.einsum("ij,ij->i", design @ self.posterior_covariance, design), 0)
        nodes, weights = np.polynomial.hermite.hermgauss(20)
        probability = (expit(mean[:, None] + np.sqrt(2 * variance[:, None]) * nodes) @ weights) / np.sqrt(np.pi)
        return np.clip(probability, 1e-7, 1 - 1e-7)

    def predict_map(self, raw):
        return expit(self.design.transform(raw) @ self.posterior_mean)

    def feature_importance(self, columns=None):
        return {}


def _nb_features(raw):
    cats, continuous = bayesian_features(raw, "compact")
    # One latest-status representation, one older-history summary and a small
    # economic/demographic set avoid six repeated likelihood factors per family.
    return cats[["PAY_0", "EDUCATION", "MARRIAGE"]], continuous[[
        "late_older_count", "log_limit", "age", "pay_history_log", "cover_recent"]]


@dataclass
class HistogramBayes:
    category_maps: list[dict]
    bin_edges: list[np.ndarray]
    likelihoods: list[np.ndarray]
    class_log_prior: np.ndarray
    metadata: dict

    def _encode(self, raw):
        cats, continuous = _nb_features(raw)
        codes = [np.array([mapping.get(value, len(mapping)) for value in cats.iloc[:, j]], dtype=int)
                 for j, mapping in enumerate(self.category_maps)]
        codes.extend(np.searchsorted(edges, continuous.iloc[:, j].to_numpy(), side="right")
                     for j, edges in enumerate(self.bin_edges))
        return codes

    def predict(self, raw):
        score = np.broadcast_to(self.class_log_prior, (len(raw), 2)).copy()
        for codes, likelihood in zip(self._encode(raw), self.likelihoods):
            score += likelihood[:, codes].T
        return np.clip(np.exp(score[:, 1] - logsumexp(score, axis=1)), 1e-7, 1-1e-7)

    def feature_importance(self, columns=None):
        return {}


def fit_bayesian(raw, y, raw_valid=None, y_valid=None, *, params=None, seed=2026, threads=2, iterations=None):
    """Fit on ``raw,y`` only; outer validation parameters are intentionally unused."""
    params = dict(params or {})
    y = np.asarray(y, dtype=float)
    if y.shape != (len(raw),) or set(np.unique(y)) != {0., 1.}:
        raise ValueError("Both binary target classes are required")
    kind = params.get("kind", "laplace_logistic")
    if kind == "histogram_nb":
        cats, continuous = _nb_features(raw)
        n_bins, alpha = int(params.get("bins", 6)), float(params.get("alpha", 5.))
        if n_bins < 2 or alpha <= 0:
            raise ValueError("At least two bins and positive Dirichlet alpha required")
        maps = [{value: index for index, value in enumerate(sorted(cats.iloc[:, j].unique()))}
                for j in range(cats.shape[1])]
        edges = [np.unique(np.quantile(continuous.iloc[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
                 for j in range(continuous.shape[1])]
        model = HistogramBayes(maps, edges, [], np.log((np.bincount(y.astype(int), minlength=2) + 1.) / (len(y) + 2.)),
                               {"kind": kind, "bins": n_bins, "alpha": alpha, "feature_count": cats.shape[1] + continuous.shape[1]})
        sizes = [len(mapping) + 1 for mapping in maps] + [len(edge) + 1 for edge in edges]
        for codes, size in zip(model._encode(raw), sizes):
            counts = np.vstack([np.bincount(codes[y == label], minlength=size) for label in (0, 1)])
            model.likelihoods.append(np.log((counts + alpha) / (counts.sum(1, keepdims=True) + alpha * size)))
        return model, 1
    if kind != "laplace_logistic":
        raise ValueError(f"Unsupported Bayesian kind {kind}")
    design = BayesianDesign(params.get("representation", "behavior"), int(params.get("knots", 4)), params.get("interactions", True), params.get("interaction_splines", False))
    matrix = np.ascontiguousarray(design.fit_transform(raw))
    precision = np.full(matrix.shape[1], float(params.get("prior_precision", 20.)))
    # Proper weak N(0, 100^2) prior on the intercept.
    precision[0] = .0001
    if np.any(precision <= 0):
        raise ValueError("Gaussian prior precision must be positive")
    initial = np.zeros(matrix.shape[1])
    initial[0] = np.log(y.mean() / (1-y.mean()))

    def objective(beta):
        logits = matrix @ beta
        loss = np.logaddexp(0, logits).sum() - y @ logits + .5 * (precision * beta) @ beta
        gradient = matrix.T @ (expit(logits) - y) + precision * beta
        return loss, gradient

    with threadpool_limits(limits=threads):
        # A previous fold's optimizer iterations are a diagnostic, not a training
        # hyperparameter. Full-data fitting must converge independently.
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": int(params.get("maxiter", 800)), "ftol": 1e-11, "gtol": 1e-5, "maxls": 40})
        if not result.success:
            raise RuntimeError(f"Bayesian MAP optimization failed: {result.message}")
        p = expit(matrix @ result.x)
        hessian = (matrix.T * (p * (1-p))) @ matrix + np.diag(precision)
        covariance = cho_solve(cho_factor(hessian, lower=True, check_finite=False), np.eye(matrix.shape[1]), check_finite=False)
    metadata = {"kind": kind, "posterior_approximation": "full_covariance_laplace", "predictive_integration": "20_point_gauss_hermite",
                "basis_count": matrix.shape[1], "prior_precision": float(precision[1]), "map_iterations": int(result.nit),
                "gradient_max_abs": float(np.abs(result.jac).max()), "optimizer_success": bool(result.success)}
    return LaplaceLogistic(design, result.x, covariance, metadata, threads), int(result.nit)


def bayesian_recipes():
    return [
        {"name": "bayes_histogram_6", "params": {"kind": "histogram_nb", "bins": 6, "alpha": 5.}},
        {"name": "bayes_laplace_additive", "params": {"kind": "laplace_logistic", "representation": "behavior", "knots": 4, "interactions": False, "prior_precision": 20.}},
        {"name": "bayes_laplace_interactions", "params": {"kind": "laplace_logistic", "representation": "behavior", "knots": 4, "interactions": True, "prior_precision": 20.}},
        {"name": "bayes_laplace_smooth", "params": {"kind": "laplace_logistic", "representation": "behavior", "knots": 5, "interactions": True, "prior_precision": 60.}},
        {"name": "bayes_laplace_curved", "params": {"kind": "laplace_logistic", "representation": "behavior", "knots": 5, "interactions": True, "interaction_splines": True, "prior_precision": 60.}},
    ]
