"""Estimator construction, early stopping, and serializable prediction bundles."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import inspect
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

from .features import model_view
from .data import DEMOGRAPHICS, PAY_STATUS


@dataclass
class Recipe:
    name: str
    family: str
    features: str
    params: dict
    status_categories: bool = False

    def to_dict(self):
        return asdict(self)


def signed_log(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def fit_model(recipe: Recipe, X: pd.DataFrame, y, X_valid=None, y_valid=None,
              seed: int = 2026, threads: int = 6, iterations: int | None = None):
    if recipe.family == "neural":
        from .neural_models import fit_neural
        return fit_neural(X, y, X_valid, y_valid, params=recipe.params,
                          seed=seed, threads=threads, iterations=iterations)
    if recipe.family == "bayesian":
        from .bayesian_models import fit_bayesian
        return fit_bayesian(X, y, X_valid, y_valid, params=recipe.params,
                            seed=seed, threads=threads, iterations=iterations)
    train, categorical = model_view(X, recipe.family, recipe.status_categories)
    valid = model_view(X_valid, recipe.family, recipe.status_categories)[0] if X_valid is not None else None
    params = dict(recipe.params)
    if recipe.family == "catboost":
        from catboost import CatBoostClassifier
        default_iterations = params.pop("iterations", 3500)
        model = CatBoostClassifier(
            iterations=iterations or default_iterations,
            loss_function="Logloss", eval_metric="Logloss", random_seed=seed,
            thread_count=threads, verbose=False, allow_writing_files=False,
            **params)
        kwargs = {"cat_features": categorical}
        if valid is not None:
            kwargs.update(eval_set=(valid, y_valid), early_stopping_rounds=180, use_best_model=True)
        model.fit(train, y, **kwargs)
        best_iteration = model.tree_count_
    elif recipe.family == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        default_iterations = params.pop("n_estimators", 4000)
        model = LGBMClassifier(
            n_estimators=iterations or default_iterations,
            objective="binary", metric="binary_logloss", random_state=seed,
            n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True,
            **params)
        kwargs = {"categorical_feature": categorical}
        if valid is not None:
            kwargs.update(lgb_validation_arguments(model, valid, y_valid))
            kwargs.update(eval_metric="binary_logloss",
                          callbacks=[early_stopping(150, verbose=False), log_evaluation(0)])
        model.fit(train, y, **kwargs)
        best_iteration = model.best_iteration_ or model.n_estimators
    elif recipe.family == "xgboost":
        from xgboost import XGBClassifier
        default_iterations = params.pop("n_estimators", 4000)
        model = XGBClassifier(
            n_estimators=iterations or default_iterations,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=threads, random_state=seed,
            **({"early_stopping_rounds": 150} if valid is not None else {}), **params)
        model.fit(train, y, **({"eval_set": [(valid, y_valid)], "verbose": False} if valid is not None else {}))
        best_iteration = model.best_iteration + 1 if valid is not None else model.n_estimators
    elif recipe.family == "logistic":
        categorical = DEMOGRAPHICS + PAY_STATUS
        continuous = [c for c in train.columns if c not in categorical]
        transform = ColumnTransformer([
            ("numeric", Pipeline([("signed_log", FunctionTransformer(signed_log)),
                                   ("scale", StandardScaler())]), continuous),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        ])
        model = Pipeline([("transform", transform), ("classifier", LogisticRegression(
            C=params.get("C", 0.1), max_iter=2500, solver="lbfgs", random_state=seed))])
        model.fit(train, y)
        best_iteration = int(model.named_steps["classifier"].n_iter_[0])
    elif recipe.family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        default_trees = params.pop("n_estimators", 400)
        # Fit nominal-category encoding inside this training fold only.
        transform = ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        ], remainder="passthrough", sparse_threshold=0)
        model = Pipeline([
            ("transform", transform),
            ("classifier", RandomForestClassifier(
                n_estimators=iterations or default_trees,
                random_state=seed, n_jobs=threads, **params))
        ])
        # Forests use a fixed tree count; validation labels are not used for fitting.
        model.fit(train, y)
        best_iteration = len(model.named_steps["classifier"].estimators_)
    else:
        raise ValueError(recipe.family)
    return ModelBundle(recipe, model), int(best_iteration)


def lgb_validation_arguments(model, X, y):
    if "eval_X" in inspect.signature(model.fit).parameters:
        return {"eval_X": X, "eval_y": y}
    return {"eval_set": [(X, y)]}


@dataclass
class ModelBundle:
    recipe: Recipe
    model: object

    def predict(self, features: pd.DataFrame):
        view = model_view(features, self.recipe.family, self.recipe.status_categories)[0]
        classes = list(self.model.classes_)
        probabilities = self.model.predict_proba(view)[:, classes.index(1)]
        return np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)

    def feature_importance(self, columns):
        if hasattr(self.model, "feature_importances_"):
            values = np.asarray(self.model.feature_importances_, dtype=float)
            return dict(zip(columns, (values / max(values.sum(), 1e-12)).tolist()))
        return {}


def default_recipes(preset="strong"):
    recipes = [Recipe("logistic_raw", "logistic", "raw", {"C": 0.3})]
    cat_combinations = [("raw", 4), ("raw", 5), ("raw", 6), ("compact", 4),
                        ("compact", 5), ("compact", 6), ("engineered", 4), ("engineered", 5)]
    for mode, depth in cat_combinations:
        recipes.append(Recipe(f"cat_{mode}_d{depth}", "catboost", mode, {
            "depth": depth, "learning_rate": 0.035, "l2_leaf_reg": 6.0,
            "random_strength": 0.5, "bootstrap_type": "Bayesian", "bagging_temperature": 0.5,
            "border_count": 128}))
    recipes.append(Recipe("cat_compact_ordered", "catboost", "compact", {
        "depth": 4, "learning_rate": .045, "l2_leaf_reg": 8., "boosting_type": "Ordered",
        "random_strength": .5, "bootstrap_type": "Bayesian", "bagging_temperature": .4,
        "border_count": 128}))
    recipes.append(Recipe("cat_compact_status_categories", "catboost", "compact", {
        "depth": 5, "learning_rate": .035, "l2_leaf_reg": 8., "random_strength": .7,
        "bootstrap_type": "Bayesian", "bagging_temperature": .5, "border_count": 128}, True))
    for mode in ("raw", "compact", "engineered"):
        for leaves in (15, 31):
            recipes.append(Recipe(f"lgb_{mode}_l{leaves}", "lightgbm", mode, {
                "num_leaves": leaves, "learning_rate": .025, "max_depth": -1,
                "min_child_samples": 100 if leaves == 15 else 140,
                "reg_lambda": 10., "reg_alpha": .2, "colsample_bytree": .85,
                "subsample": .85, "subsample_freq": 1, "max_bin": 127}))
    for mode in ("raw", "compact", "engineered"):
        for depth in (3, 4):
            recipes.append(Recipe(f"xgb_{mode}_d{depth}", "xgboost", mode, {
                "max_depth": depth, "learning_rate": .03, "min_child_weight": 15,
                "reg_lambda": 15., "reg_alpha": .2, "subsample": .85,
                "colsample_bytree": .9, "max_bin": 128}))
    if preset == "quick":
        names = {"logistic_raw", "cat_raw_d5", "cat_compact_d4", "lgb_compact_l15", "xgb_compact_d3"}
        recipes = [r for r in recipes if r.name in names]
    return recipes
