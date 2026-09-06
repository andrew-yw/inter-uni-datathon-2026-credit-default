"""Data contract, model construction, evaluation, and submission utilities.

Only organizer-provided training labels are accepted. The external UCI source
table found in the superseded archive is deliberately unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ID_COLUMN = "client_id"
TARGET_COLUMN = "default"
PREDICTION_COLUMN = "default_probability"
STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", *STATUS_COLUMNS]
AMOUNT_COLUMNS = ["LIMIT_BAL", *BILL_COLUMNS, *PAYMENT_COLUMNS]
NUMERIC_COLUMNS = [*AMOUNT_COLUMNS, "AGE"]
FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *STATUS_COLUMNS, *BILL_COLUMNS, *PAYMENT_COLUMNS,
]


@dataclass(frozen=True)
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame

    @property
    def X(self) -> pd.DataFrame:
        # The explicit allowlist prevents identifiers or labels entering a model.
        return self.train.loc[:, FEATURE_COLUMNS].copy()

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET_COLUMN].to_numpy(dtype=np.int8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_competition_data(data_dir: Path, expected_hashes: dict[str, str]) -> CompetitionData:
    """Load only the three organizer-provided competition files."""

    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for filename in ("train.csv", "test.csv", "sample_submission.csv"):
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing organizer file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hashes[filename]:
            raise ValueError(f"Unexpected content for {filename}: {actual_hash}")
        frames[filename] = pd.read_csv(path, dtype={ID_COLUMN: "string"})

    train = frames["train.csv"]
    test = frames["test.csv"]
    sample = frames["sample_submission.csv"]
    expected_columns = {
        "train": [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN],
        "test": [ID_COLUMN, *FEATURE_COLUMNS],
        "sample": [ID_COLUMN, PREDICTION_COLUMN],
    }
    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        if list(frame.columns) != expected_columns[name]:
            raise ValueError(f"Unexpected {name} schema: {list(frame.columns)}")
        if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"Missing or duplicate identifiers in {name}")
    if train[FEATURE_COLUMNS].isna().any().any() or test[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("This version expects complete organizer feature matrices")
    if set(train[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training target must contain exactly classes 0 and 1")
    if set(train[ID_COLUMN]).intersection(test[ID_COLUMN]):
        raise ValueError("Training and test identifiers overlap")
    if sample[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Submission template order does not match test order")
    return CompetitionData(train=train, test=test, sample=sample)


def signed_log(values: object) -> np.ndarray:
    """Compress currency outliers without discarding negative account balances."""

    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.log1p(np.abs(array))


def build_model(C: float, seed: int, max_iter: int = 2500) -> Pipeline:
    """Construct a fold-safe, coefficient-based probability model.

    Undocumented source codes are nominal, so demographics and repayment-status
    codes are one-hot encoded. Currency values use a signed log transform and all
    continuous variables are standardized before L2 logistic regression.
    """

    amounts = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("signed_log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    age = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categories = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transform = ColumnTransformer(
        [
            ("amount", amounts, AMOUNT_COLUMNS),
            ("age", age, ["AGE"]),
            ("category", categories, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    classifier = LogisticRegression(
        C=C,
        # scikit-learn 1.9 expresses L2 through l1_ratio=0; setting the older
        # penalty argument emits a deprecation warning.
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([("transform", transform), ("classifier", classifier)])


def make_grouped_folds(X: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int):
    """Keep exact duplicate customer profiles in one validation partition."""

    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups))
    coverage = np.zeros(len(y), dtype=np.int8)
    for training, validation in folds:
        if set(groups[training]).intersection(groups[validation]):
            raise AssertionError("Duplicate feature profile crossed a validation fold")
        coverage[validation] += 1
    if not np.all(coverage == 1):
        raise AssertionError("OOF validation does not cover each row exactly once")
    return folds


def positive_probability(model: Pipeline, X: pd.DataFrame, epsilon: float) -> np.ndarray:
    """Return P(default=1) under one documented clipping rule."""

    classes = list(model.named_steps["classifier"].classes_)
    probability = model.predict_proba(X)[:, classes.index(1)]
    return np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Expose the fitted linear effects for global model interpretation."""

    names = model.named_steps["transform"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    result = pd.DataFrame({"feature": names, "coefficient": coefficients})
    result["odds_ratio_per_transformed_unit"] = np.exp(result["coefficient"])
    result["absolute_coefficient"] = result["coefficient"].abs()
    return result.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
