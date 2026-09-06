"""Load local competition data and enforce the submission contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# In the public repository the competition files are local, ignored inputs.
# Keeping this path explicit prevents a fallback to any external source table.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ID = "client_id"
TARGET = "default"
PREDICTION = "default_probability"
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHICS = ["SEX", "EDUCATION", "MARRIAGE"]
FEATURES = ["LIMIT_BAL", *DEMOGRAPHICS, "AGE", *PAY_STATUS, *BILLS, *PAYMENTS]


@dataclass
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    fingerprints: dict[str, str]

    @property
    def X(self) -> pd.DataFrame:
        return self.train[FEATURES].copy()

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy(dtype=np.int8)

    @property
    def groups(self) -> np.ndarray:
        # Identical raw feature vectors stay together in every validation split.
        return pd.util.hash_pandas_object(self.X, index=False).to_numpy()


def load_data(directory: Path = DATA_DIR) -> CompetitionData:
    frames, fingerprints = {}, {}
    for name in ("train", "test", "sample_submission"):
        path = Path(directory) / f"{name}.csv"
        frames[name] = pd.read_csv(path, dtype={ID: str})
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    train, test, sample = (frames[n] for n in ("train", "test", "sample_submission"))
    expected = {"train": [ID, *FEATURES, TARGET], "test": [ID, *FEATURES],
                "sample_submission": [ID, PREDICTION]}
    for name, frame in frames.items():
        if list(frame.columns) != expected[name]:
            raise ValueError(f"Unexpected {name} columns: {list(frame.columns)}")
        if frame.isna().any().any() or frame[ID].duplicated().any():
            raise ValueError(f"Missing values or duplicate identifiers in {name}")
    if set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("Training target must contain exactly 0 and 1")
    if set(train[ID]) & set(test[ID]):
        raise ValueError("Train/test identifiers overlap")
    if set(sample[ID]) != set(test[ID]) or len(sample) != len(test):
        raise ValueError("Submission template does not match test identifiers")
    for frame in (train, test):
        if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite raw features")
    return CompetitionData(train, test, sample, fingerprints)


def make_folds(X, y, groups, n_splits: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X, y, groups))
    coverage = np.zeros(len(y), dtype=int)
    for training, validation in folds:
        if set(groups[training]) & set(groups[validation]):
            raise AssertionError("Duplicate feature groups cross a fold boundary")
        if len(np.unique(y[training])) != 2 or len(np.unique(y[validation])) != 2:
            raise ValueError("Each fold needs both target classes")
        coverage[validation] += 1
    if not np.all(coverage == 1):
        raise AssertionError("Invalid OOF coverage")
    return folds


def write_submission(data: CompetitionData, predictions, path: Path) -> pd.DataFrame:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(data.test),):
        raise ValueError("One prediction per test row is required")
    if not np.isfinite(predictions).all() or not ((predictions >= 0) & (predictions <= 1)).all():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    keyed = pd.Series(predictions, index=data.test[ID])
    result = data.sample[[ID]].copy()
    result[PREDICTION] = keyed.reindex(result[ID]).to_numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.10f")
    reread = pd.read_csv(path, dtype={ID: str})
    if list(reread.columns) != [ID, PREDICTION] or reread[ID].tolist() != data.sample[ID].tolist():
        raise AssertionError("Submission round-trip failed")
    return result


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
