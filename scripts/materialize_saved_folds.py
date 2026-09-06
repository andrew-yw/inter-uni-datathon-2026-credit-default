#!/usr/bin/env python3
"""Materialize the repository's established deterministic grouped folds once.

The model-training entry point never creates or changes folds.  This migration
utility exists because the restored public repository documented the splitter
but did not retain its row-level assignment.  It refuses to overwrite a file.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pipeline import ID_COLUMN, load_competition_data, make_grouped_folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "saved_folds.csv")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite saved folds: {args.output}")

    config = __import__("json").loads((ROOT / "config.json").read_text())
    data = load_competition_data(ROOT / "data" / "raw", config["input_sha256"])
    folds = make_grouped_folds(data.X, data.y, config["validation"]["n_splits"], config["validation"]["seed"])
    assignment = [-1] * len(data.train)
    for fold, (_, validation_rows) in enumerate(folds):
        for row in validation_rows:
            assignment[int(row)] = fold
    result = pd.DataFrame({ID_COLUMN: data.train[ID_COLUMN], "fold": assignment})
    if (result["fold"] < 0).any():
        raise RuntimeError("Fold coverage is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, lineterminator="\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"saved {len(result)} immutable assignments to {args.output}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
