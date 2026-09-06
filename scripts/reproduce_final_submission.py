#!/usr/bin/env python3
"""Retrain the selected ensemble and reproduce the exact finalist submission.

Training artifacts are built in an isolated temporary directory. The public
`submission.csv` is replaced only after the complete pipeline produces the
frozen expected SHA-256, so a failed or drifting run cannot overwrite the
reviewed leaderboard file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import sha256_file


def run(command: list[str]) -> None:
    """Run one documented pipeline stage and surface failures immediately."""

    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    config = json.loads((ROOT / "disagreement_config.json").read_text(encoding="utf-8"))
    expected_hash = config["expected_submission_sha256"]
    with tempfile.TemporaryDirectory(prefix="datathon-final-reproduction-") as temporary:
        scratch = Path(temporary)
        base_artifacts = scratch / "archive_ensemble_v1"
        base_submission = scratch / "submission_archive_blend_v1.csv"
        generated_submission = scratch / "submission.csv"
        disagreement_artifacts = scratch / "archive_disagreement_v2"

        # Stage 1 fits every branch from the organizer files on the immutable
        # saved folds, then refits each branch on all labelled training rows.
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_archive_ensemble.py"),
                "--data-dir",
                str(args.data_dir),
                "--threads",
                str(args.threads),
                "--output-dir",
                str(base_artifacts),
                "--submission",
                str(base_submission),
            ]
        )

        # Stage 2 consumes the newly trained component probabilities, applies
        # the frozen blend/disagreement equation and the shared clipping rule.
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_disagreement_submission.py"),
                "--data-dir",
                str(args.data_dir),
                "--component-oof",
                str(base_artifacts / "oof_predictions.csv"),
                "--component-test",
                str(base_artifacts / "test_predictions.csv"),
                "--artifacts-dir",
                str(disagreement_artifacts),
                "--output",
                str(generated_submission),
            ]
        )

        generated_hash = sha256_file(generated_submission)
        if generated_hash != expected_hash:
            raise RuntimeError(
                f"Final submission drifted: expected {expected_hash}, got {generated_hash}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated_submission, args.output)

    copied_hash = sha256_file(args.output)
    if copied_hash != expected_hash:
        raise RuntimeError(f"Copied submission failed verification: {copied_hash}")
    print(f"Verified exact submission: {args.output} ({copied_hash})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
