# Inter-Uni Datathon 2026 - Credit Default Ensemble

This repository contains the integrity-safe final pipeline and proposed submission for Stream 1. The current candidate reconstructs the honest model-only portion of the supplied archive: CatBoost, Random Forest, and Bayesian logistic models trained solely on organizer-provided labels, followed by a fixed probability blend. The archive's external source-table label reconstruction is deliberately excluded.

## Final files

- `submission.csv` - the exact proposed file to upload and submit for finalist review.
- `scripts/train_archive_ensemble.py` - trains all three models on the saved folds, reports OOF results, refits on all training rows, and regenerates `submission.csv`.
- `ensemble_config.json` - reviewed model recipes, fixed blend weights, fold hash, input hashes, and final output hash.
- `artifacts/archive_ensemble_v1/` - OOF/test probabilities, fold and overall metrics, correlations, feature importance, fitted models, and run manifests.
- `scripts/train_and_submit.py` - retained interpretable logistic baseline; it writes its separate versioned submission under `submissions/`.
- `METHOD.md` - methodology, results, limitations, and model-selection rationale.
- `DISCLOSURE.md` - external-data, software, AI-tool, and post-processing disclosure.
- `docs/SUBMISSION_RULES.md` - checklist transcribed from the organizer PDF and announcement.

## Reproduce the exact submission

Use Python 3.13.2. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the three organizer competition files and place them in `data/raw/` as described in `data/raw/README.md`. Then run:

```bash
python scripts/train_archive_ensemble.py
python -m pytest -q
shasum -a 256 submission.csv
```

The final hash must equal `expected_submission_sha256` in `ensemble_config.json`. Input hashes, package versions, fold scores, and the output hash are recorded in `artifacts/archive_ensemble_v1/run_manifest.json`. The training entry point reads the committed, hash-checked `artifacts/saved_folds.csv` and refuses to recreate missing folds.

The complete verified run takes approximately four minutes on the development machine. Its fixed-blend OOF log loss is **0.421507**, versus **0.434075** for the retained logistic baseline.

## Submission status

The repository file is a proposed replacement for the ineligible perfect-score output. `submission.csv` is the reconstructed fixed ensemble; the earlier logistic file remains at `submissions/submission_logistic_interpretable_v1.csv`. To satisfy the organizer PDF, upload the root `submission.csv` and submit this same public repository for review. Record the returned leaderboard score in `METHOD.md` without modifying the CSV.

Only one team member should submit, and the team should submit only once before midnight at the end of Sunday, 6 September 2026.
