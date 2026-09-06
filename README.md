# Inter-Uni Datathon 2026 - Credit Default Ensemble

This repository contains the integrity-safe final pipeline and proposed submission for Stream 1. The current candidate extends the honest model-only portion of the supplied archive: an eight-member CatBoost seed/view bag, Random Forest, and Bayesian logistic models trained solely on organizer-provided labels, followed by the archive's fixed blend and small disagreement-calibration layer. The archive's external source-table label reconstruction is deliberately excluded.

## Final files

- `submission.csv` - the exact proposed file to upload and submit for finalist review.
- `scripts/train_archive_ensemble.py` - trains all three models on the saved folds, reports OOF results, refits on all training rows, and regenerates the base-blend component artifacts.
- `scripts/build_disagreement_submission.py` - applies the audited eight-parameter disagreement layer and regenerates the final `submission.csv`.
- `scripts/train_seed_bagged_challenger.py` - fits four seeds across numeric and categorical repayment-status CatBoost views and generates the selected final submission.
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
python scripts/build_disagreement_submission.py
python scripts/train_seed_bagged_challenger.py
python -m pytest -q
shasum -a 256 submission.csv
```

The final hash must equal `expected_submission_sha256` in `seed_bagging_config.json`. Input hashes, package versions, fold scores, and component hashes are recorded in the versioned artifact directories. The training entry points read the committed, hash-checked `artifacts/saved_folds.csv` and refuse to recreate missing folds.

The selected seed-bagged challenger has OOF log loss **0.421138** and improves all five folds over the `0.421371` single-seed disagreement reconstruction. The fixed base blend scores **0.421507**, versus **0.434075** for the retained logistic baseline.

Optional booster research is reproducible with `requirements-research.txt`, `scripts/research_boosters.py`, and `scripts/analyze_booster_screen.py`. On macOS, LightGBM also requires Homebrew `libomp`. These experiments are documented but do not replace the selected submission because their best same-OOF blend gain is only 0.000184 and is not nested validation.

## Submission status

The repository file is a proposed replacement for the ineligible perfect-score output. `submission.csv` is the seed-bagged disagreement ensemble; the single-seed disagreement, fixed base blend, and logistic files remain versioned under `submissions/`. To satisfy the organizer PDF, upload the root `submission.csv` and submit this same public repository for review. Record the returned leaderboard score in `METHOD.md` without modifying the CSV.

Only one team member should submit, and the team should submit only once before midnight at the end of Sunday, 6 September 2026.
