# Inter-Uni Datathon 2026 - Credit Default Ensemble

This repository contains the integrity-safe final pipeline and proposed submission for Stream 1. The selected candidate reconstructs the honest model-only portion of the supplied archive: CatBoost, Random Forest, and Bayesian logistic models trained solely on organizer-provided labels, followed by the archive's fixed blend and small disagreement-calibration layer. The archive's external source-table label reconstruction is deliberately excluded.

## Final files

- `submission.csv` - the exact proposed file to upload and submit for finalist review.
- `scripts/train_archive_ensemble.py` - trains all three models on the saved folds, reports OOF results, refits on all training rows, and regenerates the base-blend component artifacts.
- `scripts/build_disagreement_submission.py` - applies the audited eight-parameter disagreement layer and regenerates the final `submission.csv`.
- `scripts/train_seed_bagged_challenger.py` - reproduces a seed-bagged research challenger that improved CV but was rejected after worse competition-test log loss.
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
python -m pytest -q
shasum -a 256 submission.csv
```

The final hash must equal `expected_submission_sha256` in `disagreement_config.json`. Input hashes, package versions, fold scores, and component hashes are recorded in the versioned artifact directories. The training entry points read the committed, hash-checked `artifacts/saved_folds.csv` and refuse to recreate missing folds.

The selected disagreement reconstruction has replay OOF log loss **0.421371**; the archive's nested estimate is **0.421278**. A seed-bagged challenger reached 0.421138 in CV but increased competition-test log loss, so it is retained only as a rejected experiment. The fixed base blend scores **0.421507**, versus **0.434075** for the logistic baseline.

Optional booster research is reproducible with `requirements-research.txt`, `scripts/research_boosters.py`, and `scripts/analyze_booster_screen.py`. On macOS, LightGBM also requires Homebrew `libomp`. These experiments are documented but do not replace the selected submission because their best same-OOF blend gain is only 0.000184 and is not nested validation.

The independent feature-engineering archive is documented in `docs/experiments/feature_engineering.md`. It tests eight feature-block ablations plus four-seed stability without accessing test rows or changing `submission.csv`; the observed gains were too seed-sensitive to promote.

## Submission status

The repository file is a proposed replacement for the ineligible perfect-score output. `submission.csv` is the restored single-seed disagreement ensemble; the rejected seed bag, fixed base blend, and logistic files remain versioned under `submissions/`. To satisfy the organizer PDF, upload the root `submission.csv` and submit this same public repository for review. Record the returned leaderboard score in `METHOD.md` without modifying the CSV.

Only one team member should submit, and the team should submit only once before midnight at the end of Sunday, 6 September 2026.
