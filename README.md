# Inter-Uni Datathon 2026 - Interpretable Credit Default Model

This repository contains the integrity-safe final pipeline and proposed submission for Stream 1. It intentionally uses a single regularized logistic model trained only on organizer-provided training labels. The superseded perfect-score package is not included because it reconstructed test labels from an external source table rather than predicting unseen outcomes.

## Final files

- `submission.csv` - the exact proposed file to upload and submit for finalist review.
- `scripts/train_and_submit.py` - validates inputs, reproduces validation, trains the model, explains coefficients, and regenerates `submission.csv`.
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
python scripts/train_and_submit.py
python -m unittest discover -v
shasum -a 256 submission.csv
```

The final hash must equal `expected_submission_sha256` in `config.json`. Input hashes, package versions, fold scores, and the output hash are also recorded in `artifacts/run_manifest.json`.

## Submission status

The repository file is a proposed replacement for the ineligible perfect-score output. To satisfy the organizer PDF, upload this exact `submission.csv` to the competition and submit this same public repository for review. Record the resulting leaderboard score in `METHOD.md` without modifying the CSV.

Only one team member should submit, and the team should submit only once before midnight at the end of Sunday, 6 September 2026.
