# Organizer data (not committed)

Download the competition files from the organizer and place these exact files here:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The final reproduction script validates their SHA-256 fingerprints against the versioned configuration files before use. The expected schemas are also checked, including submission-row order. Do not add the external UCI dataset, hidden labels or source labels to this directory.

No standalone cleaned table is required. The final feature matrices are regenerated in memory by the code under `src/archive_ensemble/`; the immutable validation assignments are already committed as `artifacts/saved_folds.csv`.
