# Feature engineering experiment archive

This archive evaluates row-local feature blocks without changing the cleaned competition files, saved outer folds, model recipe, or test predictions. Each candidate replaces only the CatBoost OOF column in the selected three-model disagreement pipeline.

The experiment covers repayment trajectory and persistence, payment coverage and underpayment, utilization/cashflow trends, fixed risk bands, and five small domain-motivated interactions. All divisions use explicit denominator floors. Bands use fixed thresholds rather than full-data quantiles. No labels, IDs, external source rows, or test-distribution statistics enter feature construction.

Run with:

```bash
python scripts/research_feature_engineering.py
```

Results are written to `artifacts/feature_engineering_v1/`. Screening does not fit on or predict the competition test set and cannot overwrite `submission.csv`.

## Results

The first fixed-seed screen found two apparent improvements:

| Candidate | Final OOF log loss | Gain vs selected 0.421371 |
|---|---:|---:|
| Trajectory only | 0.421225 | 0.000146 |
| All blocks except manual interactions | 0.421127 | 0.000244 |

Coverage, trends, bands, and interactions were neutral or harmful in isolation. Adding all blocks including hand-built interactions scored 0.421447, confirming that additional features can increase split noise.

The two promising candidates were then paired against the unmodified feature pipeline across seeds `2026`, `137`, `4099`, and `7919`. Trajectory features had mean paired gain **-0.000023**, including one materially worse seed. The no-interaction combination had mean paired gain **0.000034** across individual seeds and **0.0000265** after equal seed averaging. This is too small and unstable to separate from CatBoost randomness, especially after the earlier seed-bag candidate improved CV but worsened Kaggle log loss.

Decision: **archive only; do not generate a test candidate or replace the confirmed best submission**. The useful finding is that repayment persistence and fixed bands deserve future targeted work, while broad coverage/trend/interaction expansion is not justified.
