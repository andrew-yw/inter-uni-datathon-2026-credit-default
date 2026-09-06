# Full-refit iteration policy experiment

This experiment tests the proposed workflow: use cross-validation to select CatBoost iteration count, then include the validation rows when refitting. It uses only the five immutable saved fold IDs.

For each outer fold, the other four folds are rotated as inner validation sets. Their best iterations are aggregated by median, mean, or 75th percentile. A fresh model is then trained on all four outer-training folds for exactly that many iterations and evaluated once on the untouched outer fold. The archived fixed 437-tree policy is evaluated through the same outer-refit path.

The experiment never reads test rows, never writes test predictions, and cannot modify `submission.csv`. Results are stored in `artifacts/full_refit_policy_v1/`.

## Result

The current final OOF log loss is **0.421371269446**. All full-refit policies were worse, so none is promoted and the current submission remains unchanged.

| iteration policy | CatBoost OOF log loss | final ensemble OOF log loss | change vs current | fold SD | proposed full-data trees |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed 437 | 0.422968885132 | 0.421616979455 | +0.000245710009 worse | 0.007126160847 | 437 |
| inner 75th percentile | 0.423043923886 | 0.421648694034 | +0.000277424588 worse | 0.007088232491 | 413 |
| inner median | 0.423065184665 | 0.421664557550 | +0.000293288104 worse | 0.007247399586 | 316 |
| inner mean | 0.423051061189 | 0.421669720819 | +0.000298451374 worse | 0.007192937244 | 406 |

The inner best iterations varied substantially, from 213 to 614 trees. That makes a fold-specific epoch estimate noisy at this sample size. More importantly, the production path already trains the final CatBoost model on all 24,000 labelled rows using 437 trees; no labelled row is being discarded from the final test model. This experiment changes how the tree count is selected and how OOF models are refitted, but does not unlock additional final-training data.

Decision: retain the current production model and submission. Keep this as a reproducible negative-result archive; do not generate test predictions from any candidate here.

```bash
python scripts/research_full_refit_policy.py
```
