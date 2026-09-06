# Final methodology report

## Approach

The final model is one L2-regularized logistic regression. It was selected to make every prediction traceable to a documented linear coefficient while retaining competitive probability estimates. It uses only the 24,000 organizer-provided labelled training rows; no external labels, public solutions, or source-table matching are used.

## Data cleaning and preprocessing

The loader verifies the exact SHA-256, schema, row identifiers, target classes, missingness, train/test separation, and official submission order before training. Identifiers and the target are excluded through a 23-feature allowlist.

Currency variables receive a signed `log(1 + abs(x))` transform and standardization. Age is standardized without a logarithm. Undocumented demographic and repayment-status codes are treated as nominal and one-hot encoded. Imputation and encoding live inside the model pipeline, so every validation fold fits them using training rows only.

## Feature engineering

The final model deliberately avoids complicated trends, target encodings, opaque interactions, and external reconstruction. Raw monthly bills and payments enter as separate transformed variables. This avoids unsupported calendar and accounting claims while keeping coefficient explanations direct.

## Validation

Five-fold `StratifiedGroupKFold` validation uses seed 2026. Exact duplicate 23-feature profiles remain in the same fold to prevent local memorization. The official selection metric is binary log loss; ROC AUC, PR AUC, and Brier score are diagnostics. Exact executed values are recorded in `artifacts/fold_metrics.csv` and `artifacts/run_manifest.json`.

The verified run produced mean fold log loss **0.434075 +/- 0.005795**, overall OOF log loss **0.434075**, ROC AUC **0.7724**, PR AUC **0.5447**, and Brier score **0.135756**.

Leaderboard score: **PENDING**. This integrity-safe CSV has not been uploaded by this code workflow; once submitted, record the returned score here without regenerating or editing `submission.csv`.

## Models considered and final selection

The supplied archive documented a CatBoost/Random-Forest/Bayesian ensemble with OOF log loss 0.421442 and a disagreement-adjusted challenger at 0.421278. It also contained a perfect hard-label output created by matching the external UCI source table. That perfect output is rejected because it does not represent unseen prediction and conflicts with the organizer's integrity announcement.

The single logistic model knowingly sacrifices some ensemble performance for auditability. Its regularization strength is fixed at `C=0.3`; no leaderboard feedback or hidden labels were used to select it.

## Explainability

`artifacts/coefficients.csv` contains every transformed feature coefficient, its absolute magnitude, and the corresponding odds ratio per transformed unit. Positive coefficients are associated with higher predicted default odds and negative coefficients with lower predicted odds, conditional on the other included variables. These are predictive associations, not causal effects.

## Final inference and post-processing

After validation, the unchanged pipeline is refitted on all organizer-provided training rows. It predicts `P(default=1)` for each test row. The only post-processing is the documented numerical clip to `[1e-7, 1-1e-7]`; the same clipping rule is applied to OOF and test probabilities. No manual row changes are made.

## Limitations

The data dictionary does not fully document category meanings or month ordering, so the report makes no unsupported calendar or deployment claims. Logistic regression cannot capture all nonlinear interactions and may score below a tree ensemble. Cross-validation remains an estimate rather than a guarantee of leaderboard performance. The resulting probabilities should not be treated as causal evidence or deployed for credit decisions without fairness, calibration, and policy review.
