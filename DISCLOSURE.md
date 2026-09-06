# Required disclosure

- **External datasets:** None used for the final model. The superseded ZIP contained the UCI Default of Credit Card Clients table and source labels; those files and the associated matching logic were excluded.
- **External code/notebooks/repositories/public solutions:** No public solution or notebook was copied into the final pipeline. Python scientific libraries listed in `requirements.txt` are used under their respective licenses.
- **Pretrained models:** None. The final estimator is fitted from scratch on the organizer-provided training split.
- **AI/coding agents:** OpenAI Codex was used to audit the supplied archive, implement and document this integrity-safe replacement, and run reproducibility checks. Model fitting and predictions are deterministic code paths, not AI-generated labels.
- **Manual prediction/post-processing:** None. The only post-processing is the fixed numerical clipping policy implemented identically for OOF and test probabilities.
- **Information beyond competition files:** The organizer's Expected Submission Materials PDF and team announcement were used only to document submission requirements. They were not model-training data.
