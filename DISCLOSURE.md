# Required disclosure

- **External datasets:** None used for the final model. The superseded ZIP contained the UCI Default of Credit Card Clients table and source labels; those files and the associated matching logic were excluded.
- **External code/notebooks/repositories/public solutions:** No public solution or notebook was copied into the final pipeline. Python scientific libraries listed in `requirements.txt` are used under their respective licenses.
- **Pretrained models:** None used as prediction inputs. CatBoost, Random Forest, and Bayesian logistic branches are fitted from scratch by the repository code on organizer-provided training labels. Reconstructed predictions were compared with archived model artifacts solely as a reproducibility audit.
- **AI/coding agents:** OpenAI Codex was used to audit the supplied archive, reconstruct and document its model-only pipeline, and run reproducibility checks. Model fitting and predictions are deterministic code paths, not AI-generated labels.
- **Manual prediction/post-processing:** None. The only post-processing is the fixed numerical clipping policy implemented identically for OOF and test probabilities.
- **Information beyond competition files:** The organizer's Expected Submission Materials PDF and team announcement were used only to document submission requirements. They were not model-training data.
