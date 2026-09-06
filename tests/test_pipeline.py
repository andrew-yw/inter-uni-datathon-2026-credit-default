from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.pipeline import FEATURE_COLUMNS, build_model, positive_probability, signed_log


ROOT = Path(__file__).resolve().parents[1]


class PipelineUnitTests(unittest.TestCase):
    def test_signed_log_is_finite_and_sign_preserving(self) -> None:
        values = signed_log([-100.0, 0.0, 100.0])
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(values[0], 0)
        self.assertEqual(values[1], 0)
        self.assertGreater(values[2], 0)

    def test_model_uses_only_explicit_features(self) -> None:
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.integers(1, 5, size=(40, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        target = np.tile([0, 1], 20)
        model = build_model(C=0.3, seed=2026)
        model.fit(frame, target)
        prediction = positive_probability(model, frame, 1e-7)
        self.assertEqual(prediction.shape, (40,))
        self.assertTrue(((prediction > 0) & (prediction < 1)).all())
        transformed_names = model.named_steps["transform"].get_feature_names_out().tolist()
        self.assertFalse(any("client_id" in name or "default" in name for name in transformed_names))

    def test_committed_submission_matches_manifest(self) -> None:
        submission = ROOT / "submission.csv"
        manifest_path = ROOT / "artifacts" / "run_manifest.json"
        if not submission.exists() or not manifest_path.exists():
            self.skipTest("Final integration artifacts have not been generated")
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256(submission.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["submission"]["sha256"])
        frame = pd.read_csv(submission)
        self.assertEqual(list(frame.columns), ["client_id", "default_probability"])
        self.assertEqual(len(frame), 6000)
        self.assertTrue(frame["default_probability"].between(1e-7, 1 - 1e-7).all())
        self.assertGreater(frame["default_probability"].nunique(), 100)


if __name__ == "__main__":
    unittest.main()
