"""Numeric reference-threshold regressions; no model training or data required."""

from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mia.helpers.reference_thresholds import batched_smoothed_thresholds


class ReferenceThresholdTests(unittest.TestCase):
    def test_float32_loss_and_entropy_above_one_stay_finite(self):
        for scores in ([1.5, 2., 2.5, 3.], [0., 0.5, 1., 2.3], [0., 0., 0., 0.]):
            with self.subTest(scores=scores), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                references = np.asarray(scores, dtype=np.float32)
                actual = batched_smoothed_thresholds(references, .2)
                expected = batched_smoothed_thresholds(references.astype(np.float64), .2)
                self.assertTrue(np.isfinite(actual))
                np.testing.assert_allclose(actual, expected)

    def test_confidence_distribution_preserves_previous_finite_estimator(self):
        references = np.array([[.1, .8], [.2, .7], [.3, .6], [.4, .5]], dtype=np.float64)
        alpha = .2
        sorted_ref = np.sort(references, axis=0)
        logits = np.log(sorted_ref / (1 - sorted_ref))
        z = norm.ppf(1 - alpha, loc=logits.mean(axis=0), scale=logits.std(axis=0, ddof=1))
        expected = np.minimum(np.quantile(references, alpha, axis=0), np.exp(z) / (1 + np.exp(z)))
        np.testing.assert_allclose(batched_smoothed_thresholds(references, alpha), expected)

    def test_float32_matrix_has_one_finite_threshold_per_column(self):
        references = np.array([[1.5, 0.1, 0.], [2., 0.2, 1.], [3., 0.3, 2.]], dtype=np.float32)
        thresholds = batched_smoothed_thresholds(references, .2)
        self.assertEqual(thresholds.shape, (3,))
        self.assertTrue(np.isfinite(thresholds).all())
        for column in range(3):
            self.assertAlmostEqual(thresholds[column], batched_smoothed_thresholds(references[:, column], .2))

    def test_single_reference_uses_observed_score(self):
        np.testing.assert_array_equal(batched_smoothed_thresholds(np.array([[2., .5]]), .2), [2., .5])
        self.assertEqual(batched_smoothed_thresholds(np.array([2.]), .2), 2.)

    def test_endpoints_and_extreme_quantiles_are_finite(self):
        references = np.array([1.5, 2., 3.], dtype=np.float32)
        self.assertEqual(batched_smoothed_thresholds(references, 0), 1.5)
        self.assertEqual(batched_smoothed_thresholds(references, 1), 3.)
        for alpha in (1e-300, 1 - np.finfo(float).eps):
            self.assertTrue(np.isfinite(batched_smoothed_thresholds(references, alpha)))

    def test_invalid_references_and_alphas_fail_at_source(self):
        for references in ([], np.zeros((2, 0)), np.zeros((2, 2, 2)), [0., np.nan], [0., np.inf]):
            with self.subTest(references=references), self.assertRaisesRegex(ValueError, "Reference scores"):
                batched_smoothed_thresholds(references, .2)
        for alpha in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "alpha"):
                batched_smoothed_thresholds([.1, .2], alpha)


if __name__ == "__main__":
    unittest.main()
