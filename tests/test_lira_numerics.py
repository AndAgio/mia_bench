"""LiRA zero-variance, tail-density and stable-log-odds regressions."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock
import warnings

import numpy as np
from scipy.stats import norm
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.black_box.lira import LiRA, lira_log_likelihood_ratio


class LiRANumericsTests(unittest.TestCase):
    def test_single_in_and_out_reference_produces_finite_ordered_scores(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            scores = lira_log_likelihood_ratio([[1.], [1.]], [[0.], [0.]], [[.9], [.1]])
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], 0)
        self.assertLess(scores[1], 0)

    def test_regular_distributions_match_log_of_likelihood_ratio(self):
        inside = np.array([[.5, 1., 1.5], [0., .5, 1.]])
        outside = np.array([[-.5, 0., .5], [-1., -.5, 0.]])
        targets = np.array([.8, -.2])
        ratio = norm.pdf(targets, inside.mean(1), inside.std(1)) / norm.pdf(
            targets, outside.mean(1), outside.std(1))
        np.testing.assert_allclose(lira_log_likelihood_ratio(inside, outside, targets), np.log(ratio))

    def test_tail_scores_do_not_divide_underflowed_pdf_values(self):
        inside = np.array([[100., 101.], [100., 101.]])
        outside = np.array([[0., 1.], [0., 1.]])
        targets = np.array([1000., 10000.])
        self.assertTrue((norm.pdf(targets, inside.mean(1), inside.std(1)) == 0).all())
        self.assertTrue((norm.pdf(targets, outside.mean(1), outside.std(1)) == 0).all())
        scores = lira_log_likelihood_ratio(inside, outside, targets)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])

    def test_equal_zero_variance_distributions_have_equal_likelihood(self):
        np.testing.assert_array_equal(
            lira_log_likelihood_ratio([[1.], [2.]], [[1.], [2.]], [3., 4.]), [0., 0.])

    def test_phi_matches_probability_log_odds_and_handles_saturated_logits(self):
        logits = torch.tensor([[.1, .2, .3], [1., 0., -1.]], dtype=torch.float64)
        labels = torch.tensor([1, 0])
        probabilities = torch.softmax(logits, 1).gather(1, labels[:, None]).reshape(-1)
        expected = torch.log(probabilities) - torch.log1p(-probabilities)
        torch.testing.assert_close(LiRA.compute_phi(torch.nn.Identity(), logits, labels, "cpu"), expected)
        extreme = torch.tensor([[1000., -1000., -1000.], [-1000., 1000., 1000.]])
        actual = LiRA.compute_phi(torch.nn.Identity(), extreme, torch.tensor([0, 0]), "cpu")
        torch.testing.assert_close(actual, torch.tensor([2000. - np.log(2), -2000. - np.log(2)],
                                                       dtype=torch.float64))
        shifted = LiRA.compute_phi(torch.nn.Identity(), extreme + 100000., torch.tensor([0, 0]), "cpu")
        torch.testing.assert_close(actual, shifted)

    def test_nonfinite_or_missing_references_are_rejected(self):
        for inside, outside, target in (([[]], [[1.]], [1.]), ([[np.nan]], [[1.]], [1.]),
                                        ([[1.]], [[np.inf]], [1.]), ([[1.]], [[1.]], [np.nan])):
            with self.subTest(inside=inside, outside=outside), self.assertRaises(ValueError):
                lira_log_likelihood_ratio(inside, outside, target)
        with self.assertRaisesRegex(ValueError, "finite logits"):
            LiRA.compute_phi(torch.nn.Identity(), torch.tensor([[float("inf"), 0.]]), torch.tensor([0]), "cpu")

    def test_minimal_attack_evaluation_passes_finite_scores_to_metrics(self):
        attack = object.__new__(LiRA)
        attack.logger = Mock()
        dataset = TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long),
                                torch.arange(2), torch.arange(2))
        attack.audit_manager = SimpleNamespace(get=lambda **_: dataset, get_all_ids=lambda: [0, 1])
        attack.shadow_manager = SimpleNamespace(get_all_model_indeces=lambda: [0, 1],
                                               find_all_in_dataset_indices_for_sample_id=lambda **_: [0])
        attack.defender_model = torch.nn.Identity()
        attack.compute_phis = Mock(side_effect=[np.array([[1.], [1.]]), np.array([[0.], [0.]]),
                                                np.array([[.9], [.1]])])
        attack.compute_stats = Mock(return_value={"auc": 1.})
        attack.measure_effectiveness(device="cpu")
        scores = attack.compute_stats.call_args.args[0]
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
