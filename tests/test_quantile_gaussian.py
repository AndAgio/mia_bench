"""Gaussian quantile scoring regressions, without data downloads or training."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock
import warnings

import numpy as np
from scipy.special import ndtri
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.black_box.quantile import QuantileMIA, gaussian_margin_scores


class GaussianQuantileTests(unittest.TestCase):
    def test_finite_margins_keep_original_ranking_and_zero_threshold(self):
        rng = np.random.default_rng(123)
        targets = rng.normal(size=128)
        predictions = rng.normal(size=(128, 2))
        for quantile in (.001, .2, .5, .8, .999):
            with self.subTest(quantile=quantile):
                raw = targets - predictions[:, 0] - np.exp(predictions[:, 1]) * ndtri(quantile)
                scores = gaussian_margin_scores(targets, predictions, quantile)
                np.testing.assert_allclose(scores, np.sign(raw) * np.log1p(np.abs(raw)), atol=1e-14)
                np.testing.assert_array_equal(np.argsort(scores), np.argsort(raw))
                np.testing.assert_array_equal(scores >= 0, raw >= 0)

    def test_scales_beyond_float64_exp_range_remain_finite_and_distinct(self):
        predictions = np.array([[0., 100.], [0., 1000.], [0., 2000.]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for quantile in (.1, .9):
                scores = gaussian_margin_scores([0., 0., 0.], predictions, quantile)
                self.assertTrue(np.isfinite(scores).all())
                self.assertEqual(len(np.unique(scores)), 3)
                if quantile > .5:
                    self.assertTrue((np.diff(scores) < 0).all())
                    self.assertTrue((scores < 0).all())
                else:
                    self.assertTrue((np.diff(scores) > 0).all())
                    self.assertTrue((scores > 0).all())

    def test_median_and_zero_residual_do_not_form_inf_times_zero(self):
        predictions = np.array([[0., 1000.], [0., -1000.], [1., 1000.]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            scores = gaussian_margin_scores([0., 0., 3.], predictions, .5)
        np.testing.assert_allclose(scores, [0., 0., np.log(3.)])

    def test_invalid_predictions_and_quantile_endpoints_fail_explicitly(self):
        for quantile in (0., 1., -1., np.nan):
            with self.subTest(quantile=quantile), self.assertRaisesRegex(ValueError, "quantile"):
                gaussian_margin_scores([0.], [[0., 0.]], quantile)
        for predictions in ([[0., np.inf]], [[np.nan, 0.]], [[0.]]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                gaussian_margin_scores([0.], predictions, .8)

    def test_attack_evaluation_uses_stable_gaussian_scores(self):
        class LargeScaleHead(torch.nn.Module):
            def forward(self, inputs):
                return torch.stack((torch.zeros(len(inputs)), inputs[:, 0]), dim=1)

        dataset = TensorDataset(torch.tensor([[100.], [200.]]), torch.zeros(2, dtype=torch.long),
                                torch.arange(2), torch.arange(2))
        attack = object.__new__(QuantileMIA)
        attack.logger = Mock()
        attack.reset_logger = Mock()
        attack.audit_manager = SimpleNamespace(get=lambda **_: dataset)
        attack.shadow_manager = SimpleNamespace(get_model=lambda **_: LargeScaleHead())
        attack.defender_model = torch.nn.Identity()
        attack.defender_scoring_fn = lambda *args, **kwargs: (torch.tensor([0.]), None)
        attack.attack_configs = SimpleNamespace(use_gaussian=True, quantile_alpha=.2)
        attack.quantile = torch.tensor([[.1, .8, .9]])
        attack.compute_stats = Mock(return_value={"auc": 1.})
        attack.measure_effectiveness(device="cpu")
        scores = attack.compute_stats.call_args.args[0]
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
