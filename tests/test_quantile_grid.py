"""The Quantile attack's level grid and its alpha check, without data downloads or training."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.base_mia import BaseMIA
from src.mia.attacks.black_box.quantile import QuantileMIA, quantile_levels, with_level


class GridTests(unittest.TestCase):
    def test_log_scale_reproduces_the_authors_grid(self):
        levels = quantile_levels(n=41, low=0., high=.9999, use_logscale=True).numpy()
        np.testing.assert_allclose(levels, np.sort(1 - np.logspace(-4, 0, 41)), atol=1e-6)

    def test_log_scale_crowds_the_levels_near_one(self):
        levels = quantile_levels(n=100, low=.01, high=.99, use_logscale=True).numpy()
        self.assertAlmostEqual(levels[0], .01, places=6)
        self.assertAlmostEqual(levels[-1], .99, places=6)
        self.assertTrue((np.diff(levels) > 0).all())
        self.assertEqual((levels >= .9).sum(), 50)

    def test_linear_grid_is_unchanged(self):
        torch.testing.assert_close(quantile_levels(n=100, low=.01, high=.99, use_logscale=False),
                                   torch.linspace(.01, .99, 100))

    def test_log_scale_rejects_a_top_level_of_one_or_a_negative_bottom_level(self):
        for low, high in ((.01, 1.), (-.1, .9)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                quantile_levels(n=5, low=low, high=high, use_logscale=True)


class DecisionLevelTests(unittest.TestCase):
    def test_the_level_is_added_in_order_when_absent(self):
        grid = torch.linspace(.01, .99, 100)
        for level in (.95, .985, .999):
            with self.subTest(level=level):
                levels = with_level(grid, level)
                self.assertEqual(len(levels), 101)
                self.assertTrue((torch.diff(levels) > 0).all())
                self.assertAlmostEqual(levels[torch.argmin(torch.abs(levels - level))].item(), level, places=6)

    def test_a_level_already_in_the_grid_is_not_added_twice(self):
        self.assertEqual(len(with_level(torch.linspace(.01, .99, 100), .99)), 100)
        self.assertEqual(len(with_level(torch.tensor([.5]), .5)), 1)

    def test_pinball_scores_use_the_exact_one_minus_alpha_head(self):
        levels = with_level(torch.linspace(.01, .99, 100), .95).reshape(1, -1)

        class LevelHead(torch.nn.Module):
            """Each head predicts ten times its level, so the 0.95 head predicts 9.5."""
            def forward(self, inputs):
                return 10 * levels.repeat(len(inputs), 1)

        dataset = TensorDataset(torch.tensor([[9.6], [9.4]]), torch.zeros(2, dtype=torch.long),
                                torch.arange(2), torch.arange(2))
        attack = object.__new__(QuantileMIA)
        attack.logger = Mock()
        attack.reset_logger = Mock()
        attack.shadow_manager = SimpleNamespace(get_model=lambda **_: LevelHead())
        attack.defender_model = torch.nn.Identity()
        attack.defender_scoring_fn = lambda sample, *args, **kwargs: (sample[:, 0], None)
        attack.attack_configs = SimpleNamespace(use_gaussian=False, quantile_alpha=.05)
        attack.quantile = levels
        [(scores, decisions, _)] = attack.score(dataset, device="cpu")
        # The nearest grid level, 0.9504, would give 0.096 and -0.104.
        np.testing.assert_allclose(scores, [.1, -.1], atol=1e-5)
        np.testing.assert_array_equal(decisions, [True, False])


class AlphaTests(unittest.TestCase):
    def test_alpha_must_lie_strictly_between_zero_and_one(self):
        for alpha in (0., 1., -.1, 1.5):
            def fake_init(attack, defender_model, attacker_configs, alpha=alpha):
                attack.logger = Mock()
                attack.attack_configs = SimpleNamespace(quantile_alpha=alpha)
            with self.subTest(alpha=alpha), patch.object(BaseMIA, '__init__', fake_init), \
                    self.assertRaisesRegex(ValueError, 'strictly between 0 and 1'):
                QuantileMIA(defender_model=None, attacker_configs=None)


if __name__ == "__main__":
    unittest.main()
