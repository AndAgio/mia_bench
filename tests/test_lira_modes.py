"""LiRA's online and offline modes: startup checks, the offline one-sided test and the mode wiring."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import norm
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks import get_attacker_class
from src.mia.attacks.base_mia import BaseMIA
from src.mia.attacks.black_box.lira import LiRA, lira_offline_scores
from src.utils.configs import generate_configs_from_settings
from src.utils.settings import build_parser


def build(mode, n_shadows, shadow_mode=None):
    """Build a LiRA with BaseMIA's setup and the shadow manager replaced by stand-ins."""
    def fake_init(attack, defender_model, attacker_configs):
        attack.logger = Mock()
        attack.attack_configs = SimpleNamespace(mode=mode)
        attack.shadow_configs = SimpleNamespace(mode=shadow_mode or mode, n_shadow_datasets=n_shadows)
        for name in ('attacker_data_distribution', 'audit_manager', 'calibration_manager', 'attacker_hash',
                     'model_configs'):
            setattr(attack, name, None)
    with patch.object(BaseMIA, '__init__', fake_init), patch('src.mia.attacks.black_box.lira.ShadowManager'):
        return LiRA(defender_model=None, attacker_configs=None)


class StartupTests(unittest.TestCase):
    def test_too_few_shadow_models_fail_at_startup(self):
        for mode, n_shadows in (('online', 1), ('online', 2), ('online', 3), ('offline', 1)):
            with self.subTest(mode=mode, n_shadows=n_shadows), self.assertRaisesRegex(ValueError, 'at least'):
                build(mode, n_shadows)

    def test_the_fewest_shadow_models_each_mode_needs_are_accepted(self):
        for mode, n_shadows in (('online', 4), ('online', 5), ('offline', 2), ('offline', 3)):
            with self.subTest(mode=mode, n_shadows=n_shadows):
                self.assertEqual(build(mode, n_shadows).mode, mode)

    def test_online_error_points_to_offline_lira(self):
        with self.assertRaisesRegex(ValueError, 'offline_lira'):
            build('online', 3)

    def test_shadow_datasets_of_the_other_mode_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'own mode'):
            build('offline', 4, shadow_mode='online')


class OfflineTestTests(unittest.TestCase):
    def test_score_is_minus_the_log_survival_function_of_the_out_gaussian(self):
        out = np.array([[0., 1., 2.], [-1., 0., 1.], [5., 5.5, 6.]])
        targets = np.array([3., -2., 5.5])
        expected = -norm.logsf(targets, out.mean(axis=1), out.std(axis=1))
        np.testing.assert_allclose(lira_offline_scores(out, targets), expected)

    def test_one_sided_higher_phi_scores_higher_even_far_below_the_mean(self):
        out = np.tile([[0., 1., 2.]], (4, 1))
        scores = lira_offline_scores(out, np.array([-50., -1., 1., 50.]))
        self.assertTrue(np.all(np.diff(scores) > 0))
        self.assertTrue(np.isfinite(scores).all())

    def test_offline_scoring_uses_only_the_models_without_the_sample(self):
        attack = object.__new__(LiRA)
        attack.logger = Mock()
        attack.mode = 'offline'
        dataset = TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long), torch.arange(2), torch.arange(2))
        dataset.get_indices = lambda mode: [0, 1]
        attack.shadow_manager = SimpleNamespace(get_all_model_indeces=lambda: [0, 1, 2],
                                               find_all_in_dataset_indices_for_sample_id=lambda **_: [])
        attack.defender_model = torch.nn.Identity()
        attack.compute_phis = Mock(side_effect=[np.array([[0., 1., 2.], [0., 1., 2.]]), np.array([[3.], [-1.]])])
        [(scores, decisions, _)] = attack.score(dataset, device='cpu')
        self.assertEqual(attack.compute_phis.call_count, 2)
        self.assertEqual(attack.compute_phis.call_args_list[0].kwargs['models_for_sample'], [[0, 1, 2], [0, 1, 2]])
        self.assertIsNone(decisions)
        self.assertGreater(scores[0], scores[1])


class ModeTests(unittest.TestCase):
    def configs(self, mode):
        configs = generate_configs_from_settings(build_parser().parse_args(['--attacker_mode', mode]))
        return configs.attacker.attack, configs.attacker.dataset.shadow

    def test_lira_and_online_lira_are_online_and_offline_lira_offline(self):
        for mode, expected in (('lira', 'online'), ('online_lira', 'online'), ('offline_lira', 'offline')):
            with self.subTest(mode=mode):
                attack, shadow = self.configs(mode)
                self.assertEqual((attack.mode, shadow.mode), (expected, expected))
                self.assertIs(get_attacker_class(mode), LiRA)


if __name__ == '__main__':
    unittest.main()
