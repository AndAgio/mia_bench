"""RobustMIA's checks on its configuration, its default a and the labels of its results, without data or training."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.base_mia import BaseMIA
from src.mia.attacks.black_box.robust import RobustMIA, rmia_scores
from tests.tune_robust_a import TUNING_A_GRID, tuning_aucs


def build(mode, n_shadows=2, a=.5, dataset='cifar10'):
    """Build a RobustMIA with BaseMIA's setup and the shadow manager replaced by stand-ins."""
    def fake_init(attack, defender_model, attacker_configs):
        attack.logger = Mock()
        attack.attack_configs = SimpleNamespace(mode=mode, a=a, gamma=2., random_pop_size=4)
        attack.shadow_configs = SimpleNamespace(mode=mode, n_shadow_datasets=n_shadows)
        attack.base_dataset_configs = SimpleNamespace(name=dataset)
        attack.seed = 0
        for name in ('attacker_data_distribution', 'audit_manager', 'calibration_manager', 'attacker_hash',
                     'model_configs'):
            setattr(attack, name, None)
    with patch.object(BaseMIA, '__init__', fake_init), \
            patch('src.mia.attacks.black_box.robust.ShadowManager'):
        return RobustMIA(defender_model=None, attacker_configs=None)


class ConfigurationTests(unittest.TestCase):
    def test_online_mode_with_one_shadow_model_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, 'at least 2 shadow models'):
            build('online', n_shadows=1)

    def test_online_mode_with_two_or_three_and_offline_mode_with_one_are_accepted(self):
        for mode, n_shadows in (('online', 2), ('online', 3), ('offline', 1)):
            with self.subTest(mode=mode, n_shadows=n_shadows):
                self.assertEqual(build(mode, n_shadows).mode, mode)

    def test_offline_a_may_be_anything_from_zero_to_one_or_unset(self):
        for a in (0., 1., .3, [0., .5, 1.], None):
            with self.subTest(a=a):
                build('offline', a=a)
        for a in (-.1, 1.1, [.5, 2.]):
            with self.subTest(a=a), self.assertRaisesRegex(ValueError, 'between 0 and 1'):
                build('offline', a=a)


class DefaultATests(unittest.TestCase):
    def test_unset_a_is_the_datasets_default(self):
        for dataset, a in (('cifar10', .3), ('cinic10', .3), ('cifar100', .6), ('imagenet', 1.), ('purchase', .2)):
            with self.subTest(dataset=dataset):
                self.assertEqual(build('offline', a=None, dataset=dataset).a_values(), [a])

    def test_a_dataset_without_a_default_needs_a_given_a(self):
        with self.assertRaisesRegex(ValueError, 'no default a for unlisted.*tune_robust_a'):
            build('offline', a=None, dataset='unlisted')
        self.assertEqual(build('offline', a=[.1, .7], dataset='unlisted').a_values(), [.1, .7])
        self.assertEqual(build('online', a=None, dataset='unlisted').mode, 'online')


class TuneATests(unittest.TestCase):
    """The tuning script's scoring. Members (ids 0-49) of the attacked shadow model 0: probability 0.6 there, 0.3 on
    the reference model 1. Non-members (ids 100-149): 0.95 and 0.9. Members rank above non-members only from a = 0.2
    (at a = 0.1, 0.976 < 1.005)."""

    def test_the_auc_of_each_a(self):
        sample_ids = list(range(50)) + list(range(100, 150))
        is_member = [True] * 50 + [False] * 50
        probs = np.array([[.6, .3]] * 50 + [[.95, .9]] * 50)
        maps = [list(range(50)), list(range(1000, 1050))]
        # The population's likelihood ratios under model 0 are spread evenly from 0.01 to 1.5.
        ratios = np.linspace(.01, 1.5, 1000)
        population_probs = np.stack([ratios * .6, np.full(1000, .6)], axis=1)
        aucs = tuning_aucs(probs, sample_ids, is_member, population_probs, list(range(2000, 3000)), maps, attacked=0,
                           gamma=2.)
        self.assertEqual(aucs, [0., 0.] + [1.] * 9)
        self.assertEqual(TUNING_A_GRID[int(np.argmax(aucs))], .2)


class ResultLabelTests(unittest.TestCase):
    def test_offline_results_carry_their_a_and_the_online_result_carries_none(self):
        offline = build('offline', a=[0., 1.])
        offline.compute_with_a = Mock(return_value=np.zeros(3))
        self.assertEqual([result.params for result in offline.score(dataset=None)], [{'a': 0.}, {'a': 1.}])
        online = build('online', a=.5)
        online.compute_with_a = Mock(return_value=np.zeros(3))
        [result] = online.score(dataset=None)
        self.assertIsNone(result.params)
        self.assertIsNone(online.compute_with_a.call_args.kwargs['a'])

    def test_scores_are_the_share_of_population_samples_gamma_dominated(self):
        # 7 dominates 0.5, 1 and 2 (7 / 4 < 2); 4 only 0.5 and 1 (4 / 2 is not above 2); 1 none.
        scores = rmia_scores(np.array([1., 4., 7.]), np.array([.5, 1., 2., 4.]), gamma=2.)
        np.testing.assert_allclose(scores, [0., .5, .75])


if __name__ == "__main__":
    unittest.main()
