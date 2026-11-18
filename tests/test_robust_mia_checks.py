"""RobustMIA's checks on its configuration and the labels of its results, without data downloads or training."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.base_mia import BaseMIA
from src.mia.attacks.black_box.robust import RobustMIA


def build(mode, n_shadows=2, a=.5):
    """Build a RobustMIA with BaseMIA's setup and the shadow manager replaced by stand-ins."""
    def fake_init(attack, defender_model, attacker_configs):
        attack.logger = Mock()
        attack.attack_configs = SimpleNamespace(mode=mode, a=a, gamma=1., random_pop_size=4)
        attack.shadow_configs = SimpleNamespace(mode=mode, n_shadow_datasets=n_shadows)
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

    def test_offline_a_may_be_anything_from_zero_to_one(self):
        for a in (0., 1., .3, [0., .5, 1.]):
            with self.subTest(a=a):
                build('offline', a=a)
        for a in (-.1, 1.1, [.5, 2.]):
            with self.subTest(a=a), self.assertRaisesRegex(ValueError, 'between 0 and 1'):
                build('offline', a=a)


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


if __name__ == "__main__":
    unittest.main()
