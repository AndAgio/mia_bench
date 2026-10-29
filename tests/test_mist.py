"""MIST configuration regressions, with no dataset downloads."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.configs import MistDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser

TABULAR = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']


def mist_configs(*argv, mode='mist'):
    return generate_configs_from_settings(build_parser().parse_args([*argv, '--defender_mode', mode])).defender.defense


class MistConfigsTest(unittest.TestCase):
    def test_unset_parameters_follow_the_paper_for_the_dataset(self):
        for argv, expected in [(['--dataset', 'cifar10'], (2, 3.5)), (['--dataset', 'cifar100'], (2, 13.0)),
                               (['--dataset', 'purchase', *TABULAR], (4, 40.0)),
                               (['--dataset', 'texas', *TABULAR], (4, 25.0)),
                               # Datasets the paper does not evaluate take CIFAR-10's values.
                               (['--dataset', 'svhn'], (2, 3.5))]:
            configs = mist_configs(*argv)
            self.assertEqual((configs.num_submodels, configs.lmbd), expected, argv)
        configs = mist_configs('--dataset', 'purchase', '--defender_mist_lambda', '1.0', *TABULAR)
        self.assertEqual((configs.num_submodels, configs.lmbd), (4, 1.0))

    def test_mixup_mode_turns_mixup_on(self):
        configs = mist_configs('--dataset', 'cifar10', '--defender_mixup_alpha', '0.5', mode='mist_mixup')
        self.assertEqual((configs.mix_up, configs.alpha_mixup), (True, 0.5))
        self.assertFalse(mist_configs('--dataset', 'cifar10').mix_up)

    def test_fewer_than_two_submodels_are_rejected(self):
        with self.assertRaises(ValueError):
            MistDefenseConfigs(num_submodels=1)
        with self.assertRaises(ValueError):
            mist_configs('--dataset', 'cifar10', '--defender_mist_num_submodels', '1')


if __name__ == '__main__':
    unittest.main()
