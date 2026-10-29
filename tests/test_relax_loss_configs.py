"""RelaxLoss configuration regressions, with no dataset downloads."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.configs import generate_configs_from_settings
from src.utils.settings import build_parser

TABULAR = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']


def relax_loss_settings(*argv):
    # (alpha, upper, flatten incorrect only) that the command line selects.
    configs = generate_configs_from_settings(build_parser().parse_args([*argv, '--defender_mode', 'relax_loss']))
    defense = configs.defender.defense
    return defense.relax_alpha, defense.relax_upper, defense.flatten_incorrect_only


class RelaxLossConfigsTest(unittest.TestCase):
    def test_unset_options_follow_the_official_configs_for_the_dataset(self):
        for argv, expected in [(['--dataset', 'cifar10'], (1.0, 1.0, True)),
                               (['--dataset', 'cifar100'], (3.0, 1.0, True)),
                               (['--dataset', 'purchase', *TABULAR], (0.8, 0.3, False)),
                               (['--dataset', 'texas', *TABULAR], (2.5, 0.1, False)),
                               # Other image datasets take CIFAR-10's values, other non-image ones Purchase's.
                               (['--dataset', 'svhn'], (1.0, 1.0, True)),
                               (['--dataset', 'news', *TABULAR], (0.8, 0.3, False))]:
            self.assertEqual(relax_loss_settings(*argv), expected, argv)

    def test_set_options_win(self):
        self.assertEqual(relax_loss_settings('--dataset', 'texas', '--defender_relax_loss_alpha', '0.5',
                                             '--defender_relax_loss_upper', '0.2', '--defender_relax_loss_flatten',
                                             'incorrect', *TABULAR), (0.5, 0.2, True))


if __name__ == '__main__':
    unittest.main()
