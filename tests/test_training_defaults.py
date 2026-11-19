"""The per-dataset training defaults fill only the training options left unset on the command line."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.settings import build_parser
from src.utils.training_defaults import IMAGE_TRAINING, TRAINING_DEFAULTS, TRAINING_OPTIONS


def parse(*args):
    return build_parser().parse_args(list(args))


class TrainingDefaultsTests(unittest.TestCase):
    def test_every_dataset_has_every_option(self):
        datasets = next(action.choices for action in build_parser()._actions if action.dest == 'dataset')
        self.assertEqual(sorted(TRAINING_DEFAULTS), sorted(datasets))
        for dataset, defaults in TRAINING_DEFAULTS.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(sorted(defaults), sorted(TRAINING_OPTIONS))

    def test_unset_options_take_the_datasets_defaults_for_the_target_and_the_shadow_models(self):
        for dataset in ('cifar10', 'texas'):
            settings = parse('--dataset', dataset)
            for role in ('defender', 'attacker'):
                for option in TRAINING_OPTIONS:
                    with self.subTest(dataset=dataset, option=f'{role}_{option}'):
                        self.assertEqual(getattr(settings, f'{role}_{option}'), TRAINING_DEFAULTS[dataset][option])

    def test_image_datasets_keep_the_former_defaults_so_their_runs_keep_their_hashes(self):
        former = dict(optimizer='sgd', lr=0.01, lr_sched='cosine', weight_decay=5e-4, momentum=0.9, epochs=100,
                      batch_size=256)
        self.assertEqual(IMAGE_TRAINING, former)

    def test_given_options_win_and_each_role_is_filled_on_its_own(self):
        settings = parse('--dataset', 'texas', '--defender_lr', '0.05', '--attacker_optimizer', 'sgd')
        self.assertEqual(settings.defender_lr, .05)
        self.assertEqual(settings.attacker_lr, TRAINING_DEFAULTS['texas']['lr'])
        self.assertEqual(settings.attacker_optimizer, 'sgd')
        self.assertEqual(settings.defender_optimizer, TRAINING_DEFAULTS['texas']['optimizer'])

    def test_editing_one_dataset_leaves_the_others_alone(self):
        self.assertIsNot(TRAINING_DEFAULTS['cifar10'], TRAINING_DEFAULTS['cifar100'])


if __name__ == "__main__":
    unittest.main()
