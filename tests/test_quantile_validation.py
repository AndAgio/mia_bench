"""The Quantile regression model is trained on part of the attacker's data and keeps its best validation epoch."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset
from src.mia.attacks.black_box.quantile import QuantileMIA, validation_split


class ValidationSplitTests(unittest.TestCase):
    def test_parts_are_disjoint_complete_and_seeded(self):
        fit, val = validation_split(50, .2, seed=3)
        self.assertEqual(len(val), 10)
        self.assertEqual(sorted(np.concatenate([fit, val])), list(range(50)))
        again_fit, again_val = validation_split(50, .2, seed=3)
        np.testing.assert_array_equal(val, again_val)
        np.testing.assert_array_equal(fit, again_fit)
        self.assertFalse(np.array_equal(val, validation_split(50, .2, seed=4)[1]))

    def test_shares_that_leave_a_part_empty_fail(self):
        for share in (0., 1., -.1, .01, .99):
            with self.subTest(share=share), self.assertRaises(ValueError):
                validation_split(10, share, seed=0)


ORIGINAL_INDICES = list(range(100, 120))


class ConstantThresholds(torch.nn.Module):
    """Predicts the same thresholds for every sample: by default 150, 0, 50, 100, for the levels 0.1, 0.5, 0.9 and
    0.95 (unsorted)."""
    def __init__(self, thresholds=(150., 0., 50., 100.)):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(thresholds))

    def forward(self, inputs):
        return self.bias.repeat(len(inputs), 1)


def optimize(alpha=.05, thresholds=(150., 0., 50., 100.)):
    """Run optimize on 20 samples (16 to train, 4 to validate) with the trainer mocked; return it and the attack."""
    shadow_dataset = IndexedDataset(TensorDataset(torch.arange(20.).reshape(-1, 1), torch.zeros(20).long()),
                                    original_indices=ORIGINAL_INDICES)
    attack = object.__new__(QuantileMIA)
    attack.logger = Mock()
    attack.reset_logger = Mock()
    attack.defender_model = torch.nn.Identity()
    attack.defender_scoring_fn = lambda data, *args, **kwargs: (data[:, 0] * 10, None)
    attack.attack_configs = SimpleNamespace(use_gaussian=False, val_split=None, quantile_alpha=alpha)
    attack.quantile = torch.tensor([[.1, .5, .9, .95]])
    attack.shadow_manager = Mock()
    attack.shadow_manager.get_dataset.return_value = shadow_dataset
    train_config = SimpleNamespace(device='cpu', seed=7)
    with patch('src.mia.attacks.black_box.quantile.TrainManager') as manager, \
            patch('src.mia.attacks.black_box.quantile.get_logger'):
        manager.return_value.train.return_value = ConstantThresholds(thresholds)
        attack.optimize(train_config)
    return manager, attack


def warnings_of(attack):
    return [call.args[0] for call in attack.logger.print_it.call_args_list if '[WARNING]' in call.args[0]]


class OptimizeTests(unittest.TestCase):
    def test_trains_on_one_part_and_keeps_the_best_validation_epoch(self):
        original_indices = ORIGINAL_INDICES
        manager, attack = optimize()
        dataset = manager.return_value.initialize_train.call_args.kwargs['dataset']
        fit, val = dataset.get('train'), dataset.get('val')
        self.assertEqual(len(val), 4)
        self.assertEqual(sorted(fit.original_indices + val.original_indices), original_indices)
        # Each sample keeps its own target score, the defender's score of that sample.
        for part in (fit, val):
            for x, y, original_index, _ in part:
                self.assertEqual(float(y), 10 * (original_index - 100))
        train_kwargs = manager.return_value.train.call_args.kwargs
        self.assertTrue(train_kwargs['return_best_model'])
        self.assertFalse(train_kwargs['return_last_model'])
        best_model = manager.return_value.train.return_value
        attack.shadow_manager.update_model.assert_called_once_with(index=0, model=best_model)

    def test_trains_with_the_authors_recipe_and_leaves_the_attackers_settings_alone(self):
        manager, attack = optimize()
        config = manager.call_args.kwargs['train_configs']
        self.assertEqual((config.optimizer_config.name, config.optimizer_config.lr,
                          config.optimizer_config.weight_decay), ('adamw', 1e-4, 1e-4))
        self.assertEqual((config.scheduler_config.name, config.scheduler_config.epochs), ('const', 30))
        self.assertEqual((config.batch_size, config.grad_clip), (64, 1.))
        self.assertIs(manager.return_value.initialize_train.call_args.kwargs['configs'], config)

    def test_logs_the_share_of_validation_samples_above_their_sorted_threshold(self):
        manager, attack = optimize(alpha=.05)
        # Sorted, the 0.95 threshold is 150 (unsorted, that head predicts 100); the targets are 10 * (index - 100).
        val = manager.return_value.initialize_train.call_args.kwargs['dataset'].get('val')
        expected = np.mean([float(y) > 150 for _, y, _, _ in val])
        messages = [call.args[0] for call in attack.logger.print_it.call_args_list]
        logged = [message for message in messages if 'validation samples' in message]
        self.assertEqual(len(logged), 1)
        self.assertIn(f'{expected:.2%} of the 4 validation samples', logged[0])

    def test_warns_when_significantly_more_validation_samples_than_alpha_exceed_their_threshold(self):
        # The 4 validation targets lie between 0 and 190. All of them exceed -1000: p = 0.05^4, a warning.
        _, attack = optimize(alpha=.05, thresholds=(-1000.,) * 4)
        self.assertEqual(len(warnings_of(attack)), 1)
        self.assertIn('the calibrated rule, or the Gaussian head', warnings_of(attack)[0])
        # None of them exceeds 1000, and 1 of 4 at alpha 0.5 is not significant: no warning.
        for alpha, thresholds in ((.05, (1000.,) * 4), (.5, (150.,) * 4)):
            with self.subTest(alpha=alpha, thresholds=thresholds):
                _, attack = optimize(alpha=alpha, thresholds=thresholds)
                self.assertEqual(warnings_of(attack), [])


if __name__ == "__main__":
    unittest.main()
