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


class OptimizeTests(unittest.TestCase):
    def test_trains_on_one_part_and_keeps_the_best_validation_epoch(self):
        original_indices = list(range(100, 120))
        shadow_dataset = IndexedDataset(TensorDataset(torch.arange(20.).reshape(-1, 1), torch.zeros(20).long()),
                                        original_indices=original_indices)
        attack = object.__new__(QuantileMIA)
        attack.logger = Mock()
        attack.reset_logger = Mock()
        attack.defender_model = torch.nn.Identity()
        attack.defender_scoring_fn = lambda data, *args, **kwargs: (data[:, 0] * 10, None)
        attack.attack_configs = SimpleNamespace(use_logscale=False, low_quantile=.1, high_quantile=.9, n_quantile=3,
                                                use_gaussian=False, val_split=None)
        attack.shadow_manager = Mock()
        attack.shadow_manager.get_dataset.return_value = shadow_dataset
        train_config = SimpleNamespace(device='cpu', seed=7)
        with patch('src.mia.attacks.black_box.quantile.TrainManager') as manager, \
                patch('src.mia.attacks.black_box.quantile.get_logger'):
            attack.optimize(train_config)
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


if __name__ == "__main__":
    unittest.main()
