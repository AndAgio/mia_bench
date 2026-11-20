"""MMD's per-dataset lambda and its split of the validation set, with no dataset downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
from src.mia.defenses.training.mmd import MmdDefender
from src.utils.configs import generate_configs_from_settings
from src.utils.settings import build_parser
from src.utils.variables import MMD_PAPER_LAMBDAS

TABULAR = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']


def mmd_lambda(*argv, mode='mmd'):
    """Return the MMD weight that the command line selects."""
    settings = build_parser().parse_args([*argv, '--defender_mode', mode])
    return generate_configs_from_settings(settings).defender.defense.lmbd


def labelled(labels, first_index):
    """Return an indexed dataset with the given labels and original indices from first_index on."""
    inputs = torch.arange(len(labels), dtype=torch.float32).unsqueeze(1)
    return IndexedDataset(TensorDataset(inputs, torch.tensor(labels)),
                          original_indices=list(range(first_index, first_index + len(labels))))


class MmdDefenderTests(unittest.TestCase):
    def test_unset_lambda_follows_table_5_for_the_dataset(self):
        for argv, expected in [(['--dataset', 'cifar10'], MMD_PAPER_LAMBDAS['cifar10']),
                               (['--dataset', 'cifar100'], MMD_PAPER_LAMBDAS['cifar100']),
                               (['--dataset', 'purchase', *TABULAR], MMD_PAPER_LAMBDAS['purchase']),
                               (['--dataset', 'texas', *TABULAR], MMD_PAPER_LAMBDAS['texas']),
                               # Datasets the paper does not evaluate take CIFAR-10's or Purchase's value.
                               (['--dataset', 'svhn'], MMD_PAPER_LAMBDAS['cifar10']),
                               (['--dataset', 'news', *TABULAR], MMD_PAPER_LAMBDAS['purchase'])]:
            for mode in ('mmd', 'mmd_mixup'):
                self.assertEqual(mmd_lambda(*argv, mode=mode), expected, (argv, mode))

    def test_set_lambda_wins(self):
        self.assertEqual(mmd_lambda('--dataset', 'cifar10', '--defender_mmd_lambda', '2.5'), 2.5)
        self.assertEqual(mmd_lambda('--dataset', 'cifar10', '--defender_mmd_lambda', '0'), 0.0)

    def test_reference_and_selection_split_val_into_disjoint_parts(self):
        # A class with a single validation sample must still get a reference sample.
        val_labels = [0] * 6 + [1] * 5 + [2]
        defender = MmdDefender.__new__(MmdDefender)
        defender.logger = MagicMock()
        defender.name = 'mmd'
        defender.mmd_configs = SimpleNamespace(lmbd=0.5)
        defender.dataset_configs = SimpleNamespace(seed=3)
        defender.untrained_model = torch.nn.Linear(1, 3)
        train, val, test = labelled([0, 1, 2] * 4, 0), labelled(val_labels, 100), labelled([0, 1, 2], 200)
        defender.dataset = MultiDatasets([train, val, test], ids=['train', 'val', 'test'], info={'name': 'toy'})
        with patch('src.mia.defenses.training.mmd.MmdTrainManager') as manager:
            defender.train_model(train_configs=SimpleNamespace())
        kwargs = manager.return_value.initialize_train.call_args.kwargs
        selection, reference = kwargs['dataset'].get('val'), kwargs['reference_dataset']
        self.assertIs(kwargs['dataset'].get('train'), train)
        self.assertIs(kwargs['dataset'].get('test'), test)
        self.assertFalse(set(selection.original_indices) & set(reference.original_indices))
        self.assertEqual(sorted(selection.original_indices + reference.original_indices), list(range(100, 112)))
        reference_labels = {int(reference[i][1]) for i in range(len(reference))}
        self.assertEqual(reference_labels, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
