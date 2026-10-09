"""Regression checks for validation class-index caching in MMD training."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader, Dataset

from src.mia.defenses.training.mmd import MmdTrainManager


class CountingDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.reads = 0
        self.target_scans = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.reads += 1
        label = self.labels[index]
        return torch.tensor([float(label), 1.0]), label, index, index

    def get_all_targets(self, to_torch=False):
        self.target_scans += 1
        labels = [self[i][1] for i in range(len(self))]
        return torch.tensor(labels) if to_torch else labels


class MmdIndexCacheTests(unittest.TestCase):
    def test_label_sources_preserve_local_order(self):
        expected = {2: [0, 2], 0: [1], 1: [3]}
        tracked = CountingDataset([2, 0, 2, 1])
        metadata = SimpleNamespace(targets=[2, 0, 2, 1])

        class ItemOnlyDataset(CountingDataset):
            def get_all_targets(self, to_torch=False):
                raise AttributeError

        item_only = ItemOnlyDataset([2, 0, 2, 1])
        for dataset in (tracked, metadata, item_only):
            with self.subTest(dataset=type(dataset).__name__):
                self.assertEqual(MmdTrainManager.get_indices_by_label(dataset), expected)

    def test_cache_across_batches_and_epochs_and_replaced_split(self):
        train = CountingDataset([0, 1, 1, 0])
        validation = CountingDataset([1, 0, 1, 0, 1, 0])
        manager = MmdTrainManager.__new__(MmdTrainManager)
        manager._validation_index_dataset = None
        manager._validation_indices_by_label = {}
        manager.dataset = {'train': train, 'val': validation}
        manager.train_loader = DataLoader(train, batch_size=2)
        manager.device = torch.device('cpu')
        manager.distributed = False
        manager.model = torch.nn.Linear(2, 2)
        manager.optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.01)
        manager.mmd_configs = SimpleNamespace(lmbd=1.0)
        manager.reset_epoch_stats = MagicMock()
        manager.epoch_stats_tracker = MagicMock()
        manager.logger = MagicMock()
        manager.build_message_for_batch_end_mmd = MagicMock(return_value='')
        manager.build_message_for_stage_end = MagicMock(return_value='')

        for epoch in range(1, 3):
            manager.train_with_mmd_distance()
            self.assertEqual(validation.target_scans, 1)
            # One full label scan, then only the selected images on each epoch.
            self.assertEqual(validation.reads, len(validation) + epoch * len(train))
            self.assertEqual(manager._validation_indices_by_label, {1: [0, 2, 4], 0: [1, 3, 5]})

        replacement = CountingDataset([0, 0, 1, 1])
        manager.dataset['val'] = replacement
        manager.train_with_mmd_distance()
        self.assertEqual(replacement.target_scans, 1)
        self.assertEqual(manager._validation_indices_by_label, {0: [0, 1], 1: [2, 3]})
        self.assertEqual(validation.target_scans, 1)


if __name__ == '__main__':
    unittest.main()
