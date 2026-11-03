"""Under DDP, the batch size is the global batch: each process takes an equal share of it."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
from src.trainer.train_manager import TrainManager

SAMPLES = [(torch.randn(4), i % 3) for i in range(1000)]


def loaders(world_size: int, batch_size: int, rank: int = 0) -> TrainManager:
    """A TrainManager with only its loaders set up, as process `rank` of `world_size`."""
    manager = object.__new__(TrainManager)
    manager.logger = MagicMock()
    manager.distributed, manager.world_size, manager.global_rank = world_size > 1, world_size, rank
    data = MultiDatasets(datasets=[IndexedDataset(SAMPLES), IndexedDataset(SAMPLES[:100]),
                                   IndexedDataset(SAMPLES[:100])], ids=['train', 'val', 'test'])
    manager.setup_dataloaders_from_multidatasets(dataset=data, batch_size=batch_size)
    return manager


class GlobalBatchTest(unittest.TestCase):
    def test_each_gpu_takes_an_equal_share_of_the_batch(self):
        single = loaders(world_size=1, batch_size=64)
        self.assertEqual(single.train_loader.batch_size, 64)
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                manager = loaders(world_size=world_size, batch_size=64, rank=world_size - 1)
                self.assertEqual(manager.train_loader.batch_size, 64 // world_size)
                self.assertEqual(manager.val_loader.batch_size, 64 // world_size)
                self.assertIsInstance(manager.train_loader.sampler, DistributedSampler)
                # Opacus' Poisson sampling rate is 1 / len(loader): as on one GPU, so DP-SGD's epsilon is too.
                self.assertEqual(len(manager.train_loader), len(single.train_loader))

    def test_a_batch_that_does_not_split_evenly_is_refused(self):
        with self.assertRaisesRegex(ValueError, r'must be divisible by the number of GPUs \(3\)'):
            loaders(world_size=3, batch_size=64)


if __name__ == '__main__':
    unittest.main()
