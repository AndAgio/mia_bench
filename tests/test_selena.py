"""SELENA data split regressions, with small in-memory data and no dataset downloads."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets, SubsampledDataset
from src.mia.defenses.training.selena import SplitDataManager
from src.utils.configs import SelenaDefenseConfigs

K, L = 5, 2


def split_manager():
    # A train split whose original indices (29, 27, ..., 7) differ from its positions, as after the dataset splits.
    base = IndexedDataset(TensorDataset(torch.randn(30, 3), torch.zeros(30, dtype=torch.long)))
    train = SubsampledDataset.from_positions(base, [29 - 2 * position for position in range(12)])
    return SplitDataManager(MultiDatasets([train], ['train']), SelenaDefenseConfigs(K=K, L=L), seed=0, logger=Mock())


class SplitDataManagerTest(unittest.TestCase):
    def test_lookups_split_the_models_into_teachers_and_trained_models(self):
        manager = split_manager()
        for original_index in manager.dataset.get('train').original_indices:
            teachers = manager.get_teachers(original_index)
            trained = manager.get_trained_models(original_index)
            self.assertEqual(len(teachers), L)
            self.assertEqual(sorted(teachers + trained), list(range(K)))

    def test_lookups_agree_with_the_mask_and_the_split_models_data(self):
        manager = split_manager()
        train, mask = manager.dataset.get('train'), manager.get_exclusion_mask()
        for k in range(K):
            seen = set(manager.get_dataset_for_model(k).get('train').original_indices)
            for position, original_index in enumerate(train.original_indices):
                self.assertEqual(k in manager.get_trained_models(original_index), original_index in seen)
                self.assertEqual(k in manager.get_teachers(original_index), bool(mask[position, k]))

    def test_unknown_original_index_is_an_error(self):
        # 28 belongs to the base data but not to the train split, and 0 is a position, not an original index here.
        with self.assertRaises(KeyError):
            split_manager().get_teachers(28)
        with self.assertRaises(KeyError):
            split_manager().get_trained_models(0)


if __name__ == '__main__':
    unittest.main()
