"""Original-index tracking: every dataset must agree with its own items on each sample's original index.

The unit tests need no data. The pipeline test builds the real defender/attacker/auditing/shadow/Selena datasets
from CIFAR-10 in ./datas and is skipped when it is not there.

  python -m unittest tests.test_index_tracking -v
"""

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.helpers import (ConstantLabelDataset, DatasetSplitter, IndexedDataset, MergedDataset,
                              SubsampledDataset, TargetOverrideDataset)

DATASETS_FOLDER = REPO_ROOT / 'datas'


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _tracked(n=10, first_id=100):
    """Sample i has value x=i and original index first_id+i, so items can be checked against their ids."""
    data = TensorDataset(torch.arange(n, dtype=torch.float32), torch.zeros(n, dtype=torch.long))
    return IndexedDataset(data, original_indices=list(range(first_id, first_id + n)))


class _Passthrough(Dataset):
    """An untracked wrapper that forwards tracked items without exposing their original indices."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class IndexTrackingAssertions(unittest.TestCase):
    def assert_consistent(self, dataset, n_checks=64):
        """Item i carries original_indices[i] as its original index and i as its local index."""
        self.assertEqual(len(dataset.original_indices), len(dataset))
        positions = list(range(len(dataset)))
        if len(positions) > n_checks:
            positions = random.Random(0).sample(positions, n_checks)
        for i in positions:
            item = dataset[i]
            self.assertEqual(int(item[2]), int(dataset.original_indices[i]), f"{type(dataset).__name__}[{i}]")
            self.assertEqual(int(item[3]), i, f"{type(dataset).__name__}[{i}]")
            self.assertEqual(dataset._orig_to_local[dataset.original_indices[i]], i)


class WrapperTests(IndexTrackingAssertions):
    def test_subset_of_tracked_dataset_inherits_original_indices(self):
        wrapped = IndexedDataset(Subset(_tracked(), [2, 5, 7]))
        self.assertEqual(wrapped.original_indices, [102, 105, 107])
        self.assert_consistent(wrapped)

    def test_concat_of_tracked_datasets_inherits_original_indices(self):
        wrapped = IndexedDataset(ConcatDataset([_tracked(3, 100), _tracked(2, 200)]))
        self.assertEqual(wrapped.original_indices, [100, 101, 102, 200, 201])
        self.assert_consistent(wrapped)

    def test_raw_datasets_still_get_fresh_indices(self):
        raw = TensorDataset(torch.zeros(4), torch.zeros(4))
        self.assertEqual(IndexedDataset(raw).original_indices, [0, 1, 2, 3])
        self.assertEqual(IndexedDataset(ConcatDataset([raw, raw])).original_indices, list(range(8)))
        self.assert_consistent(IndexedDataset(raw))

    def test_concat_mixing_tracked_and_raw_is_rejected(self):
        with self.assertRaises(ValueError):
            IndexedDataset(ConcatDataset([_tracked(), TensorDataset(torch.zeros(2), torch.zeros(2))]))

    def test_untracked_wrapper_of_tracked_items_is_rejected(self):
        with self.assertRaises(ValueError):
            IndexedDataset(_Passthrough(_tracked()))

    def test_wrappers_over_subset_agree_with_items(self):
        subset = Subset(_tracked(), [2, 5, 7])
        for wrapped in (TargetOverrideDataset(subset, torch.zeros(3, 2)), ConstantLabelDataset(subset, constant_label=1)):
            with self.subTest(wrapper=type(wrapped).__name__):
                self.assertEqual(list(wrapped.original_indices), [102, 105, 107])
                self.assert_consistent(wrapped)

    def test_subsampling_by_original_index_through_subset(self):
        subset = Subset(_tracked(), [2, 5, 7])
        picked = SubsampledDataset(subset, [105])
        self.assertEqual(picked[0][0].item(), 5.0)
        self.assert_consistent(picked)
        # An index that is not in the subset must fail loudly, not resolve to whatever sample sits at that position.
        with self.assertRaises(ValueError):
            SubsampledDataset(subset, [1])

    def test_from_positions(self):
        picked = SubsampledDataset.from_positions(_tracked(), [7, 2])
        self.assertEqual(picked.original_indices, [107, 102])
        self.assertEqual([picked[i][0].item() for i in range(2)], [7.0, 2.0])
        self.assert_consistent(picked)

    def test_merged_dataset_of_subsampled_parts(self):
        base = _tracked()
        merged = MergedDataset(SubsampledDataset.from_positions(base, [0, 1]), SubsampledDataset.from_positions(base, [8]))
        self.assertEqual(merged.original_indices, [100, 101, 108])
        self.assert_consistent(merged)

    def test_split_by_metric_with_original_index_map(self):
        base = SubsampledDataset.from_positions(_tracked(), [9, 0, 4, 7])   # original indices 109, 100, 104, 107
        scores = {orig: float(orig) for orig in range(100, 110)}           # keyed by original index, not position
        top, rest = DatasetSplitter.split_by_metric(base, scores, proportions=[0.5, 0.5])
        self.assertEqual(sorted(top.original_indices), [107, 109])
        self.assertEqual(sorted(rest.original_indices), [100, 104])
        for chunk in (top, rest):
            self.assert_consistent(chunk)
        with self.assertRaises(ValueError):
            DatasetSplitter.split_by_metric(base, {109: 1.0}, proportions=[0.5, 0.5])
        with self.assertRaises(ValueError):
            DatasetSplitter.split_by_metric(base, [1.0, 2.0], proportions=[0.5, 0.5])


@unittest.skipUnless((DATASETS_FOLDER / 'cifar10').exists(), "CIFAR-10 not found in ./datas")
class PipelineTests(IndexTrackingAssertions):
    """Walks the datasets the pipeline actually builds and checks each one."""

    @classmethod
    def setUpClass(cls):
        from src.data import get_attacker_datas, get_defender_datas
        cls.defender = get_defender_datas(dataset='cifar10', datasets_folder=DATASETS_FOLDER, seed=7, logger=_SilentLogger())
        cls.attacker = get_attacker_datas(dataset='cifar10', datasets_folder=DATASETS_FOLDER, seed=7, logger=_SilentLogger())

    def test_defender_and_attacker_splits(self):
        ids = {}
        for name in ('train', 'val', 'test'):
            ids[name] = set(self.defender.get(name).original_indices)
            self.assert_consistent(self.defender.get(name))
        ids['attacker'] = set(self.attacker.get('all').original_indices)
        self.assert_consistent(self.attacker.get('all'))
        names = list(ids)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                self.assertFalse(ids[names[a]] & ids[names[b]], f"{names[a]} and {names[b]} share samples")

    def test_selena_split_datasets(self):
        from src.mia.defenses.training.selena import SplitDataManager
        manager = SplitDataManager(dataset=self.defender, selena_configs=SimpleNamespace(K=3, L=1), seed=0,
                                   logger=_SilentLogger())
        train_ids = self.defender.get('train').original_indices
        for k in range(3):
            split = manager.get_dataset_for_model(model_index=k).get('train')
            self.assert_consistent(split)
            expected = [train_ids[i] for i in range(len(train_ids)) if not manager.get_exclusion_mask()[i, k]]
            self.assertEqual(list(split.original_indices), expected)

    def test_auditing_and_shadow_datasets(self):
        from src.mia.helpers import shadow_data_manager
        from src.mia.helpers.auditing_data_manager import AuditingDatasetManager
        from src.utils.configs import AuditingDataConfigs, ShadowDataConfigs
        auditing = AuditingDatasetManager(self.defender, AuditingDataConfigs(n_auditing_samples=20, seed=7),
                                          logger=_SilentLogger())
        for labels in ('original', 'mia'):
            self.assert_consistent(auditing.get(labels=labels))
        with tempfile.TemporaryDirectory() as shadow_folder, \
                patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', shadow_folder):
            shadows = shadow_data_manager.ShadowDatasetsManager(self.attacker, auditing,
                                                            ShadowDataConfigs(n_shadow_datasets=2, mode='online', seed=7),
                                                            attacker_hash='test_index_tracking', logger=_SilentLogger())
            self.assert_consistent(shadows.get(0, labels='original'))


if __name__ == '__main__':
    unittest.main()
