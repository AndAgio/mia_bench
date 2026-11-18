"""The draws made after the shadow datasets do not depend on whether those were sampled or loaded from the cache."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
import src.mia.helpers.shadow_data_manager as shadow_data_manager

AUDIT_IDS = list(range(1000, 1020))


class QuietLogger:
    """Swallows the manager's messages."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def build(mode, seed=0):
    """A shadow data manager over a pool of 1000 samples, with 20 auditing samples and 4 shadow datasets."""
    pool = IndexedDataset(TensorDataset(torch.zeros(1000, 1), torch.zeros(1000, dtype=torch.long)))
    auditing = Mock(get_all_ids=lambda: AUDIT_IDS)
    auditing.get.return_value.get_indices.return_value = AUDIT_IDS
    return shadow_data_manager.ShadowDatasetsManager(
        attacker_data_distribution=MultiDatasets([pool], ids=['all']), auditing_dataset=auditing,
        shadow_configs=SimpleNamespace(mode=mode, n_shadow_datasets=4, seed=seed), attacker_hash=f'test_{mode}',
        logger=QuietLogger())


def draws(manager):
    """The population sample and the samples outside shadow dataset 0, as the attacks draw them."""
    population = manager.sample_random_indices(num_data=20)
    outside = manager.sample_outside_shadow_dataset(index=0, num_data=20).original_indices
    return population, outside


class CachedShadowDatasetsTests(unittest.TestCase):
    def test_draws_are_the_same_whether_the_shadow_datasets_were_sampled_or_loaded(self):
        for mode in ('offline', 'online'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, \
                    patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
                sampled = build(mode)
                loaded = build(mode)
                self.assertEqual(sampled.shadow_datasets_map, loaded.shadow_datasets_map)
                self.assertEqual(draws(sampled), draws(loaded))

    def test_the_population_does_not_depend_on_other_draws_and_follows_the_seed(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
            alone = build('offline').sample_random_indices(num_data=20)
            after_outside = build('offline')
            after_outside.sample_outside_shadow_dataset(index=0, num_data=20)
            self.assertEqual(alone, after_outside.sample_random_indices(num_data=20))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
            self.assertNotEqual(alone, build('offline', seed=1).sample_random_indices(num_data=20))

    def test_no_draw_comes_from_the_auditing_samples(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
            population, outside = draws(build('online'))
        self.assertFalse(set(population + outside) & set(AUDIT_IDS))


if __name__ == "__main__":
    unittest.main()
