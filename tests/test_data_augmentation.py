"""Data augmentation regressions, with no dataset downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import AugmentWrappedDataset, IndexedDataset, MultiDatasets
from src.mia.defenses.base import BaseDefender
from src.mia.defenses.training.data_augmentation import (DataAugmentationDefender, FeatureFlip, FeatureMask,
                                                         FeatureNoise)
from src.utils.configs import DataAugmentationDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser


TABULAR = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']


def make_defender(im_size=(8, 8), **options):
    defender = DataAugmentationDefender.__new__(DataAugmentationDefender)
    defender.logger = MagicMock()
    defender.data_augmentation_configs = DataAugmentationDefenseConfigs(**options)
    defender.dataset_configs = SimpleNamespace(im_size=im_size)
    images = [(torch.randn(3, 8, 8), 0) for _ in range(4)]
    defender.dataset = MultiDatasets(datasets=[IndexedDataset(images), IndexedDataset(images[:2])],
                                     ids=['train', 'test'], info={'name': 'toy'})
    return defender


class DataAugmentationTest(unittest.TestCase):
    def test_training_sees_augmented_samples_and_the_splits_stay_clean(self):
        defender = make_defender()
        clean_train = defender.dataset.get('train')
        seen = {}

        def train_model(self, train_configs, return_stats=False):
            seen['train'], seen['test'] = self.dataset.get('train'), self.dataset.get('test')
            return 'model'

        with patch.object(BaseDefender, 'train_model', train_model):
            self.assertEqual(defender.train_model(SimpleNamespace()), 'model')
        self.assertIsInstance(seen['train'], AugmentWrappedDataset)
        self.assertIs(seen['train'].base_dataset, clean_train)
        self.assertIs(seen['test'], defender.dataset.get('test'))
        self.assertIs(defender.dataset.get('train'), clean_train)

    def test_splits_are_restored_when_training_fails(self):
        defender = make_defender()
        clean = defender.dataset
        with patch.object(BaseDefender, 'train_model', side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):
                defender.train_model(SimpleNamespace())
        self.assertIs(defender.dataset, clean)

    def test_crops_come_first_only_when_padding_is_set(self):
        self.assertFalse(any(isinstance(t, transforms.RandomCrop)
                             for t in make_defender().get_augmentations().transforms))
        first = make_defender(crop_padding=4).get_augmentations().transforms[0]
        self.assertIsInstance(first, transforms.RandomCrop)
        self.assertEqual((tuple(first.size), first.padding), ((8, 8), 4))
        self.assertEqual(make_defender(crop_padding=4).get_augmentations()(torch.randn(3, 8, 8)).shape, (3, 8, 8))

    def test_options_reach_the_configs(self):
        settings = build_parser().parse_args(['--dataset', 'cifar10', '--defender_mode', 'data_augmentation',
                                              '--defender_augment_rotation', '7.5'])
        configs = generate_configs_from_settings(settings).defender.defense
        self.assertEqual((configs.rotation, configs.crop_padding), (7.5, None))
        settings = build_parser().parse_args(['--dataset', 'cifar10', '--defender_mode', 'data_augmentation',
                                              '--defender_augment_crop_padding', '4'])
        self.assertEqual(generate_configs_from_settings(settings).defender.defense.crop_padding, 4)



def augment_configs(*argv):
    settings = build_parser().parse_args([*argv, '--defender_mode', 'data_augmentation'])
    return generate_configs_from_settings(settings).defender.defense


class TabularAugmentationTest(unittest.TestCase):
    def test_tabular_datasets_skip_the_image_transforms_and_mask_features(self):
        steps = make_defender(im_size=None).get_augmentations().transforms
        self.assertEqual([type(step) for step in steps], [FeatureMask])
        self.assertEqual(steps[0].p, 0.2)
        defender = make_defender(im_size=None, feature_flip_prob=0.01, feature_noise_std=0.1)
        steps = defender.get_augmentations().transforms
        self.assertEqual([type(step) for step in steps], [FeatureFlip, FeatureNoise, FeatureMask])

    def test_feature_transforms(self):
        torch.manual_seed(0)
        x = (torch.rand(20000) < 0.3).float()
        masked = FeatureMask(p=0.2)(x)
        self.assertTrue(torch.all(masked <= x))
        self.assertAlmostEqual(1 - masked.sum().item() / x.sum().item(), 0.2, delta=0.02)
        flipped = FeatureFlip(p=0.1)(x)
        self.assertTrue(torch.all((flipped == x) | (flipped == 1 - x)))
        self.assertAlmostEqual((flipped != x).float().mean().item(), 0.1, delta=0.01)
        self.assertAlmostEqual((FeatureNoise(std=0.5)(x) - x).std().item(), 0.5, delta=0.02)

    def test_feature_options_reach_the_configs_of_the_right_datasets(self):
        configs = augment_configs('--dataset', 'purchase', *TABULAR)
        self.assertEqual((configs.feature_mask_prob, configs.feature_flip_prob, configs.feature_noise_std),
                         (0.2, None, None))
        configs = augment_configs('--dataset', 'texas', *TABULAR, '--defender_augment_feature_flip_prob', '0.01',
                                  '--defender_augment_feature_mask_prob', '0.1')
        self.assertEqual((configs.feature_mask_prob, configs.feature_flip_prob), (0.1, 0.01))
        self.assertEqual(augment_configs('--dataset', 'news', *TABULAR,
                                         '--defender_augment_feature_noise_std', '0.1').feature_noise_std, 0.1)
        for argv in (['--dataset', 'news', *TABULAR, '--defender_augment_feature_flip_prob', '0.01'],
                     ['--dataset', 'cifar10', '--defender_augment_feature_mask_prob', '0.1'],
                     ['--dataset', 'cifar10', '--defender_augment_feature_noise_std', '0.1']):
            with self.assertRaisesRegex(ValueError, 'does not apply'):
                augment_configs(*argv)


if __name__ == '__main__':
    unittest.main()
