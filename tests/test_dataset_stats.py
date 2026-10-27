"""Normalization statistics exist for every image dataset the command line accepts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_dataset_mean_std
from src.utils.configs import get_im_channels_from_name, get_im_size_from_name
from src.utils.settings import build_parser


class DatasetStatsTest(unittest.TestCase):
    def test_every_image_dataset_has_mean_and_std(self):
        datasets = next(action.choices for action in build_parser()._actions if action.dest == 'dataset')
        # Tabular datasets have no image size and are not normalized.
        for dataset in [name for name in datasets if get_im_size_from_name(name) is not None]:
            with self.subTest(dataset=dataset):
                mean, std = get_dataset_mean_std(dataset)
                self.assertEqual(len(mean), get_im_channels_from_name(dataset))
                self.assertEqual(len(std), get_im_channels_from_name(dataset))


if __name__ == '__main__':
    unittest.main()
