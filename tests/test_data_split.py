"""The defender/attacker split: today's split at the default shares, defender sets that do not depend on the
attacker's share, the unused rest with a warning, and invalid shares refused."""
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import split_defender_attacker
from src.data.helpers import DatasetSplitter, IndexedDataset
from src.utils.configs import get_hash_from_settings
from src.utils.settings import build_parser
from src.utils.variables import (PERCENTAGE_OF_DATA_TO_USE_FOR_TRAINING_SPLIT as TRAIN,
                                 PERCENTAGE_OF_DATA_TO_USE_FOR_VALIDATION_SPLIT as VAL,
                                 PERCENTAGE_OF_DATA_TO_USE_FOR_TEST_SPLIT as TEST)

# Uneven class sizes, so that the rounding differs between classes.
CLASS_SIZES = [7, 13, 101, 600]
DATASET = IndexedDataset([(0.0, label) for label, size in enumerate(CLASS_SIZES) for _ in range(size)])


def indices(chunks):
    """Original indices of each chunk, in order."""
    return [chunk.get_indices(mode='original') for chunk in chunks]


def split(def_split, att_split, seed=12345):
    """The defender/attacker split of DATASET, without its log and warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return split_defender_attacker(DATASET, def_split=def_split, att_split=att_split, seed=seed,
                                       logger=MagicMock())


class DefenderAttackerSplitTest(unittest.TestCase):
    def test_the_default_shares_give_the_previous_split(self):
        for seed in (12345, 0):
            with self.subTest(seed=seed):
                # The previous code: one normalized cut, whose validation share used the attacker's share.
                previous = DatasetSplitter.split_stratified(DATASET, proportions=[0.5 * TRAIN, 0.5 * VAL, 0.5 * TEST,
                                                                                  0.5], seed=seed)
                self.assertEqual(indices(split(0.5, 0.5, seed=seed)), indices(previous))

    def test_the_defender_sets_do_not_depend_on_the_attacker_share(self):
        for def_split in (0.5, 0.7):
            reference = indices(split(def_split, 1 - def_split))[:3]
            for att_split in (0.1, 0.2):
                with self.subTest(def_split=def_split, att_split=att_split):
                    self.assertEqual(indices(split(def_split, att_split))[:3], reference)

    def test_shares_below_one_leave_the_rest_unused_with_a_warning(self):
        with self.assertWarnsRegex(UserWarning, 'add up to 0.8: the rest of the dataset is left unused'):
            chunks = split_defender_attacker(DATASET, def_split=0.5, att_split=0.3, seed=0, logger=MagicMock())
        used = [i for chunk in indices(chunks) for i in chunk]
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(len(chunks[3]), sum(int(np.floor(0.3 * n)) for n in CLASS_SIZES))
        self.assertLess(len(used), len(DATASET))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            chunks = split_defender_attacker(DATASET, def_split=0.7, att_split=0.3, seed=0, logger=MagicMock())
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(DATASET))

    def test_invalid_shares_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'more than the whole dataset'):
            split(0.7, 0.4)
        for def_split, att_split in ((0, 0.5), (0.5, -0.1), (1.5, 0.5)):
            with self.subTest(def_split=def_split, att_split=att_split):
                with self.assertRaisesRegex(ValueError, r'must be in \(0, 1\]'):
                    split(def_split, att_split)


class SplitHashTest(unittest.TestCase):
    def test_each_hash_covers_the_shares_its_data_depends_on(self):
        def hashes(*argv):
            settings = build_parser().parse_args(['--dataset', 'cifar10', *argv])
            return {mode: get_hash_from_settings(settings, mode=mode) for mode in ('defender', 'attacker')}
        base = hashes()
        defender_share = hashes('--defender_data_split_perc', '0.4')
        attacker_share = hashes('--attacker_data_split_perc', '0.4')
        # The defender's share moves the attacker's pool and the defender's sets the auditing samples come from.
        self.assertNotEqual(defender_share['defender'], base['defender'])
        self.assertNotEqual(defender_share['attacker'], base['attacker'])
        # The attacker's share moves no defender sample, so the trained defender is reused.
        self.assertEqual(attacker_share['defender'], base['defender'])
        self.assertNotEqual(attacker_share['attacker'], base['attacker'])


if __name__ == '__main__':
    unittest.main()
