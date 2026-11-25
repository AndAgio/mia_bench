"""The transfer attack chooses its threshold on as many shadow members as non-members, all there are, without a cap."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset
from src.mia.attacks.label_only.transfer import TransferMIA
from test_hop_skip_jump import Silent


def tracked(n, first_id):
    """n one-feature samples with original indices first_id, first_id + 1, ..."""
    ids = torch.arange(first_id, first_id + n)
    return IndexedDataset(TensorDataset(ids.float()[:, None], torch.zeros(n, dtype=torch.long)),
                          original_indices=ids.tolist())


class TransferThresholdTest(unittest.TestCase):
    def attack(self, n_members, n_outside):
        """A transfer attack whose shadow model has n_members members and n_outside samples outside its dataset."""
        attack = object.__new__(TransferMIA)
        attack.logger, attack.seed = Silent(), 0
        outside = tracked(n_outside, 10_000)
        attack.shadow_manager = SimpleNamespace(
            shadow_data=SimpleNamespace(indices_outside_shadow_dataset=lambda index: list(range(n_outside))),
            shadow_models=SimpleNamespace(get=lambda index: torch.nn.Identity()),
            get_dataset=lambda index, labels: tracked(n_members, 0),
            sample_outside_shadow_dataset=lambda index, num_data, labels: IndexedDataset(
                TensorDataset(*outside[:num_data][:2]), original_indices=list(range(10_000, 10_000 + num_data))))
        # Members score 1, non-members 0: the original index tells them apart.
        attack._features = lambda model, inputs, true_labels: (inputs[:, 0] < 10_000).float()
        return attack

    def test_equal_numbers_of_members_and_non_members_all_there_are(self):
        # At the production shadow share, 60% of the data is in the shadow dataset and 40% outside it.
        for n_members, n_outside in ((6000, 4000), (300, 700)):
            with self.subTest(n_members=n_members, n_outside=n_outside):
                attack = self.attack(n_members, n_outside)
                threshold, accuracy = attack.find_optimal_threshold(device='cpu')
                inputs = torch.cat([item[0] for item in attack.regression_dataset])
                expected = min(n_members, n_outside)
                self.assertEqual(int((inputs < 10_000).sum()), expected)
                self.assertEqual(int((inputs >= 10_000).sum()), expected)
                self.assertEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
