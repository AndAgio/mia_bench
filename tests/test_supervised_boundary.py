"""SBA queries valid pixels, chooses its threshold on equal numbers of shadow samples, and decides above it."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset
from src.mia.attacks.label_only import supervised_boundary
from src.mia.attacks.label_only.supervised_boundary import SupervisedBoundaryMIA
from test_hop_skip_jump import Linear, Silent, samples


class SupervisedBoundaryPixelTest(unittest.TestCase):
    def test_images_are_attacked_as_valid_pixels_and_distances_are_in_pixels(self):
        torch.manual_seed(0)
        attack = object.__new__(SupervisedBoundaryMIA)
        attack.logger = Silent()
        attack.attack_configs = SimpleNamespace(boundary=SimpleNamespace(mode='hop_skip_jump', norm='l2',
                                                                          n_queries=3000))
        mean, std = torch.tensor([0.4]).view(1, 1, 1, 1), torch.tensor([0.25]).view(1, 1, 1, 1)
        attack.normalization = (mean, std)
        attack.qeba_basis = None
        model = Linear()
        pixels, _ = samples(model)
        normalized = (pixels - mean) / std
        labels = (model.margin(normalized) < 0).long()
        model.seen.clear()
        distances = attack.estimate_boundary_distance(model, normalized, labels)
        queried_pixels = torch.cat(model.seen) * std + mean
        self.assertGreaterEqual(queried_pixels.min().item(), -1e-5)
        self.assertLessEqual(queried_pixels.max().item(), 1 + 1e-5)
        # In pixels, the margin is w . (p - mean) / std + b: a hyperplane with normal w / std.
        exact = model.margin(normalized).abs() / (model.w / std.flatten()).norm()
        torch.testing.assert_close(distances.float(), exact, rtol=0.05, atol=1e-4)


def tracked(n, first_id):
    """n one-feature samples with original indices first_id, first_id + 1, ..."""
    ids = torch.arange(first_id, first_id + n)
    return IndexedDataset(TensorDataset(ids.float()[:, None], torch.zeros(n, dtype=torch.long)),
                          original_indices=ids.tolist())


class SupervisedBoundaryThresholdTest(unittest.TestCase):
    def attack(self, n_members, n_outside):
        """An SBA whose shadow model has n_members members and n_outside samples outside its dataset."""
        attack = object.__new__(SupervisedBoundaryMIA)
        attack.logger, attack.seed = Silent(), 0
        outside = tracked(n_outside, 1000)
        shadow_data = SimpleNamespace(indices_outside_shadow_dataset=lambda index: list(range(n_outside)))
        attack.shadow_manager = SimpleNamespace(
            shadow_data=shadow_data, shadow_models=SimpleNamespace(get=lambda index: torch.nn.Identity()),
            get_dataset=lambda index, labels: tracked(n_members, 0),
            sample_outside_shadow_dataset=lambda index, num_data, labels: IndexedDataset(
                TensorDataset(*outside[:num_data][:2]), original_indices=list(range(1000, 1000 + num_data))))
        # Members get distance 1, non-members 0: the original index tells them apart.
        attack._features = lambda model, inputs, targets: (inputs[:, 0] < 1000).float()
        return attack

    def test_equal_numbers_of_members_and_non_members_at_most_the_cap(self):
        for n_members, n_outside, cap, expected in ((10, 6, 2500, 6), (10, 12, 2500, 10), (10, 12, 4, 4)):
            with self.subTest(n_members=n_members, n_outside=n_outside, cap=cap), \
                    patch.object(supervised_boundary, 'THRESHOLD_SAMPLES_PER_SIDE', cap):
                attack = self.attack(n_members, n_outside)
                attack.find_optimal_threshold(device='cpu')
                inputs = torch.cat([item[0] for item in attack.regression_dataset])
                self.assertEqual(int((inputs < 1000).sum()), expected)
                self.assertEqual(int((inputs >= 1000).sum()), expected)

    def test_a_member_is_strictly_above_the_threshold(self):
        attack = self.attack(1, 1)
        attack.attack_threshold = 0.0
        attack._features = lambda model, inputs, targets: inputs[:, 0]
        dataset = IndexedDataset(TensorDataset(torch.tensor([[0.], [0.5]]), torch.zeros(2, dtype=torch.long)))
        scores, decisions = attack.infer_dataset(torch.nn.Identity(), dataset, device='cpu')
        np.testing.assert_array_equal(decisions, [0, 1])

if __name__ == '__main__':
    unittest.main()
