"""Fast checks for data-free boundary calibration and transfer relabeling."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from src.data import get_dataset_mean_std
from src.mia.attacks.label_only.unsupervised_boundary import UnsupervisedBoundaryMIA
from src.mia.attacks.label_only.transfer import TransferMIA
from src.utils.configs import UnsupervisedBoundaryAttackConfig


class BoundaryCalibrationTests(unittest.TestCase):
    def attack(self):
        attack = object.__new__(UnsupervisedBoundaryMIA)
        attack.seed = 42
        attack.input_shape = (3, 32, 32)
        attack.base_dataset_configs = SimpleNamespace(name='cifar10')
        attack.attack_configs = UnsupervisedBoundaryAttackConfig(n_calibration_samples=4, quantile=0.5)
        attack.logger = Mock()
        return attack

    def test_random_pixels_are_normalized_and_reproducible(self):
        attack = self.attack()
        first = torch.stack(list(attack.random_calibration_inputs()))
        second = torch.stack(list(attack.random_calibration_inputs()))
        self.assertTrue(torch.equal(first, second))
        mean, std = get_dataset_mean_std('cifar10')
        pixels = first * torch.tensor(std)[None, :, None, None] + torch.tensor(mean)[None, :, None, None]
        self.assertTrue(((pixels >= 0) & (pixels <= 1)).all())
        self.assertEqual(first.shape, (4, 3, 32, 32))

    def test_calibration_without_auxiliary_data(self):
        attack = self.attack()
        attack.defender_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 2))
        distances = iter([1., 2., 3., 4.])
        def features(model, inputs, targets):
            self.assertTrue(torch.equal(targets, model(inputs).argmax(1)))
            return torch.tensor([next(distances)])
        attack._features = features
        attack.find_threshold_via_quantile(torch.device('cpu'))
        self.assertEqual(attack.attack_threshold, 2.5)

    def test_candidate_labels_do_not_define_boundary(self):
        attack = self.attack()
        model = torch.nn.Identity()
        attack.estimate_boundary_distance = Mock(return_value=torch.tensor([2.]))
        attack._features(model, torch.tensor([[0., 5.]]), torch.tensor([0]))
        self.assertEqual(attack.estimate_boundary_distance.call_args.args[2].item(), 1)
        attack.attack_threshold = 1.
        scores, decisions = attack.infer_batch(model, torch.tensor([[0., 5.]]), torch.tensor([0]), torch.device('cpu'))
        np.testing.assert_array_equal(decisions, [1])

    def test_positive_calibration_count(self):
        with self.assertRaises(ValueError):
            UnsupervisedBoundaryAttackConfig(n_calibration_samples=0)

    def test_transfer_relabels_in_eval_mode(self):
        attack = object.__new__(TransferMIA)
        attack.logger = Mock()
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Identity())
        attack.defender_model = model.train()
        dataset = [(torch.tensor([0., 5.]), 0, 0, 0), (torch.tensor([5., 0.]), 1, 1, 1)]
        attack.shadow_manager = Mock()
        attack.shadow_manager.get_dataset.return_value = dataset
        attack.get_device = lambda _: torch.device('cpu')
        result = attack.relabel_shadow_dataset(SimpleNamespace(device='cpu', batch_size=1))
        self.assertEqual([int(result[i][1]) for i in range(2)], [1, 0])
        self.assertFalse(model.training)


if __name__ == '__main__':
    unittest.main()
