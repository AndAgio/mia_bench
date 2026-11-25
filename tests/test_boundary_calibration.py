"""Fast checks for UBA's data-free threshold, the labels its distances are taken around, and transfer relabeling."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_dataset_mean_std
from src.data.helpers import IndexedDataset
from src.mia.attacks.label_only.unsupervised_boundary import UnsupervisedBoundaryMIA
from src.mia.attacks.label_only.transfer import TransferMIA
from src.utils.configs import UnsupervisedBoundaryAttackConfig


class BoundaryCalibrationTests(unittest.TestCase):
    def attack(self, dataset='cifar10', input_shape=(3, 32, 32)):
        attack = object.__new__(UnsupervisedBoundaryMIA)
        attack.seed = 42
        attack.input_shape = input_shape
        attack.base_dataset_configs = SimpleNamespace(name=dataset)
        attack.normalization = None
        if len(input_shape) == 3:
            mean, std = get_dataset_mean_std(dataset)
            attack.normalization = (torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1))
        attack.attack_configs = UnsupervisedBoundaryAttackConfig(n_calibration_samples=4, quantile=0.5)
        attack.attack_threshold = None
        attack.qeba_basis = None
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

    def test_tabular_data_gets_no_threshold_so_the_attack_rule_falls_back_to_calibration(self):
        attack = self.attack('purchase', (2,))
        attack.find_threshold_via_quantile = Mock()
        attack.optimize(SimpleNamespace(device='cpu'))
        attack.find_threshold_via_quantile.assert_not_called()
        self.assertIsNone(attack.attack_threshold)
        attack.estimate_boundary_distance = Mock(return_value=torch.tensor([2.]))
        data = IndexedDataset(torch.utils.data.TensorDataset(torch.tensor([[0., 5.]]), torch.tensor([1])))
        scores, decisions = attack.infer_dataset(torch.nn.Identity(), data, torch.device('cpu'))
        np.testing.assert_array_equal(scores, [2.])
        self.assertIsNone(decisions)

    def test_audited_samples_keep_their_true_label(self):
        attack = self.attack()
        model = torch.nn.Identity()
        attack.estimate_boundary_distance = Mock(return_value=torch.tensor([2.]))
        attack._features(model, torch.tensor([[0., 5.]]), torch.tensor([0]))
        self.assertEqual(attack.estimate_boundary_distance.call_args.args[2].item(), 0)

    def test_a_misclassified_sample_has_distance_0_and_is_not_a_member_at_threshold_0(self):
        attack = self.attack('news', (2,))
        attack.attack_configs.boundary.mode = 'hop_skip_jump'
        attack.attack_configs.boundary.n_queries = 100
        attack.attack_threshold = 0.
        # The identity model predicts class 1 for [0, 5]; its true label is 0.
        scores, decisions = attack.infer_batch(torch.nn.Identity(), torch.tensor([[0., 5.]]), torch.tensor([0]),
                                               torch.device('cpu'))
        np.testing.assert_array_equal(scores, [0.])
        np.testing.assert_array_equal(decisions, [0])

    def test_positive_calibration_count(self):
        with self.assertRaises(ValueError):
            UnsupervisedBoundaryAttackConfig(n_calibration_samples=0)

    def test_transfer_relabels_in_eval_mode(self):
        attack = object.__new__(TransferMIA)
        attack.logger = Mock()
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Identity())
        attack.defender_model = model.train()
        # Shadow datasets are tracked (they carry original indices), like the MergedDataset the attack gets in practice.
        dataset = IndexedDataset(torch.utils.data.TensorDataset(torch.tensor([[0., 5.], [5., 0.]]), torch.tensor([0, 1])))
        attack.shadow_manager = Mock()
        attack.shadow_manager.get_dataset.return_value = dataset
        attack.get_device = lambda _: torch.device('cpu')
        result = attack.relabel_shadow_dataset(SimpleNamespace(device='cpu', batch_size=1))
        self.assertEqual([int(result[i][1]) for i in range(2)], [1, 0])
        self.assertFalse(model.training)


if __name__ == '__main__':
    unittest.main()
