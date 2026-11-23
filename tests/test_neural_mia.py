"""NeuralMIA trains one attack model per class on the shadow models' outputs and memberships, and scores with it."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
from src.mia.attacks.black_box.neural import NeuralMIA
from src.mia.helpers import shadow_data_manager
from src.mia.helpers.auditing_data_manager import AuditingDatasetManager
from src.mia.helpers.calibration_data_manager import CalibrationDatasetManager
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AuditingDataConfigs, ShadowDataConfigs
from test_decision_rules import _SilentLogger

N_SHADOWS = 4
N_CLASSES = 3


def tracked(n, first_id):
    """n two-feature samples with original indices first_id, first_id + 1, ..., of class (original index mod 3)."""
    ids = torch.arange(first_id, first_id + n)
    data = TensorDataset(torch.stack([ids.float(), -ids.float()], dim=1), ids % N_CLASSES)
    return IndexedDataset(data, original_indices=ids.tolist())


DEFENDER = MultiDatasets(datasets=[tracked(60, 0), tracked(15, 60), tracked(25, 75)], ids=['train', 'val', 'test'])
ATTACKER = MultiDatasets(datasets=[tracked(400, 1000)], ids=['all'])


def neural_attack(mode):
    """A NeuralMIA on the datasets above, with identity shadow models, its shadow datasets sampled in `mode`."""
    auditing = AuditingDatasetManager(DEFENDER, AuditingDataConfigs(n_auditing_samples=20, seed=3),
                                      logger=_SilentLogger())
    calibration = CalibrationDatasetManager(ATTACKER, n_samples=50, seed=3, logger=_SilentLogger())
    with tempfile.TemporaryDirectory() as folder, \
            patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
        shadow_data = shadow_data_manager.ShadowDatasetsManager(
            ATTACKER, auditing, ShadowDataConfigs(n_shadow_datasets=N_SHADOWS, mode=mode, seed=3),
            attacker_hash='neural', calibration_dataset=calibration, logger=_SilentLogger())
    models = {index: torch.nn.Identity() for index in range(N_SHADOWS)}
    shadow_models = SimpleNamespace(get_all=lambda: models, get=lambda index: models[index],
                                    get_all_ids=lambda: list(models))
    attack = object.__new__(NeuralMIA)
    attack.logger = _SilentLogger()
    attack.reset_logger = lambda: None
    attack.attacker_models = {}
    attack.shadow_manager = ShadowManager(shadow_data=shadow_data, logger=_SilentLogger())
    attack.shadow_manager.shadow_models = shadow_models
    attack.attack_configs = SimpleNamespace(neural_input_mode='logit', model_layers=[4], model_lr=.01,
                                            model_epochs=1)
    return attack, shadow_data


class NeuralTrainingTest(unittest.TestCase):
    def test_each_class_model_trains_on_that_class_rows_with_balanced_weights(self):
        for mode in ('offline', 'online'):
            with self.subTest(mode=mode):
                attack, shadow_data = neural_attack(mode)
                calls = []

                def capture(model, train_loader, device, pos_weight):
                    calls.append((train_loader.dataset, pos_weight))
                    return model

                with patch.object(ShadowManager, 'train_all'), \
                        patch.object(NeuralMIA, 'train_attacker_model', side_effect=capture):
                    attack.optimize(train_config=SimpleNamespace(device='cpu'))

                samples = list(shadow_data.get_all_samples_in_all_shadow_datasets())
                self.assertEqual(sorted(attack.attacker_models), list(range(N_CLASSES)))
                self.assertEqual(len(calls), N_CLASSES)
                for label, (rows, pos_weight) in enumerate(calls):
                    class_ids = [int(sample[2]) for sample in samples if int(sample[1]) == label]
                    expected_labels = [float(model in shadow_data.find_shadow_datasets_containing_sample_id(i))
                                       for model in range(N_SHADOWS) for i in class_ids]
                    expected_inputs = [[float(i), -float(i)] for _ in range(N_SHADOWS) for i in class_ids]
                    self.assertEqual([float(item[1]) for item in rows], expected_labels)
                    self.assertEqual([item[0].tolist() for item in rows], expected_inputs)
                    n_members = sum(expected_labels)
                    self.assertAlmostEqual(pos_weight.item(), (len(expected_labels) - n_members) / n_members)


class NeuralScoringTest(unittest.TestCase):
    def scoring_attack(self, classes):
        """An attack whose class-c model returns the first input plus 100 c, over an identity target."""
        attack = object.__new__(NeuralMIA)
        attack.logger = _SilentLogger()
        attack.attack_configs = SimpleNamespace(neural_input_mode='logit')
        attack.defender_model = torch.nn.Identity()

        class Offset(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, x):
                return x[:, :1] + self.offset

        attack.attacker_models = {c: Offset(100. * c) for c in classes}
        return attack

    def test_each_sample_is_scored_by_its_class_model(self):
        inputs = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        labels = torch.tensor([2, 0, 1, 2])
        dataset = IndexedDataset(TensorDataset(inputs, labels))
        [(scores, decisions, _)] = self.scoring_attack(range(3)).score(dataset, device='cpu')
        np.testing.assert_allclose(scores, [201., 2., 103., 204.])
        np.testing.assert_array_equal(decisions, scores >= 0)

    def test_batches_score_as_single_samples(self):
        generator = torch.Generator().manual_seed(0)
        inputs = torch.randn(600, 2, generator=generator)
        labels = torch.randint(0, 3, (600,), generator=generator)
        attack = self.scoring_attack(range(3))
        torch.manual_seed(0)
        attack.attacker_models = {c: torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
                                  for c in range(3)}
        [(batched, _, _)] = attack.score(IndexedDataset(TensorDataset(inputs, labels)), device='cpu')
        one_by_one = [attack.score(IndexedDataset(TensorDataset(inputs[i:i + 1], labels[i:i + 1])), device='cpu')[0][0]
                      for i in range(0, 600, 37)]
        np.testing.assert_allclose(batched[::37], np.concatenate(one_by_one), rtol=1e-6)

    def test_a_class_without_attack_model_is_refused(self):
        dataset = IndexedDataset(TensorDataset(torch.zeros(2, 2), torch.tensor([0, 1])))
        with self.assertRaisesRegex(ValueError, 'class 1'):
            self.scoring_attack([0]).score(dataset, device='cpu')


if __name__ == '__main__':
    unittest.main()
