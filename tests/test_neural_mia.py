"""NeuralMIA trains its attack model on each shadow sample's true membership in each shadow model."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.black_box.neural import NeuralMIA
from src.mia.helpers import shadow_data_manager
from src.mia.helpers.auditing_data_manager import AuditingDatasetManager
from src.mia.helpers.calibration_data_manager import CalibrationDatasetManager
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AuditingDataConfigs, ShadowDataConfigs
from test_decision_rules import ATTACKER, DEFENDER, _SilentLogger

N_SHADOWS = 4


class NeuralLabelsTest(unittest.TestCase):
    def test_attack_model_labels_are_the_shadow_memberships(self):
        auditing = AuditingDatasetManager(DEFENDER, AuditingDataConfigs(n_auditing_samples=20, seed=3),
                                          logger=_SilentLogger())
        calibration = CalibrationDatasetManager(ATTACKER, n_samples=50, seed=3, logger=_SilentLogger())
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
            shadow_data = shadow_data_manager.ShadowDatasetsManager(
                ATTACKER, auditing, ShadowDataConfigs(n_shadow_datasets=N_SHADOWS, mode='online', seed=3),
                attacker_hash='neural', calibration_dataset=calibration, logger=_SilentLogger())
        models = {index: torch.nn.Identity() for index in range(N_SHADOWS)}
        shadow_models = SimpleNamespace(get_all=lambda: models, get=lambda index: models[index],
                                        get_all_ids=lambda: list(models))
        attack = object.__new__(NeuralMIA)
        attack.logger = _SilentLogger()
        attack.reset_logger = lambda: None
        attack.shadow_manager = ShadowManager(shadow_data=shadow_data, logger=_SilentLogger())
        attack.shadow_manager.shadow_models = shadow_models
        attack.attack_configs = SimpleNamespace(neural_input_mode='logit', model_layers=[4], model_lr=.01,
                                                model_epochs=1)
        trained_on = {}

        def capture(model, train_loader, device):
            trained_on['dataset'] = train_loader.dataset
            return model

        with patch.object(ShadowManager, 'train_all'), \
                patch.object(NeuralMIA, 'get_model_out', side_effect=lambda model, data, device, mode:
                             torch.zeros(len(data), 2)), \
                patch.object(NeuralMIA, 'train_attacker_model', side_effect=capture):
            attack.optimize(train_config=SimpleNamespace(device='cpu'))

        samples = shadow_data.get_all_samples_in_all_shadow_datasets()
        expected = [float(index in shadow_data.find_shadow_datasets_containing_sample_id(int(sample[2])))
                    for index in range(N_SHADOWS) for sample in samples]
        labels = [float(item[1]) for item in trained_on['dataset']]
        self.assertEqual(labels, expected)
        # Each auditing and calibration sample is a member of exactly half of the online shadow models.
        self.assertGreater(sum(labels), 0)


if __name__ == '__main__':
    unittest.main()
