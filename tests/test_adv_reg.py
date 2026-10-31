"""AdvReg regressions for the paper's defaults and attack model, with no dataset downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.defenses.training.adversarial_regularization import (AdvRegDefender, AdvRegTrainManager, AttackNet,
                                                                  PaperAttackNet)
from src.utils.configs import AdvRegDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser

TABULAR = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']


def adv_reg_configs(*argv):
    # The AdvReg configs that the command line selects.
    settings = build_parser().parse_args([*argv, '--defender_mode', 'adv_reg'])
    return generate_configs_from_settings(settings).defender.defense


class AdvRegConfigsTest(unittest.TestCase):
    def test_unset_lambda_follows_table_3_for_the_dataset(self):
        for argv, expected in [(['--dataset', 'cifar100'], 6.0), (['--dataset', 'purchase', *TABULAR], 3.0),
                               (['--dataset', 'texas', *TABULAR], 2.0),
                               # Other image datasets take CIFAR-100's value, other non-image ones Purchase's.
                               (['--dataset', 'cifar10'], 6.0), (['--dataset', 'news', *TABULAR], 3.0)]:
            self.assertEqual(adv_reg_configs(*argv).adv_lambda, expected, argv)

    def test_unset_attack_model_settings_follow_the_paper(self):
        configs = adv_reg_configs('--dataset', 'cifar10')
        self.assertIsNone(configs.shadow_attacker_model_layers)
        self.assertEqual(configs.attacker_lr, 0.0001)
        self.assertEqual(configs.penalty, 'linear')
        self.assertEqual(adv_reg_configs('--dataset', 'cifar10', '--defender_adv_reg_penalty', 'log').penalty, 'log')

    def test_set_options_win(self):
        configs = adv_reg_configs('--dataset', 'texas', *TABULAR, '--defender_adv_reg_lambda', '0.5',
                                  '--defender_adv_reg_attacker_lr', '0.001',
                                  '--defender_adv_reg_shadow_attacker_model_layers', '64', '32')
        self.assertEqual((configs.adv_lambda, configs.attacker_lr, configs.shadow_attacker_model_layers),
                         (0.5, 0.001, [64, 32]))


class PaperAttackNetTest(unittest.TestCase):
    def test_layers_and_initialization_follow_the_paper(self):
        torch.manual_seed(0)
        model = PaperAttackNet(num_classes=100)
        layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                         [(100, 1024), (1024, 512), (512, 64), (100, 512), (512, 64), (128, 256), (256, 64), (64, 1)])
        weights = torch.cat([layer.weight.flatten() for layer in layers])
        self.assertAlmostEqual(weights.std().item(), 0.01, delta=0.0005)
        self.assertTrue(all(torch.all(layer.bias == 0) for layer in layers))

    def test_branches_read_the_prediction_and_the_label(self):
        torch.manual_seed(0)
        model = PaperAttackNet(num_classes=3)
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight)
        features = torch.tensor([[0.7, 0.2, 0.1, 1.0, 0.0, 0.0]])
        logit = model(features, return_logits=True)
        self.assertEqual(logit.shape, (1,))
        torch.testing.assert_close(model(features), torch.sigmoid(logit))
        for changed in (torch.tensor([[0.1, 0.2, 0.7, 1.0, 0.0, 0.0]]), torch.tensor([[0.7, 0.2, 0.1, 0.0, 0.0, 1.0]])):
            self.assertFalse(torch.allclose(model(changed, return_logits=True), logit))


class ClassifierLossTest(unittest.TestCase):
    def test_penalty_is_lambda_h_or_lambda_log_h(self):
        torch.manual_seed(0)
        manager = AdvRegTrainManager.__new__(AdvRegTrainManager)
        manager.model = torch.nn.Linear(4, 3)
        manager.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        manager.attacker_model = AttackNet(in_dim=6, hidden=[5])
        inputs, targets, references = torch.randn(2, 4), torch.tensor([0, 2]), torch.randn(2, 4)
        outputs = manager.model(inputs)
        h = manager.attacker_model(AdvRegTrainManager.attack_features(outputs, targets))
        task = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        for penalty, expected in (('linear', task + 2.0 * h), ('log', task + 2.0 * h.log())):
            manager.adv_reg_configs = AdvRegDefenseConfigs(adv_lambda=2.0, penalty=penalty)
            loss, _ = manager.classifier_loss(inputs, targets, references)
            torch.testing.assert_close(loss, expected)
        # The penalty trains the classifier only.
        loss.sum().backward()
        self.assertIsNotNone(manager.model.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in manager.attacker_model.parameters()))


class AdvRegDefenderTest(unittest.TestCase):
    def attack_model(self, layers):
        defender = AdvRegDefender.__new__(AdvRegDefender)
        defender.logger = MagicMock()
        defender.name = 'adv_reg'
        defender.adv_reg_configs = AdvRegDefenseConfigs(shadow_attacker_model_layers=layers)
        defender.dataset_configs = SimpleNamespace(num_classes=10, seed=0)
        defender.dataset = MagicMock()
        defender.untrained_model = torch.nn.Linear(2, 10)
        module = 'src.mia.defenses.training.adversarial_regularization'
        with patch(f'{module}.AdvRegTrainManager'), patch(f'{module}.MultiDatasets'), \
                patch(f'{module}.DatasetSplitter.split_stratified', return_value=([0] * 3, [0] * 3)):
            defender.train_model(SimpleNamespace())
        return defender.attacker_model

    def test_unset_layers_select_the_paper_attack_model_and_set_ones_an_mlp(self):
        self.assertIsInstance(self.attack_model(None), PaperAttackNet)
        mlp = self.attack_model([16])
        self.assertIsInstance(mlp, AttackNet)
        self.assertEqual(mlp.net[0].in_features, 20)


if __name__ == '__main__':
    unittest.main()
