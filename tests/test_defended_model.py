"""Checks of the common base of the defended models, the target query count and the note on undefended features."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import TargetQueryCounter, check_pairing
from src.mia.attacks.black_box.lira import LiRA
from src.mia.attacks.black_box.neural import NeuralMIA
from src.mia.defenses.base import BaseDefender, DefendedModel
from src.mia.defenses.hybrid.hamp import HampDefender, HampTestWrapper
from src.mia.defenses.post_hoc.ldl import LdlDefender, LdlModel
from src.mia.defenses.post_hoc.mem_guard import AttackNet, MemGuard, MemGuardDefender
from src.mia.defenses.post_hoc.purifier import ConfidenceReformer, PurifiedModelWrapper, PurifierDefender, TorchLabelSwapper
from src.mia.defenses.training.relax_loss import RelaxLossDefender
from src.mia.defenses.vanilla import VanillaDefender
from src.utils.configs import HampDefenseConfigs


class FeatureModel(torch.nn.Module):
    # A classifier with the feature method of the repository's models.
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(6, 4)
        self.head = torch.nn.Linear(4, 3)

    def feature(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.feature(x))


def defended_models(model):
    # One of each defended model around the same classifier.
    swapper = TorchLabelSwapper(Pindex=torch.zeros(1, 3))
    swapper.set_device(torch.device('cpu'))
    return [HampTestWrapper(model),
            LdlModel(model, n_queries=2),
            MemGuard(model, AttackNet(in_dim=3, hidden=(4,)), max_iter=2),
            PurifiedModelWrapper(model, ConfidenceReformer(num_classes=3, latent_dim=2, hidden=4), swapper, 0.01, 1e-3)]


class TestDefendedModel(unittest.TestCase):
    def test_defended_models_start_in_eval_mode_and_refuse_training_mode(self):
        x = torch.randn(5, 6)
        for defended in defended_models(FeatureModel()):
            with self.subTest(type(defended).__name__):
                self.assertIsInstance(defended, DefendedModel)
                self.assertFalse(defended.training)
                self.assertEqual(defended(x).shape, (5, 3))
                defended.train()
                with self.assertRaises(RuntimeError):
                    defended(x)

    def test_feature_is_the_wrapped_models(self):
        model, x = FeatureModel(), torch.randn(5, 6)
        for defended in defended_models(model):
            with self.subTest(type(defended).__name__):
                torch.testing.assert_close(defended.feature(x), model.feature(x))
                torch.testing.assert_close(NeuralMIA.get_model_out(defended, x, 'cpu', mode='feat'),
                                           model.feature(x).detach())

    def test_probabilities_to_logits_round_trips_through_softmax(self):
        probabilities = torch.tensor([[0.7, 0.3, 0.0], [0.2, 0.2, 0.6]])
        torch.testing.assert_close(DefendedModel.probabilities_to_logits(probabilities).softmax(dim=1), probabilities)


class FailingModel(FeatureModel):
    # Raises on its first forward, like a query that fails inside an attack.
    failed = False

    def forward(self, x):
        if not self.failed:
            self.failed = True
            raise RuntimeError('query failed')
        return super().forward(x)


class TestTargetQueryCounter(unittest.TestCase):
    def test_counts_forward_and_feature_samples_once_each(self):
        # FeatureModel's forward calls feature: the inner call is not a second query.
        model = FeatureModel()
        with TargetQueryCounter(model) as counter:
            model(torch.randn(5, 6))
            model.feature(torch.randn(3, 6))
        self.assertEqual(counter.count, 8)

    def test_counts_queries_to_defended_models_once(self):
        # HAMP and LDL run the wrapped model several times per query; only the query counts.
        for defended in [HampTestWrapper(FeatureModel()), LdlModel(FeatureModel(), n_queries=4, max_batch_size=8)]:
            with self.subTest(type(defended).__name__):
                with TargetQueryCounter(defended) as counter:
                    defended(torch.randn(5, 6))
                    defended.feature(torch.randn(3, 6))
                self.assertEqual(counter.count, 8)

    def test_counts_neural_feat_queries(self):
        model = HampTestWrapper(FeatureModel())
        with TargetQueryCounter(model) as counter:
            NeuralMIA.get_model_out(model, torch.randn(4, 6), 'cpu', mode='feat')
            NeuralMIA.get_model_out(model, torch.randn(4, 6), 'cpu', mode='logit')
        self.assertEqual(counter.count, 8)

    def test_restores_the_model_and_stops_counting_on_exit(self):
        model = FeatureModel()
        with TargetQueryCounter(model) as counter:
            model(torch.randn(2, 6))
        model(torch.randn(5, 6))
        model.feature(torch.randn(5, 6))
        self.assertEqual(counter.count, 2)
        self.assertNotIn('feature', vars(model))
        self.assertEqual(len(model._forward_pre_hooks) + len(model._forward_hooks), 0)

    def test_keeps_counting_after_a_failed_query(self):
        model = FailingModel()
        with TargetQueryCounter(model) as counter:
            with self.assertRaises(RuntimeError):
                model(torch.randn(2, 6))
            model(torch.randn(3, 6))
        self.assertEqual(counter.count, 5)


def hamp_defender(mode):
    # HampDefender without BaseDefender.__init__, which would load the dataset.
    with patch.object(BaseDefender, '__init__', return_value=None):
        defender = HampDefender(SimpleNamespace(defense=HampDefenseConfigs(mode=mode)))
    defender.logger = Mock()
    return defender


def defender_stub(inference_only):
    return SimpleNamespace(inference_only=inference_only, logger=Mock())


def neural_configs(mode):
    return SimpleNamespace(attack=SimpleNamespace(neural_input_mode=mode))


class TestPairing(unittest.TestCase):
    def test_defenses_declare_whether_they_act_only_at_inference(self):
        for defender_class in [LdlDefender, MemGuardDefender, PurifierDefender]:
            self.assertTrue(defender_class.inference_only, defender_class.__name__)
        for defender_class in [VanillaDefender, RelaxLossDefender]:
            self.assertFalse(defender_class.inference_only, defender_class.__name__)
        self.assertTrue(hamp_defender('test_only').inference_only)
        self.assertFalse(hamp_defender('full').inference_only)
        self.assertFalse(hamp_defender('train_only').inference_only)

    def test_only_neural_feat_reads_features(self):
        self.assertTrue(NeuralMIA.reads_features(neural_configs('feat')))
        self.assertFalse(NeuralMIA.reads_features(neural_configs('logit')))
        self.assertFalse(LiRA.reads_features(neural_configs('feat')))

    def test_note_only_when_the_attack_reads_undefended_features(self):
        settings = SimpleNamespace(attacker_mode='neural_feat', defender_mode='ldl')
        noted = defender_stub(inference_only=True)
        check_pairing(settings, noted, NeuralMIA, neural_configs('feat'))
        noted.logger.print_it.assert_called_once()
        self.assertRegex(noted.logger.print_it.call_args.args[0], "'neural_feat'.*'ldl'")
        for defender, configs in [(defender_stub(inference_only=False), neural_configs('feat')),
                                  (defender_stub(inference_only=True), neural_configs('logit')),
                                  (hamp_defender('full'), neural_configs('feat'))]:
            check_pairing(settings, defender, NeuralMIA, configs)
            defender.logger.print_it.assert_not_called()
        hamp_test = hamp_defender('test_only')
        check_pairing(settings, hamp_test, NeuralMIA, neural_configs('feat'))
        hamp_test.logger.print_it.assert_called_once()


if __name__ == '__main__':
    unittest.main()
