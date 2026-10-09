"""MemGuard regressions with small deterministic models and no dataset downloads."""
import sys
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.defenses.post_hoc.mem_guard import AttackNet, MemGuard, MemGuardDefender
from src.utils.configs import MemGuardDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser


class ConfidenceAttacker(torch.nn.Module):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.inputs = []

    def forward(self, features, return_logits=False):
        self.inputs.append(features.detach().clone())
        logits = self.scale * (features.max(dim=1).values - self.threshold)
        return logits if return_logits else logits.sigmoid()


def defense(attacker=None, **kwargs):
    settings = dict(max_iter=20, c3_max=0.1, randomize_mix=0,
                    apply_expected_budget=False)
    settings.update(kwargs)
    return MemGuard(torch.nn.Identity(), attacker or ConfidenceAttacker(), **settings).eval()


class MemGuardTests(unittest.TestCase):
    def test_single_sigmoid_and_early_success(self):
        attacker = AttackNet(2, hidden=())
        with torch.no_grad():
            attacker.net[0].weight.copy_(torch.tensor([[0., 1.]]))
            attacker.net[0].bias.fill_(-0.7)
        model = defense(attacker)
        logits = torch.tensor([[1., 0.]])
        probs, l1 = model._optimize_confidence(logits)
        self.assertGreater(l1.item(), 0)
        self.assertLessEqual(attacker(probs.sort(dim=1).values).item(), 0.5)
        self.assertEqual(probs.argmax(1).item(), 0)

    def test_training_and_defense_use_sorted_probabilities(self):
        inputs = torch.tensor([[0., 1., -2.], [3., 0., 1.]])
        features = MemGuardDefender.get_model_out(torch.nn.Identity(), inputs, 'cpu')
        torch.testing.assert_close(features, inputs.softmax(1).sort(dim=1).values)
        attacker = ConfidenceAttacker()
        defense(attacker, max_iter=1)(inputs)
        torch.testing.assert_close(attacker.inputs[0], features)
        for seen in attacker.inputs:
            torch.testing.assert_close(seen.sum(1), torch.ones(seen.size(0)))
            self.assertTrue((seen[:, 1:] >= seen[:, :-1]).all())

    def test_stopping_and_escalation_are_independent_per_sample(self):
        logits = torch.tensor([[1., 0.], [20., 0.], [0., 1.]])
        settings = dict(max_iter=10, c3_max=10.)
        batched = defense(**settings).defend(logits)
        separate = torch.cat([defense(**settings).defend(row[None]) for row in logits])
        torch.testing.assert_close(batched, separate)
        self.assertFalse(torch.equal(batched[0], logits.softmax(1)[0]))
        torch.testing.assert_close(batched[1], logits.softmax(1)[1])
        torch.testing.assert_close(batched.argmax(1), logits.argmax(1))

    def test_failed_label_constraint_returns_clean_even_with_mixing(self):
        # A binary confidence cannot cross below 0.4; a large step also flips the label.
        logits = torch.tensor([[1., 0.]])
        model = defense(ConfidenceAttacker(0.4), step_size=10., randomize_mix=0.1)
        result, l1 = model._optimize_confidence(logits)
        torch.testing.assert_close(result, logits.softmax(1))
        self.assertEqual(l1.item(), 0)

    def test_boundary_samples_skip_optimization(self):
        logits = torch.tensor([[0.7, 0.3]]).log()
        attacker = ConfidenceAttacker()
        model = defense(attacker, randomize_mix=0.1)
        torch.testing.assert_close(model.defend(logits), logits.softmax(1))
        self.assertEqual(len(attacker.inputs), 1)

    def test_forward_obeys_logits_contract_including_extreme_inputs(self):
        logits = torch.tensor([[1., 0.], [1000., -1000.]])
        model = defense()
        probabilities = model.defend(logits)
        output = model(logits)
        self.assertTrue(output.isfinite().all())
        torch.testing.assert_close(output.softmax(1), probabilities)
        torch.testing.assert_close(output.argmax(1), logits.argmax(1))

    def test_no_model_gradients_under_no_grad(self):
        attacker = ConfidenceAttacker()
        attacker.scale.grad = torch.tensor(3.)
        with torch.no_grad():
            defense(attacker)(torch.tensor([[1., 0.]]))
        self.assertEqual(attacker.scale.grad.item(), 3.)

    def test_expected_budget_selects_clean_or_defended_output(self):
        logits = torch.tensor([[1., 0.]])
        model = defense(apply_expected_budget=True, budget_l1=0.)
        torch.testing.assert_close(model.defend(logits), logits.softmax(1))
        model.budget_l1 = 0.001
        q_adv, l1 = model._optimize_confidence(logits)
        self.assertGreater(l1.item(), model.budget_l1)
        with patch.object(model, '_one_time_uniform', side_effect=lambda x, p: torch.zeros_like(p)):
            torch.testing.assert_close(model.defend(logits), q_adv)
        with patch.object(model, '_one_time_uniform', side_effect=lambda x, p: torch.ones_like(p)):
            torch.testing.assert_close(model.defend(logits), logits.softmax(1))

    def test_membership_training_uses_validation_not_audit_test(self):
        def data(values):
            inputs = torch.tensor(values)
            return TensorDataset(inputs, torch.zeros(len(values)))
        splits = {'train': data([[2., 0.], [3., 0.]]),
                  'val': data([[0., 1.]]), 'test': data([[100., 0.]])}
        defender = object.__new__(MemGuardDefender)
        defender.logger = Mock()
        defender.model_configs = SimpleNamespace(num_classes=2)
        defender.mem_guard_configs = SimpleNamespace(shadow_attacker_model_layers=[],
            shadow_attacker_model_lr=0.01, shadow_attacker_model_epochs=1)
        defender.dataset = Mock()
        defender.dataset.get.side_effect = splits.__getitem__
        defender.trained_model = torch.nn.Identity()
        with patch('src.mia.defenses.post_hoc.mem_guard.DataLoader', wraps=DataLoader) as loaders, \
                patch('torch.optim.SGD', wraps=torch.optim.SGD) as optimizer:
            attacker = defender.fit_shadow_attacker_model('cpu')
        training_loader = next(call for call in loaders.call_args_list
                               if isinstance(call.args[0], TensorDataset))
        _, training_labels = training_loader.args[0].tensors
        self.assertEqual(training_labels.tolist(), [1., 0.])
        self.assertEqual(training_loader.kwargs['batch_size'], 64)
        optimizer.assert_called_once()
        self.assertFalse(attacker.training)
        self.assertEqual([call.args[0] for call in defender.dataset.get.call_args_list],
                         ['train', 'val'])

    def test_unsorted_features_and_class_order_are_preserved(self):
        logits = torch.tensor([[0., 1., -2.]])
        attacker = ConfidenceAttacker()
        model = defense(attacker, use_sorted=False)
        result = model.defend(logits)
        torch.testing.assert_close(attacker.inputs[0], logits.softmax(1))
        self.assertEqual(result.argmax(1).item(), 1)
        sorted_result = defense().defend(logits)
        torch.testing.assert_close(result, sorted_result)

    def test_published_defaults_match_cli_and_config(self):
        config = MemGuardDefenseConfigs()
        settings = build_parser().parse_args(['--defender_mode', 'mem_guard'])
        generated = generate_configs_from_settings(settings).defender.defense
        self.assertEqual(generated, config)
        custom = build_parser().parse_args(['--defender_mode', 'mem_guard',
                                           '--defender_mem_guard_randomness_quantization', '0.01'])
        self.assertEqual(generate_configs_from_settings(custom).defender.defense.randomness_quantization, 0.01)
        for field, cli in [('budget', 'budget'), ('shadow_attacker_model_layers', 'shadow_model_layers'),
                           ('shadow_attacker_model_epochs', 'shadow_model_epochs'),
                           ('shadow_attacker_model_lr', 'shadow_model_lr'),
                           ('randomness_quantization', 'randomness_quantization')]:
            self.assertEqual(getattr(config, field), getattr(settings, 'defender_mem_guard_' + cli))
        self.assertEqual(config.budget, 0.1)
        self.assertEqual(config.shadow_attacker_model_layers, [256, 128, 64])
        self.assertEqual(config.shadow_attacker_model_epochs, 400)
        self.assertEqual(config.shadow_attacker_model_lr, 0.001)
        self.assertEqual(MemGuard(torch.nn.Identity(), ConfidenceAttacker()).randomize_mix, 0)

    def test_success_escalates_and_later_failure_retains_last_success(self):
        logits = torch.tensor([[2., 0.]])
        first_attacker = ConfidenceAttacker()
        first = defense(first_attacker, max_iter=30, c3_max=0.1)
        q_first, _ = first._optimize_confidence(logits)
        searching_attacker = ConfidenceAttacker()
        searching = defense(searching_attacker, max_iter=30, c3_max=1e5)
        q_last, _ = searching._optimize_confidence(logits)
        # c3=1 prevents a crossing for this linear confidence classifier. The
        # search must run that round, retain c3=0.1's success and stop there.
        torch.testing.assert_close(q_last, q_first)
        self.assertEqual(len(searching_attacker.inputs) - len(first_attacker.inputs), 2 * 29)
        self.assertFalse(torch.equal(q_last, logits.softmax(1)))

    def test_first_failure_stops_without_escalation(self):
        logits = torch.tensor([[1., 0.]])
        first_attacker = ConfidenceAttacker(0.4)
        capped_attacker = ConfidenceAttacker(0.4)
        q_first, _ = defense(first_attacker)._optimize_confidence(logits)
        q_capped, _ = defense(capped_attacker, c3_max=1e5)._optimize_confidence(logits)
        torch.testing.assert_close(q_first, logits.softmax(1))
        torch.testing.assert_close(q_capped, q_first)
        self.assertEqual(len(first_attacker.inputs), len(capped_attacker.inputs))

    def test_phase_two_rejects_boundary_overshoot(self):
        logits = torch.tensor([[0.701, 0.299]]).log()
        model = defense(apply_expected_budget=True, budget_l1=2.)
        candidate, distortion = model._optimize_confidence(logits)
        self.assertGreater(distortion.item(), 0)
        attacker = model.M
        self.assertGreater((attacker(candidate) - 0.5).abs().item(),
                           (attacker(logits.softmax(1)) - 0.5).abs().item())
        with patch.object(model, '_one_time_uniform', side_effect=lambda x, p: torch.zeros_like(p)):
            torch.testing.assert_close(model.defend(logits), logits.softmax(1))

    def test_one_time_randomness_is_repeatable_and_batch_independent(self):
        logits = torch.tensor([[1., 0.], [2., 0.], [0., 1.]])
        model = defense(apply_expected_budget=True, budget_l1=0.05)
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        batched = model.defend(logits)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        self.assertEqual(random.getstate(), python_state)
        torch.manual_seed(234)
        torch.rand(20)
        torch.testing.assert_close(model.defend(logits), batched)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model.defend(logits[order]), batched[order])
        separate = torch.cat([model.defend(row[None]) for row in logits])
        torch.testing.assert_close(separate, batched)
        model.budget_l1 = 0.059
        repeated = model.defend(logits[:1].expand(8, -1))
        torch.testing.assert_close(repeated, repeated[:1].expand_as(repeated))

    def test_quantized_query_draw_is_stable_for_small_changes(self):
        model = defense(randomness_quantization=0.01)
        like = torch.zeros(1)
        x = torch.tensor([[1.001, 0.001]])
        torch.testing.assert_close(model._one_time_uniform(x, like),
                                   model._one_time_uniform(x + 0.001, like))
        self.assertNotEqual(model._one_time_uniform(x, like).item(),
                            model._one_time_uniform(x + 0.02, like).item())

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_query_hash_transfers_from_mps_before_float64_conversion(self):
        model = defense()
        query = torch.tensor([[1.001, 0.001]], dtype=torch.float32)
        expected = model._one_time_uniform(query, torch.zeros(1))
        actual = model._one_time_uniform(query.to('mps'), torch.zeros(1, device='mps'))
        self.assertEqual(actual.device.type, 'mps')
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual.cpu(), expected)

    def test_optimizer_keeps_initial_sorting_when_lower_classes_swap(self):
        class FirstFeatureAttacker(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, x, return_logits=False):
                self.inputs.append(x.detach().clone())
                scores = x[:, 0] - 0.25
                return scores if return_logits else scores.sigmoid()

        attacker = FirstFeatureAttacker()
        model = defense(attacker, max_iter=2, step_size=1.)
        logits = torch.tensor([[0.79, 0.10, 0.11]]).log()
        result, _ = model._optimize_confidence(logits)
        self.assertTrue(any((features[:, 0] > features[:, 1]).any() for features in attacker.inputs))
        self.assertEqual(result.argmax(1).item(), 0)
        self.assertGreater(result[0, 1].item(), result[0, 2].item())

    def test_sorted_failures_preserve_exact_clean_output(self):
        logits = torch.tensor([[0.2, 1.1, -2.3], [2.3, -1.1, 0.2]])
        result, distortion = defense(max_iter=1, randomize_mix=0.1)._optimize_confidence(logits)
        self.assertTrue(torch.equal(result, logits.softmax(1)))
        self.assertTrue(torch.equal(distortion, torch.zeros(2)))

    def test_invalid_search_and_randomness_parameters_fail_early(self):
        for kwargs in ({'c3_init': 0}, {'c3_max': float('inf')}, {'budget_l1': -1},
                       {'randomness_quantization': 0}, {'step_size': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                defense(**kwargs)


if __name__ == '__main__':
    unittest.main()
