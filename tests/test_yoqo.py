"""YOQO: the query search's steps, losses and stop, its labels, batching, one target query per sample, defaults."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset
from src.mia.attacks.label_only import yoqo
from src.mia.attacks.label_only.yoqo import Yoqo
from src.utils.configs import YoqoAttackConfig, generate_configs_from_settings
from src.utils.settings import build_parser


class Silent:
    """A logger that prints nothing."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class CountingModel(torch.nn.Module):
    """Wraps a model and counts the samples it is queried on."""
    def __init__(self, model):
        super().__init__()
        self.model, self.queried = model, 0

    def forward(self, x):
        self.queried += len(x)
        return self.model(x)


def mlp(seed, dims=4, classes=3):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(dims, 16), torch.nn.ReLU(), torch.nn.Linear(16, classes)).double().eval()


def attack(mode, models, inside=None, target=None, **configs):
    """A YOQO attack on the given shadow models; inside[i] lists the models that trained on the sample of index i."""
    configs = {'alpha': 2., 'gamma': 5., 'adv_opt_lr': 0.1, **configs}
    instance = object.__new__(Yoqo)
    instance.logger = Silent()
    instance.mode = mode
    instance.attack_configs = YoqoAttackConfig(mode=mode, **configs)
    instance.defender_model = target
    inside = inside or {}
    instance.shadow_manager = SimpleNamespace(
        get_n_models=lambda: len(models), get_n_datasets=lambda: len(models),
        get_all_model_indeces=lambda: list(range(len(models))), get_model=lambda i: models[i],
        find_all_in_dataset_indices_for_sample_id=lambda id: inside.get(id, []))
    return instance


class YoqoTests(unittest.TestCase):
    def test_each_step_follows_the_gradient_at_the_current_query(self):
        model = mlp(0)
        x0, y = torch.randn(1, 4, dtype=torch.float64), torch.tensor([0])
        yoqo_attack = attack('offline', [model], adv_opt_max_iter=2, adv_opt_loss_threshold=0., gamma=0.)
        target = Yoqo.target_labels([model], x0, y)[:, 0]
        x = x0.clone()
        for _ in range(2):
            x.requires_grad_(True)
            # The authors' offline scale, 6 / (1 + gamma), is 6 at gamma = 0.
            gradient, = torch.autograd.grad(6 * F.cross_entropy(model(x), target), x)
            x = x.detach() - 0.1 * gradient
        query = yoqo_attack.search_batch([model], x0, y, torch.zeros(1, 1, dtype=torch.bool))
        torch.testing.assert_close(query, x)

    def test_out_models_target_their_top_label_other_than_the_sample_s(self):
        model = lambda x: torch.tensor([[0., 3., 2.], [5., 1., 4.]])
        labels = Yoqo.target_labels([model], torch.zeros(2, 4), torch.tensor([1, 2]))
        np.testing.assert_array_equal(labels[:, 0], [2, 0])

    def test_the_search_stops_on_the_loss_per_out_model_whatever_their_number(self):
        x0, y = torch.randn(1, 4, dtype=torch.float64), torch.tensor([0])
        model = mlp(1)
        target = Yoqo.target_labels([model], x0, y)[0, 0]
        loss = 6 * F.cross_entropy(model(x0), target[None]).item()
        for copies in (1, 3):
            with self.subTest(copies=copies):
                models = [model] * copies
                yoqo_attack = attack('offline', models, adv_opt_max_iter=10, adv_opt_loss_threshold=loss + 1e-9,
                                     gamma=0.)
                with torch.enable_grad():
                    query = yoqo_attack.search_batch(models, x0, y, torch.zeros(1, copies, dtype=torch.bool))
                # One step, taken before the stop, of the gradient summed over the copies.
                x = x0.clone().requires_grad_(True)
                gradient, = torch.autograd.grad(6 * F.cross_entropy(model(x), target[None]) * copies, x)
                torch.testing.assert_close(query, x0 - 0.1 * gradient)

    def test_online_in_models_keep_the_label_and_out_models_push_away_from_it(self):
        models = [mlp(2), mlp(3)]
        x0, y = torch.randn(1, 4, dtype=torch.float64), torch.tensor([1])
        inside = torch.tensor([[True, False]])
        yoqo_attack = attack('online', models, adv_opt_max_iter=1, adv_opt_loss_threshold=0., alpha=1.)
        targets = Yoqo.target_labels(models, x0, y)
        x = x0.clone().requires_grad_(True)
        # The authors' online scale, 3 / (1 + alpha), is 1.5 at alpha = 1.
        loss = 1.5 * (F.cross_entropy(models[0](x), y) + 1. * F.cross_entropy(models[1](x), targets[:, 1]))
        gradient, = torch.autograd.grad(loss, x)
        torch.testing.assert_close(yoqo_attack.search_batch(models, x0, y, inside), x0 - 0.1 * gradient)

    def test_offline_counts_the_distance_with_each_out_model(self):
        models = [mlp(4), mlp(5)]
        x0, y = torch.randn(1, 4, dtype=torch.float64), torch.tensor([2])
        yoqo_attack = attack('offline', models, adv_opt_max_iter=2, adv_opt_loss_threshold=0., gamma=2.)
        targets = Yoqo.target_labels(models, x0, y)
        x = x0.clone()
        for _ in range(2):
            x.requires_grad_(True)
            distance = ((x - x0) ** 2).mean()
            loss = sum((2. * distance + F.cross_entropy(model(x), targets[:, i])) * 6 / 3
                       for i, model in enumerate(models))
            gradient, = torch.autograd.grad(loss, x)
            x = x.detach() - 0.1 * gradient
        torch.testing.assert_close(yoqo_attack.search_batch(models, x0, y, torch.zeros(1, 2, dtype=torch.bool)), x)

    def test_batched_queries_equal_queries_searched_one_sample_at_a_time(self):
        models = [mlp(seed) for seed in range(4)]
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(10, 4, generator=generator, dtype=torch.float64)
        y = torch.randint(0, 3, (10,), generator=generator)
        data = IndexedDataset(torch.utils.data.TensorDataset(x, y))
        for mode in ('online', 'offline'):
            with self.subTest(mode=mode):
                inside = {i: [i % 4, (i + 1) % 4] for i in range(10)} if mode == 'online' else {}
                yoqo_attack = attack(mode, models, inside, adv_opt_max_iter=20, adv_opt_loss_threshold=0.3)
                mask = yoqo_attack.in_models_mask(data.get_indices(mode='original'))
                batched, _ = yoqo_attack.search_queries(data, mask, torch.device('cpu'))
                with patch.object(yoqo, 'SEARCH_BATCH_SIZE', 1):
                    single, _ = yoqo_attack.search_queries(data, mask, torch.device('cpu'))
                torch.testing.assert_close(batched, single, rtol=0, atol=1e-12)

    def test_one_target_query_per_sample_and_the_scores_are_the_decisions(self):
        target = CountingModel(mlp(9))
        data = IndexedDataset(torch.utils.data.TensorDataset(torch.randn(6, 4, dtype=torch.float64),
                                                             torch.tensor([0, 1, 2, 0, 1, 2])))
        yoqo_attack = attack('offline', [mlp(5), mlp(6)], target=target, adv_opt_max_iter=5,
                             adv_opt_loss_threshold=0.1)
        [output] = yoqo_attack.score(data, device='cpu')
        self.assertEqual(target.queried, 6)
        self.assertTrue(set(output.scores.tolist()) <= {0, 1})
        np.testing.assert_array_equal(output.scores, output.decisions)

    def test_unset_search_settings_take_the_paper_s_for_the_mode(self):
        for mode, max_iter, threshold in (('online_yoqo', 30, 4 / 16), ('offline_yoqo', 35, 8 / 16)):
            with self.subTest(mode=mode):
                settings = build_parser().parse_args(['--attacker_mode', mode])
                configs = generate_configs_from_settings(settings).attacker.attack
                self.assertEqual((configs.adv_opt_max_iter, configs.adv_opt_loss_threshold), (max_iter, threshold))
        settings = build_parser().parse_args(['--attacker_mode', 'offline_yoqo', '--attacker_yoqo_adv_opt_max_iter',
                                              '7', '--attacker_yoqo_adv_opt_loss_threshold', '0.1'])
        configs = generate_configs_from_settings(settings).attacker.attack
        self.assertEqual((configs.adv_opt_max_iter, configs.adv_opt_loss_threshold), (7, 0.1))

    def test_the_calibrated_rule_needs_no_calibration_samples(self):
        instance = object.__new__(Yoqo)
        instance.evaluation_configs = SimpleNamespace(decision_rules=['calibrated'])
        self.assertFalse(instance.needs_calibration())


if __name__ == '__main__':
    unittest.main()
