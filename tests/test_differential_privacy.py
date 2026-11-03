"""DP-SGD regressions: configs, optimizers, loss logging and privacy accounting, with no dataset downloads."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
from src.mia.defenses.base import BaseDefender
from src.mia.defenses.training.differential_privacy import DifferentialPrivacyDefender, DifferentialPrivacyTrainManager
from src.utils.configs import (DPDefenseConfigs, OptimizerConfigs, SchedulerConfigs, TrainConfigs,
                               generate_configs_from_settings)
from src.utils.settings import build_parser
from src.utils.variables import DP_SAT_PAPER_RHO

torch.manual_seed(0)
SAMPLES = [(torch.randn(8), i % 3) for i in range(64)]


def make_manager(folder: Path, epochs=2, optimizer='sgd', resume=False, optimizer_options=None, **dp_options):
    configs = TrainConfigs(optimizer_config=OptimizerConfigs(name=optimizer, lr=0.1, **(optimizer_options or {})),
                           scheduler_config=SchedulerConfigs(name='const', epochs=epochs),
                           batch_size=16, device='cpu', seed=0, resume=resume,
                           ckpts_folder=folder / 'ckpts', resume_ckpts_folder=folder / 'resume')
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    model.name = 'mlp'
    data = MultiDatasets(datasets=[IndexedDataset(SAMPLES), IndexedDataset(SAMPLES[:16]),
                                   IndexedDataset(SAMPLES[:16])], ids=['train', 'val', 'test'])
    manager = DifferentialPrivacyTrainManager(configs, 'dp', logger=MagicMock())
    manager.initialize_train(data, model, configs, DPDefenseConfigs(**dp_options))
    return manager


def dp_configs(*argv):
    return generate_configs_from_settings(build_parser().parse_args([*argv, '--dataset', 'cifar10'])).defender.defense


class DPConfigsTest(unittest.TestCase):
    def test_every_alias_selects_dp_and_delta_defaults_to_the_paper(self):
        for mode in ('dp', 'differential_privacy', 'differential-privacy'):
            configs = dp_configs('--defender_mode', mode)
            self.assertIsInstance(configs, DPDefenseConfigs)
            self.assertEqual((configs.delta, configs.target_epsilon, configs.grad_sample_mode), (1e-5, None, 'ghost'))
        configs = dp_configs('--defender_mode', 'dp', '--defender_dp_delta', '1e-6', '--defender_dp_epsilon', '4')
        self.assertEqual((configs.delta, configs.target_epsilon), (1e-6, 4.0))
        self.assertEqual(DPDefenseConfigs().grad_sample_mode, 'ghost')


def mean_loss_gradient(manager, weights, inputs, targets):
    """Gradient of the mean loss of a plain copy of the manager's model, at the given weights."""
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)
    torch.nn.functional.cross_entropy(model(inputs), targets).backward()
    return [p.grad for p in model.parameters()]


class DPTrainingTest(unittest.TestCase):
    def test_only_sam_runs_with_dp(self):
        for optimizer in ('adaptive_sam', 'esam', 'wsam', 'looksam', 'friendlysam'):
            with self.subTest(optimizer=optimizer), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, f'no private version of {optimizer}'):
                    make_manager(Path(folder), optimizer=optimizer)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(make_manager(Path(folder), optimizer='sam').sat_rho, DP_SAT_PAPER_RHO)
            manager = make_manager(Path(folder), optimizer='sam', optimizer_options={'extra': {'sam_rho': 0.1}})
            self.assertEqual(manager.sat_rho, 0.1)
            self.assertIsNone(make_manager(Path(folder)).sat_rho)

    def test_dp_sat_takes_the_gradient_at_weights_moved_along_the_previous_noisy_gradient(self):
        # Without noise and clipping, each step's gradient is the batch's mean gradient, which can be checked exactly.
        with tempfile.TemporaryDirectory() as folder:
            manager = make_manager(Path(folder), optimizer='sam', grad_sample_mode='hooks', noise_multiplier=0.0,
                                   max_grad_norm=1e6, optimizer_options={'momentum': 0.0, 'weight_decay': 0.0,
                                                                         'extra': {'sam_rho': 0.5}})
            # train_step outside train(): no stats or progress message.
            manager.epoch_stats_tracker, manager.extra_configs = MagicMock(), None
            manager.build_message_for_batch_end = MagicMock(return_value='')
            inputs, targets = torch.stack([x for x, _ in SAMPLES[:16]]), torch.tensor([y for _, y in SAMPLES[:16]])
            previous = None
            for _ in range(3):
                weights = [p.detach().clone() for p in manager.optimizer.params]
                moved = weights
                if previous is not None:
                    norm = torch.norm(torch.stack([g.norm() for g in previous]))
                    moved = [w + 0.5 * g / norm for w, g in zip(weights, previous)]
                manager.train_step(inputs, targets)
                for g, expected in zip(manager.sat_direction, mean_loss_gradient(manager, moved, inputs, targets)):
                    torch.testing.assert_close(g, expected)
                # The step updates the weights it started from, not the moved ones.
                for p, w, g in zip(manager.optimizer.params, weights, manager.sat_direction):
                    torch.testing.assert_close(p.detach(), w - 0.1 * g)
                previous = manager.sat_direction

    def test_dp_sat_spends_the_epsilon_of_dp_sgd_in_every_mode(self):
        for mode in ('ghost', 'hooks', 'functorch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                sgd = make_manager(Path(folder) / 'sgd', grad_sample_mode=mode)
                sgd.train()
                sat = make_manager(Path(folder) / 'sat', optimizer='sam', grad_sample_mode=mode)
                sat.train()
                self.assertIsNotNone(sat.sat_direction)
                self.assertAlmostEqual(sat.epsilon, sgd.epsilon, places=6)

    def test_only_mps_empties_its_cache_after_each_step(self):
        manager = object.__new__(DifferentialPrivacyTrainManager)
        manager.private_step = MagicMock()
        for device, calls in (('cpu', 0), ('cuda', 0), ('mps', 2)):
            with self.subTest(device=device), patch('torch.mps.empty_cache') as empty_cache:
                manager.device = torch.device(device)
                manager.train_step(None, None)
                manager.train_step(None, None)
                self.assertEqual(empty_cache.call_count, calls)

    def test_dp_sat_keeps_its_direction_in_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = make_manager(Path(folder) / 'first', optimizer='sam')
            manager.train()
            resumed = make_manager(Path(folder) / 'second', optimizer='sam')
            resumed.load_checkpoint_extra(manager.get_checkpoint_extra())
            for restored, kept in zip(resumed.sat_direction, manager.sat_direction):
                torch.testing.assert_close(restored, kept)

    def test_every_mode_trains_logs_the_loss_and_spends_epsilon(self):
        for mode in ('ghost', 'hooks', 'functorch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                manager = make_manager(Path(folder), grad_sample_mode=mode)
                _, stats = manager.train(return_best_model=True, return_stats=True)
                losses = [epoch['metric:loss'] for epoch in stats.to_checkpoint_dict()['history']
                          if epoch['stage'] == 'train']
                self.assertTrue(losses and all(isinstance(loss, float) for loss in losses), losses)
                self.assertGreater(manager.epsilon, 0)

    def test_target_epsilon_is_reached(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = make_manager(Path(folder), target_epsilon=4.0, delta=1e-5)
            manager.train()
            self.assertAlmostEqual(manager.epsilon, 4.0, delta=0.1)
            self.assertNotEqual(manager.optimizer.noise_multiplier, DPDefenseConfigs().noise_multiplier)

    def test_epsilon_counts_the_epochs_before_a_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            uninterrupted = make_manager(Path(folder) / 'full', epochs=3)
            uninterrupted.train()
            for drop_accounting in (False, True):
                with self.subTest(drop_accounting=drop_accounting):
                    split = Path(folder) / f'split_{drop_accounting}'
                    first = make_manager(split, epochs=2)
                    first.train()
                    if drop_accounting:
                        # A checkpoint from before the accounting was saved.
                        path = Path(first.resume_folder) / 'last.pth'
                        checkpoint = torch.load(path, weights_only=False)
                        del checkpoint['extra']['privacy_accountant']
                        torch.save(checkpoint, path)
                    resumed = make_manager(split, epochs=3, resume=True)
                    resumed.train()
                    self.assertAlmostEqual(resumed.epsilon, uninterrupted.epsilon, places=6)


class PrivacyBudgetTest(unittest.TestCase):
    def test_only_dp_reports_a_privacy_budget(self):
        self.assertIsNone(BaseDefender.get_privacy_budget(SimpleNamespace()))
        defender = DifferentialPrivacyDefender.__new__(DifferentialPrivacyDefender)
        defender.logger, defender.name, defender.dataset = MagicMock(), 'dp', MagicMock()
        defender.untrained_model, defender.dp_configs = torch.nn.Linear(2, 2), DPDefenseConfigs(delta=1e-6)
        with patch('src.mia.defenses.training.differential_privacy.DifferentialPrivacyTrainManager') as manager:
            manager.return_value.epsilon = 2.5
            defender.train_model(SimpleNamespace())
        self.assertEqual(defender.get_privacy_budget(), {'epsilon': 2.5, 'delta': 1e-6})


if __name__ == '__main__':
    unittest.main()
