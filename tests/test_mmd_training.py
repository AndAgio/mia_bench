"""Regression checks for MMD sampling, training mode and clean-input metrics."""

import unittest
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader

from src.mia.defenses.training.mmd import MmdTrainManager, mix_rbf_mmd2
from src.utils.configs import OptimizerConfigs
from src.utils.configs import MmdDefenseConfigs
from src.trainer.stats_tracker import EpochStats
from tests.test_mmd_index_cache import CountingDataset


OPTIMIZERS = ('sgd', 'adam', 'sam', 'adaptive_sam', 'esam', 'adaptive_esam',
              'wsam', 'adaptive_wsam', 'looksam', 'adaptive_looksam',
              'friendlysam', 'adaptive_friendlysam')


def make_manager(optimizer='sgd', use_mixup=False, model=None):
    manager = MmdTrainManager.__new__(MmdTrainManager)
    manager._validation_index_dataset = None
    manager._validation_indices_by_label = {}
    manager.device = torch.device('cpu')
    manager.distributed = False
    manager.model = model if model is not None else torch.nn.Linear(2, 2)
    manager.criterion = torch.nn.CrossEntropyLoss(reduction='none')
    manager.logger = MagicMock()
    manager.setup_optimizer(OptimizerConfigs(name=optimizer, lr=0.01))
    manager.mmd_configs = SimpleNamespace(lmbd=1.0, use_mixup=use_mixup, mixup_alpha=0.4)
    manager.extra_configs = {}
    manager.reset_epoch_stats = MagicMock()
    manager.epoch_stats_tracker = MagicMock()
    manager.build_message_for_batch_end = MagicMock(return_value='')
    manager.build_message_for_batch_end_mmd = MagicMock(return_value='')
    manager.build_message_for_stage_end = MagicMock(return_value='')
    return manager


class MmdTrainingTests(unittest.TestCase):
    def test_kernel_value_gradient_and_permutation_against_direct_formula(self):
        torch.manual_seed(21)
        for batch_size in (1, 7):
            x = torch.softmax(torch.randn(batch_size, 3, dtype=torch.float64), dim=1).requires_grad_()
            y = torch.softmax(torch.randn(batch_size, 3, dtype=torch.float64), dim=1)
            kernel = lambda a, b: torch.exp(-((a[:, None] - b[None, :]) ** 2).sum(-1) / 2)
            expected = kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
            actual = mix_rbf_mmd2(x, y, [1])
            self.assertTrue(torch.allclose(actual, expected, atol=1e-12))
            actual_gradient = torch.autograd.grad(actual, x, retain_graph=True)[0]
            expected_gradient = torch.autograd.grad(expected, x)[0]
            self.assertTrue(torch.allclose(actual_gradient, expected_gradient, atol=1e-12))
            self.assertTrue(torch.allclose(actual, mix_rbf_mmd2(x, y.flip(0), [1]), atol=1e-12))

    def test_sgd_update_uses_only_training_side_gradient(self):
        torch.manual_seed(14)
        manager = make_manager()
        manager.optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.02)
        manager.mmd_configs.lmbd = 2.0
        reference = copy.deepcopy(manager.model)
        train = CountingDataset([0, 1])

        class ShiftedValidation(CountingDataset):
            def __getitem__(self, index):
                image, label, original, resampled = super().__getitem__(index)
                return image + 0.7, label, original, resampled

        manager.dataset = {'train': train, 'val': ShiftedValidation([0, 1])}
        manager.train_loader = DataLoader(train, batch_size=2)
        forwards = []
        manager.model.register_forward_hook(lambda module, args, outputs:
                                            forwards.append((args[0].clone(), outputs.requires_grad)))
        manager.train_with_mmd_distance()
        self.assertEqual([requires_grad for _, requires_grad in forwards], [True, False])
        x = reference(forwards[0][0]).softmax(dim=1)
        y = reference(forwards[1][0]).softmax(dim=1).detach()
        kernel = lambda a, b: torch.exp(-((a[:, None] - b[None, :]) ** 2).sum(-1) / 2)
        loss = 2.0 * (kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean())
        gradients = torch.autograd.grad(loss, tuple(reference.parameters()))
        for actual, initial, gradient in zip(manager.model.parameters(), reference.parameters(), gradients):
            self.assertTrue(torch.allclose(actual, initial - 0.02 * gradient, atol=1e-7))

    def test_epoch_gate_and_final_epoch(self):
        cases = [(0.0, 0.1, 1e-5, 1, True), (0.0, 0.03, 1.0, 1, True),
                 (0.0, 0.02, 1.0, 1, False), (0.0, 0.1, 0.0, 1, False),
                 (0.0, 0.1, 1.0, 3, False)]
        for train_acc, val_acc, weight, epoch, expected in cases:
            with self.subTest(train_acc=train_acc, val_acc=val_acc, weight=weight, epoch=epoch):
                manager = make_manager()
                manager.run_val = True
                manager.run_test = True
                manager.epoch = epoch
                manager.train_configs = SimpleNamespace(scheduler_config=SimpleNamespace(epochs=3))
                manager.mmd_configs.lmbd = weight
                for method in ('train_epoch', 'val_epoch', 'test_epoch', 'train_with_mmd_distance'):
                    setattr(manager, method, MagicMock())
                manager.get_train_accuracy = MagicMock(return_value=train_acc)
                manager.get_val_accuracy = MagicMock(return_value=val_acc)
                manager.scheduler = MagicMock()
                manager.train_executions()
                self.assertEqual(manager.train_with_mmd_distance.call_count, int(expected))
                manager.train_epoch.assert_called_once()
                manager.val_epoch.assert_called_once()
                manager.test_epoch.assert_called_once()
                manager.scheduler.step.assert_called_once()

    def test_short_classes_and_partial_batch_match_training_counts(self):
        manager = make_manager()
        train = CountingDataset([0, 0, 0, 1, 1, 0])
        manager.dataset = {'train': train, 'val': CountingDataset([1, 0])}
        manager.train_loader = DataLoader(train, batch_size=4)
        forwards = []
        manager.model.register_forward_pre_hook(lambda model, args: forwards.append(args[0].clone()))
        with patch('src.mia.defenses.training.mmd.mix_rbf_mmd2', wraps=mix_rbf_mmd2) as kernel:
            manager.train_with_mmd_distance()
        self.assertEqual([call.args[0].size(0) for call in kernel.call_args_list], [4, 2])
        for train_inputs, valid_inputs in zip(forwards[::2], forwards[1::2]):
            self.assertTrue(torch.equal(train_inputs[:, 0].sort().values, valid_inputs[:, 0].sort().values))

    def test_missing_class_has_clear_error(self):
        manager = make_manager()
        train = CountingDataset([0, 1])
        manager.dataset = {'train': train, 'val': CountingDataset([0])}
        manager.train_loader = DataLoader(train, batch_size=2)
        before = [p.detach().clone() for p in manager.model.parameters()]
        with self.assertRaisesRegex(ValueError, 'no samples for training class 1'):
            manager.train_with_mmd_distance()
        for parameter, saved in zip(manager.model.parameters(), before):
            self.assertTrue(torch.equal(parameter, saved))

    def test_mmd_uses_eval_mode_and_freezes_bn_for_all_optimizers(self):
        for optimizer in OPTIMIZERS:
            with self.subTest(optimizer=optimizer):
                torch.manual_seed(1)
                bn = torch.nn.BatchNorm1d(2)
                dropout = torch.nn.Dropout(0.2)
                model = torch.nn.Sequential(bn, dropout, torch.nn.Linear(2, 2))
                manager = make_manager(optimizer, model=model)
                train = CountingDataset([0, 1, 0, 1, 1])
                manager.dataset = {'train': train, 'val': CountingDataset([0, 1])}
                manager.train_loader = DataLoader(train, batch_size=4)
                modes = []
                dropout.register_forward_pre_hook(lambda module, args: modes.append(module.training))
                before = [p.detach().clone() for p in model.parameters()]
                buffers = [buffer.clone() for buffer in model.buffers()]
                model.train()
                manager.train_with_mmd_distance()
                self.assertTrue(modes and not any(modes))
                self.assertTrue(any(not torch.equal(p, saved) for p, saved in zip(model.parameters(), before)))
                for buffer, saved in zip(model.buffers(), buffers):
                    self.assertTrue(torch.equal(buffer, saved))
                self.assertEqual(bn.momentum, 0.1)
                self.assertFalse(hasattr(bn, 'backup_momentum'))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_mmd_records_loss_and_batch_timing(self):
        manager = make_manager()
        train = CountingDataset([0, 1, 0, 1])
        manager.dataset = {'train': train, 'val': CountingDataset([0, 1])}
        manager.train_loader = DataLoader(train, batch_size=2)
        manager.epoch_stats_tracker = EpochStats()
        manager.reset_epoch_stats = lambda phase: manager.epoch_stats_tracker.stage_begin(phase)
        manager.train_with_mmd_distance()
        summary = manager.epoch_stats_tracker.finalize_epoch(1).stages['mmd']
        self.assertEqual(summary.num_batches, 2)
        self.assertIn('mmd_loss', summary.metrics)
        self.assertTrue(torch.isfinite(torch.tensor(summary.metrics['mmd_loss'])))
        self.assertGreater(summary.batch_time_avg_sec, 0)
        self.assertGreater(summary.samples_per_sec_avg, 0)

    def test_direct_mixup_config_has_positive_alpha(self):
        self.assertGreater(MmdDefenseConfigs(use_mixup=True).mixup_alpha, 0)

    def test_stats_forward_does_not_update_bn_with_or_without_mixup(self):
        for optimizer in OPTIMIZERS:
            for use_mixup in (False, True):
                with self.subTest(optimizer=optimizer, use_mixup=use_mixup):
                    torch.manual_seed(3)
                    bn = torch.nn.BatchNorm1d(2)
                    model = torch.nn.Sequential(bn, torch.nn.Linear(2, 2))
                    manager = make_manager(optimizer, use_mixup, model)
                    model.train()
                    snapshots = []
                    model.register_forward_hook(lambda module, args, output:
                                                snapshots.append((bn.running_mean.clone(), bn.running_var.clone())))
                    inputs = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
                    targets = torch.tensor([0, 1, 0, 1])
                    before = [p.detach().clone() for p in model.parameters()]
                    # Two steps also exercise LookSAM's reused direction and restored BN state.
                    for _ in range(2):
                        start = len(snapshots)
                        manager.train_step(inputs, targets)
                        for previous, after in zip(snapshots[start], snapshots[-1]):
                            self.assertTrue(torch.equal(previous, after))
                        if optimizer in ('sgd', 'adam'):
                            self.assertEqual(len(snapshots) - start, 2 if use_mixup else 1)
                        self.assertEqual(bn.momentum, 0.1)
                        self.assertFalse(hasattr(bn, 'backup_momentum'))
                    self.assertTrue(any(not torch.equal(p, saved) for p, saved in zip(model.parameters(), before)))
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                    self.assertEqual(manager.epoch_stats_tracker.update.call_count, 2)


if __name__ == '__main__':
    unittest.main()
