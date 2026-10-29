"""Empty Poisson draws must keep DP noise/accounting steps and valid metrics."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.defenses.training.differential_privacy import DifferentialPrivacyTrainManager
from src.models.resnet import ResNet18


class DPEmptyBatchTests(unittest.TestCase):
    def private_manager(self, mode, per_layer, noise, architecture="cnn"):
        torch.manual_seed(123)
        class RejectEmptyInputs(torch.nn.Sequential):
            calls = 0

            def forward(self, inputs):
                if not len(inputs):
                    raise RuntimeError("This backbone does not support empty inputs")
                self.calls += 1
                return super().forward(inputs)

        class RejectEmptyLoss(torch.nn.CrossEntropyLoss):
            def forward(self, inputs, targets):
                if not len(targets):
                    raise RuntimeError("This criterion does not support empty targets")
                return super().forward(inputs, targets)

        layers = ((torch.nn.Conv2d(3, 4, 1), torch.nn.GroupNorm(1, 4), torch.nn.ReLU(),
                   torch.nn.Flatten(1), torch.nn.Linear(64, 2)) if architecture == "cnn" else
                  (torch.nn.Flatten(1), torch.nn.Linear(48, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)))
        backbone = RejectEmptyInputs(*layers)
        model = backbone
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=0., weight_decay=0.)
        criterion = RejectEmptyLoss()
        data = DataLoader(TensorDataset(torch.zeros(4, 3, 4, 4), torch.zeros(4, dtype=torch.long)),
                          batch_size=2)
        engine = PrivacyEngine(accountant="rdp")
        options = dict(module=model, optimizer=optimizer, data_loader=data,
                       noise_multiplier=noise, grad_sample_mode=mode,
                       clipping="per_layer" if per_layer else "flat",
                       max_grad_norm=[1.] * len(list(model.parameters())) if per_layer else 1.)
        if mode == "ghost":
            model, optimizer, criterion, _ = engine.make_private(criterion=criterion, **options)
        else:
            model, optimizer, _ = engine.make_private(**options)

        manager = object.__new__(DifferentialPrivacyTrainManager)
        manager.model, manager.optimizer, manager.criterion = model, optimizer, criterion
        manager.device = torch.device("cpu")
        manager.epoch_stats_tracker = Mock()
        manager.logger = Mock()
        manager.extra_configs = None
        manager.build_message_for_batch_end = Mock(return_value="training")
        manager.backbone = backbone
        return manager, engine

    def test_empty_steps_keep_noise_accounting_and_next_nonempty_step(self):
        for mode, per_layer in (("hooks", False), ("hooks", True), ("ghost", False)):
            for noise in (0., 1.):
                with self.subTest(mode=mode, per_layer=per_layer, noise=noise):
                    manager, engine = self.private_manager(mode, per_layer, noise)
                    before = [p.detach().clone() for p in manager.model.parameters()]
                    manager.train_step(torch.empty(0, 3, 4, 4), torch.empty(0, dtype=torch.long))
                    self.assertEqual(sum(item[2] for item in engine.accountant.history), 1)
                    self.assertTrue(all(torch.isfinite(p).all() for p in manager.model.parameters()))
                    changed = any(not torch.equal(old, new) for old, new in zip(before, manager.model.parameters()))
                    self.assertEqual(changed, noise > 0)
                    self.assertEqual(manager.backbone.calls, 0)
                    manager.epoch_stats_tracker.update.assert_not_called()
                    manager.epoch_stats_tracker.batch_end.assert_called_once_with(batch_size=0)

                    # Empty batches must not leave processed gradients, activations,
                    # or a changed reduction mode that breaks the next real step.
                    manager.train_step(torch.randn(2, 3, 4, 4), torch.tensor([0, 1]))
                    self.assertEqual(sum(item[2] for item in engine.accountant.history), 2)
                    self.assertTrue(all(torch.isfinite(p).all() for p in manager.model.parameters()))
                    self.assertEqual(manager.epoch_stats_tracker.update.call_count, 1)
                    self.assertEqual(manager.backbone.calls, 1)
                    if mode == "ghost":
                        self.assertEqual(manager.criterion.loss_reduction, "mean")
                    else:
                        self.assertEqual(manager.criterion.reduction, "mean")
                    for _ in range(2):
                        manager.train_step(torch.empty(0, 3, 4, 4), torch.empty(0, dtype=torch.long))
                    self.assertEqual(sum(item[2] for item in engine.accountant.history), 4)
                    self.assertTrue(all(torch.isfinite(p).all() for p in manager.model.parameters()))
                    self.assertEqual(manager.epoch_stats_tracker.update.call_count, 1)
                    self.assertEqual(manager.backbone.calls, 1)

    def test_empty_mlp_steps_do_not_call_model_and_keep_accounting(self):
        for mode, per_layer in (("hooks", False), ("hooks", True), ("ghost", False)):
            with self.subTest(mode=mode, per_layer=per_layer):
                manager, engine = self.private_manager(mode, per_layer, 1., architecture="mlp")
                manager.train_step(torch.empty(0, 3, 4, 4), torch.empty(0, dtype=torch.long))
                self.assertEqual(manager.backbone.calls, 0)
                self.assertEqual(sum(item[2] for item in engine.accountant.history), 1)
                manager.train_step(torch.randn(2, 3, 4, 4), torch.tensor([0, 1]))
                self.assertEqual(manager.backbone.calls, 1)
                self.assertEqual(sum(item[2] for item in engine.accountant.history), 2)
                self.assertTrue(all(torch.isfinite(p).all() for p in manager.model.parameters()))

    def test_virtual_batch_skip_flags_still_control_noise_and_accounting(self):
        for mode in ("hooks", "ghost"):
            with self.subTest(mode=mode):
                manager, engine = self.private_manager(mode, False, 1.)
                manager.optimizer.signal_skip_step(do_skip=True)
                manager.train_step(torch.empty(0, 3, 4, 4), torch.empty(0, dtype=torch.long))
                self.assertEqual(len(engine.accountant.history), 0)
                manager.train_step(torch.randn(2, 3, 4, 4), torch.tensor([0, 1]))
                self.assertEqual(sum(item[2] for item in engine.accountant.history), 1)
                self.assertTrue(all(torch.isfinite(p).all() for p in manager.model.parameters()))

    def test_resnet_forward_and_features_accept_empty_batches(self):
        model = ResNet18(num_classes=3).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.empty(0, 3, 8, 8)).shape), (0, 3))
            self.assertEqual(tuple(model.feature(torch.empty(0, 3, 8, 8)).shape), (0, 512))
            inputs = torch.randn(2, 3, 8, 8)
            torch.testing.assert_close(model(inputs), model.fc(model.feature(inputs)))


if __name__ == "__main__":
    unittest.main()
