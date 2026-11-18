"""TrainManager's AdamW optimizer, gradient clipping and the error when no epoch gives a best model."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
from src.trainer.train_manager import TrainManager
from src.utils.configs import OptimizerConfigs, SchedulerConfigs, TrainConfigs


class QuietLogger:
    """Swallows TrainManager's messages."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class NanLoss(torch.nn.Module):
    """A per-sample loss that is NaN from the first batch."""
    def forward(self, outputs, targets):
        return outputs.sum(dim=1) * float('nan')


def make_manager(folder, optimizer='adamw', grad_clip=None, loss='crossentropy', epochs=1):
    configs = TrainConfigs(optimizer_config=OptimizerConfigs(name=optimizer, lr=1e-3, weight_decay=1e-2),
                           scheduler_config=SchedulerConfigs(name='const', epochs=epochs), batch_size=8,
                           loss=loss, metrics=(), metric_to_track=None, grad_clip=grad_clip, resume=False,
                           ckpts_folder=Path(folder) / 'ckpts', resume_ckpts_folder=Path(folder) / 'resume')
    generator = torch.Generator().manual_seed(0)
    data = [IndexedDataset(TensorDataset(torch.randn(16, 4, generator=generator), torch.randint(0, 3, (16,))))
            for _ in range(2)]
    model = torch.nn.Linear(4, 3)
    model.name = 'linear'
    manager = TrainManager(train_configs=configs, name='options', logger=QuietLogger())
    manager.initialize_train(dataset=MultiDatasets(data, ids=['train', 'val']), model=model, configs=configs)
    return manager


class TrainManagerOptionsTests(unittest.TestCase):
    def test_adamw_uses_the_configured_learning_rate_and_weight_decay(self):
        with tempfile.TemporaryDirectory() as folder:
            optimizer = make_manager(folder).optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-3)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-2)

    def test_gradients_are_clipped_before_each_step_only_when_asked(self):
        for grad_clip in (None, .5):
            with self.subTest(grad_clip=grad_clip), tempfile.TemporaryDirectory() as folder, \
                    patch('torch.nn.utils.clip_grad_norm_') as clip:
                make_manager(folder, grad_clip=grad_clip).train(return_best_model=True)
                if grad_clip is None:
                    clip.assert_not_called()
                else:
                    self.assertEqual(clip.call_count, 2)          # 16 samples in batches of 8
                    self.assertEqual(clip.call_args.args[1], .5)

    def test_clipping_with_a_sam_like_optimizer_fails(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'SAM-like'):
            make_manager(folder, optimizer='sam', grad_clip=1.)

    def test_a_nan_validation_loss_from_the_first_epoch_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(RuntimeError, 'no best model'):
            make_manager(folder, loss=NanLoss(), epochs=2).train(return_best_model=True)


if __name__ == "__main__":
    unittest.main()
