"""Full MMD training and checkpoint-resume checks on a tiny deterministic dataset."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.data.helpers import MultiDatasets
from src.mia.defenses.training.mmd import MmdTrainManager
from src.utils.configs import MmdDefenseConfigs, OptimizerConfigs, SchedulerConfigs, TrainConfigs
from tests.test_mmd_index_cache import CountingDataset
from tests.test_mmd_training import OPTIMIZERS
from tests.test_relax_loss import QuietLogger


def make_training_manager(folder, optimizer, use_mixup, resume=False):
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Dropout(0.2), torch.nn.Linear(2, 2))
    model.name = 'mmd_tiny'
    with torch.no_grad():
        model[-1].weight.copy_(torch.tensor([[0.1, -0.1], [-0.1, 0.1]]))
        model[-1].bias.copy_(torch.tensor([2., 0.]))
    data = MultiDatasets(datasets=[CountingDataset([0, 0, 0, 1, 0, 0, 0, 1]),
                                  CountingDataset([0, 1, 1, 1, 0, 1, 1, 1]),
                                  CountingDataset([0, 1, 0, 1])], ids=['train', 'val', 'test'])
    configs = TrainConfigs(optimizer_config=OptimizerConfigs(name=optimizer, lr=0.01),
                           scheduler_config=SchedulerConfigs(name='const', epochs=3),
                           # ESAM selects half the CE batch; keep at least two samples for training BN1d.
                           batch_size=4, device='cpu', seed=17, resume=resume,
                           ckpts_folder=folder / 'ckpts', resume_ckpts_folder=folder / 'resume')
    manager = MmdTrainManager(configs, name='mmd_lifecycle', logger=QuietLogger(),
                              mmd_configs=MmdDefenseConfigs(use_mixup=use_mixup))
    manager.initialize_train(data, model, configs)
    return manager


class MmdLifecycleTests(unittest.TestCase):
    def test_training_stats_best_model_and_resume_all_optimizers(self):
        with tempfile.TemporaryDirectory(prefix='mmd-lifecycle-') as folder:
            for optimizer in OPTIMIZERS:
                for use_mixup in (False, True):
                    with self.subTest(optimizer=optimizer, use_mixup=use_mixup):
                        root = Path(folder) / optimizer / str(use_mixup)
                        full = make_training_manager(root / 'full', optimizer, use_mixup)
                        best, stats = full.train(return_stats=True)
                        self.assertTrue(all(torch.isfinite(p).all() for p in best.parameters()))
                        self.assertEqual(sorted(stats.history), [1, 2, 3])
                        for epoch in (1, 2):
                            summary = stats.history[epoch].stages['mmd']
                            self.assertEqual(summary.num_batches, 2)
                            self.assertIn('mmd_loss', summary.metrics)
                        self.assertNotIn('mmd', stats.history[3].stages)
                        self.assertEqual(full.dataset.get('val').target_scans, 1)

                        interrupted = make_training_manager(root / 'resumed', optimizer, use_mixup)
                        execute_epoch = interrupted.train_executions

                        def stop_after_checkpoint():
                            if interrupted.epoch == 2:
                                raise RuntimeError('test interruption after epoch 1 checkpoint')
                            execute_epoch()

                        interrupted.train_executions = stop_after_checkpoint
                        with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                            interrupted.train()
                        resumed = make_training_manager(root / 'resumed', optimizer, use_mixup, resume=True)
                        resumed.train()
                        self.assertEqual(sorted(resumed.train_stats_tracker.history), [1, 2, 3])
                        for name, tensor in full.model.state_dict().items():
                            self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
