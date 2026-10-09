#!/usr/bin/env python3
"""One classification epoch plus one explicit MMD pass on synthetic images.

Run from the repository root:
    python3 tests/smoke_mmd.py
    python3 tests/smoke_mmd.py --optimizer sam --mixup

The explicit MMD pass bypasses the accuracy gate and final-epoch skip for this
smoke test only. Production training keeps the reference's scheduling behavior.
No data is downloaded, and temporary training folders are removed on exit.
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.helpers import MultiDatasets
from src.mia.defenses.training.mmd import MmdTrainManager
from src.trainer.stats_tracker import EpochStats
from src.utils.configs import MmdDefenseConfigs, OptimizerConfigs, SchedulerConfigs, TrainConfigs


OPTIMIZERS = ('sgd', 'adam', 'sam', 'adaptive_sam', 'esam', 'adaptive_esam',
              'wsam', 'adaptive_wsam', 'looksam', 'adaptive_looksam',
              'friendlysam', 'adaptive_friendlysam')


class QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class SyntheticImages(Dataset):
    def __init__(self, size, seed):
        prototypes = torch.randn(4, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        generator = torch.Generator().manual_seed(seed)
        self.targets = torch.arange(size) % 4
        self.inputs = prototypes[self.targets] + 0.5 * torch.randn(size, 3, 8, 8, generator=generator)
        self.target_scans = 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], index, index

    def get_all_targets(self, to_torch=False):
        self.target_scans += 1
        return self.targets if to_torch else self.targets.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--optimizer', choices=OPTIMIZERS, default='sgd')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable on this host')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable on this host')
    # TrainManager prefers CUDA over MPS when both are available.
    if args.device == 'mps' and torch.cuda.is_available():
        parser.error('TrainManager selects CUDA on this host; use --device cuda')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8),
                                torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(8, 4))
    model.name = 'synthetic_mmd_smoke'
    data = MultiDatasets(datasets=[SyntheticImages(64, 1), SyntheticImages(32, 2), SyntheticImages(32, 3)],
                         ids=['train', 'val', 'test'])

    with tempfile.TemporaryDirectory(prefix='mmd-smoke-') as folder:
        root = Path(folder)
        configs = TrainConfigs(optimizer_config=OptimizerConfigs(name=args.optimizer, lr=0.01),
                               scheduler_config=SchedulerConfigs(name='const', epochs=1),
                               batch_size=16, device=args.device, seed=17, resume=False,
                               ckpts_folder=root / 'ckpts', resume_ckpts_folder=root / 'resume')
        manager = MmdTrainManager(configs, name='mmd_smoke', logger=QuietLogger(),
                                  mmd_configs=MmdDefenseConfigs(use_mixup=args.mixup))
        manager.initialize_train(data, model, configs)
        manager.reset_running_stats()
        manager.train_stats_tracker.start_timer()
        manager.epoch_stats_tracker = EpochStats()
        manager.epoch = 1
        manager.extra_configs = {}
        manager.epoch_stats_tracker.epoch_start()

        print(f'Device: {manager.device} | optimizer: {args.optimizer} | mixup: {args.mixup}')
        print('Running 1 classification epoch (64 images, 4 batches)...', flush=True)
        train_summary = manager.train_epoch()
        before_parameters = [parameter.detach().clone() for parameter in manager.model.parameters()]
        before_buffers = [buffer.clone() for buffer in manager.model.buffers()]
        print('Running 1 explicit MMD pass (bypasses the gate/final-epoch skip for this test)...', flush=True)
        manager.train_with_mmd_distance()
        if manager.model.training:
            raise RuntimeError('MMD left the model in training mode')
        for actual, saved in zip(manager.model.buffers(), before_buffers):
            if not torch.equal(actual, saved):
                raise RuntimeError('MMD changed BatchNorm buffers')
        if not any(not torch.equal(actual, saved) for actual, saved in zip(manager.model.parameters(), before_parameters)):
            raise RuntimeError('MMD did not update parameters')
        if not all(torch.isfinite(parameter).all() for parameter in manager.model.parameters()):
            raise RuntimeError('Non-finite model parameters')
        if data.get('val').target_scans != 1:
            raise RuntimeError('Validation indices were not computed exactly once')

        manager.val_epoch()
        manager.test_epoch()
        manager.scheduler.step()
        manager.epoch_stats_tracker.epoch_end()
        summary = manager.epoch_stats_tracker.finalize_epoch(epoch=1)
        mmd = summary.stages['mmd']
        if mmd.num_batches != 4 or not math.isfinite(mmd.metrics['mmd_loss']):
            raise RuntimeError('Incorrect MMD batch count or non-finite loss')
        print(f"Classification loss: {train_summary.metrics['loss']:.6f}")
        print(f"MMD loss: {mmd.metrics['mmd_loss']:.6f} | MMD batches: {mmd.num_batches}")
        print(f"Validation accuracy: {summary.stages['val'].metrics['multi_class_accuracy']:.3f}")
        print(f"Test accuracy: {summary.stages['test'].metrics['multi_class_accuracy']:.3f}")
        print('PASS: finite parameters updated, BN unchanged during MMD, class-index cache and stats verified.')


if __name__ == '__main__':
    main()
