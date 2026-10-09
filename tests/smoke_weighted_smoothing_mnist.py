#!/usr/bin/env python3
"""Run one MNIST epoch with prediction noise + PyTorch CE and no warmup.

    python tests/smoke_weighted_smoothing_mnist.py
    python tests/smoke_weighted_smoothing_mnist.py --device mps
    python tests/smoke_weighted_smoothing_mnist.py --optimizer sam
    python tests/smoke_weighted_smoothing_mnist.py --train-samples 60000 --test-samples 10000

Downloads MNIST on first use. This checks execution and finite results, not
membership-inference protection or reproduction of the paper's reported accuracy.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.helpers import IndexedDataset, MultiDatasets
from src.mia.defenses.training.weigthed_smoothing import WeightedSmoothingTrainManager
from src.utils.configs import OptimizerConfigs, SchedulerConfigs, TrainConfigs, WeightedSmoothingDefenseConfigs
from src.utils.log import get_logger


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'mnist_smoke_cnn'
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cpu', help='cpu, or mps/cuda to use an available accelerator')
    parser.add_argument('--optimizer', choices=('adam', 'sgd', 'sam', 'esam'), default='adam')
    parser.add_argument('--lr', type=float, default=None, help='Default: 0.001 for Adam, 0.01 otherwise')
    parser.add_argument('--sigma', type=float, default=0.1, help='Prediction noise standard deviation')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-samples', type=int, default=5000)
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datas' / 'mnist')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'smoke_test_results' / 'weighted_smoothing_mnist')
    args = parser.parse_args()
    if not 1 <= args.train_samples <= 60000 or not 1 <= args.test_samples <= 10000:
        parser.error('train-samples must be in 1..60000 and test-samples in 1..10000')
    if args.batch_size < 1 or args.sigma <= 0 or not math.isfinite(args.sigma):
        parser.error('batch-size and sigma must be positive; sigma must be finite')
    if args.lr is None:
        args.lr = 0.001 if args.optimizer == 'adam' else 0.01
    if args.lr <= 0 or not math.isfinite(args.lr):
        parser.error('lr must be finite and positive')
    return args


def main():
    args = parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_indices = torch.randperm(len(train), generator=generator)[:args.train_samples].tolist()
    test_indices = torch.randperm(len(test), generator=generator)[:args.test_samples].tolist()
    data = MultiDatasets(
        datasets=[IndexedDataset(Subset(train, train_indices)), IndexedDataset(Subset(test, test_indices))],
        ids=['train', 'test'],
    )
    configs = TrainConfigs(
        optimizer_config=OptimizerConfigs(name=args.optimizer, lr=args.lr),
        scheduler_config=SchedulerConfigs(name='const', epochs=1),
        batch_size=args.batch_size, device=args.device, seed=args.seed, resume=False,
        ckpts_folder=args.output_dir / 'ckpts', resume_ckpts_folder=args.output_dir / 'resume_ckpts',
    )
    logger = get_logger(name='weighted_smoothing_mnist', log_folder=str(args.output_dir / 'logs'))
    manager = WeightedSmoothingTrainManager(
        configs, name='weighted_smoothing_mnist', logger=logger,
        # Zero warmup ensures the only training epoch actually uses smoothing.
        weighted_smoothing_configs=WeightedSmoothingDefenseConfigs(warmup_epochs=0, sigma_noise=args.sigma),
    )
    manager.initialize_train(dataset=data, model=MNISTCNN(), configs=configs)
    before = [parameter.detach().clone() for parameter in manager.model.parameters()]
    print(f'One smoothing epoch: train={len(data.get("train"))}, test={len(data.get("test"))}, '
          f'device={manager.device}, optimizer={args.optimizer}, sigma={args.sigma}')
    model, stats = manager.train(return_stats=True)

    if not torch.isfinite(manager.weights).all():
        raise RuntimeError('Non-finite smoothing weights')
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError('Non-finite model parameters')
    if not any(not torch.equal(old, new) for old, new in zip(before, model.parameters())):
        raise RuntimeError('Training did not change any parameters')
    summary = stats.last_epoch_summary()
    for stage in ('train', 'test'):
        for metric in ('loss', 'multi_class_accuracy'):
            if not math.isfinite(summary.stages[stage].metrics[metric]):
                raise RuntimeError(f'Non-finite {stage} {metric}')
    model.eval()
    inputs, _, _, _ = next(iter(manager.test_loader))
    with torch.no_grad():
        predictions = model(inputs.to(manager.device))
    if predictions.shape != (len(inputs), 10) or not torch.isfinite(predictions).all():
        raise RuntimeError('Invalid inference predictions')
    print(f'PASS: one smoothing epoch completed; train loss={summary.stages["train"].metrics["loss"]:.4f}, '
          f'clean test accuracy={summary.stages["test"].metrics["multi_class_accuracy"]:.4f}')
    print(f'Checkpoints and logs: {args.output_dir}')


if __name__ == '__main__':
    main()
