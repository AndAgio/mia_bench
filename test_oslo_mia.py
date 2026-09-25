"""Regression checks and optional one-epoch OSLO smoke runs.

Regression only: python3 -m unittest test_oslo_mia
One-epoch smoke runs: python3 test_oslo_mia.py --smoke --device cpu
"""

import argparse
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.mia.attacks.label_only.oslo import OsloMIA


class RecordingTarget(torch.nn.Module):
    def __init__(self, flip=False):
        super().__init__()
        self.flip = flip
        self.queries = []

    def forward(self, inputs):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.queries.append(inputs.detach().clone())
        return inputs.flip(dims=[1]) if self.flip else inputs


class OsloInferenceTests(unittest.TestCase):
    def setUp(self):
        self.attack = object.__new__(OsloMIA)
        self.attack.logger = Mock()
        self.attack.attack_configs = SimpleNamespace(threshold=0.01)
        self.inputs = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
        self.labels = torch.tensor([0, 0])
        self.adversarial = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
        self.attack._score = Mock(return_value=(self.adversarial, torch.tensor([0.0, 100.0])))

    def test_queries_adversarial_inputs_once_and_depends_on_target(self):
        for flip, expected in [(False, [1.0, 0.0]), (True, [0.0, 1.0])]:
            target = RecordingTarget(flip=flip)
            scores = self.attack.infer_batch(target, self.inputs, self.labels, device=torch.device('cpu'))
            self.assertEqual(scores.tolist(), expected)
            self.assertEqual(len(target.queries), 1)
            torch.testing.assert_close(target.queries[0], self.adversarial)

    def test_distance_does_not_affect_membership(self):
        for distances in [torch.tensor([0.0, 100.0]), torch.tensor([100.0, 0.0])]:
            self.attack._score.return_value = (self.adversarial, distances)
            scores = self.attack.infer_batch(RecordingTarget(), self.inputs, self.labels, device=torch.device('cpu'))
            self.assertEqual(scores.tolist(), [1.0, 0.0])

    def test_dataset_preserves_binary_decisions_even_at_zero_threshold(self):
        self.attack.attack_configs.threshold = 0.0
        self.attack._score.side_effect = lambda inputs, targets: (inputs.flip(dims=[1]), torch.zeros(len(inputs)))
        labels = torch.tensor([1, 0])
        dataset = TensorDataset(self.inputs, labels, torch.arange(2), torch.zeros(2))
        target = RecordingTarget()
        scores, decisions = self.attack.infer_dataset(target, dataset, device=torch.device('cpu'))
        np.testing.assert_array_equal(scores, [1.0, 0.0])
        np.testing.assert_array_equal(decisions, [1, 0])
        self.assertEqual(decisions.dtype, np.int64)
        self.assertEqual(len(target.queries), len(dataset))
        torch.testing.assert_close(torch.cat(target.queries), self.inputs.flip(dims=[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='Also train and evaluate OSLO using three tiny one-epoch profiles')
    parser.add_argument('--device', default='cpu', help='cpu or CUDA device index (e.g. 0)')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datasets-folder', type=Path, default=Path(__file__).resolve().parent / 'datas')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'smoke_test_results' / 'oslo_mia')
    parser.add_argument('--mode', choices=('difgsm', 'mifgsm', 'tifgsm', 'tmifgsm'), default='difgsm')
    parser.add_argument('--timeout', type=int, default=1800, help='Maximum seconds per smoke profile')
    parser.add_argument('--keep-work', action='store_true', help='Retain checkpoints and detailed result artifacts')
    parser.add_argument('--dry-run', action='store_true', help='Print smoke commands without training (implies --smoke)')
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OsloInferenceTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    if not (args.smoke or args.dry_run):
        return 0

    import smoke_test_all

    smoke_args = [
        '--preset', 'quick', '--device', args.device,
        '--dataset', args.dataset, '--datasets-folder', str(args.datasets_folder),
        '--output-dir', str(args.output_dir), '--timeout', str(args.timeout),
        '--only', f'attack:oslo_{args.mode}',
    ]
    if args.keep_work:
        smoke_args.append('--keep-work')
    if args.dry_run:
        smoke_args.append('--dry-run')
    print('OSLO smoke: three small dataset subsets; one epoch per defender and surrogate; '
          'two surrogates (one source, one validation); K=1, N=1. '
          'This checks execution, not attack quality.', flush=True)
    return smoke_test_all.main(smoke_args)


if __name__ == '__main__':
    raise SystemExit(main())
