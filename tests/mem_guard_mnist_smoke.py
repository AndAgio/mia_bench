#!/usr/bin/env python3
"""Run the repository's MemGuard on MNIST with a small CNN.

Example (one epoch for both the CNN and membership classifier):
    python3 tests/mem_guard_mnist_smoke.py --device mps --epochs 1 --membership-epochs 1

Downloads MNIST when needed. Uses 10,000 training and 2,000 validation
examples by default; audit non-members come only from the MNIST test set.
This checks execution and invariants, not MemGuard's privacy effectiveness.
"""
import argparse
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mia.defenses.post_hoc.mem_guard import MemGuardDefender
from src.utils.configs import MemGuardDefenseConfigs


class Logger:
    def print_it(self, message, **kwargs):
        print(message, flush=True)

    def print_it_same_line(self, message, **kwargs):
        print('\r' + message, end='', flush=True)

    def set_logger_newline(self, **kwargs):
        print(flush=True)


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, data, device, batch_size):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=batch_size, num_workers=0):
            predictions = model(x.to(device)).argmax(1).cpu()
            correct += (predictions == y).sum().item()
            total += len(y)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--membership-epochs', type=int, default=1)
    parser.add_argument('--train-samples', type=int, default=10000)
    parser.add_argument('--validation-samples', type=int, default=2000)
    parser.add_argument('--audit-samples', type=int, default=32,
                        help='Even total: half members, half test-set non-members')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--defense-batch-size', type=int, default=16)
    parser.add_argument('--budget', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datas' / 'mnist')
    args = parser.parse_args()
    if min(args.epochs, args.membership_epochs, args.train_samples,
           args.validation_samples, args.batch_size, args.defense_batch_size) < 1:
        parser.error('Epochs, sample counts and batch sizes must be positive.')
    if args.audit_samples < 2 or args.audit_samples % 2:
        parser.error('--audit-samples must be an even number >= 2.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable. Use an Apple Silicon Mac with an MPS-enabled PyTorch installation.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print(f'Device: {device}; PyTorch: {torch.__version__}', flush=True)
    print(f'Training epochs: CNN={args.epochs}, membership={args.membership_epochs}', flush=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    source = datasets.MNIST(str(args.data_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(args.data_dir), train=False, download=True, transform=transform)
    if args.train_samples + args.validation_samples > len(source):
        parser.error('Train and validation sample counts must sum to at most 60,000.')
    half = args.audit_samples // 2
    if half > min(args.train_samples, len(test)):
        parser.error('Audit members/non-members exceed the available samples.')
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(source), generator=generator).tolist()
    train = Subset(source, order[:args.train_samples])
    validation = Subset(source, order[args.train_samples:args.train_samples + args.validation_samples])
    test_indices = torch.randperm(len(test), generator=generator)[:half].tolist()
    print(f'Splits: train={len(train)}, validation={len(validation)}, audit={args.audit_samples}', flush=True)

    victim = MNISTCNN().to(device)
    optimizer = torch.optim.Adam(victim.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    for epoch in range(args.epochs):
        victim.train()
        loss_sum = correct = count = 0
        start = time.perf_counter()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = victim(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        print(f'CNN epoch {epoch + 1}: loss={loss_sum/count:.4f}, '
              f'train accuracy={correct/count:.2%}, seconds={time.perf_counter()-start:.1f}', flush=True)
    print(f'Validation accuracy: {accuracy(victim, validation, device, args.batch_size):.2%}', flush=True)

    # MNIST is not a dataset option in the main CLI. Supply this harness's
    # already-trained CNN and splits to the actual MemGuardDefender methods.
    defender = object.__new__(MemGuardDefender)
    defender.logger = Logger()
    defender.model_configs = SimpleNamespace(num_classes=10)
    defender.mem_guard_configs = MemGuardDefenseConfigs(
        budget=args.budget, shadow_attacker_model_epochs=args.membership_epochs)
    defender.dataset = {'train': train, 'val': validation, 'test': test}
    defender.trained_model = victim.eval()
    model = defender.defend_model(device=device)
    assert next(model.victim.parameters()).device.type == device.type
    assert next(model.M.parameters()).device.type == device.type
    # Discard stale training gradients, then check that inference creates none.
    for parameter in model.parameters():
        parameter.grad = None

    audit = [train[i] for i in range(half)] + [test[i] for i in test_indices]
    audit_x = torch.stack([x for x, _ in audit]).to(device)
    audit_y = torch.tensor([y for _, y in audit], device=device)
    total_changed = total_candidates = 0
    maximum_expected_l1 = realized_l1_sum = 0.
    start = time.perf_counter()
    print('Running MemGuard checks (300 inner iterations, published c3 search)...', flush=True)
    # MemGuard needs input gradients internally: use no_grad, not inference_mode.
    with torch.no_grad():
        for batch_number, x in enumerate(audit_x.split(args.defense_batch_size), 1):
            logits = victim(x)
            clean = logits.softmax(1)
            candidate, l1 = model._optimize_confidence(logits)
            features0 = clean.sort(1).values
            features1 = candidate.sort(1).values
            g0 = model.M(features0)
            g1 = model.M(features1)
            improves = (g1 - .5).abs() < (g0 - .5).abs()
            probability = torch.where((l1 > 1e-12) & improves,
                                      (args.budget / l1.clamp_min(1e-12)).clamp(max=1),
                                      torch.zeros_like(l1))
            expected_l1 = probability * l1
            assert (expected_l1 <= args.budget + 1e-6).all(), 'Expected L1 budget exceeded'
            draws = model._one_time_uniform(x, probability)
            expected_output = torch.where((draws < probability)[:, None], candidate, clean)

            defended = model(x).softmax(1)
            assert torch.isfinite(defended).all(), 'Non-finite defended output'
            assert (defended >= 0).all(), 'Negative probability'
            torch.testing.assert_close(defended.sum(1), torch.ones(len(x), device=device))
            assert torch.equal(defended.argmax(1), clean.argmax(1)), 'Predicted label changed'
            torch.testing.assert_close(defended, expected_output, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(model(x).softmax(1), defended, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(model(x.flip(0)).softmax(1).flip(0), defended,
                                       atol=1e-5, rtol=1e-5)
            # Also test a query by itself to expose dependence on batch composition.
            torch.testing.assert_close(model(x[:1]).softmax(1), defended[:1], atol=1e-5, rtol=1e-5)
            realized_l1 = (defended - clean).abs().sum(1)
            total_changed += (realized_l1 > 1e-5).sum().item()
            total_candidates += (l1 > 1e-5).sum().item()
            maximum_expected_l1 = max(maximum_expected_l1, expected_l1.max().item())
            realized_l1_sum += realized_l1.sum().item()
            print(f'  Batch {batch_number}: all checks passed', flush=True)
        assert all(parameter.grad is None for parameter in model.parameters()), 'Model gradients accumulated'
        clean_accuracy = (victim(audit_x).argmax(1) == audit_y).float().mean().item()
    print(f'PASS: {args.audit_samples} queries; label preservation, finite probabilities, '
          'repeatability, batching and expected budget verified.', flush=True)
    print(f'Audit accuracy (clean and defended): {clean_accuracy:.2%}', flush=True)
    print(f'Nonzero candidates: {total_candidates}/{args.audit_samples}; '
          f'applied perturbations: {total_changed}/{args.audit_samples}', flush=True)
    print(f'Maximum expected L1: {maximum_expected_l1:.6f}; '
          f'mean realized L1: {realized_l1_sum/args.audit_samples:.6f}; '
          f'defense-check seconds: {time.perf_counter()-start:.1f}', flush=True)
    print('One-epoch training is a smoke test; zero applied perturbations is not a test failure.', flush=True)


if __name__ == '__main__':
    main()
