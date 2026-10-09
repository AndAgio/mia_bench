#!/usr/bin/env python3
"""Smoke test of the mixup defense with every optimizer supported by TrainManager.

Example: python3 tests/test_mixup.py --device cpu
         python3 tests/test_mixup.py --optimizers sgd sam esam --alpha 0.4 --device mps

Uses the small synthetic image dataset of tests/test_relax_loss.py (nothing is downloaded), so it checks the
implementation, not the quality of the defense:
  1. reference: mixup_data follows the paper (Zhang et al., ICLR 2018): lambda ~ Beta(alpha, alpha), one lambda per
     batch, inputs mixed with a permutation of the batch, and no mixing with alpha=0;
  2. step: with every optimizer, one train_step hands the optimizer the gradient of
     lambda * CE(f(x_mix), y_a) + (1 - lambda) * CE(f(x_mix), y_b), updates the BatchNorm running stats with the mixed
     batch only (the clean pass for the train stats must not touch them), restores the BatchNorm momentum, computes
     the train stats on the clean inputs, and leaves finite, changed parameters;
  3. train: a full TrainManager.train() run (val/test, checkpoints, best model) stays finite;
  4. resume: training stopped halfway and resumed restarts from the right epoch and draws the same lambdas as an
     uninterrupted run (the numpy random state is in the checkpoints).
Exits with status 1 if any check fails.
"""

import argparse
import contextlib
import copy
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.mia.defenses.training.mixup as mixup_module
from src.mia.defenses.training.mixup import MixupTrainManager, mixup_data
from src.trainer.stats_tracker import EpochStats
from src.utils.configs import MixupDefenseConfigs, OptimizerConfigs, SchedulerConfigs, TrainConfigs
from src.utils.log import get_logger
from tests.test_relax_loss import OPTIMIZERS, QuietLogger, build_data, build_model, params_finite, pretrain


def make_manager(args, model, data, optimizer, epochs, folder, resume=False) -> MixupTrainManager:
    configs = TrainConfigs(optimizer_config=OptimizerConfigs(name=optimizer, lr=args.adam_lr if optimizer == 'adam' else args.lr),
                           scheduler_config=SchedulerConfigs(name='const', epochs=epochs),
                           batch_size=args.batch_size, device=args.device, seed=args.seed, resume=resume,
                           ckpts_folder=folder / 'ckpts', resume_ckpts_folder=folder / 'resume_ckpts')
    manager = MixupTrainManager(train_configs=configs, name='mixup_smoke', logger=args.logger,
                                mixup_configs=MixupDefenseConfigs(alpha=args.alpha))
    manager.initialize_train(dataset=data, model=model, configs=configs)
    return manager


@contextlib.contextmanager
def record_mixup(manager: MixupTrainManager):
    # Replaces mixup_data in the mixup module (train_step looks it up there) to log (epoch, mixed inputs, y_a, y_b,
    # lambda) for every batch.
    calls = []

    def wrapped(x, y, alpha=1.0):
        out = mixup_data(x, y, alpha=alpha)
        calls.append((getattr(manager, 'epoch', None), *out))
        return out

    mixup_module.mixup_data = wrapped
    try:
        yield calls
    finally:
        mixup_module.mixup_data = mixup_data


def mixup_loss(outputs, targets_a, targets_b, lmbd) -> torch.Tensor:
    return lmbd * F.cross_entropy(outputs, targets_a) + (1 - lmbd) * F.cross_entropy(outputs, targets_b)


def bn_layers(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, _BatchNorm)]


def capture_gradient(manager: MixupTrainManager) -> dict:
    # Wraps optimizer.step to store the gradient that train_step hands to the optimizer. SAM-like optimizers receive a
    # closure instead, which is called here as they do: per-sample losses (as ESAM does) and a mean-loss backward. These
    # extra calls run in train mode, so the buffers (BatchNorm running stats) are restored afterwards.
    captured = {}
    step = manager.optimizer.step

    def flat_grad():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in manager.model.parameters()])

    def wrapped(*args, **kwargs):
        if args:  # step(closure, mixed_inputs, targets_a, targets_b)
            closure, *closure_args = args
            buffers = [b.clone() for b in manager.model.buffers()]
            manager.optimizer.zero_grad()
            with torch.enable_grad():
                per_sample_loss, _ = closure(*closure_args, mean=False, backward=False, run_stats=True)
                closure(*closure_args, mean=True, backward=True, run_stats=True)
            captured['per_sample_shape'] = tuple(per_sample_loss.shape)
            captured['grad'] = flat_grad()
            manager.optimizer.zero_grad()
            with torch.no_grad():
                for b, saved in zip(manager.model.buffers(), buffers):
                    b.copy_(saved)
        else:
            captured['grad'] = flat_grad()
        return step(*args, **kwargs)

    manager.optimizer.step = wrapped
    return captured


def capture_stats(manager: MixupTrainManager) -> dict:
    # Wraps epoch_stats_tracker.update to store the predictions and targets the train stats are computed on.
    captured = {}
    update = manager.epoch_stats_tracker.update

    def wrapped(preds, targets, **kwargs):
        captured['preds'], captured['targets'] = preds.detach().clone(), targets.detach().clone()
        return update(preds=preds, targets=targets, **kwargs)

    manager.epoch_stats_tracker.update = wrapped
    return captured


def check_reference(args) -> list:
    errors = []
    # Mixing: lambda * x + (1 - lambda) * x[perm], with y_b = y[perm] and the same perm (targets 0..n-1 reveal it).
    n = 64
    x, y = torch.randn(n, 3, 8, 8), torch.arange(n)
    mixed, y_a, y_b, lmbd = mixup_data(x, y, alpha=args.alpha)
    if not (np.isscalar(lmbd) and 0.0 <= lmbd <= 1.0):
        errors.append(f"reference: lambda should be one scalar in [0, 1] per batch, got {lmbd}")
    if not torch.equal(y_a, y):
        errors.append("reference: y_a should be the original targets")
    if not torch.equal(torch.sort(y_b).values, y):
        errors.append("reference: y_b is not a permutation of the batch targets")
    elif not torch.allclose(mixed, lmbd * x + (1 - lmbd) * x[y_b], atol=1e-6):
        errors.append("reference: mixed inputs differ from lambda * x + (1 - lambda) * x[perm] with the permutation of y_b")
    # alpha = 0: no mixing (lambda = 1).
    mixed, _, _, lmbd = mixup_data(x, y, alpha=0.0)
    if lmbd != 1.0 or not torch.equal(mixed, x):
        errors.append(f"reference: alpha=0 should leave the inputs unmixed (lambda=1), got lambda={lmbd}")
    # lambda ~ Beta(alpha, alpha): mean 1/2 and variance 1 / (4 (2 alpha + 1)).
    np.random.seed(args.seed)
    small_x, small_y = torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)
    for alpha in (0.2, 1.0, 2.0):
        lmbds = np.array([mixup_data(small_x, small_y, alpha=alpha)[3] for _ in range(20000)])
        var = 1.0 / (4.0 * (2.0 * alpha + 1.0))
        if abs(lmbds.mean() - 0.5) > 0.01 or abs(lmbds.var() - var) / var > 0.05:
            errors.append(f"reference: alpha={alpha}: lambda mean {lmbds.mean():.4f} / var {lmbds.var():.4f}, "
                          f"Beta(alpha, alpha) has 0.5 / {var:.4f}")
    return errors


def check_step(args, model: nn.Module, data, optimizer: str, folder: Path) -> list:
    errors = []
    train = data.get('train')
    try:
        manager = make_manager(args, model, data, optimizer, epochs=2, folder=folder)
        # State that TrainManager.train() sets up before calling train_step.
        manager.extra_configs = None
        manager.epoch = 1
        manager.reset_running_stats()
        manager.train_stats_tracker.start_timer()
        manager.epoch_stats_tracker = EpochStats()
        manager.epoch_stats_tracker.epoch_start()
        manager.reset_epoch_stats(phase='train')
        manager.model.train()

        inputs, targets = train.inputs[:args.batch_size], train.targets[:args.batch_size]
        model_before = copy.deepcopy(manager.model)
        params_before = [p.detach().clone() for p in manager.model.parameters()]
        momenta = [m.momentum for m in bn_layers(manager.model)]
        captured_grad = capture_gradient(manager)
        captured_stats = capture_stats(manager)
        with record_mixup(manager) as calls:
            manager.train_step(inputs, targets, batch_idx=0, total_batches=1)
        _, mixed, targets_a, targets_b, lmbd = calls[0]
        mixed, targets_a, targets_b = mixed.to(manager.device), targets_a.to(manager.device), targets_b.to(manager.device)

        # Expected: one train-mode forward on the mixed batch at the weights before the step, which gives both the
        # gradient of the mixup loss and the BatchNorm running stats after the step.
        model_before.train().zero_grad()
        mixup_loss(model_before(mixed), targets_a, targets_b, lmbd).backward()
        expected_grad = torch.cat([p.grad.flatten() for p in model_before.parameters()])

        if len(calls) != 1:
            errors.append(f"step: mixup_data called {len(calls)} times, expected once per batch")
        if captured_grad.get('per_sample_shape', (len(targets),)) != (len(targets),):
            errors.append(f"step: closure with mean=False returned shape {captured_grad['per_sample_shape']}, expected one loss per sample")
        if 'grad' not in captured_grad:
            errors.append("step: optimizer.step was not called")
        elif not torch.allclose(captured_grad['grad'], expected_grad, rtol=1e-4, atol=1e-5):
            errors.append(f"step: gradient handed to the optimizer differs from the one of the mixup loss "
                          f"(max diff {(captured_grad['grad'] - expected_grad).abs().max():.2e})")
        for i, (bn, bn_ref) in enumerate(zip(bn_layers(manager.model), bn_layers(model_before))):
            if not (torch.allclose(bn.running_mean, bn_ref.running_mean, rtol=1e-4, atol=1e-6)
                    and torch.allclose(bn.running_var, bn_ref.running_var, rtol=1e-4, atol=1e-6)):
                errors.append(f"step: BatchNorm {i} running stats differ from one update with the mixed batch "
                              f"(the clean stats pass or a perturbed pass updated them)")
                break
        if [m.momentum for m in bn_layers(manager.model)] != momenta or any(hasattr(m, 'backup_momentum') for m in bn_layers(manager.model)):
            errors.append("step: BatchNorm momentum not restored after the step")
        if 'preds' not in captured_stats:
            errors.append("step: train stats not updated")
        else:
            with torch.no_grad():
                clean_outputs = copy.deepcopy(manager.model).train()(inputs.to(manager.device))
            if not torch.equal(captured_stats['targets'].cpu(), targets):
                errors.append("step: train stats computed against other targets than the clean ones")
            elif not torch.allclose(captured_stats['preds'], clean_outputs, rtol=1e-4, atol=1e-5):
                errors.append("step: train stats not computed on the clean inputs with the updated model")
        if not params_finite(manager.model):
            errors.append("step: non-finite parameters after the step")
        elif all(torch.equal(p, q) for p, q in zip(params_before, manager.model.parameters())):
            errors.append("step: parameters did not change")
    except Exception as exc:
        errors.append(f"step: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    return errors


def epoch_metric(history: dict, stage: str, names=('multi_class_accuracy', 'accuracy')) -> list:
    values = []
    for epoch in sorted(history):
        metrics = history[epoch].stages[stage].metrics if stage in history[epoch].stages else {}
        values.append(next((metrics[name] for name in names if name in metrics), float('nan')))
    return values


def check_train(args, data, optimizer: str, folder: Path):
    errors, info = [], {}
    try:
        manager = make_manager(args, build_model(args), data, optimizer, args.epochs, folder)
        with record_mixup(manager) as calls:
            best_model = manager.train(return_best_model=True)
        history = manager.train_stats_tracker.history
        train_losses = epoch_metric(history, 'train', names=('loss',))
        info = {'train_acc': epoch_metric(history, 'train'), 'val_acc': epoch_metric(history, 'val'),
                'lambdas': [(epoch, lmbd) for epoch, *_, lmbd in calls]}
        if sorted(history) != list(range(1, args.epochs + 1)):
            errors.append(f"train: epochs in history are {sorted(history)}")
        if not all(np.isfinite(train_losses)):
            errors.append(f"train: non-finite training losses {train_losses}")
        if not (params_finite(best_model) and params_finite(manager.model)):
            errors.append("train: non-finite parameters in the best or last model")
        if any(hasattr(m, 'backup_momentum') for m in bn_layers(manager.model)):
            errors.append("train: BatchNorm momentum left disabled after training")
        for name in ('best.pth', 'last.pth'):
            if not (Path(manager.resume_folder) / name).exists():
                errors.append(f"train: checkpoint {name} not saved")
    except Exception as exc:
        errors.append(f"train: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    return errors, info


def check_resume(args, data, optimizer: str, folder: Path, full_lambdas: list) -> list:
    errors = []
    half = max(args.epochs // 2, 1)
    try:
        make_manager(args, build_model(args), data, optimizer, half, folder, resume=True).train()
        manager = make_manager(args, build_model(args), data, optimizer, args.epochs, folder, resume=True)
        with record_mixup(manager) as calls:
            manager.train()
        epochs_run = sorted({epoch for epoch, *_ in calls})
        if epochs_run != list(range(half + 1, args.epochs + 1)):
            errors.append(f"resume: expected to run epochs {half + 1}..{args.epochs}, ran {epochs_run}")
        if sorted(manager.train_stats_tracker.history) != list(range(1, args.epochs + 1)):
            errors.append(f"resume: epochs in history are {sorted(manager.train_stats_tracker.history)}")
        resumed = [(epoch, lmbd) for epoch, *_, lmbd in calls]
        expected = [(epoch, lmbd) for epoch, lmbd in full_lambdas if epoch > half]
        if full_lambdas and resumed != expected:
            errors.append("resume: the resumed run draws other lambdas than an uninterrupted one (numpy random state not restored?)")
    except Exception as exc:
        errors.append(f"resume: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cpu', help='cpu or CUDA index, e.g. 0 (MPS is used on Apple silicon)')
    parser.add_argument('--optimizers', nargs='+', default=list(OPTIMIZERS), choices=OPTIMIZERS)
    parser.add_argument('--model', default='tiny', help="'tiny' (default) or a model name of the repository, e.g. resnet18")
    parser.add_argument('--im-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=6, help='epochs of the train and resume checks')
    parser.add_argument('--alpha', type=float, default=1.0, help='mixup alpha (1.0 is the CIFAR value of the paper)')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--adam-lr', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--verbose', action='store_true', help='show the training logs of TrainManager')
    args = parser.parse_args()

    failures, n_ok = [], 0
    with tempfile.TemporaryDirectory(prefix='mixup_smoke_') as tmp:
        tmp = Path(tmp)
        args.logger = get_logger('mixup_smoke', str(tmp), mode='dumb') if args.verbose else QuietLogger()
        data = build_data(args)
        pretrained = pretrain(args, data)

        errors = check_reference(args)
        failures += errors
        print(f"[reference] mixup_data vs paper (Beta(alpha, alpha), batch permutation): {'OK' if not errors else 'FAIL'}", flush=True)

        for optimizer in args.optimizers:
            start = time.time()
            folder = tmp / optimizer
            step_errors = check_step(args, pretrained, data, optimizer, folder / 'step')
            train_errors, info = check_train(args, data, optimizer, folder / 'train')
            resume_errors = check_resume(args, data, optimizer, folder / 'resume', info.get('lambdas', []))
            failures += [f"{optimizer} {e}" for e in step_errors + train_errors + resume_errors]
            n_ok += not (step_errors or train_errors or resume_errors)
            status = lambda errors: 'FAIL' if errors else 'OK  '
            fmt = lambda values: ' '.join(f'{v:.2f}' for v in values) or '-'
            print(f"[{optimizer:>20s}] step {status(step_errors)} train {status(train_errors)} resume {status(resume_errors)} | "
                  f"train acc (clean): {fmt(info.get('train_acc', []))} | val acc: {fmt(info.get('val_acc', []))} | "
                  f"{time.time() - start:.1f}s", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILED CHECK(S):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"\nAll checks passed for {n_ok} optimizers.")


if __name__ == '__main__':
    main()
