#!/usr/bin/env python3
"""Smoke test of the RelaxLoss defense with every optimizer supported by TrainManager.

Example: python3 tests/test_relax_loss.py --device cpu
         python3 tests/test_relax_loss.py --optimizers sgd sam esam --variants incorrect --device 0

Uses a small synthetic image dataset (nothing is downloaded), so it checks the implementation, not the
quality of the defense:
  1. reference: relax_loss matches Algorithm 1 and App. B.3 of the paper (with gradient ascent on the correct
     predictions while flattening the incorrect ones on image data, as in the official code), values and gradients;
  2. steps: with every optimizer, one train_step per branch (gradient descent, gradient ascent, posterior flattening,
     and flattening of a batch without incorrect predictions) takes the expected branch, hands the optimizer the
     gradient of the paper's objective, stays finite and decreases the objective of that branch;
  3. train: a full TrainManager.train() run (val/test, checkpoints, best model) stays finite;
  4. resume: training stopped halfway and resumed restarts from the right epoch, keeping the phase parity.
Exits with status 1 if any check fails.
"""

import argparse
import copy
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.helpers import MultiDatasets
from src.mia.defenses.training.relax_loss import RelaxLossTrainManager
from src.models import get_model
from src.trainer.stats_tracker import EpochStats
from src.utils.configs import OptimizerConfigs, SchedulerConfigs, TrainConfigs
from src.utils.log import get_logger


# All the names accepted by TrainManager.setup_optimizer.
OPTIMIZERS = ('sgd', 'adam', 'sam', 'adaptive_sam', 'esam', 'adaptive_esam', 'wsam', 'adaptive_wsam',
              'looksam', 'adaptive_looksam', 'friendlysam', 'adaptive_friendlysam')
# Variants of the paper (App. B.3): flattening of incorrect predictions only for image data, and flattening of
# all samples with a clamped ground-truth score for non-image data.
VARIANTS = {'incorrect': dict(flatten_incorrect_only=True, relax_upper=1.0),
            'all': dict(flatten_incorrect_only=False, relax_upper=0.3)}
# Step case -> (branch, epoch forcing its phase, alpha forcing the batch loss above or below it). Epochs count from 1 as
# in Algorithm 1: gradient ascent on even epochs and posterior flattening on odd ones, both only below alpha.
STEP_CASES = {'descent': ('descent', 1, 1e-3), 'ascent': ('ascent', 2, 1e3), 'flatten': ('flatten', 1, 1e3),
              # Labels set to the model predictions: with flatten_incorrect_only the whole batch gets gradient ascent.
              'flatten_all_correct': ('flatten', 1, 1e3)}
NUM_CLASSES = 10


class SyntheticImages(Dataset):
    # Noisy class prototypes (a colour plus a smooth pattern, learnable in a few epochs), with a fraction of flipped
    # labels so that some training samples stay misclassified.
    def __init__(self, n: int, im_size: int, seed: int, label_noise: float = 0.1):
        g = torch.Generator().manual_seed(0)  # same prototypes for all splits
        prototypes = 2.0 * torch.randn(NUM_CLASSES, 3, 1, 1, generator=g) \
            + F.interpolate(torch.randn(NUM_CLASSES, 3, 4, 4, generator=g), size=im_size, mode='bilinear', align_corners=False)
        g = torch.Generator().manual_seed(seed)
        self.targets = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        self.inputs = prototypes[self.targets] + torch.randn(n, 3, im_size, im_size, generator=g)
        flip = torch.rand(n, generator=g) < label_noise
        self.targets[flip] = torch.randint(0, NUM_CLASSES, (int(flip.sum()),), generator=g)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Same (inputs, targets, original_indices, resampled_indices) format as the datasets of the repository.
        return self.inputs[idx], self.targets[idx], idx, idx


class TinyCNN(nn.Module):
    # The BatchNorm layers exercise enable/disable_running_stats in the closures of the SAM-like optimizers.
    def __init__(self):
        super().__init__()
        self.name = 'tiny_cnn'
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


class QuietLogger:
    # Swallows the per-batch progress messages of TrainManager (use --verbose to see them).
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def reference_relax_loss(outputs, targets, epoch, alpha, upper, incorrect_only, ref_outputs=None):
    # Objective of one step of Algorithm 1 of the paper (epochs counted from 1), with the techniques of its App. B.3:
    # flattening restricted to incorrect predictions (incorrect_only), with gradient ascent on the correct ones as in the
    # official code, and ground-truth score clamped to upper. Branch, soft labels and correctness mask are computed on
    # ref_outputs (default: outputs) and are constants, so that the objective of a step can also be evaluated after the
    # update with them frozen.
    ref = (outputs if ref_outputs is None else ref_outputs).detach()
    loss_ce = F.cross_entropy(outputs, targets)
    if F.cross_entropy(ref, targets).item() >= alpha:
        return loss_ce  # gradient descent
    if epoch % 2 == 0:
        return -loss_ce  # gradient ascent
    # Posterior flattening, with the soft labels of Sec. 4.2: (clamped) ground-truth score, rest spread evenly.
    num_classes = outputs.size(1)
    p_gt = F.softmax(ref, dim=1)[torch.arange(targets.size(0)), targets].clamp(max=upper)
    soft_targets = torch.where(F.one_hot(targets, num_classes).bool(), p_gt.unsqueeze(1), ((1 - p_gt) / (num_classes - 1)).unsqueeze(1))
    soft_ce = -(soft_targets * F.log_softmax(outputs, dim=1)).sum(dim=1)
    if incorrect_only:
        incorrect = (ref.argmax(dim=1) != targets).to(outputs.dtype)
        soft_ce = incorrect * soft_ce - (1 - incorrect) * F.cross_entropy(outputs, targets, reduction='none')
    return soft_ce.mean()


def build_model(args) -> nn.Module:
    torch.manual_seed(args.seed)
    if args.model == 'tiny':
        return TinyCNN()
    return get_model(model_name=args.model, im_channels=3, num_classes=NUM_CLASSES,
                     im_size=(args.im_size, args.im_size), logger=QuietLogger())


def build_data(args) -> MultiDatasets:
    return MultiDatasets(datasets=[SyntheticImages(512, args.im_size, seed=1), SyntheticImages(128, args.im_size, seed=2),
                                   SyntheticImages(128, args.im_size, seed=3)],
                         ids=['train', 'val', 'test'])


def pretrain(args, data: MultiDatasets) -> nn.Module:
    # A briefly trained model has peaked, partly wrong predictions, so that posterior flattening has something to do.
    model = build_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loader = torch.utils.data.DataLoader(data.get('train'), batch_size=args.batch_size, shuffle=True)
    model.train()
    for _ in range(3):
        for inputs, targets, _, _ in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(inputs), targets).backward()
            optimizer.step()
    return model


def make_manager(args, model, data, optimizer, variant, alpha, epochs, folder, resume=False) -> RelaxLossTrainManager:
    configs = TrainConfigs(optimizer_config=OptimizerConfigs(name=optimizer, lr=args.adam_lr if optimizer == 'adam' else args.lr),
                           scheduler_config=SchedulerConfigs(name='const', epochs=epochs),
                           batch_size=args.batch_size, device=args.device, seed=args.seed, resume=resume,
                           ckpts_folder=folder / 'ckpts', resume_ckpts_folder=folder / 'resume_ckpts')
    manager = RelaxLossTrainManager(train_configs=configs, name='relax_loss_smoke', logger=args.logger,
                                    relax_alpha=alpha, **VARIANTS[variant])
    manager.initialize_train(dataset=data, model=model, configs=configs)
    return manager


def record_branches(manager: RelaxLossTrainManager) -> list:
    # Wraps relax_loss to log (epoch, branch it should take, whether its output is finite) for every call.
    calls = []
    relax_loss = manager.relax_loss

    def wrapped(outputs, targets):
        out = relax_loss(outputs=outputs, targets=targets)
        if F.cross_entropy(outputs.detach(), targets).item() >= manager.relax_alpha:
            branch = 'descent'
        else:
            branch = 'ascent' if manager.epoch % 2 == 0 else 'flatten'
        calls.append((manager.epoch, branch, bool(torch.isfinite(out).all())))
        return out

    manager.relax_loss = wrapped
    return calls


def capture_gradient(manager: RelaxLossTrainManager) -> dict:
    # Wraps optimizer.step to store the gradient that train_step hands to the optimizer. SAM-like optimizers receive a
    # closure instead, which is called here as they do: per-sample losses (as ESAM does) and a mean-loss backward.
    captured = {}
    step = manager.optimizer.step

    def flat_grad():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in manager.model.parameters()])

    def wrapped(*args, **kwargs):
        if args:  # step(closure, inputs, targets)
            closure, inputs, targets = args
            manager.optimizer.zero_grad()
            with torch.enable_grad():
                per_sample_loss, _ = closure(inputs, targets, mean=False, backward=False, run_stats=True)
                closure(inputs, targets, mean=True, backward=True, run_stats=True)
            captured['per_sample_shape'] = tuple(per_sample_loss.shape)
            captured['grad'] = flat_grad()
            manager.optimizer.zero_grad()
        else:
            captured['grad'] = flat_grad()
        return step(*args, **kwargs)

    manager.optimizer.step = wrapped
    return captured


def reference_gradient(model: nn.Module, inputs, targets, epoch, alpha, upper, incorrect_only) -> torch.Tensor:
    model = copy.deepcopy(model).train()
    model.zero_grad()
    reference_relax_loss(model(inputs), targets, epoch, alpha, upper, incorrect_only).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def train_mode_outputs(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    # Outputs with batch statistics, as seen by train_step, without touching the running stats of the model.
    model = copy.deepcopy(model).train()
    with torch.no_grad():
        return model(inputs)


def params_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def check_reference(args, model: nn.Module, data: MultiDatasets) -> list:
    errors = []
    inputs, targets = data.get('train').inputs[:args.batch_size], data.get('train').targets[:args.batch_size]
    model = copy.deepcopy(model).train()
    batch_loss = F.cross_entropy(model(inputs), targets).item()
    for variant, kwargs in VARIANTS.items():
        for epoch in range(1, 5):
            for alpha in (0.5 * batch_loss, 2.0 * batch_loss):
                manager = RelaxLossTrainManager.__new__(RelaxLossTrainManager)
                manager.relax_alpha, manager.relax_upper = alpha, kwargs['relax_upper']
                manager.flatten_incorrect_only, manager.epoch = kwargs['flatten_incorrect_only'], epoch
                values, grads = [], []
                for loss_fn in (lambda out: manager.relax_loss(outputs=out, targets=targets).mean(),
                                lambda out: reference_relax_loss(out, targets, epoch, alpha, kwargs['relax_upper'],
                                                                 kwargs['flatten_incorrect_only'])):
                    model.zero_grad()
                    loss = loss_fn(model(inputs))
                    loss.backward()
                    values.append(loss.detach())
                    grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
                if not (torch.allclose(values[0], values[1], atol=1e-6) and torch.allclose(grads[0], grads[1], atol=1e-6)):
                    errors.append(f"reference: variant={variant} epoch={epoch} alpha={alpha:.3f}: value or gradient "
                                  f"differs from the paper (max |dgrad|={(grads[0] - grads[1]).abs().max():.2e})")
    return errors


def check_steps(args, model: nn.Module, data: MultiDatasets, optimizer: str, variant: str, folder: Path) -> list:
    errors = []
    train = data.get('train')
    for case, (branch, epoch, alpha) in STEP_CASES.items():
        where = f"steps[{case}]"
        try:
            manager = make_manager(args, model, data, optimizer, variant, alpha, epochs=2, folder=folder / case)
            # State that TrainManager.train() sets up before calling train_step.
            manager.extra_configs = None
            manager.epoch = epoch
            manager.reset_running_stats()
            manager.train_stats_tracker.start_timer()
            manager.epoch_stats_tracker = EpochStats()
            manager.epoch_stats_tracker.epoch_start()
            manager.reset_epoch_stats(phase='train')
            manager.model.train()

            inputs = train.inputs[:args.batch_size].to(manager.device)
            targets = train.targets[:args.batch_size].to(manager.device)
            params_before = [p.detach().clone() for p in manager.model.parameters()]
            outputs_before = train_mode_outputs(manager.model, inputs)
            if case == 'flatten_all_correct':
                targets = outputs_before.argmax(dim=1)
            upper, incorrect_only = VARIANTS[variant]['relax_upper'], VARIANTS[variant]['flatten_incorrect_only']
            objective_before = reference_relax_loss(outputs_before, targets, epoch, alpha, upper, incorrect_only).item()
            expected_grad = reference_gradient(manager.model, inputs, targets, epoch, alpha, upper, incorrect_only)

            calls = record_branches(manager)
            captured = capture_gradient(manager)
            manager.train_step(inputs, targets, batch_idx=0, total_batches=1)

            objective_after = reference_relax_loss(train_mode_outputs(manager.model, inputs), targets, epoch, alpha,
                                                   upper, incorrect_only, ref_outputs=outputs_before).item()
            taken = sorted({call[1] for call in calls})
            if taken != [branch]:
                errors.append(f"{where}: expected branch '{branch}', relax_loss took {taken}")
            if not all(call[2] for call in calls):
                errors.append(f"{where}: relax_loss returned non-finite values")
            if captured.get('per_sample_shape', (len(targets),)) != (len(targets),):
                errors.append(f"{where}: closure with mean=False returned shape {captured['per_sample_shape']}, expected one loss per sample")
            if 'grad' not in captured:
                errors.append(f"{where}: optimizer.step was not called")
            elif not torch.allclose(captured['grad'], expected_grad, rtol=1e-4, atol=1e-5):
                errors.append(f"{where}: gradient handed to the optimizer differs from the one of the paper's objective "
                              f"(max diff {(captured['grad'] - expected_grad).abs().max():.2e})")
            if not params_finite(manager.model):
                errors.append(f"{where}: non-finite parameters after the step")
            elif all(torch.equal(p, q) for p, q in zip(params_before, manager.model.parameters())):
                errors.append(f"{where}: parameters did not change")
            elif not objective_after < objective_before and not optimizer.endswith('wsam'):
                # WSAM descends L + 9 * sharpness (gamma=0.9), so one step need not decrease L itself, e.g. when ascending.
                errors.append(f"{where}: objective of the branch did not decrease ({objective_before:.5f} -> {objective_after:.5f})")
        except Exception as exc:
            errors.append(f"{where}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    return errors


def check_train(args, data: MultiDatasets, optimizer: str, variant: str, folder: Path):
    errors, info = [], {}
    try:
        manager = make_manager(args, build_model(args), data, optimizer, variant, args.alpha, args.epochs, folder)
        calls = record_branches(manager)
        best_model = manager.train(return_best_model=True)
        history = manager.train_stats_tracker.history
        train_losses = [history[e].stages['train'].metrics['loss'] for e in sorted(history)]
        info = {'losses': train_losses, 'branches': sorted({call[1] for call in calls})}
        if sorted(history) != list(range(1, args.epochs + 1)):
            errors.append(f"train: epochs in history are {sorted(history)}")
        if not all(call[2] for call in calls):
            errors.append("train: relax_loss returned non-finite values")
        if not all(torch.isfinite(torch.tensor(train_losses))):
            errors.append(f"train: non-finite training losses {train_losses}")
        if not (params_finite(best_model) and params_finite(manager.model)):
            errors.append("train: non-finite parameters in the best or last model")
        for name in ('best.pth', 'last.pth'):
            if not (Path(manager.resume_folder) / name).exists():
                errors.append(f"train: checkpoint {name} not saved")
    except Exception as exc:
        errors.append(f"train: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    return errors, info


def check_resume(args, data: MultiDatasets, optimizer: str, variant: str, folder: Path) -> list:
    errors = []
    half = max(args.epochs // 2, 1)
    try:
        make_manager(args, build_model(args), data, optimizer, variant, args.alpha, half, folder, resume=True).train()
        manager = make_manager(args, build_model(args), data, optimizer, variant, args.alpha, args.epochs, folder, resume=True)
        calls = record_branches(manager)
        manager.train()
        epochs_run = sorted({call[0] for call in calls})
        if epochs_run != list(range(half + 1, args.epochs + 1)):
            errors.append(f"resume: expected to run epochs {half + 1}..{args.epochs}, ran {epochs_run}")
        if sorted(manager.train_stats_tracker.history) != list(range(1, args.epochs + 1)):
            errors.append(f"resume: epochs in history are {sorted(manager.train_stats_tracker.history)}")
    except Exception as exc:
        errors.append(f"resume: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cpu', help='cpu or CUDA index, e.g. 0 (MPS is used on Apple silicon)')
    parser.add_argument('--optimizers', nargs='+', default=list(OPTIMIZERS), choices=OPTIMIZERS)
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--model', default='tiny', help="'tiny' (default) or a model name of the repository, e.g. resnet18")
    parser.add_argument('--im-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8, help='epochs of the train and resume checks')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha of the train and resume checks')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--adam-lr', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--verbose', action='store_true', help='show the training logs of TrainManager')
    args = parser.parse_args()

    failures, rows = [], []
    with tempfile.TemporaryDirectory(prefix='relax_loss_smoke_') as tmp:
        tmp = Path(tmp)
        args.logger = get_logger('relax_loss_smoke', str(tmp), mode='dumb') if args.verbose else QuietLogger()
        data = build_data(args)
        pretrained = pretrain(args, data)

        errors = check_reference(args, pretrained, data)
        failures += errors
        print(f"[reference] relax_loss vs paper (Algorithm 1, App. B.3): {'OK' if not errors else 'FAIL'}", flush=True)

        for optimizer in args.optimizers:
            for variant in args.variants:
                start = time.time()
                folder = tmp / f'{optimizer}_{variant}'
                step_errors = check_steps(args, pretrained, data, optimizer, variant, folder / 'steps')
                train_errors, info = check_train(args, data, optimizer, variant, folder / 'train')
                resume_errors = check_resume(args, data, optimizer, variant, folder / 'resume')
                failures += [f"{optimizer}/{variant} {e}" for e in step_errors + train_errors + resume_errors]
                trajectory = ' '.join(f'{loss:.2f}' for loss in info.get('losses', []))
                branches = info.get('branches', [])
                relaxed = any(branch in ('ascent', 'flatten') for branch in branches)
                rows.append((optimizer, variant, relaxed, not (train_errors or step_errors or resume_errors)))
                status = lambda errors: 'FAIL' if errors else 'OK  '
                print(f"[{optimizer:>20s} | {variant:9s}] steps {status(step_errors)} train {status(train_errors)} "
                      f"resume {status(resume_errors)} | train CE/epoch: {trajectory or '-'} | "
                      f"branches: {','.join(branches) or '-'} | {time.time() - start:.1f}s", flush=True)

    print(f"\nTrain CE (alpha={args.alpha}) is the epoch average: RelaxLoss acts on the batches whose loss is below alpha.")
    stuck = [f"{optimizer}/{variant}" for optimizer, variant, relaxed, ok in rows if ok and not relaxed]
    if stuck:
        print(f"WARNING: no batch loss got below alpha in {args.epochs} epochs for {', '.join(stuck)}, so the train check "
              f"only ran the descent branches for them (the steps check covers all branches anyway). Slow optimizer? "
              f"Note that TrainManager.setup_optimizer does not pass the learning rate to LookSAM and FriendlySAM.")
    if failures:
        print(f"\n{len(failures)} FAILED CHECK(S):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"\nAll checks passed for {len(rows)} optimizer/variant combinations.")


if __name__ == '__main__':
    main()
