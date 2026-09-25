#!/usr/bin/env python3
"""One tiny RobustMIA run: one epoch per defender/shadow model.

Example: python3 test_robust_mia.py --attacker_robust_gamma 1.2 --device cpu
Uses the normal dataset loader (which may download CIFAR-10), limited to 400
samples by the existing smoke worker. This checks execution, not attack quality.
"""

import argparse
from datetime import datetime
import math
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacker_robust_gamma', type=float, default=1.2)
    parser.add_argument('--device', default='cpu', help='cpu or CUDA index, e.g. 0')
    parser.add_argument('--datasets-folder', type=Path, default=ROOT / 'datas')
    parser.add_argument('--dry-run', action='store_true', help='Print command without training')
    args = parser.parse_args()
    if not math.isfinite(args.attacker_robust_gamma) or args.attacker_robust_gamma <= 0:
        parser.error('--attacker_robust_gamma must be finite and positive')

    # Fresh output directory prevents previous checkpoints from skipping training.
    output = ROOT / 'smoke_test_results' / 'robust_mia' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    command = [
        sys.executable, str(ROOT / 'smoke_test_all.py'),
        '--worker', '--data-limit', '400', '--entrypoint', 'run', '--',
        '--dataset', 'cifar10', '--datasets_folder', str(args.datasets_folder.resolve()),
        '--out_folder', str(output), '--device', args.device,
        '--defender_mode', 'none', '--attacker_mode', 'offline_robust',
        '--defender_model', 'resnet18', '--attacker_model', 'resnet18',
        '--defender_epochs', '1', '--attacker_epochs', '1',
        '--defender_batch_size', '8', '--attacker_batch_size', '8',
        '--defender_lr', '0.01', '--attacker_lr', '0.01',
        '--defender_lr_sched', 'cosine', '--attacker_lr_sched', 'cosine',
        '--seed', '101', '--n_shadows', '2',
        '--n_auditing_samples', '8', '--audit_in_perc', '0.5',
        '--attacker_robust_rand_pop_size', '8', '--attacker_robust_alphas', '0.5',
        '--attacker_robust_gamma', str(args.attacker_robust_gamma),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    print(f'Running one-epoch RobustMIA smoke check; results: {output}', flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode == 0:
        print(f'PASS: RobustMIA completed with gamma={args.attacker_robust_gamma}.')
        print(f'Results and checkpoints: {output}')
    else:
        print('FAIL: see the error above.', file=sys.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
