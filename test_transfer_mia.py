#!/usr/bin/env python3
"""Focused regression and end-to-end smoke tests for TransferMIA.

The fast regression check verifies the score orientation and threshold rule for
both loss and confidence.  The end-to-end portion reuses smoke_test_all.py so it
also trains a defender and transfer model, calibrates the attack threshold, and
writes the normal benchmark results for both attack modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_curve

import smoke_test_all
from src.mia.attacks.label_only.transfer import TransferMIA


REPO_ROOT = Path(__file__).resolve().parent


class FixedLogitModel(torch.nn.Module):
    """Treat its inputs as logits, making the regression check deterministic."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def check_score_direction(feature_mode: str) -> None:
    # The first two examples are confident/correct (member-like); the last two
    # are less confident (non-member-like).
    logits = torch.tensor([
        [5.0, 0.0],
        [4.0, 0.0],
        [0.2, 0.0],
        [0.0, 0.0],
    ])
    true_labels = torch.zeros(4, dtype=torch.long)
    membership_labels = np.array([1, 1, 0, 0], dtype=np.int64)

    attack = object.__new__(TransferMIA)
    attack.attack_configs = SimpleNamespace(feature_mode=feature_mode)
    scores = attack._features(FixedLogitModel(), logits, true_labels).numpy()

    _, _, thresholds = roc_curve(membership_labels, scores)
    predictions = [
        (scores >= threshold).astype(np.int64)
        for threshold in thresholds
    ]
    accuracies = [
        accuracy_score(membership_labels, prediction)
        for prediction in predictions
    ]
    best_index = int(np.argmax(accuracies))
    best_threshold = float(thresholds[best_index])
    best_predictions = predictions[best_index]

    if accuracies[best_index] != 1.0:
        raise AssertionError(
            f"{feature_mode}: expected perfect separation, got scores={scores}, "
            f"threshold={best_threshold}, predictions={best_predictions}"
        )
    if not np.all(scores[:2] > scores[2:].max()):
        raise AssertionError(
            f"{feature_mode}: member-like scores are not larger: {scores}"
        )

    print(
        f"PASS direction check: {feature_mode}; scores={scores.tolist()}, "
        f"threshold={best_threshold:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test TransferMIA with loss and confidence scores."
    )
    parser.add_argument("--device", default="cpu",
                        help="cpu or CUDA device index (default: cpu)")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--datasets-folder", type=Path,
                        default=REPO_ROOT / "datas")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "smoke_test_results" / "transfer_mia")
    parser.add_argument("--preset", choices=("quick", "medium", "moderate"),
                        default="quick")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Maximum seconds for each end-to-end case")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep model checkpoints and result artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print end-to-end commands without executing them")
    parser.add_argument("--direction-only", action="store_true",
                        help="Run only the fast deterministic regression checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Running deterministic TransferMIA score checks...")
    check_score_direction("loss")
    check_score_direction("max_confidence")

    if args.direction_only:
        print("TransferMIA direction checks passed.")
        return 0

    smoke_args = [
        "--preset", args.preset,
        "--device", args.device,
        "--dataset", args.dataset,
        "--datasets-folder", str(args.datasets_folder),
        "--output-dir", str(args.output_dir),
        "--timeout", str(args.timeout),
        "--only", "attack:transfer_loss", "attack:transfer_confidence",
    ]
    if args.keep_work:
        smoke_args.append("--keep-work")
    if args.dry_run:
        smoke_args.append("--dry-run")

    print("\nRunning end-to-end TransferMIA smoke cases...")
    return smoke_test_all.main(smoke_args)


if __name__ == "__main__":
    raise SystemExit(main())
