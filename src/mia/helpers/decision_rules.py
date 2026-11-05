"""The last stage of an experiment: turning an attack's membership scores into membership decisions.

The attacks only score samples, larger scores for likelier members. A decision rule then sets the decisions:
- attack: the attack's own decisions, made without the auditing labels. Attacks without a rule of their own are
  calibrated instead.
- calibrated: a threshold set at a false-positive rate on the calibration samples, known non-members of the target.
- oracle: the threshold that maximizes TPR - FPR on the auditing labels. It is the best any threshold on the scores
  achieves, so it measures the strongest attacker a defender may face, not a realistic one.
"""
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve

from src.utils.variables import DECISION_RULES


def oracle_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold maximizing TPR - FPR (Youden's J) on the auditing labels; members score at least it."""
    if scores.dtype == np.bool_ or np.all(np.isin(scores, [0, 1])):
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.flatnonzero(np.isfinite(thresholds))
    return float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])


def calibrated_threshold(calibration_scores: np.ndarray, fpr: float) -> float:
    """Return the threshold that at most a share fpr of the calibration scores exceed; members score above it."""
    if not 0 < fpr < 1:
        raise ValueError(f'The calibration false-positive rate must be in (0, 1), got {fpr}.')
    ordered = np.sort(np.asarray(calibration_scores, dtype=np.float64).reshape(-1))
    if not len(ordered):
        raise ValueError('The calibrated decision rule needs calibration scores.')
    # The small tolerance keeps e.g. 0.29 * 100 at 29 samples rather than 28.
    allowed = int(np.floor(fpr * len(ordered) + 1e-9))
    return float(ordered[len(ordered) - allowed - 1])


def decide(rule: str, scores: np.ndarray, attack_decisions: Optional[np.ndarray],
           calibration_scores: Optional[np.ndarray], fpr: float,
           labels: np.ndarray) -> Tuple[np.ndarray, str, Optional[float]]:
    """Return the decisions under the rule, the rule they come from, and their threshold (None for the attack's own).

    The rule they come from is 'calibrated' when the attack rule falls back to calibration.
    """
    if rule not in DECISION_RULES:
        raise ValueError(f'Decision rule {rule!r} not recognized: choose among {", ".join(DECISION_RULES)}.')
    scores = np.asarray(scores).reshape(-1)
    if rule == 'attack' and attack_decisions is not None:
        return np.asarray(attack_decisions).reshape(-1), 'attack', None
    if rule == 'oracle':
        threshold = oracle_threshold(labels, scores)
        return (scores >= threshold).astype(np.int64), 'oracle', threshold
    if calibration_scores is None:
        raise ValueError(f'The {rule} decision rule needs the calibration samples\' scores.')
    threshold = calibrated_threshold(calibration_scores, fpr)
    return (scores > threshold).astype(np.int64), 'calibrated', threshold
