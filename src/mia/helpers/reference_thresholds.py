"""Attack-R and Attack-P's test, computed as Ye et al. do for their figures.

The source is privacy_meter's research/2022_enhanced_mia/plot_attack_via_reference_or_distill.py (commit e0342ca). For
each target sample, alpha_star is the smallest false-positive rate alpha at which the sample is a member: its loss lies
below the threshold of both the paper's percentile (App. B.3, with two dummy references) and its logit-rescaled
Gaussian fit. Ranking samples by -alpha_star gives the paper's ROC, and deciding alpha_star <= alpha the paper's test
at level alpha, so both come from the same number.
"""

import numpy as np
from scipy.stats import norm

# Ye's code adds these two dummy losses to the references, so that a target below or above every reference still gets
# an alpha inside (0, 1) from the linear interpolation: with m references, the k-th lowest sits at alpha = k / (m + 1).
DUMMY_LOW, DUMMY_HIGH = 0., 1000.
# Losses are raised to this before their logit, which is infinite at a loss of 0 (a probability of exactly 1).
MIN_LOSS = 1e-12


def loss_like(scores, score_type: str) -> np.ndarray:
    """The score as a loss, lower for likelier members, which the paper's formula expects: the loss as it is, the
    entropy as a loss, and the confidence c as -log c."""
    scores = np.asarray(scores, dtype=np.float64)
    if score_type in ('loss', 'entropy'):
        return scores
    if score_type == 'confidence':
        return -np.log(np.clip(scores, np.finfo(np.float64).tiny, 1.))
    raise ValueError(f"Unsupported scoring type '{score_type}'.")


def logit_of_loss(losses) -> np.ndarray:
    """log(e^-l / (1 - e^-l)), the logit of the probability e^-l, computed stably."""
    losses = np.maximum(np.asarray(losses, dtype=np.float64), MIN_LOSS)
    return -losses - np.log(-np.expm1(-losses))


def alpha_star(references, targets) -> np.ndarray:
    """The smallest alpha at which each target is a member: the larger of the linear and the logit alpha.

    references holds the reference losses of each target, shape (m, n), or (m,) when all targets share them (Attack-P's
    population); targets holds the target model's losses, shape (n,). A target is a member at level alpha when
    alpha_star <= alpha.
    """
    references = np.asarray(references, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if references.ndim not in (1, 2) or references.shape[0] == 0:
        raise ValueError("Reference losses must be a nonempty vector or matrix")
    if references.ndim == 2 and references.shape[1] != len(targets):
        raise ValueError("A reference matrix needs one column per target")
    if not np.isfinite(references).all() or not np.isfinite(targets).all():
        raise ValueError("Reference and target losses must be finite")

    levels = np.linspace(0, 1, references.shape[0] + 2)
    if references.ndim == 1:
        padded = np.sort(np.concatenate([references, [DUMMY_LOW, DUMMY_HIGH]]))
        linear = np.interp(targets, padded, levels)
    else:
        dummies = np.full((1, references.shape[1]), DUMMY_LOW), np.full((1, references.shape[1]), DUMMY_HIGH)
        padded = np.sort(np.concatenate([references, *dummies]), axis=0)
        linear = np.array([np.interp(target, padded[:, column], levels) for column, target in enumerate(targets)])

    reference_logits = logit_of_loss(references)
    # The survival function equals Ye's 1 - cdf, without losing the smallest alphas to rounding.
    logit = norm.sf(logit_of_loss(targets), reference_logits.mean(axis=0), reference_logits.std(axis=0) + 1e-30)
    return np.maximum(linear, logit)
