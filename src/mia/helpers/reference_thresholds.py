"""Numerically stable reference-score smoothing for Attack-P and Attack-R."""

import numpy as np
from scipy.special import expit
from scipy.stats import norm


def batched_smoothed_thresholds(ref_matrix: np.ndarray, alpha: float):
    """Return the existing empirical/logit-smoothed cutoff, per column.

    Attack-P supplies float32 model scores. Promote them before clipping: in
    float32, 1 - 1e-8 rounds to 1, producing infinite logits and NaN variances.
    Scores such as loss and multiclass entropy routinely exceed one.
    """
    references = np.asarray(ref_matrix, dtype=np.float64)
    if references.ndim not in (1, 2) or references.size == 0:
        raise ValueError("Reference scores must be a nonempty vector or matrix")
    if not np.isfinite(references).all():
        raise ValueError("Reference scores must all be finite before threshold smoothing")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Threshold alpha must be finite and between 0 and 1")

    sorted_ref = np.sort(references, axis=0)
    m = sorted_ref.shape[0]
    # A single reference cannot estimate sample variance. Endpoint quantiles
    # likewise need no Gaussian fit (whose inverse CDF is infinite there).
    if m == 1 or alpha == 0:
        return sorted_ref[0]
    if alpha == 1:
        return sorted_ref[-1]

    u = alpha * (m - 1)
    i = int(np.floor(u))
    j = min(i + 1, m - 1)
    w = u - i
    p_linear = sorted_ref[i] * (1 - w) + sorted_ref[j] * w

    clipped = np.clip(sorted_ref, 1e-8, 1 - 1e-8)
    logit_vals = np.log(clipped) - np.log1p(-clipped)
    mu = np.mean(logit_vals, axis=0)
    sigma = np.maximum(np.std(logit_vals, axis=0, ddof=1), 1e-6)
    z = norm.ppf(1 - alpha, loc=mu, scale=sigma)
    return np.minimum(p_linear, expit(z))
