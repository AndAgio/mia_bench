"""Attack-R and Attack-P's alpha_star against Ye et al.'s figure code; no model training or data required."""

from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
import scipy.stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mia.helpers.reference_thresholds import alpha_star, logit_of_loss, loss_like


def ye_linear(losses, check_losses):
    """generate_reference_or_distill_linear of privacy_meter's research/2022_enhanced_mia (commit e0342ca)."""
    dummy_min = np.zeros((1, len(losses[0])))
    dummy_max = dummy_min + 1000
    dat_reference_or_distill = np.sort(np.concatenate((losses, dummy_max, dummy_min), axis=0), axis=0)
    prediction = np.array([])
    discrete_alpha = np.linspace(0, 1, len(dat_reference_or_distill))
    for i in range(len(dat_reference_or_distill[0])):
        losses_i = dat_reference_or_distill[:, i]
        prediction = np.append(prediction, np.interp(check_losses[0][i], losses_i, discrete_alpha))
    return prediction


def ye_logit(losses, check_losses):
    """generate_reference_or_distill_logit of the same file."""
    dat_reference_or_distill = np.log(np.exp(-losses) / (1 - np.exp(-losses)))
    mean_reference_or_distill = np.mean(dat_reference_or_distill, 0)
    std_reference_or_distill = np.std(dat_reference_or_distill, 0)
    check_losses = np.transpose(check_losses[0])
    return 1 - scipy.stats.norm.cdf(np.log(np.exp(-check_losses) / (1 - np.exp(-check_losses))),
                                    mean_reference_or_distill, std_reference_or_distill + 1e-30)


class AlphaStarTests(unittest.TestCase):
    def test_matches_the_maximum_of_ye_linear_and_logit_alphas(self):
        rng = np.random.default_rng(0)
        references = rng.gamma(2., .5, size=(9, 200))
        targets = rng.gamma(2., .5, size=200)
        expected = np.maximum(ye_linear(references, targets[None]), ye_logit(references, targets[None]))
        np.testing.assert_allclose(alpha_star(references, targets), expected, atol=1e-12)

    def test_the_kth_lowest_reference_sits_at_k_over_m_plus_one_and_the_dummies_at_0_and_1(self):
        references = np.array([1., 2., 3., 4.])
        # alpha_star is the larger of the linear and the logit alpha, so it is at least the linear k / (m + 1).
        self.assertTrue((alpha_star(references, references) >= np.array([1, 2, 3, 4]) / 5 - 1e-12).all())
        np.testing.assert_allclose(alpha_star(references, [0., 1000.]), [0., 1.], atol=1e-12)

    def test_a_shared_reference_vector_equals_a_matrix_of_identical_columns(self):
        rng = np.random.default_rng(1)
        population, targets = rng.gamma(2., .5, size=50), rng.gamma(2., .5, size=20)
        np.testing.assert_allclose(alpha_star(population, targets),
                                   alpha_star(np.repeat(population[:, None], 20, axis=1), targets))

    def test_non_members_drawn_like_their_references_are_members_at_most_alpha_of_the_time(self):
        # The linear alpha is a valid p-value for exchangeable losses, and alpha_star is never below it.
        rng = np.random.default_rng(2)
        losses = rng.gamma(2., .5, size=(10, 20000)) * rng.uniform(.2, 3., size=20000)
        stars = alpha_star(losses[:9], losses[9])
        for alpha in (.1, .05):
            with self.subTest(alpha=alpha):
                self.assertLessEqual((stars <= alpha).mean(), alpha + .005)

    def test_zero_losses_and_confidences_of_one_stay_finite(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertTrue(np.isfinite(logit_of_loss([0., 1e-300, 50.])).all())
            stars = alpha_star(np.zeros((3, 2)), [0., 1.])
            confidence_losses = loss_like(np.array([1., .5, 1e-320]), 'confidence')
        self.assertTrue(np.isfinite(stars).all())
        np.testing.assert_allclose(confidence_losses[:2], [0., np.log(2)])
        self.assertTrue(np.isfinite(confidence_losses).all())

    def test_invalid_inputs_fail_explicitly(self):
        for references, targets in ((np.ones((0, 2)), [1., 1.]), (np.ones((3, 2)), [1.]), ([1., np.nan], [1.]),
                                    ([1., 2.], [np.inf])):
            with self.subTest(references=references, targets=targets), self.assertRaises(ValueError):
                alpha_star(references, targets)
        with self.assertRaises(ValueError):
            loss_like([.5], 'margin')


if __name__ == '__main__':
    unittest.main()
