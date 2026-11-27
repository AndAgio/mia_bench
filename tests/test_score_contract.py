"""Deterministic scoring regressions; no datasets or model training required."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mia.attacks.black_box.attack_p import AttackPMIA
from src.mia.attacks.black_box.attack_r import AttackRMIA
from src.mia.attacks.label_only.yoqo import Yoqo


class ScoreContractTests(unittest.TestCase):
    def test_reference_attacks_measure_with_real_float32_thresholds(self):
        dataset = TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long),
                                torch.arange(4), torch.zeros(4))
        for cls, prefix in ((AttackPMIA, 'p'), (AttackRMIA, 'r')):
            for mode in ('loss', 'entropy', 'confidence'):
                with self.subTest(attack=prefix, mode=mode):
                    attack = object.__new__(cls)
                    attack.logger = Mock()
                    attack.name = prefix
                    attack.attack_configs = SimpleNamespace(
                        **{f'{prefix}_score_type': mode, f'{prefix}_alpha': .2})
                    attack.shadow_manager = Mock()
                    attack.shadow_manager.get_dataset.return_value = dataset
                    attack.shadow_manager.get_all_models.return_value = {0: torch.nn.Identity()}
                    attack.defender_model = torch.nn.Identity()
                    population = [1.5, 2., 2.5, 3.]
                    audit = [2.9, 2.4, .3, .1]
                    if mode == 'confidence':
                        population, audit = [.1, .2, .3, .4], [.05, .1, .8, .9]
                    attack.compute_batch_scores = Mock(side_effect=[
                        np.asarray(population, dtype=np.float32), np.asarray(audit, dtype=np.float32)])
                    [(scores, decisions, _)] = attack.score(dataset)
                    self.assertTrue(np.isfinite(scores).all())
                    self.assertEqual(roc_auc_score([0, 0, 1, 1], scores), 1.)
                    np.testing.assert_array_equal(decisions, [False, False, True, True])

    def test_scores_follow_the_loss_and_members_are_the_samples_with_alpha_star_at_most_alpha(self):
        # Attack-P compares every sample with the same population, so alpha_star grows with the loss: the ranking is the
        # loss's, and the members are exactly the samples whose alpha_star, minus their score, is at most alpha.
        dataset = TensorDataset(torch.zeros(5, 2), torch.zeros(5, dtype=torch.long), torch.arange(5), torch.zeros(5))
        population, audit = np.linspace(.5, 3., 20), np.array([.1, .6, 1.2, 2.5, 4.])
        for mode in ('loss', 'entropy', 'confidence'):
            with self.subTest(mode=mode):
                attack = object.__new__(AttackPMIA)
                attack.logger = Mock()
                attack.name = 'p'
                attack.attack_configs = SimpleNamespace(p_score_type=mode, p_alpha=.1)
                attack.shadow_manager = Mock()
                attack.shadow_manager.get_dataset.return_value = dataset
                attack.defender_model = torch.nn.Identity()
                # A confidence c reads as the loss -log c, so the confidences are e^-loss.
                inputs = (np.exp(-population), np.exp(-audit)) if mode == 'confidence' else (population, audit)
                attack.compute_batch_scores = Mock(side_effect=list(inputs))
                [(scores, decisions, _)] = attack.score(dataset)
                self.assertTrue((np.diff(scores) < 0).all())
                np.testing.assert_array_equal(decisions, -scores <= .1)

    def test_yoqo_scores_use_only_label_agreement(self):
        attack = object.__new__(Yoqo)
        attack.defender_model = torch.nn.Identity()
        labels = torch.zeros(2, dtype=torch.long)
        # The incorrect prediction has a larger true-class logit. A per-sample
        # common logit shift must have no effect on label-only evidence.
        logits = torch.tensor([[1., 0.], [100., 101.]])
        for queries in (logits, logits + torch.tensor([[500.], [-500.]])):
            scores = attack.query_target(queries, labels, torch.device('cpu'))
            np.testing.assert_array_equal(scores, [1, 0])
            self.assertEqual(roc_auc_score([1, 0], scores), 1.0)


if __name__ == '__main__':
    unittest.main()
