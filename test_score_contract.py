"""Deterministic scoring regressions; no datasets or model training required."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset

from src.mia.attacks.black_box.attack_p import AttackPMIA
from src.mia.attacks.black_box.attack_r import AttackRMIA
from src.mia.attacks.label_only.yoqo import Yoqo


class ScoreContractTests(unittest.TestCase):
    def test_reference_attacks_keep_ranking_and_calibrated_decisions(self):
        dataset = TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long),
                                torch.arange(4), torch.zeros(4))
        for cls, prefix in ((AttackPMIA, 'p'), (AttackRMIA, 'r')):
            for mode in ('loss', 'entropy', 'confidence'):
                with self.subTest(attack=prefix, mode=mode):
                    attack = object.__new__(cls)
                    attack.logger = Mock()
                    attack.name = prefix
                    attack.attack_configs = SimpleNamespace(
                        **{f'{prefix}_score_type': mode, f'{prefix}_alpha': 0.1})
                    attack.audit_manager = Mock()
                    attack.audit_manager.get.return_value = dataset
                    attack.shadow_manager = Mock()
                    attack.shadow_manager.get_dataset.return_value = dataset
                    attack.shadow_manager.get_all_models.return_value = {0: torch.nn.Identity()}
                    attack.defender_model = torch.nn.Identity()
                    # All points fall on the same side of the calibrated cutoff;
                    # ROC must still retain the perfect underlying ranking.
                    raw = np.array([0.1, 0.2, 0.3, 0.4])
                    if mode == 'confidence':
                        raw = 1 - raw
                    threshold = 0.5
                    attack.compute_batch_scores = Mock(return_value=raw)
                    attack.compute_stats = Mock(return_value={'auc': 1.0})
                    with patch.object(cls, 'batched_smoothed_thresholds', return_value=threshold):
                        attack.measure_effectiveness()
                    scores = attack.compute_stats.call_args.args[0]
                    decisions = attack.compute_stats.call_args.kwargs['decisions']
                    self.assertEqual(roc_auc_score([1, 1, 0, 0], scores), 1.0)
                    self.assertEqual(len(np.unique(scores)), 4)
                    np.testing.assert_array_equal(decisions, [1, 1, 1, 1])

    def test_yoqo_scores_use_only_label_agreement(self):
        attack = object.__new__(Yoqo)
        attack.defender_model = torch.nn.Identity()
        attack.audit_manager = Mock()
        attack.audit_manager.get.return_value = TensorDataset(
            torch.zeros(2, 2), torch.zeros(2, dtype=torch.long),
            torch.arange(2), torch.zeros(2))
        # The incorrect prediction has a larger true-class logit. A per-sample
        # common logit shift must have no effect on label-only evidence.
        logits = torch.tensor([[1., 0.], [100., 101.]])
        for inputs in (logits, logits + torch.tensor([[500.], [-500.]])):
            scores, decisions = attack.compute_scores_from_adversaries(inputs)
            np.testing.assert_array_equal(scores, [1, 0])
            np.testing.assert_array_equal(decisions, scores)
            self.assertEqual(roc_auc_score([1, 0], scores), 1.0)


if __name__ == '__main__':
    unittest.main()
