"""Fast runtime checks for the edge harness's output-contract validation."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

import smoke_test_all as smoke

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.base_mia import BaseMIA


class EdgeRuntimeTests(unittest.TestCase):
    def test_membership_scores_and_decisions_are_checked_before_casting(self):
        smoke.validate_attack_outputs([.1, .9], [0, 1], 2)
        smoke.validate_attack_outputs(np.ones((2, 1)), None, 2)
        for scores, decisions in (([np.nan, .5], None), ([np.inf, .5], None),
                                  ([.5], None), ([.1, .9], [0, .5]),
                                  ([.1, .9], [0]), ([.1, .9], [0, np.nan])):
            with self.subTest(scores=scores, decisions=decisions), self.assertRaises(ValueError):
                smoke.validate_attack_outputs(scores, decisions, 2)

    def test_stats_instrumentation_checks_metrics_and_tracks_evaluation(self):
        metrics = dict(auc=.5, true_positives=1, true_negatives=1,
                       false_positives=0, false_negatives=0)
        attack = SimpleNamespace(audit_manager=SimpleNamespace(get_all_ids=lambda: [1, 2]))
        for updates, should_fail in (({}, False), ({"auc": np.nan}, True),
                                     ({"auc": 1.1}, True), ({"true_positives": 0}, True)):
            with self.subTest(updates=updates), patch.object(
                    BaseMIA, "compute_stats", return_value={**metrics, **updates}) as original:
                calls = smoke.install_strict_attack_checks()
                if should_fail:
                    with self.assertRaises(ValueError):
                        BaseMIA.compute_stats(attack, [.1, .9], decisions=[0, 1])
                    self.assertEqual(calls, [])
                else:
                    BaseMIA.compute_stats(attack, [.1, .9], decisions=[0, 1])
                    self.assertEqual(calls, [2])
                original.assert_called_once()

    def defense_probe(self, model):
        dataset = TensorDataset(torch.arange(8, dtype=torch.float32).reshape(4, 2),
                                torch.zeros(4, dtype=torch.long), torch.arange(4), torch.arange(4))
        fake = SimpleNamespace(train_model=Mock(), defend_model=Mock(),
                               get_defended_model=lambda: model,
                               get_device=lambda _: torch.device("cpu"),
                               get_dataset=lambda: SimpleNamespace(get=lambda _: dataset))
        configs = SimpleNamespace(defender=SimpleNamespace(train=SimpleNamespace(device="cpu"),
                                                           model=SimpleNamespace(num_classes=2)))
        with ExitStack() as patches:
            patches.enter_context(patch("src.utils.settings.gather_settings",
                                        return_value=SimpleNamespace(device="cpu", defender_mode="none")))
            patches.enter_context(patch("src.utils.settings.setup_configs_and_folder_from_settings",
                                        return_value=(configs, Path("/unused"))))
            patches.enter_context(patch("src.mia.defenses.get_defender_class", return_value=lambda **_: fake))
            smoke.probe_defender(strict_checks=True)
        fake.train_model.assert_called_once()
        fake.defend_model.assert_called_once()

    def test_defense_probe_exercises_singletons_odd_batches_and_replays(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = []

            def forward(self, inputs):
                self.seen.append(inputs.clone())
                return inputs.clone()

        model = RecordingModel()
        self.defense_probe(model)
        self.assertEqual([len(x) for x in model.seen], [4, 1, 3, 4, 4, 1, 1] * 2)
        torch.testing.assert_close(model.seen[3], model.seen[0].flip(0))
        torch.testing.assert_close(model.seen[4], model.seen[0])
        self.assertTrue(torch.equal(model.seen[5], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(model.seen[6], torch.ones(1, 2)))

    def test_defense_probe_rejects_mutation_and_nonfinite_outputs(self):
        class BrokenModel(torch.nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def forward(self, inputs):
                if self.mode == "mutation":
                    return inputs.add_(1)
                if self.mode == "shape":
                    return inputs[:, :1]
                return inputs * float("nan")

        for mode in ("mutation", "shape", "nonfinite"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.defense_probe(BrokenModel(mode))


if __name__ == "__main__":
    unittest.main()
