"""The last stage: decision rules on the attack's scores, and the calibration samples they are calibrated on."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.helpers import IndexedDataset, MultiDatasets
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers import shadow_data_manager
from src.mia.helpers.auditing_data_manager import AuditingDatasetManager
from src.mia.helpers.calibration_data_manager import CalibrationDatasetManager
from src.mia.helpers.decision_rules import calibrated_threshold, decide, oracle_threshold
from src.mia.helpers.results_manager import ResultManager
from src.utils.configs import AuditingDataConfigs, EvaluationConfigs, ShadowDataConfigs, get_hash_from_settings
from src.utils.settings import build_parser


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _tracked(n, first_id):
    """n samples with original indices first_id, first_id + 1, ..."""
    data = TensorDataset(torch.arange(n, dtype=torch.float32), torch.zeros(n, dtype=torch.long))
    return IndexedDataset(data, original_indices=list(range(first_id, first_id + n)))


DEFENDER = MultiDatasets(datasets=[_tracked(60, 0), _tracked(15, 60), _tracked(25, 75)], ids=['train', 'val', 'test'])
ATTACKER = MultiDatasets(datasets=[_tracked(400, 1000)], ids=['all'])


class DecisionRuleTest(unittest.TestCase):
    def test_calibrated_threshold_lets_at_most_the_rate_of_calibration_samples_above_it(self):
        rng = np.random.default_rng(0)
        for scores in (rng.normal(size=1000), rng.integers(0, 5, size=1000), np.zeros(10)):
            for fpr in (0.01, 0.1, 0.29, 0.5):
                with self.subTest(fpr=fpr, distinct=len(np.unique(scores))):
                    threshold = calibrated_threshold(scores, fpr)
                    self.assertLessEqual(np.mean(scores > threshold), fpr + 1e-12)
        # Distinct scores give exactly the rate.
        self.assertEqual(np.sum(np.arange(1000.) > calibrated_threshold(np.arange(1000.), 0.29)), 290)
        for fpr in (0, 1, -0.1):
            with self.subTest(fpr=fpr), self.assertRaisesRegex(ValueError, 'in \\(0, 1\\)'):
                calibrated_threshold(np.arange(10.), fpr)

    def test_oracle_threshold_maximizes_youden_j(self):
        rng = np.random.default_rng(1)
        labels = np.repeat([1, 0], 50)
        scores = rng.normal(size=100) + labels
        threshold = oracle_threshold(labels, scores)
        best = max(np.mean(scores[labels == 1] >= t) - np.mean(scores[labels == 0] >= t) for t in scores)
        decisions = scores >= threshold
        self.assertAlmostEqual(np.mean(decisions[labels == 1]) - np.mean(decisions[labels == 0]), best)
        self.assertEqual(oracle_threshold(labels, (scores > 0).astype(float)), 0.5)

    def test_each_rule_takes_its_decisions_from_its_own_source(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([.9, .2, .8, .1])
        calibration = np.array([.0, .3, .5, .7])
        own = np.array([1, 0, 0, 0])
        decisions, source, threshold = decide('attack', scores, own, calibration, .25, labels)
        np.testing.assert_array_equal(decisions, own)
        self.assertEqual((source, threshold), ('attack', None))
        # Without decisions of its own, the attack rule calibrates: one of four calibration scores is above 0.5.
        for rule in ('attack', 'calibrated'):
            decisions, source, threshold = decide(rule, scores, None if rule == 'attack' else own, calibration, .25,
                                                  labels)
            np.testing.assert_array_equal(decisions, [1, 0, 1, 0])
            self.assertEqual((source, threshold), ('calibrated', .5))
        # The oracle ignores the attack's decisions: 0.9 and 0.2 both reach J = 0.5, and the first is kept.
        decisions, source, threshold = decide('oracle', scores, own[::-1], None, .25, labels)
        self.assertEqual((source, threshold), ('oracle', .9))
        np.testing.assert_array_equal(decisions, [1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'calibration'):
            decide('calibrated', scores, own, None, .25, labels)
        with self.assertRaisesRegex(ValueError, 'not recognized'):
            decide('best', scores, own, calibration, .25, labels)


class _FakeAttack(BaseMIA):
    """An attack scoring each sample by its value, with or without decisions of its own."""
    def score(self, dataset, device='cpu'):
        scores = np.asarray([float(x) for x, _, _, _ in dataset])
        return [MembershipScores(scores, decisions=scores > 50 if self.has_own_decisions else None)]


def _fake_attack(rules, has_own_decisions=True):
    attack = object.__new__(_FakeAttack)
    attack.logger = _SilentLogger()
    attack.has_own_decisions = has_own_decisions
    attack.evaluation_configs = EvaluationConfigs(decision_rules=rules, calibration_fpr=0.1)
    attack.audit_manager = AuditingDatasetManager(DEFENDER, AuditingDataConfigs(n_auditing_samples=20, seed=3),
                                                  logger=_SilentLogger())
    attack.calibration_manager = CalibrationDatasetManager(ATTACKER, n_samples=50, seed=3, logger=_SilentLogger())
    attack.audit_scores, attack.calibration_scores = None, None
    attack.results_manager = ResultManager()
    attack.audit_class_labels = [0] * 20
    attack.audit_model_correctness = [True] * 20
    return attack


class EvaluationTest(unittest.TestCase):
    def test_each_rule_gives_one_result_from_the_same_scores(self):
        attack = _fake_attack(['attack', 'calibrated', 'oracle'])
        with patch.object(_FakeAttack, 'score', wraps=attack.score) as score:
            results = attack.measure_effectiveness()
        # The auditing samples, then the calibration samples: the rules share the attack's scores.
        self.assertEqual(score.call_count, 2)
        self.assertEqual([result['decision_source'] for result in results], ['attack', 'calibrated', 'oracle'])
        self.assertEqual(len({result['auc'] for result in results}), 1)
        recorded = {params['decision_rule']: metrics for params, metrics in attack.results_manager.items()}
        self.assertEqual(recorded['calibrated']['calibration_fpr'], 0.1)
        self.assertNotIn('calibration_fpr', recorded['oracle'])

    def test_calibration_samples_are_scored_only_when_a_rule_needs_them(self):
        for rules, own, needed in ((['attack'], True, False), (['oracle'], False, False),
                                   (['attack'], False, True), (['oracle', 'calibrated'], True, True)):
            with self.subTest(rules=rules, own=own):
                attack = _fake_attack(rules, has_own_decisions=own)
                self.assertEqual(attack.needs_calibration(), needed)
                attack.measure_effectiveness()
                self.assertEqual(attack.calibration_scores is not None, needed)
                if not own and rules == ['attack']:
                    self.assertEqual(attack.results_manager.get_result({'decision_rule': 'attack'})['decision_source'],
                                     'calibrated')

    def test_a_rule_that_calibrates_needs_calibration_samples(self):
        configs = SimpleNamespace(dataset=SimpleNamespace(base=SimpleNamespace(name='cifar10', data_folder='', seed=0,
                                                                               def_split=.5, att_split=.5),
                                                          auditing=AuditingDataConfigs(n_auditing_samples=20),
                                                          shadow=None),
                                  model=None, attack=None, hash='h', log=None,
                                  evaluation=EvaluationConfigs(decision_rules=['calibrated'], n_calibration_samples=0))
        with patch('src.mia.attacks.base_mia.get_logger_from_configs', return_value=_SilentLogger()), \
                patch('src.mia.attacks.base_mia.get_defender_datas', return_value=DEFENDER), \
                patch('src.mia.attacks.base_mia.get_attacker_datas', return_value=ATTACKER), \
                self.assertRaisesRegex(ValueError, '--n_calibration_samples above 0'):
            _FakeAttack(defender_model=None, attacker_configs=configs)


class CalibrationSamplesTest(unittest.TestCase):
    def test_calibration_samples_are_known_non_members_independent_of_the_attack(self):
        calibration = CalibrationDatasetManager(ATTACKER, n_samples=50, seed=3, logger=_SilentLogger())
        ids = calibration.get_ids()
        self.assertEqual(len(set(ids)), 50)
        self.assertTrue(set(ids) <= set(ATTACKER.get('all').get_indices(mode='original')))
        self.assertEqual(calibration.get().get_indices(mode='original'), ids)
        self.assertEqual(CalibrationDatasetManager(ATTACKER, n_samples=50, seed=3, logger=_SilentLogger()).get_ids(),
                         ids)
        for n_samples in (0, 400):
            with self.subTest(n_samples=n_samples), self.assertRaisesRegex(ValueError, 'between 1 and 399'):
                CalibrationDatasetManager(ATTACKER, n_samples=n_samples, seed=3, logger=_SilentLogger())

    def shadows(self, mode, n_shadows=4):
        auditing = AuditingDatasetManager(DEFENDER, AuditingDataConfigs(n_auditing_samples=20, seed=3),
                                          logger=_SilentLogger())
        calibration = CalibrationDatasetManager(ATTACKER, n_samples=50, seed=3, logger=_SilentLogger())
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(shadow_data_manager, 'DEFAULT_SHADOW_DATASETS_FOLDER', folder):
            shadows = shadow_data_manager.ShadowDatasetsManager(
                ATTACKER, auditing, ShadowDataConfigs(n_shadow_datasets=n_shadows, mode=mode, seed=3),
                attacker_hash=mode, calibration_dataset=calibration, logger=_SilentLogger())
        return shadows, calibration.get_ids()

    def test_shadow_datasets_treat_calibration_samples_like_auditing_samples(self):
        for mode, expected in (('offline', 0), ('online', 2)):
            with self.subTest(mode=mode):
                shadows, ids = self.shadows(mode)
                for sample_id in ids:
                    self.assertEqual(len(shadows.find_shadow_datasets_containing_sample_id(sample_id)), expected)
                # 60% of the pool left once the calibration samples are held out.
                self.assertEqual(shadows.n_samples_per_dataset, int(np.ceil(0.6 * 350)))
                for index in range(4):
                    self.assertEqual(len(shadows.get(index, labels='original')), shadows.n_samples_per_dataset)

    def test_calibration_samples_stay_out_of_the_attacks_other_samples(self):
        shadows, ids = self.shadows('offline', n_shadows=1)
        self.assertFalse(set(ids) & set(shadows.sample_random_indices(num_data=100)))
        outside = shadows.sample_outside_shadow_dataset(index=0, num_data=100).get_indices(mode='original')
        self.assertFalse(set(ids) & set(outside))


class EvaluationHashTest(unittest.TestCase):
    def test_only_the_calibration_samples_change_the_attacker_hash(self):
        def hashes(*argv):
            settings = build_parser().parse_args(['--dataset', 'cifar10', *argv])
            modes = ('defender', 'attacker', 'experiment')
            return {mode: get_hash_from_settings(settings, mode=mode) for mode in modes}
        base = hashes()
        calibration = hashes('--n_calibration_samples', '500')
        self.assertEqual(calibration['defender'], base['defender'])
        self.assertNotEqual(calibration['attacker'], base['attacker'])
        # The decision rule acts after the attack: the shadow models are reused, the results are not.
        for argv in (('--decision_rule', 'oracle'), ('--calibration_fpr', '0.01')):
            with self.subTest(argv=argv):
                rule = hashes(*argv)
                self.assertEqual(rule['attacker'], base['attacker'])
                self.assertNotEqual(rule['experiment'], base['experiment'])


if __name__ == '__main__':
    unittest.main()
