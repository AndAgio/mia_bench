"""Attacks --n_shadows does not affect: one shadow dataset, the same hashes whatever it is; boundary budgets."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.configs import (N_SHADOWS_UNUSED_MODES, ONE_SHADOW_MODEL_MODES, generate_configs_from_settings,
                               get_hash_from_settings)
from src.utils.settings import build_parser
from src.utils.variables import BOUNDARY_N_QUERIES, QEBA_REDUCTION_FACTOR


class AttackShadowConfigTest(unittest.TestCase):
    def settings(self, *args):
        return build_parser().parse_args(list(args))

    def test_hashes_of_attacks_that_ignore_n_shadows_do_not_depend_on_it(self):
        for attack in N_SHADOWS_UNUSED_MODES:
            for mode in ('attacker', 'experiment'):
                with self.subTest(attack=attack, mode=mode):
                    one, many = (get_hash_from_settings(self.settings('--attacker_mode', attack, '--n_shadows', n),
                                                        mode) for n in ('1', '32'))
                    self.assertEqual(one, many)
        for mode in ('attacker', 'experiment'):
            with self.subTest(attack='lira', mode=mode):
                lira = [get_hash_from_settings(self.settings('--attacker_mode', 'lira', '--n_shadows', n), mode)
                        for n in ('4', '32')]
                self.assertNotEqual(*lira)

    def test_one_shadow_model_attacks_get_one_shadow_dataset(self):
        for attack in ONE_SHADOW_MODEL_MODES:
            with self.subTest(attack=attack):
                configs = generate_configs_from_settings(self.settings('--attacker_mode', attack, '--n_shadows', '32'))
                self.assertEqual(configs.attacker.dataset.shadow.n_shadow_datasets, 1)

    def test_budgets_when_unset_and_set(self):
        for mode, unset in (('sba', BOUNDARY_N_QUERIES), ('uba', BOUNDARY_N_QUERIES)):
            with self.subTest(mode=mode):
                default = generate_configs_from_settings(self.settings('--attacker_mode', mode))
                chosen = generate_configs_from_settings(self.settings('--attacker_mode', mode,
                                                                      '--attacker_boundary_n_queries', '300'))
                self.assertEqual(default.attacker.attack.boundary.n_queries, unset)
                self.assertEqual(chosen.attacker.attack.boundary.n_queries, 300)

    def test_qeba_reduction_factor_when_unset_and_set(self):
        for mode in ('sba_qeba', 'uba_qeba-dct'):
            with self.subTest(mode=mode):
                default = generate_configs_from_settings(self.settings('--attacker_mode', mode))
                chosen = generate_configs_from_settings(self.settings(
                    '--attacker_mode', mode, '--attacker_boundary_qeba_reduction_factor', '4'))
                self.assertEqual(default.attacker.attack.boundary.qeba.reduction_factor, QEBA_REDUCTION_FACTOR)
                self.assertEqual(chosen.attacker.attack.boundary.qeba.reduction_factor, 4)

if __name__ == '__main__':
    unittest.main()
