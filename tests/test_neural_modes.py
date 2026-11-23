"""The neural attack's modes and startup checks: neural_* is the paper's offline attack, online_neural_* online."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks import get_attacker_class
from src.mia.attacks.base_mia import BaseMIA
from src.mia.attacks.black_box.neural import NeuralMIA
from src.utils.configs import generate_configs_from_settings
from src.utils.settings import build_parser


class ModeTests(unittest.TestCase):
    def test_neural_modes_are_offline_and_online_neural_modes_online(self):
        for input_mode in ('feat', 'prob', 'logit'):
            for prefix, expected in (('', 'offline'), ('online_', 'online')):
                mode = f'{prefix}neural_{input_mode}'
                with self.subTest(mode=mode):
                    settings = build_parser().parse_args(['--attacker_mode', mode])
                    configs = generate_configs_from_settings(settings)
                    attack, shadow = configs.attacker.attack, configs.attacker.dataset.shadow
                    self.assertEqual((attack.mode, shadow.mode, attack.neural_input_mode),
                                     (expected, expected, input_mode))
                    self.assertIs(get_attacker_class(mode), NeuralMIA)

    def test_shadow_datasets_of_the_other_mode_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'own mode'):
            build('offline', 4, shadow_mode='online')

    def test_one_shadow_model_is_refused_and_two_accepted(self):
        for mode in ('offline', 'online'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'at least 2'):
                    build(mode, 1)
                self.assertEqual(build(mode, 2).mode, mode)


def build(mode, n_shadows, shadow_mode=None):
    """Build a NeuralMIA with BaseMIA's setup and the shadow manager replaced by stand-ins."""
    def fake_init(attack, defender_model, attacker_configs):
        attack.logger = Mock()
        attack.attack_configs = SimpleNamespace(mode=mode)
        attack.shadow_configs = SimpleNamespace(mode=shadow_mode or mode, n_shadow_datasets=n_shadows)
        for name in ('attacker_data_distribution', 'audit_manager', 'calibration_manager', 'attacker_hash',
                     'model_configs'):
            setattr(attack, name, None)
    with patch.object(BaseMIA, '__init__', fake_init), patch('src.mia.attacks.black_box.neural.ShadowManager'):
        return NeuralMIA(defender_model=None, attacker_configs=None)


if __name__ == '__main__':
    unittest.main()
