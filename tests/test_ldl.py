"""LDL regressions with small models and no dataset downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pydantic
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_dataset_mean_std
from src.mia.defenses.post_hoc.ldl import LdlDefender, LdlModel
from src.utils.configs import LdlDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser


def defender(dataset, im_size, noise_type='normal'):
    # LdlDefender without BaseDefender.__init__, which would load the dataset.
    model = LdlDefender.__new__(LdlDefender)
    model.logger = Mock()
    model.dataset_configs = SimpleNamespace(name=dataset, im_size=im_size)
    model.ldl_configs = LdlDefenseConfigs(n_queries=3, noise_type=noise_type)
    model.max_batch_size = 8
    model.trained_model = torch.nn.Identity()
    return model


def ldl_configs(*argv):
    settings = build_parser().parse_args([*argv, '--defender_mode', 'ldl'])
    return generate_configs_from_settings(settings).defender.defense


class LdlModelTest(unittest.TestCase):
    def test_batched_copies_keep_samples_apart(self):
        # Without noise every copy equals its input, so any mix-up between samples or chunks shows up. With 3 samples and
        # max_batch_size=10, the 7 copies go through in chunks of 3, 3 and 1.
        x = torch.randn(3, 5)
        model = LdlModel(torch.nn.Identity(), n_queries=7, noise_scale=0.0, max_batch_size=10)
        torch.testing.assert_close(model(x), x)
        torch.testing.assert_close(LdlModel(torch.nn.Identity(), n_queries=7, noise_scale=0.0, max_batch_size=1)(x), x)

    def test_average_has_noise_of_n_queries_independent_copies(self):
        torch.manual_seed(0)
        x = torch.zeros(4, 20000)
        out = LdlModel(torch.nn.Identity(), n_queries=100, noise_scale=0.5, max_batch_size=64)(x)
        self.assertAlmostEqual(out.std().item(), 0.5 / 10, delta=0.002)

    def test_noise_scale_is_in_pixel_units(self):
        torch.manual_seed(0)
        std = [0.5, 0.25, 0.1]
        model = LdlModel(torch.nn.Identity(), noise_scale=0.02, input_std=std)
        noise = model.add_noise(torch.zeros(64, 3, 32, 32))
        for channel, channel_std in enumerate(std):
            self.assertAlmostEqual(noise[:, channel].std().item(), 0.02 / channel_std, delta=0.02 / channel_std * 0.02)

    def test_flip_prob_is_separate_from_noise_scale(self):
        torch.manual_seed(0)
        model = LdlModel(torch.nn.Identity(), noise_type='bernoulli', noise_scale=0.5, flip_prob=0.01)
        x = (torch.rand(256, 600) > 0.5).float()
        noisy = model.add_noise(x)
        self.assertAlmostEqual((noisy != x).float().mean().item(), 0.01, delta=0.001)
        self.assertTrue(((noisy == 0) | (noisy == 1)).all())

    def test_flips_extend_to_non_binary_inputs(self):
        torch.manual_seed(0)
        model = LdlModel(torch.nn.Identity(), noise_type='bernoulli', flip_prob=0.1)
        x = torch.rand(64, 100)
        noisy = model.add_noise(x)
        flipped = noisy != x
        self.assertGreater(flipped.sum().item(), 0)
        torch.testing.assert_close(noisy[flipped], 1 - x[flipped])

    def test_gradients_only_when_the_input_requires_them(self):
        model = LdlModel(torch.nn.Linear(6, 3), n_queries=4)
        self.assertFalse(model(torch.randn(2, 6)).requires_grad)
        x = torch.randn(2, 6, requires_grad=True)
        model(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
        with torch.no_grad():
            self.assertFalse(model(x).requires_grad)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_input_std_moves_to_mps(self):
        model = LdlModel(torch.nn.Identity(), input_std=[0.5, 0.25, 0.1]).to('mps')
        self.assertEqual(model(torch.zeros(2, 3, 4, 4, device='mps')).device.type, 'mps')


class LdlConfigsTest(unittest.TestCase):
    def test_noise_type_rejects_unknown_names(self):
        self.assertEqual(LdlDefenseConfigs().noise_type, 'normal')
        with self.assertRaises(pydantic.ValidationError):
            LdlDefenseConfigs(noise_type='gaussian')

    def test_noise_follows_the_dataset(self):
        tabular = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']
        self.assertEqual(ldl_configs('--dataset', 'cifar10').noise_type, 'normal')
        self.assertEqual(ldl_configs('--dataset', 'news', *tabular).noise_type, 'normal')
        purchase = ldl_configs('--dataset', 'purchase', *tabular)
        self.assertEqual((purchase.noise_type, purchase.flip_prob), ('bernoulli', 0.0166))
        self.assertEqual(ldl_configs('--dataset', 'texas', *tabular).flip_prob, 0.00016)
        self.assertEqual(ldl_configs('--dataset', 'purchase', '--defender_ldl_flip_prob', '0.05', *tabular).flip_prob, 0.05)

    def test_defend_model_rescales_only_image_noise(self):
        image = defender('cifar10', (32, 32)).defend_model(torch.device('cpu'))
        torch.testing.assert_close(image.input_std.flatten(), torch.tensor(get_dataset_mean_std('cifar10')[1]))
        self.assertEqual(image.max_batch_size, 8)
        self.assertIsNone(defender('purchase', None, noise_type='bernoulli').defend_model(torch.device('cpu')).input_std)


if __name__ == '__main__':
    unittest.main()
