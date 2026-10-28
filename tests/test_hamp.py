"""HAMP regressions with small models and no dataset downloads."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_dataset_mean_std
from src.mia.defenses.hybrid.hamp import HampDefender, HampLoss, HampTestWrapper, HampTrainManager
from src.trainer.train_manager import TrainManager
from src.utils.configs import HampDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser

TABULAR = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']


def defender(dataset, im_size, mode='test_only', gamma=0.95, alpha=0.001):
    # HampDefender without BaseDefender.__init__, which would load the dataset.
    model = HampDefender.__new__(HampDefender)
    model.logger = Mock()
    model.dataset_configs = SimpleNamespace(name=dataset, im_size=im_size)
    model.hamp_configs = HampDefenseConfigs(mode=mode, gamma=gamma, alpha=alpha)
    model.trained_model = torch.nn.Identity()
    return model


def hamp_configs(*argv):
    settings = build_parser().parse_args([*argv, '--defender_mode', 'hamp'])
    return generate_configs_from_settings(settings).defender.defense


class HampParametersTest(unittest.TestCase):
    def test_unset_parameters_follow_the_paper_per_dataset(self):
        for dataset, extra, expected in [('cifar10', [], (0.95, 0.001)),
                                         ('cifar100', [], (0.5, 0.005)),
                                         ('purchase', TABULAR, (0.8, 0.01)),
                                         ('texas', TABULAR, (0.6, 0.01)),
                                         ('svhn', [], (0.95, 0.001))]:
            with self.subTest(dataset=dataset):
                configs = hamp_configs('--dataset', dataset, *extra)
                self.assertEqual((configs.gamma, configs.alpha), expected)

    def test_set_parameters_override_the_paper(self):
        configs = hamp_configs('--dataset', 'purchase', '--defender_hamp_gamma', '0.5', *TABULAR)
        self.assertEqual((configs.gamma, configs.alpha), (0.5, 0.01))
        configs = hamp_configs('--dataset', 'cifar10', '--defender_hamp_alpha', '0.01')
        self.assertEqual((configs.gamma, configs.alpha), (0.95, 0.01))

    def test_warning_only_for_values_other_than_the_paper(self):
        for dataset, gamma, alpha, warned in [('cifar10', 0.95, 0.001, False),
                                              ('purchase', 0.8, 0.01, False),
                                              ('cifar10', 0.9, 0.001, True),
                                              ('purchase', 0.8, 0.001, True),
                                              ('svhn', 0.95, 0.001, True)]:
            with self.subTest(dataset=dataset, gamma=gamma, alpha=alpha):
                model = defender(dataset, None, mode='full', gamma=gamma, alpha=alpha)
                model.warn_if_not_paper_parameters()
                self.assertEqual(model.logger.print_it.called, warned)


def train_manager(num_classes, gamma):
    # HampTrainManager without TrainManager.__init__, which would create folders and pick devices.
    manager = HampTrainManager.__new__(HampTrainManager)
    manager.logger = Mock()
    manager.num_classes = num_classes
    manager.hamp_configs = HampDefenseConfigs(gamma=gamma)
    manager.soft_label_top1, manager.soft_label_other = manager.get_soft_label_template()
    return manager


class HampSoftLabelTest(unittest.TestCase):
    def test_template_reaches_entropy_threshold_with_true_class_highest(self):
        for num_classes in [10, 100]:
            for gamma in [0.5, 0.95]:
                with self.subTest(num_classes=num_classes, gamma=gamma):
                    manager = train_manager(num_classes, gamma)
                    p, q = manager.soft_label_top1, manager.soft_label_other
                    self.assertAlmostEqual(p + (num_classes - 1) * q, 1.0, places=4)
                    entropy = -p * math.log(p) - (num_classes - 1) * q * math.log(q)
                    self.assertGreaterEqual(entropy, gamma * math.log(num_classes) - 1e-5)
                    self.assertGreater(p, q)
                    manager.logger.print_it.assert_not_called()

    def test_gamma_one_stops_while_true_class_is_highest(self):
        # These class counts never reached the threshold at gamma=1 and used to loop forever.
        for num_classes in [2, 32]:
            with self.subTest(num_classes=num_classes):
                manager = train_manager(num_classes, 1.0)
                self.assertGreater(manager.soft_label_top1, manager.soft_label_other)
                manager.logger.print_it.assert_called_once()

    def test_soft_labels_put_top1_on_label(self):
        manager = train_manager(10, 0.9)
        labels = torch.tensor([3, 0, 9, 3])
        soft = manager.get_soft_labels(labels)
        self.assertTrue(torch.equal(soft.argmax(dim=1), labels))
        self.assertTrue((soft[torch.arange(4), labels] == manager.soft_label_top1).all())
        self.assertEqual(int((soft == manager.soft_label_other).sum()), 4 * 9)

    def test_train_step_trains_on_soft_labels(self):
        manager = train_manager(10, 0.9)
        labels = torch.tensor([1, 2])
        with patch.object(TrainManager, 'train_step') as train_step:
            manager.train_step(torch.zeros(2, 5), labels, batch_idx=3, total_batches=7)
        inputs, targets = train_step.call_args.args
        self.assertTrue(torch.equal(targets, manager.get_soft_labels(labels)))
        self.assertEqual(train_step.call_args.kwargs, {'batch_idx': 3, 'total_batches': 7})

    def test_configs_reject_gamma_outside_unit_range_and_negative_alpha(self):
        HampDefenseConfigs(gamma=0.0)
        HampDefenseConfigs(gamma=1.0)
        for kwargs in [{'gamma': 1.01}, {'gamma': -0.1}, {'alpha': -0.001}]:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                HampDefenseConfigs(**kwargs)


class HampLossTest(unittest.TestCase):
    def test_soft_targets_give_kl_minus_alpha_entropy_per_sample(self):
        torch.manual_seed(0)
        outputs = torch.randn(8, 10) * 3
        targets = train_manager(10, 0.9).get_soft_labels(torch.randint(0, 10, (8,)))
        per_sample = HampLoss(gamma=0.9, alpha=0.01)(outputs, targets)
        self.assertEqual(per_sample.shape, (8,))
        probs = torch.softmax(outputs, dim=1)
        # The KL is averaged over the 10 classes, as in the authors' code.
        kl = torch.sum(targets * (targets.log() - probs.log()), dim=1) / 10
        entropy = -torch.sum(probs * probs.log(), dim=1)
        torch.testing.assert_close(per_sample, kl - 0.01 * entropy)

    def test_mean_loss_matches_authors_code(self):
        # The training loss of cifar10/cifar-train-hamp.py in the authors' repository.
        torch.manual_seed(0)
        outputs = torch.randn(8, 10) * 3
        targets = train_manager(10, 0.9).get_soft_labels(torch.randint(0, 10, (8,)))
        entropy = torch.distributions.Categorical(probs=torch.nn.functional.softmax(outputs, dim=1)).entropy()
        authors = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(outputs, dim=1), targets) - 0.01 * torch.mean(entropy)
        torch.testing.assert_close(HampLoss(gamma=0.9, alpha=0.01, reduction='mean')(outputs, targets), authors)

    def test_integer_targets_give_cross_entropy(self):
        torch.manual_seed(0)
        outputs, labels = torch.randn(8, 10), torch.randint(0, 10, (8,))
        loss = HampLoss(gamma=0.9, alpha=0.01, reduction='mean')(outputs, labels)
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(outputs, labels))


class HampRandomInputsTest(unittest.TestCase):
    def test_image_pixels_cover_unit_range_before_normalization(self):
        torch.manual_seed(0)
        mean, std = get_dataset_mean_std('cifar10')
        wrapper = HampTestWrapper(torch.nn.Identity(), input_mean=mean, input_std=std)
        x = wrapper.random_inputs(torch.zeros(64, 3, 32, 32))
        pixels = x * torch.tensor(std).view(1, -1, 1, 1) + torch.tensor(mean).view(1, -1, 1, 1)
        for channel in range(3):
            self.assertGreaterEqual(pixels[:, channel].min().item(), 0.0)
            self.assertLessEqual(pixels[:, channel].max().item(), 1.0 + 1e-6)
            self.assertLess(pixels[:, channel].min().item(), 0.01)
            self.assertGreater(pixels[:, channel].max().item(), 0.99)

    def test_binary_features_are_fair_coin_flips(self):
        torch.manual_seed(0)
        x = HampTestWrapper(torch.nn.Identity(), binary_inputs=True).random_inputs(torch.zeros(256, 600))
        self.assertTrue(((x == 0) | (x == 1)).all())
        self.assertEqual(x.dtype, torch.float32)
        self.assertAlmostEqual(x.mean().item(), 0.5, delta=0.01)

    def test_continuous_features_are_uniform_in_unit_range(self):
        torch.manual_seed(0)
        x = HampTestWrapper(torch.nn.Identity()).random_inputs(torch.zeros(256, 600))
        self.assertGreaterEqual(x.min().item(), 0.0)
        self.assertLessEqual(x.max().item(), 1.0)
        self.assertAlmostEqual(x.mean().item(), 0.5, delta=0.01)

    def test_defend_model_picks_sampling_from_dataset(self):
        for dataset, im_size, binary, normalized in [('cifar10', (32, 32), False, True),
                                                     ('purchase', None, True, False),
                                                     ('texas', None, True, False),
                                                     ('news', None, False, False)]:
            with self.subTest(dataset=dataset):
                wrapper = defender(dataset, im_size).defend_model('cpu')
                self.assertEqual(wrapper.binary_inputs, binary)
                self.assertEqual(wrapper.input_mean is not None, normalized)
                self.assertEqual(wrapper.input_std is not None, normalized)

    def test_output_has_random_scores_in_original_ranking(self):
        wrapper = HampTestWrapper(torch.nn.Linear(20, 10))
        x = torch.randn(32, 20)
        torch.manual_seed(0)
        out = wrapper(x)
        torch.manual_seed(0)
        with torch.no_grad():
            original, random = wrapper.model(x), wrapper.model(wrapper.random_inputs(x))
        self.assertTrue(torch.equal(out.argsort(dim=1), original.argsort(dim=1)))
        self.assertTrue(torch.equal(out.sort(dim=1).values, random.sort(dim=1).values))


if __name__ == '__main__':
    unittest.main()
