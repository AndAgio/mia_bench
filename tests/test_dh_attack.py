"""DHAttack: fixed inputs, fixedBD and its queries, batching, the threshold's labels, seeds and decision rules."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torchvision
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The tests folder too, so that the sibling import below works with or without -t.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.mia.attacks.label_only.dh_attack import DHAttack
from src.mia.attacks.label_only.unsupervised_boundary import UnsupervisedBoundaryMIA
from src.utils.configs import generate_configs_from_settings
from src.utils.settings import build_parser
from test_hop_skip_jump import Silent


class StepModel(torch.nn.Module):
    """Predicts class 1 when the mean feature is above a threshold, class 0 otherwise; counts the samples queried."""
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold, self.queried = threshold, 0

    def forward(self, x):
        self.queried += x.shape[0]
        above = (x.flatten(1).mean(dim=1) > self.threshold).float()
        return torch.stack([1 - above, above], dim=1)


def attack(target, shadows, is_image=False, mode='white', n_queries=10, dataset=None, name='purchase'):
    dh = object.__new__(DHAttack)
    dh.logger = Silent()
    dh.seed = 0
    dh.attack_configs = SimpleNamespace(n_queries=n_queries, fixed_input_mode=mode)
    dh.base_dataset_configs = SimpleNamespace(im_size=32 if is_image else None, name=name)
    dh.defender_model = target
    dh.shadow_manager = SimpleNamespace(get_dataset=lambda index, labels: dataset,
                                        get_all_models=lambda: dict(enumerate(shadows)))
    dh.fixed_point, dh.attack_threshold = None, None
    return dh


class DHAttackTests(unittest.TestCase):
    def test_tabular_samples_move_toward_their_own_largest_or_smallest_value(self):
        inputs = torch.tensor([[0., 1., 0.], [.2, .5, .1]])
        white = attack(None, [], mode='white').construct_fixed_input(inputs)
        black = attack(None, [], mode='black').construct_fixed_input(inputs)
        self.assertTrue(torch.equal(white, torch.tensor([[1., 1., 1.], [.5, .5, .5]])))
        self.assertTrue(torch.equal(black, torch.tensor([[0., 0., 0.], [.1, .1, .1]])))

    def test_tabular_random_point_is_one_random_record_of_valid_values(self):
        inputs = torch.zeros(3, 50)
        binary = attack(None, [], mode='random', name='purchase').construct_fixed_input(inputs)
        continuous = attack(None, [], mode='random', name='news').construct_fixed_input(inputs)
        for fixed in (binary, continuous):
            self.assertTrue(torch.equal(fixed[0], fixed[1]) and torch.equal(fixed[0], fixed[2]))
            self.assertTrue(((fixed >= 0) & (fixed <= 1)).all())
        self.assertEqual(set(binary.unique().tolist()), {0., 1.})
        self.assertGreater(len(continuous.unique()), 2)
        again = attack(None, [], mode='random', name='purchase').construct_fixed_input(inputs)
        self.assertTrue(torch.equal(binary, again))

    def test_white_image_is_rgb_255_through_the_dataset_transform(self):
        mean, std = (0.4, 0.5, 0.6), (0.2, 0.25, 0.3)
        dataset = SimpleNamespace(transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]))
        fixed = attack(None, [], is_image=True, dataset=dataset).construct_fixed_input(torch.zeros(2, 3, 4, 4))
        expected = ((1 - torch.tensor(mean)) / torch.tensor(std)).view(1, 3, 1, 1).expand(2, 3, 4, 4)
        self.assertTrue(torch.equal(fixed, expected))

    def test_fixedbd_is_the_first_step_with_another_label_and_queries_only_until_then(self):
        # Each sample's largest value is 1, so it moves toward (1, 1): its first feature is a + k/10 (1 - a) at step
        # k, and the label turns to 1 once that feature exceeds 0.6 (mean above 0.8).
        inputs = torch.tensor([[0., 1.], [.25, 1.], [.4, 1.], [.7, 1.]])
        labels = torch.tensor([0, 0, 0, 0])
        model = StepModel(threshold=.8)
        distances = attack(model, [], n_queries=10).fixedbd_over_model(model, inputs, labels)
        # Above 0.6 from k = 7, 5 and 4; the last sample is above it already.
        np.testing.assert_array_equal(distances, [7, 5, 4, 0])
        self.assertEqual(model.queried, sum(d + 1 for d in (7, 5, 4, 0)))

    def test_samples_that_never_change_label_get_k_after_k_queries(self):
        model = StepModel(threshold=2.)
        distances = attack(model, [], n_queries=10).fixedbd_over_model(model, torch.zeros(3, 2),
                                                                       torch.zeros(3, dtype=torch.long))
        np.testing.assert_array_equal(distances, [10, 10, 10])
        self.assertEqual(model.queried, 30)

    def test_batched_scores_equal_scores_of_one_sample_at_a_time(self):
        torch.manual_seed(0)
        models = [torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)).eval()
                  for _ in range(4)]
        dh = attack(models[0], models[1:], n_queries=30)
        inputs, labels = torch.randn(64, 5), torch.randint(0, 3, (64,))
        batched = dh.score_batches(models[0], [(inputs, labels)], torch.device('cpu'))
        single = np.concatenate([dh.score_batches(models[0], [(x[None], y[None])], torch.device('cpu'))
                                 for x, y in zip(inputs, labels)])
        np.testing.assert_array_equal(batched, single)

    def test_tabular_scores_have_no_decisions_and_use_the_given_target(self):
        target, other = StepModel(), StepModel()
        dh = attack(target, [StepModel(), StepModel()])
        scores, decisions = dh.infer_batch(other, torch.tensor([[.2, .2]]), torch.tensor([0]), device='cpu')
        self.assertIsNone(decisions)
        self.assertEqual(scores.shape, (1,))
        self.assertGreater(other.queried, 0)
        self.assertEqual(target.queried, 0)

    def test_score_orders_like_the_relscore_and_keeps_its_ties_at_one_apart(self):
        # Shadow models give 1, 2 and 1 steps; the target 20, 25 and 3; the last sample has shadows that all agree.
        distances = {'target': np.array([20., 25., 3.]), 0: np.array([1., 1., 2.]), 1: np.array([2., 2., 2.]),
                     2: np.array([1., 1., 2.])}
        dh = attack('target', [0, 1, 2])
        dh.fixedbd_over_model = lambda model, inputs, true_labels: distances[model]
        scores = dh.rel_score('target', torch.zeros(3, 2), torch.zeros(3, dtype=torch.long))
        shadows = np.stack([distances[i] for i in range(3)], axis=1)
        relscores = norm.cdf(distances['target'], shadows.mean(1), shadows.std(1) + 1e-30)
        self.assertTrue((relscores == 1.0).all())
        self.assertTrue(scores[1] > scores[0] and np.isfinite(scores[:2]).all())

    def test_each_shadow_model_trains_with_its_own_seed(self):
        settings = build_parser().parse_args(['--attacker_mode', 'dh'])
        configs = generate_configs_from_settings(settings).attacker.train
        seeds = [DHAttack.shadow_train_configs(configs, i).seed for i in range(3)]
        self.assertEqual(seeds, [configs.seed, configs.seed + 1, configs.seed + 2])
        self.assertEqual(DHAttack.shadow_train_configs(configs, 2).batch_size, configs.batch_size)

    def test_random_images_take_the_label_the_target_predicts(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((.5, .5, .5), (.25, .25, .25))])
        target = StepModel(threshold=0.)   # class 1 for images brighter than mid-grey on average, 0 otherwise
        dh = attack(target, [], is_image=True, dataset=[(torch.zeros(3, 4, 4), 0)])
        dh.dataset_transform = lambda: transform
        seen = []
        dh.score_batches = lambda model, batches, device: np.concatenate(
            [np.zeros(len(y)) for _, y in batches if seen.append(y) is None])
        dh.find_optimal_threshold(device='cpu')
        labels = torch.cat(seen)
        images = dh.construct_random_samples((3, 4, 4), len(labels))
        self.assertTrue(torch.equal(labels, target(images).argmax(dim=1)))

    def test_tabular_attack_rule_draws_calibration_samples_and_images_do_not(self):
        # DHAttack has no threshold of its own on tabular data; UBA sets one on random records there too.
        for cls, tabular in ((DHAttack, True), (UnsupervisedBoundaryMIA, False)):
            for im_size, expected in ((None, tabular), (32, False)):
                with self.subTest(attack=cls.__name__, im_size=im_size):
                    instance = object.__new__(cls)
                    instance.base_dataset_configs = SimpleNamespace(im_size=im_size)
                    instance.evaluation_configs = SimpleNamespace(decision_rules=['attack'])
                    self.assertEqual(instance.needs_calibration(), expected)


if __name__ == '__main__':
    unittest.main()
