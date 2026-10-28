"""Purifier regressions with small deterministic models and no dataset downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.defenses.post_hoc.purifier import ConfidenceReformer, PurifiedModelWrapper, PurifierDefender, TorchLabelSwapper
from src.utils.configs import PurifierDefenseConfigs, generate_configs_from_settings
from src.utils.settings import build_parser


class EchoReformer(torch.nn.Module):
    # Returns the classifier confidences unchanged, so outputs are deterministic.
    latent_dim = 1

    def forward(self, c, l_onehot, noise=None):
        return c.clone()


class IndexedData(Dataset):
    # Yields the repository's (x, y, original index, current index) samples.
    def __init__(self, inputs, labels, original_indices):
        self.inputs, self.labels, self.original_indices = inputs, labels, original_indices

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], self.original_indices[index], index


def wrapper(Pindex, threshold=0.01, reformer=None, noise_quantization=1e-3):
    swapper = TorchLabelSwapper(Pindex=Pindex)
    swapper.set_device(torch.device('cpu'))
    model = PurifiedModelWrapper(model=torch.nn.Identity(), reformer=reformer or EchoReformer(),
                                 label_swapper=swapper, threshold=threshold, noise_quantization=noise_quantization)
    return model.eval()


def reformer_settings(*argv):
    # (h, latent, epochs, lr) of the reformer that the command line selects.
    configs = generate_configs_from_settings(build_parser().parse_args([*argv, '--defender_mode', 'purifier']))
    defense = configs.defender.defense
    return defense.reformer_hidden_dim, defense.reformer_latent_dim, defense.reformer_epochs, defense.reformer_lr


def defender(splits, seed=0):
    defender = object.__new__(PurifierDefender)
    defender.logger = Mock()
    defender.dataset = Mock()
    defender.dataset.get.side_effect = splits.__getitem__
    defender.dataset_configs = SimpleNamespace(seed=seed)
    defender.trained_model = torch.nn.Identity()
    defender.device = torch.device('cpu')
    return defender


class PurifierTests(unittest.TestCase):
    def test_forward_obeys_logits_contract_including_extreme_inputs(self):
        logits = torch.tensor([[1., 0., 0.], [1000., -1000., 0.]])
        model = wrapper(Pindex=[torch.tensor([0., 1., 0.]).log_softmax(0)])
        probabilities = model.defended_probabilities(logits)
        output = model(logits)
        self.assertTrue(output.isfinite().all())
        torch.testing.assert_close(output.softmax(1), probabilities)
        torch.testing.assert_close(output.argmax(1), logits.argmax(1))

    def test_matched_member_gets_top_two_scores_exchanged(self):
        logits = torch.tensor([[2., 1., 0.], [0., 0., 3.]])
        c = logits.softmax(1)
        model = wrapper(Pindex=[logits[0].log_softmax(0)])
        probabilities = model.defended_probabilities(logits)
        torch.testing.assert_close(probabilities[0], c[0][[1, 0, 2]])
        torch.testing.assert_close(probabilities[1], c[1])
        self.assertTrue(model(logits).isfinite().all())

    def test_saturated_confidences_of_other_inputs_do_not_match(self):
        member, other = torch.tensor([[30., 0., 0.]]), torch.tensor([[40., 0., 0.]])
        self.assertLess((member.softmax(1) - other.softmax(1)).norm().item(), 1e-12)
        model = wrapper(Pindex=[member[0].log_softmax(0)])
        torch.testing.assert_close(model.defended_probabilities(other), other.softmax(1))
        self.assertEqual(model.defended_probabilities(member).argmax(1).item(), 1)

    def test_empty_swap_set_leaves_outputs_unchanged(self):
        logits = torch.tensor([[2., 1., 0.]])
        model = wrapper(Pindex=torch.empty(0, 3))
        torch.testing.assert_close(model.defended_probabilities(logits), logits.softmax(1))

    def test_swap_set_follows_paper_rate_from_val_accuracy_and_never_reads_test(self):
        train_logits = torch.randn(20, 3) * 5
        train = IndexedData(train_logits, train_logits.argmax(1), torch.arange(100, 120))
        val_logits = torch.randn(10, 3)
        val_labels = val_logits.argmax(1)
        val_labels[:4] = (val_labels[:4] + 1) % 3
        val = IndexedData(val_logits, val_labels, torch.arange(10))
        model = defender({'train': train, 'val': val})
        Pindex = model.build_pindex()
        # acc_train = 1.0, acc_test = 0.6 -> p_swap = 0.4 -> 8 of the 20 members
        positions = [index - 100 for index in model.swap_indices]
        self.assertEqual(len(set(positions)), 8)
        torch.testing.assert_close(Pindex, train_logits[positions].log_softmax(1))
        self.assertNotIn('test', [call.args[0] for call in model.dataset.get.call_args_list])

    def test_reformer_noise_survives_training(self):
        torch.manual_seed(0)
        logits = torch.randn(33, 3) * 4
        model = defender({'val': IndexedData(logits, logits.argmax(1), torch.arange(33))})
        model.purifier_configs = SimpleNamespace(reformer_batch_size=16, reformer_lr=0.01,
                                                 reformer_epochs=200, reformer_lambda=0.1)
        model.reformer = ConfidenceReformer(num_classes=3, latent_dim=2, hidden=16)
        model.train_reformer()  # the last batch holds a single sample
        model.reformer.eval()
        c = logits.softmax(1)
        l = torch.nn.functional.one_hot(c.argmax(1), 3).float()
        with torch.no_grad():
            outputs = torch.stack([model.reformer(c, l) for _ in range(20)])
        self.assertGreater(outputs.std(0).max(-1).values.mean().item(), 2e-2)

    def test_query_noise_is_repeatable_and_batch_independent(self):
        torch.manual_seed(0)
        model = wrapper(Pindex=torch.empty(0, 3), reformer=ConfidenceReformer(num_classes=3, latent_dim=2, hidden=8))
        logits = torch.tensor([[2., 1., 0.], [0., 1., 0.5], [1., 0., 3.]])
        torch_state = torch.random.get_rng_state().clone()
        batched = model.defended_probabilities(logits)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        torch.rand(20)
        torch.testing.assert_close(model.defended_probabilities(logits), batched)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model.defended_probabilities(logits[order]), batched[order])
        torch.testing.assert_close(torch.cat([model.defended_probabilities(row[None]) for row in logits]), batched)
        noise = model.query_noise(logits)
        self.assertFalse(torch.equal(noise[0], noise[1]))

    def test_query_noise_is_stable_within_the_quantization_width(self):
        model = wrapper(Pindex=torch.empty(0, 2), reformer=ConfidenceReformer(num_classes=2, latent_dim=4),
                        noise_quantization=0.01)
        x = torch.tensor([[1.001, 0.001]])
        torch.testing.assert_close(model.query_noise(x), model.query_noise(x + 0.001))
        self.assertFalse(torch.equal(model.query_noise(x), model.query_noise(x + 0.02)))

    def test_defend_model_keeps_a_torch_device(self):
        # get_device expects a config string: it would turn torch.device('cpu') into CUDA/MPS
        # and torch.device('cuda:0') into 'cuda:cuda:0'.
        torch.manual_seed(0)
        logits = torch.randn(24, 3) * 4
        model = defender({'train': IndexedData(logits[:16], logits[:16].argmax(1), torch.arange(16)),
                          'val': IndexedData(logits[16:], torch.zeros(8, dtype=torch.long), torch.arange(8))})
        model.purifier_configs = PurifierDefenseConfigs(reformer_epochs=2, reformer_batch_size=8)
        model.reformer = ConfidenceReformer(num_classes=3)
        model.label_swapper = TorchLabelSwapper()
        defended = model.defend_model(torch.device('cpu')).eval()
        self.assertEqual(model.device, torch.device('cpu'))
        self.assertEqual(defended(logits).device.type, 'cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is unavailable')
    def test_query_noise_transfers_from_mps_before_float64_conversion(self):
        model = wrapper(Pindex=torch.empty(0, 2), reformer=ConfidenceReformer(num_classes=2, latent_dim=4))
        query = torch.tensor([[1.001, 0.001]], dtype=torch.float32)
        actual = model.query_noise(query.to('mps'))
        self.assertEqual(actual.device.type, 'mps')
        torch.testing.assert_close(actual.cpu(), model.query_noise(query))

    def test_unset_reformer_settings_follow_the_paper_for_the_dataset(self):
        tabular = ['--defender_model', 'tabular_mlp', '--attacker_model', 'tabular_mlp']
        self.assertEqual(reformer_settings('--dataset', 'cifar100'), (512, 100, 300, 0.0005))
        self.assertEqual(reformer_settings('--dataset', 'purchase', *tabular), (128, 20, 150, 0.0001))
        # Datasets the paper does not evaluate take CIFAR-10's settings.
        self.assertEqual(reformer_settings('--dataset', 'svhn'), (32, 2, 100, 0.01))
        self.assertEqual(reformer_settings('--dataset', 'cifar100', '--defender_purifier_reformer_epochs', '5')[2], 5)


if __name__ == '__main__':
    unittest.main()
