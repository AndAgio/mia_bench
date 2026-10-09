"""Regression checks for effective noise, weight inference and sampler reuse."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.data.helpers import IndexTrackingMixin, MultiDatasets
from src.mia.defenses.training.weigthed_smoothing import WeightedSmoothingTrainManager, WeightedSmoothingDefender
from src.trainer.train_manager import TrainManager
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.utils.configs import TrainConfigs, OptimizerConfigs, SchedulerConfigs, WeightedSmoothingDefenseConfigs


class CountingDataset(Dataset, IndexTrackingMixin):
    def __init__(self):
        self.reads = 0
        self.targets = torch.tensor([0, 1, 0, 1, 2])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.reads += 1
        return torch.tensor([float(index), 1.0]), self.targets[index], index, index

    def get_all_targets(self, **kwargs):
        raise AssertionError('Normalization must reuse labels from weight inference')


def make_manager():
    manager = WeightedSmoothingTrainManager.__new__(WeightedSmoothingTrainManager)
    manager.model = nn.Sequential(nn.Linear(2, 6), nn.BatchNorm1d(6), nn.Dropout(0.5), nn.Linear(6, 3))
    manager.device = torch.device('cpu')
    manager.weighted_smoothing_configs = SimpleNamespace(sigma_noise=0.8, warmup_epochs=0)
    manager.criterion = nn.CrossEntropyLoss(reduction='none')
    manager.weights = torch.ones(5)
    manager.logger = MagicMock()
    manager.epoch_stats_tracker = MagicMock()
    manager.reset_epoch_stats = MagicMock()
    manager.build_message_for_batch_end = MagicMock(return_value='')
    manager.build_message_for_stage_end = MagicMock(return_value='')
    manager.extra_configs = {}
    manager.epoch = 1
    manager.distributed = False
    manager.seed = 12345
    manager.global_rank = 0
    return manager


class WeightedSmoothingTests(unittest.TestCase):
    def test_defender_requests_final_model(self):
        defender = WeightedSmoothingDefender.__new__(WeightedSmoothingDefender)
        defender.logger = MagicMock()
        defender.name = 'weighted_smoothing'
        defender.weighted_smoothing_configs = SimpleNamespace(warmup_epochs=1)
        defender.dataset = MagicMock()
        defender.untrained_model = nn.Linear(2, 3)
        final_model = nn.Linear(2, 3)
        with patch('src.mia.defenses.training.weigthed_smoothing.WeightedSmoothingTrainManager') as trainer:
            for return_stats in (False, True):
                trainer.return_value.train.return_value = (final_model, 'stats') if return_stats else final_model
                result = defender.train_model(SimpleNamespace(), return_stats=return_stats)
                self.assertIs(result[0] if return_stats else result, final_model)
                self.assertEqual(trainer.return_value.train.call_args.kwargs,
                                 dict(return_best_model=False, return_last_model=True, return_stats=return_stats))

    def test_schedule_requires_smoothing_epochs(self):
        manager = make_manager()
        manager.train_configs = SimpleNamespace(scheduler_config=SimpleNamespace(epochs=2))
        for warmup in (-1, 2, 3):
            manager.weighted_smoothing_configs.warmup_epochs = warmup
            with self.subTest(warmup=warmup), patch.object(TrainManager, 'train') as train:
                with self.assertRaisesRegex(ValueError, 'warmup_epochs < total epochs'):
                    manager.train()
                train.assert_not_called()

    def test_final_model_returned_when_warmup_checkpoint_is_best(self):
        with tempfile.TemporaryDirectory() as folder:
            configs = TrainConfigs(optimizer_config=OptimizerConfigs(name='sgd', lr=0.1),
                                   scheduler_config=SchedulerConfigs(name='const', epochs=2),
                                   batch_size=5, device='cpu', resume=False,
                                   ckpts_folder=Path(folder) / 'ckpts', resume_ckpts_folder=Path(folder) / 'resume')
            manager = WeightedSmoothingTrainManager(configs, 'weighted_smoothing', logger=MagicMock(),
                                                   weighted_smoothing_configs=WeightedSmoothingDefenseConfigs(warmup_epochs=1))
            model = nn.Linear(2, 3)
            model.name = 'linear'
            manager.initialize_train(MultiDatasets(datasets=[CountingDataset()], ids=['train']), model, configs)
            # Force selection of the warmup checkpoint, independent of random accuracy.
            manager.performance_metrics = {'warmup_preferred': lambda preds, targets: float(manager.epoch == 1)}
            manager.metric_to_track = 'warmup_preferred'
            final_model, stats = manager.train(return_stats=True)
            best = torch.load(Path(manager.resume_folder) / 'best.pth', weights_only=False)
            last = torch.load(Path(manager.resume_folder) / 'last.pth', weights_only=False)
            self.assertEqual(stats.best_epoch, 1)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 2)
            for name, value in final_model.state_dict().items():
                torch.testing.assert_close(value, last['model_state'][name])
            self.assertTrue(any(not torch.equal(value, best['model_state'][name])
                                for name, value in final_model.state_dict().items()))

    def test_noise_streams_are_rank_specific_and_resume_at_epoch_boundary(self):
        predictions = torch.zeros(5, 3)
        managers = [make_manager(), make_manager()]
        managers[1].global_rank = 1
        global_rng = torch.get_rng_state().clone()
        first = [manager.sample_prediction_noise(predictions) for manager in managers]
        self.assertFalse(torch.equal(*first))
        torch.testing.assert_close(torch.get_rng_state(), global_rng)
        for rank, manager in enumerate(managers):
            second = manager.sample_prediction_noise(predictions)
            self.assertFalse(torch.equal(first[rank], second))
            repeat = make_manager()
            repeat.global_rank = rank
            torch.testing.assert_close(repeat.sample_prediction_noise(predictions), first[rank])
            torch.testing.assert_close(repeat.sample_prediction_noise(predictions), second)
            manager.epoch = 2
            resumed = make_manager()
            resumed.global_rank = rank
            resumed.epoch = 2
            torch.testing.assert_close(manager.sample_prediction_noise(predictions), resumed.sample_prediction_noise(predictions))

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            def train_run(directory, epochs, resume=False):
                configs = TrainConfigs(optimizer_config=OptimizerConfigs(name='sgd', lr=0.1),
                                       scheduler_config=SchedulerConfigs(name='const', epochs=epochs),
                                       batch_size=5, device='cpu', resume=resume,
                                       ckpts_folder=directory / 'ckpts', resume_ckpts_folder=directory / 'resume')
                manager = WeightedSmoothingTrainManager(configs, 'weighted_smoothing', logger=MagicMock(),
                                                       weighted_smoothing_configs=WeightedSmoothingDefenseConfigs(warmup_epochs=1))
                model = nn.Linear(2, 3)
                model.name = 'linear'
                manager.initialize_train(MultiDatasets(datasets=[CountingDataset()], ids=['train']), model, configs)
                return manager.train()

            uninterrupted = train_run(Path(folder) / 'full', 3)
            train_run(Path(folder) / 'split', 2)
            resumed = train_run(Path(folder) / 'split', 3, resume=True)
            for name, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)

    def test_class_standardization_has_unit_population_variance(self):
        manager = make_manager()
        manager.train_loader = DataLoader(CountingDataset(), batch_size=5)
        manager.all_targets = torch.tensor([0, 0, 0, 1, 2])
        manager.weights = torch.tensor([1.0, 2.0, 3.0, 0.0, 4.0])
        manager.normalize_weights()
        standardized = 1 - manager.weights[:3]
        torch.testing.assert_close(standardized.mean(), torch.tensor(0.0))
        torch.testing.assert_close(standardized.var(correction=0), torch.tensor(1.0))
        torch.testing.assert_close(manager.weights[3:], torch.ones(2))

    def test_confident_mentr_matches_direct_float64_sum(self):
        probs = torch.tensor([[0.9999, 0.0001], [0.99999, 0.00001]], dtype=torch.float32)
        labels = torch.zeros(2, dtype=torch.long)
        reference_probs = probs.double()
        expected = -(1 - reference_probs[:, 0]) * reference_probs[:, 0].log() \
                   - reference_probs[:, 1] * torch.log1p(-reference_probs[:, 1])
        actual = WeightedSmoothingTrainManager.mentr(probs, labels, from_logits=False)
        torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=0.0)

    def test_prediction_noise_ce_loss_and_gradient(self):
        torch.manual_seed(12)
        model = nn.Linear(2, 3)
        inputs, targets = torch.randn(5, 2), torch.tensor([0, 1, 2, 1, 0])
        noise = torch.linspace(-2.0, 2.0, 15).reshape(5, 3)
        results = []
        for sigma in (0.0, 0.8):
            manager = make_manager()
            manager.model = copy.deepcopy(model)
            manager.weighted_smoothing_configs.sigma_noise = sigma
            manager.optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.0)
            with patch.object(manager, 'sample_prediction_noise', return_value=sigma * noise):
                manager.train_step_with_weighted_smoothing(inputs, targets, torch.arange(5))
            outputs = manager.epoch_stats_tracker.update.call_args.kwargs['preds']
            results.append((outputs.detach(), manager.model.weight.grad.clone()))
            reference = copy.deepcopy(model)
            expected = reference(inputs).softmax(dim=1) + sigma * noise
            expected_loss = nn.functional.cross_entropy(expected, targets)
            expected_loss.backward()
            torch.testing.assert_close(outputs, expected)
            torch.testing.assert_close(manager.model.weight.grad, reference.weight.grad)
            torch.testing.assert_close(nn.functional.cross_entropy(outputs, targets), expected_loss)
        plain = model(inputs)
        nn.functional.cross_entropy(plain, targets).backward()
        torch.testing.assert_close(results[0][0], plain.softmax(1))
        self.assertFalse(torch.allclose(results[0][1], model.weight.grad))
        self.assertFalse(torch.allclose(results[1][1], results[0][1]))
        # Noisy predictions remain unconstrained scores for PyTorch CE.
        self.assertTrue((results[1][0] < 0).any())
        self.assertNotAlmostEqual(nn.functional.cross_entropy(results[1][0], targets).item(),
                                  nn.functional.cross_entropy(results[0][0], targets).item())

    def test_weight_inference_preserves_modes_buffers_and_reuses_labels(self):
        manager = make_manager()
        data = CountingDataset()
        manager.train_loader = DataLoader(data, batch_size=2)
        manager.model.train()
        manager.model[2].eval()  # Preserve a deliberately mixed module mode.
        modes = [m.training for m in manager.model.modules()]
        buffers = [b.clone() for b in manager.model.buffers()]
        manager.compute_weights()
        first_weights = manager.weights.clone()
        manager.normalize_weights()
        self.assertEqual(data.reads, len(data))
        self.assertTrue(torch.isfinite(manager.weights).all())
        self.assertEqual(manager.weights[-1].item(), 1.0)  # Singleton class.
        manager.compute_weights()
        torch.testing.assert_close(manager.weights, first_weights)
        torch.testing.assert_close(manager.all_targets, data.targets)
        self.assertEqual(modes, [m.training for m in manager.model.modules()])
        for before, after in zip(buffers, manager.model.buffers()):
            torch.testing.assert_close(before, after)

    def test_weight_inference_restores_mode_on_failure(self):
        manager = make_manager()
        manager.train_loader = DataLoader(CountingDataset(), batch_size=2)
        with patch.object(manager.model, 'forward', side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):
                manager.compute_weights()
        self.assertTrue(all(m.training for m in manager.model.modules()))

    def test_sam_variants_reuse_noise_and_restore_batchnorm(self):
        torch.manual_seed(7)
        inputs, targets = torch.randn(5, 2), torch.tensor([0, 1, 2, 1, 0])
        for optimizer_cls in (SAM, ESAM, WSAM, LookSAM, FriendlySAM):
            for adaptive in (False, True):
                with self.subTest(optimizer=optimizer_cls.__name__, adaptive=adaptive):
                    manager = make_manager()
                    # Keep ESAM's selected batch large enough for BatchNorm.
                    options = {'gamma': 0.6} if optimizer_cls is ESAM else {}
                    manager.optimizer = optimizer_cls(manager.model.parameters(), torch.optim.SGD,
                                                      lr=0.01, adaptive=adaptive, **options)
                    before = [p.clone() for p in manager.model.parameters()]
                    with patch.object(manager, 'sample_prediction_noise', wraps=manager.sample_prediction_noise) as draw:
                        for _ in range(2):
                            manager.train_step_with_weighted_smoothing(inputs, targets, torch.arange(5))
                        self.assertEqual(draw.call_count, 2)
                        self.assertEqual(tuple(draw.call_args.args[0].shape), (5, 3))
                    self.assertEqual(manager.model[1].momentum, 0.1)
                    self.assertFalse(hasattr(manager.model[1], 'backup_momentum'))
                    self.assertTrue(all(torch.isfinite(p).all() for p in manager.model.parameters()))
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, manager.model.parameters())))

    def test_esam_selected_samples_keep_their_original_noise(self):
        manager = make_manager()
        manager.model = nn.Linear(2, 5)
        manager.weights = torch.arange(1, 6, dtype=torch.float)
        manager.optimizer = ESAM(manager.model.parameters(), torch.optim.SGD, lr=0.01, rho=0.0, gamma=0.6)
        calls = []

        def criterion(outputs, targets):
            calls.append((outputs.detach().clone(), targets.clone()))
            return nn.functional.cross_entropy(outputs, targets, reduction='none')

        manager.criterion = criterion
        manager.train_step_with_weighted_smoothing(torch.randn(5, 2), torch.arange(5), torch.arange(5))
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(calls[0][0], calls[1][0])
        self.assertEqual(len(calls[2][1]), 3)
        torch.testing.assert_close(calls[2][0], calls[0][0][calls[2][1]])

    def test_distributed_epoch_uses_rank_sampler(self):
        shards = []
        for rank in (0, 1):
            manager = make_manager()
            data = CountingDataset()
            sampler = DistributedSampler(data, num_replicas=2, rank=rank)
            manager.train_loader = DataLoader(data, batch_size=2, sampler=sampler)
            manager.distributed = True
            manager.compute_weights = MagicMock()
            manager.normalize_weights = MagicMock()
            manager.train_step_with_weighted_smoothing = MagicMock()
            manager.train_epoch()
            seen = torch.cat([call.args[2] for call in manager.train_step_with_weighted_smoothing.call_args_list]).tolist()
            self.assertEqual(seen, list(sampler))
            self.assertEqual(len(seen), 3)  # DistributedSampler pads one sample.
            self.assertEqual(sampler.epoch, manager.epoch)
            self.assertIs(manager.indexed_train_loader, manager.train_loader)
            shards.append(seen)
        self.assertEqual(set(shards[0] + shards[1]), set(range(5)))

    def test_saturated_probabilities_have_finite_mentr(self):
        logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
        self.assertTrue(torch.isfinite(WeightedSmoothingTrainManager.mentr(logits, torch.tensor([0, 0]))).all())


if __name__ == '__main__':
    unittest.main()
