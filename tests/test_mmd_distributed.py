"""Two-process CPU regression checks for the MMD distributed path."""

import multiprocessing
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.trainer.stats_tracker import EpochStats
from tests.test_mmd_index_cache import CountingDataset
from tests.test_mmd_training import OPTIMIZERS, make_manager


class ShiftedDataset(CountingDataset):
    def __getitem__(self, index):
        image, label, original, resampled = super().__getitem__(index)
        return image + torch.tensor([0., 1.]), label, original, resampled


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    try:
        for optimizer in OPTIMIZERS:
            torch.manual_seed(12)
            random.seed(12 + rank)
            model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                model.weight.zero_()
                model.bias.copy_(torch.tensor([2., 0.]))
            manager = make_manager(optimizer, model=DistributedDataParallel(model))
            manager.distributed = True
            manager.global_rank = rank
            manager.world_size = 2
            accuracy_data = CountingDataset([0, 1, 0, 1])
            sampler = DistributedSampler(accuracy_data, num_replicas=2, rank=rank, shuffle=False)
            manager.train_loader = DataLoader(accuracy_data, batch_size=2, sampler=sampler)
            manager.val_loader = manager.train_loader
            # Rank 0 is locally 100% correct, rank 1 is 0%: both must report 50%.
            assert manager.get_train_accuracy() == 0.5
            assert manager.get_val_accuracy() == 0.5
            # Padding must not turn the true 3/5 accuracy into 4/6.
            uneven = CountingDataset([0, 1, 0, 1, 0])
            for shuffle in (False, True):
                uneven_sampler = DistributedSampler(uneven, num_replicas=2, rank=rank, shuffle=shuffle)
                manager.train_loader = DataLoader(uneven, batch_size=2, sampler=uneven_sampler)
                manager.val_loader = manager.train_loader
                assert manager.get_train_accuracy() == 0.6
                assert manager.get_val_accuracy() == 0.6
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.1]]))
            before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()

            train = CountingDataset([0, 1, 0, 1, 0])
            manager.dataset = {'train': train, 'val': ShiftedDataset([0, 1])}
            manager.epoch_stats_tracker = EpochStats()
            manager.reset_epoch_stats = lambda phase: manager.epoch_stats_tracker.stage_begin(phase)
            seen_samplers = []

            def record_loader(dataset, *args, **kwargs):
                if dataset is train:
                    seen_samplers.append(kwargs['sampler'])
                return DataLoader(dataset, *args, **kwargs)

            with patch('src.mia.defenses.training.mmd.DataLoader', side_effect=record_loader):
                manager.train_with_mmd_distance()
            assert list(seen_samplers[0]) == ([0, 2, 4] if rank == 0 else [1, 3, 0])
            summary = manager.epoch_stats_tracker.finalize_epoch(1).stages['mmd']
            assert summary.num_batches == 2
            assert 'mmd_loss' in summary.metrics
            parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
            gathered = [torch.empty_like(parameters) for _ in range(2)]
            dist.all_gather(gathered, parameters)
            assert torch.isfinite(parameters).all()
            assert not torch.equal(parameters, before)
            assert torch.allclose(gathered[0], gathered[1])

        # Only rank 1 encounters the absent class; rank 0 must fail alongside it.
        manager.dataset = {'train': accuracy_data, 'val': CountingDataset([0])}
        try:
            manager.train_with_mmd_distance()
        except ValueError as error:
            assert 'no samples for training class' in str(error)
        else:
            raise AssertionError('Missing validation class did not raise on every rank')
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is unavailable')
class MmdDistributedTests(unittest.TestCase):
    def test_global_accuracy_sharded_updates_and_coordinated_errors(self):
        with tempfile.TemporaryDirectory(prefix='mmd-ddp-') as folder:
            context = multiprocessing.get_context('spawn')
            processes = [context.Process(target=distributed_worker,
                                         args=(rank, str(Path(folder) / 'rendezvous')))
                         for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=30)
                    self.assertFalse(process.is_alive(), 'DDP check hung')
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
