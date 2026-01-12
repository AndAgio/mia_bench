import os
import pathlib
import sys
import math
import time
import copy
from typing import Tuple, Union, Callable, Any
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from src.data.multi import MultiDatasets
from src.models import get_model
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts
from src.utils.configs import TrainConfigs, ModelConfigs
from src.utils import convert_to_hms
from src.trainer.stats_tracker import TrainStats, EpochStats
from src.trainer.checkpoints import CheckpointManager
from src.trainer.metrics import get_performance_metric_func
from src.trainer.distributed import maybe_init_ddp, is_rank0, ddp_barrier, maybe_cleanup_ddp
from src.utils.log import Loggable, SmartLogger, DumbLogger


class TrainManager(Loggable):
    def __init__(self, train_configs: TrainConfigs, name: str, logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        self.train_configs = train_configs
        self.name = name

        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)

    
    def reset_config(self, configs: TrainConfigs):
        self.train_configs = configs
        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)
    

    def setup_folders(self, train_configs: TrainConfigs):
        models_folder = os.path.join(train_configs.ckpts_folder, self.name)
        self.logger.print_it(f'Setting up checkpoints folder to {models_folder}')
        os.makedirs(models_folder, exist_ok=True)
        self.models_folder = models_folder
        self.ckpts_folder = train_configs.ckpts_folder
        resume_folder = os.path.join(train_configs.resume_ckpts_folder, self.name)
        self.logger.print_it(f'Setting up resume folder to {resume_folder}')
        os.makedirs(resume_folder, exist_ok=True)
        self.resume_folder = resume_folder


    def set_devices_and_seed(self, train_configs: TrainConfigs):
        self._setup_distributed_training(distributed=train_configs.distributed,
                                        device=train_configs.device)
        self._setup_seed(seed=train_configs.seed)


    def _setup_distributed_training(self, distributed: bool = False, device: str = 'cpu'):
        self.global_rank, self.world_size, self.local_rank, self.device = maybe_init_ddp(use_ddp=distributed, device=device)
        self.distributed = True if (self.world_size > 1 and distributed) else False

    def _setup_seed(self, seed: int = 12345):
        # Set random seed for initialization
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # For CUDA:
        torch.cuda.manual_seed_all(seed)
        # For MPS (Apple Silicon):
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        # cuDNN flags (safe even if CUDA is not available):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.seed = seed

    def setup_model_from_configs(self, model_configs: ModelConfigs):
        self.setup_model_from_name_and_info(model_name=model_configs.model_name,
                                            im_channels=model_configs.im_channels,
                                            num_classes=model_configs.num_classes,
                                            im_size=model_configs.im_size)


    def setup_model_from_name_and_info(self, model_name: str, im_channels: int, num_classes: int, im_size: Tuple[int, ...]):
        self.logger.print_it('Setting up model "{}"...'.format(model_name))
        # Setup model
        model = get_model(model_name=model_name,
                        im_channels=im_channels,
                        num_classes=num_classes,
                        im_size=im_size,
                        logger=self.logger)
        # Move model to device
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.device)
        self.model_name = model_name
        self.logger.print_it('Model setup done!')

    def set_model(self, model: nn.Module):
        self.logger.print_it('Setting up model "{}"...'.format(model.name))
        self.model = copy.deepcopy(model)
        # Move model to device
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.device)
        self.model_name = model.name
        self.logger.print_it('Model setup done!')

    def get_model_name(self):
        return self.model_name

    def setup_loss(self, loss: Union[str, Callable]):
        if isinstance(loss, str):
            self.logger.print_it('Setting up {} loss...'.format(loss))
            if loss == 'crossentropy':
                self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
            else:
                print('Specified loss "{}" not recognized!'.format(loss))
        elif isinstance(loss, Callable) or isinstance(loss, torch.nn.Module):
            self.logger.print_it('Setting up loss directly to function {}...'.format(loss))
            self.criterion = loss

    def setup_performance_metrics(self, metrics: list[Union[str, Callable]], metric_to_track: str = None):
        self.performance_metrics = {}
        for metric in metrics:
            if isinstance(metric, str):
                self.logger.print_it('Setting up {} metric...'.format(metric))
                self.performance_metrics[metric] = get_performance_metric_func(metric_name=metric)
            elif isinstance(metric, Callable):
                self.logger.print_it('Setting up performance metric directly to function {}...'.format(metric))
                self.performance_metrics[metric.__name__] = metric
            else:
                raise ValueError('Metrics to track should be either strings or callable functions!')
        if metric_to_track is None:
            self.metric_to_track = 'loss'
        else:
            assert metric_to_track in list(self.performance_metrics.keys()), f'Metric to track should be among tracked metrics. Found "{metric_to_track}" and {list(self.performance_metrics.keys())}!'
            self.metric_to_track = metric_to_track

    def setup_optimizer(self, optimizer: str, lr: float = None):
        self.logger.print_it('Setting up "{}" optimizer...'.format(optimizer))
        if optimizer == 'adam':
            self.optimizer = Adam(params=self.model.parameters(),
                                lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = SGD(params=self.model.parameters(), 
                                lr=lr,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'sam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = SAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=lr,
                                rho=0.05,
                                adaptive=adaptive,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'esam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = ESAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=lr,
                                rho=0.05,
                                beta=1,
                                gamma=0.5,
                                adaptive=adaptive,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'wsam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = WSAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=lr,
                                rho=0.05,
                                gamma=0.9,
                                sam_eps=1e-12,
                                adaptive=adaptive,
                                decouple=True,
                                max_norm=None,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'looksam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = LookSAM(params=self.model.parameters(),
                                    base_optimizer=SGD,
                                    rho=0.05,
                                    k=10,
                                    alpha=0.7,
                                    adaptive=adaptive,
                                    use_gc=False,
                                    perturb_eps=1e-12,
                                    nesterov=False,
                                    weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'friendlysam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = FriendlySAM(params=self.model.parameters(),
                                        base_optimizer=SGD,
                                        rho=0.05,
                                        sigma=1,
                                        lmbda=0.9,
                                        adaptive=adaptive,
                                        perturb_eps=1e-12,
                                        momentum=0.9,
                                        nesterov=False,
                                        weight_decay=0.0005)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(optimizer))

    def setup_lr_scheduler(self, lr_sched: str, epochs: int = None, lr: float = None):
        self.logger.print_it('Setting up "{}" learning rate scheduler...'.format(lr_sched))
        if lr_sched == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif lr_sched == 'warmup_step':
            assert epochs is not None and epochs > 0
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(epochs/3), gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif lr_sched == 'warmup_exp':
            assert epochs is not None and epochs > 0
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif lr_sched == 'warmup_cosine':
            assert epochs is not None and epochs > 0
            assert lr is not None and 0 < lr < 1
            cycle_steps = math.ceil(epochs/5)
            warmup_steps = math.ceil(cycle_steps/10)
            max_lr=self.settings.lr
            min_lr=max_lr/100
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=cycle_steps, cycle_mult=1.0, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=0.5)
        elif lr_sched == 'step':
            assert epochs is not None and epochs > 0
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(epochs/3), gamma=0.1)
        elif lr_sched == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        elif lr_sched == 'cosine':
            assert epochs is not None and epochs > 0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        else:
            raise ValueError('Learning rate scheduler "{}" not available!'.format(lr_sched))

    def setup_training(self, 
                        loss: Union[str, Callable],
                        metrics: list[Union[str, Callable]],
                        metric_to_track: str,
                        optimizer: str,
                        lr: float,
                        lr_sched: str,
                        epochs: int):
        self.logger.print_it('Setting up training...')
        self.setup_loss(loss=loss)
        self.setup_performance_metrics(metrics=metrics,
                                    metric_to_track=metric_to_track)
        self.setup_optimizer(optimizer=optimizer,
                            lr=lr,)
        self.setup_lr_scheduler(lr_sched=lr_sched,
                                epochs=epochs, 
                                lr=lr)
        self.logger.print_it('Training setup done!')

    def setup_dataloaders_from_multidatasets(self, dataset: MultiDatasets, batch_size: int = 128):
        try:
            train_dataset = dataset.get('train')
            self.run_train = True
        except:
            raise ValueError('At least the train split should be in the dataset!')
        try:
            test_dataset = dataset.get('test')
            self.run_test = True
        except:
            self.run_test = False
        if self.distributed:
            if self.run_train:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(train_dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=True))
            if self.run_test:
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                                pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(test_dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=False))
        else:
            if self.run_train:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            if self.run_test:
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def setup_dataloaders_from_torch_dataset(self, dataset: Dataset, batch_size: int = 128, split: bool = False):
        self.run_train = True
        if split:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
            if self.distributed:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                    pin_memory=True, shuffle=False,
                                                    sampler=DistributedSampler(train_dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=True))
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                                    pin_memory=True, shuffle=False,
                                                    sampler=DistributedSampler(test_dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=False))
            else:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            self.run_test = True
        else:
            self.run_test = False
            if self.distributed:
                self.train_loader = DataLoader(dataset, batch_size=batch_size,
                                                    pin_memory=True, shuffle=False,
                                                    sampler=DistributedSampler(dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=True))
            else:
                self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def setup_dataloaders(self, dataset: Union[MultiDatasets, Dataset], batch_size: int = 128):
        if isinstance(dataset, MultiDatasets):
            self.setup_dataloaders_from_multidatasets(dataset=dataset,
                                                        batch_size=batch_size)
        elif isinstance(dataset, Dataset):
            self.setup_dataloaders_from_torch_dataset(dataset=dataset,
                                                        batch_size=batch_size,
                                                        split=False)


    def initialize_train(self, 
                        dataset: Union[MultiDatasets, Dataset],
                        model: Union[ModelConfigs,nn.Module],
                        configs: TrainConfigs,
                        ):
        self.logger.print_it('Initializing training...')

        if not configs.__eq__(self.train_configs):
            self.logger.print_it('Found different training configurations in the initialize_train method. Resetting the trainer configs...')
            self.reset_configs(configs)

        self.setup_dataloaders(dataset=dataset,
                                batch_size=self.train_configs.batch_size)
        if isinstance(model, nn.Module):
            self.set_model(model)
        elif isinstance(model, ModelConfigs):
            self.setup_model_from_configs(model_configs=model)
        else:
            raise ValueError('Not recognizing model given!')
        self.setup_training(loss=self.train_configs.loss,
                            metrics=self.train_configs.metrics,
                            metric_to_track=self.train_configs.metric_to_track,
                            optimizer=self.train_configs.optimizer,
                            lr=self.train_configs.lr,
                            lr_sched=self.train_configs.lr_sched,
                            epochs=self.train_configs.epochs)
        self.logger.print_it('Training initialization completed!')


    def reset_running_stats(self):
        best_record = np.inf if self.metric_to_track in ["loss", "mse", "mae", "rmse"] else 0
        mode_to_track_best = "min" if self.metric_to_track in ["loss", "mse", "mae", "rmse"] else "max"
        stage_to_track_best = "test" if self.run_test else "train"
        self.train_stats_tracker = TrainStats(best_epoch=0,
                                            best_record=best_record,
                                            metric_to_track_best=self.metric_to_track,
                                            stage_to_track_best=stage_to_track_best,
                                            mode_to_track_best=mode_to_track_best)

    
    def reset_epoch_stats(self, phase: str = 'train'):
        metrics = {"loss": self.criterion}
        for met, met_fn in self.performance_metrics.items():
            metrics[met] = met_fn
        self.epoch_stats_tracker.stage_begin(stage=phase,
                                            metrics=metrics,
                                            sync_cuda=True,
                                            sync_mps=True)
    

    def train(self, extra_configs: dict[str, Any] = None, return_best_model: bool = True, return_last_model: bool = False, return_stats: bool = False):
        self.extra_configs = extra_configs
        
        self.amp_enabled = bool(self.train_configs.use_grad_scaling) and torch.cuda.is_available()
        self.grad_scaler = torch.amp.GradScaler(device=self.train_configs.device,
                                            enabled=self.amp_enabled)
        

        # Checkpoint manager works in both single and DDP
        self.ckpts_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_dir=self.resume_folder, # self.ckpts_folder, #self.train_configs.ckpts_folder,
            scheduler=self.scheduler,
            scaler=self.grad_scaler if self.amp_enabled else None,  # persist scaler only if AMP is enabled
            logger=self.logger,
        )

        # Stats
        self.reset_running_stats()

        if self.train_configs.resume:
            resume_ckpt = self.ckpts_manager.resume(latest=True)
            if resume_ckpt is not None:
                self.epoch = int(resume_ckpt['epoch']) + 1
                self.train_stats_tracker.load_from_checkpoint_dict(resume_ckpt.get("train_stats", {}))
            else:
                self.epoch = 1
        else:
            self.epoch = 1
        
        self.train_stats_tracker.start_timer()
        self.best_perf = self.train_stats_tracker.get_best()
        self.epoch_stats_tracker = EpochStats()

        while(self.epoch <= self.train_configs.epochs):
            self.epoch_stats_tracker.epoch_start()
            self.train_epoch()
            if self.run_test:
                self.test_epoch()
            self.scheduler.step()

            self.epoch_stats_tracker.epoch_end()
            self.epoch_stats_tracker.ddp_consolidate_epoch_time(op="max")
            epoch_time = self.epoch_stats_tracker.get_epoch_time()
            # Finalize epoch and checkpoint (rank-0)
            epoch_summary = self.epoch_stats_tracker.finalize_epoch(epoch=self.epoch)
            best_epoch, new_best = self.train_stats_tracker.update_history_and_best(epoch_summary=epoch_summary)
            
            self.logger.print_it(f'Best epoch so far is {best_epoch}: {'test' if self.run_test else 'train'} {self.metric_to_track} = {new_best:.4f}')
            
            if self.local_rank == 0:
                ckpt = self.ckpts_manager.build_checkpoint(epoch=self.epoch,
                                                        train_stats=self.train_stats_tracker)
                self.ckpts_manager.save(name='last.pth',
                                        ckpt=ckpt)
                if best_epoch == self.epoch:
                    self.best_perf = new_best
                    self.ckpts_manager.save(name='best.pth',
                                            ckpt=ckpt)

            ddp_barrier()

            self.epoch += 1

        maybe_cleanup_ddp()

        h, m, s = convert_to_hms(self.train_stats_tracker.total_time())
        self.logger.print_it('Training for "{}" with seed {} completed in: {}:{:02d}:{:02d}'.format(self.model.name, self.seed, h, m, s))

        best_model = self.ckpts_manager.load_best_model()
        if return_best_model:
            if return_last_model:
                if return_stats:
                    return best_model, self.model, self.train_stats_tracker
                else:
                    return best_model, self.model
            else:
                if return_stats:
                    return best_model, self.train_stats_tracker
                else:
                    return best_model
        else:
            if return_last_model:
                if return_stats:
                    return self.model, self.train_stats_tracker
                else:
                    return self.model
            else:
                if return_stats:
                    return self.train_stats_tracker
                else:
                    return


    def train_epoch(self):
        self.reset_epoch_stats(phase='train')
        self.model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.train_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.train_loader))
        self.logger.set_logger_newline()
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()
        return train_summary

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            assert not self.amp_enabled, f'GradScaler for SAM and SMA-like optimizers not yet implemented!' 
            # TODO: Implement gradscaler for SAM-like optimizers.
            # Issue URL: https://github.com/AndAgio/mia_bench/issues/10
            # assignees: AndAgio

            # Working with closure
            def closure(inputs, targets, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            self.optimizer.step(closure, inputs, targets)
            self.optimizer.zero_grad()
            loss, outputs = self.optimizer.get_first_closure_outputs()
            # An alternative to running with closure is to run manually both steps of the sam-like optimizers, like the following.
            # # first forward-backward step
            # enable_running_stats(self.model)
            # outputs = self.model(inputs)
            # loss = self.criterion(outputs, targets).mean()
            # loss.backward()
            # self.optimizer.first_step(zero_grad=True)
            # # second forward-backward step
            # disable_running_stats(self.model)
            # self.criterion(self.model(inputs), targets).mean().backward()
            # self.optimizer.second_step(zero_grad=True)
        else:
            # Forward propagation, compute loss, get predictions
            if not self.amp_enabled:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            else:
                # AMP path: autocast + GradScaler
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            loss = loss.mean()
            # TODO: Double-check that loss.mean() is ok with gradscaler.
            # Issue URL: https://github.com/AndAgio/mia_bench/issues/9
            # assignees: AndAgio
            loss.backward() if not self.amp_enabled else self.grad_scaler.scale(loss).backward()
            self.optimizer.step() if not self.amp_enabled else self.grad_scaler.step(self.optimizer)
            if self.amp_enabled:
                self.grad_scaler.update()
        
        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        # Print message on console
        self.print_message(index_batch=batch_idx+1, total_batches=total_batches)


    def test_epoch(self):
        self.reset_epoch_stats(phase='test')
        self.model.eval()
        if self.distributed:
            self.test_loader.sampler.set_epoch(self.epoch)
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.test_loader))
        self.logger.set_logger_newline()

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        test_summary = self.epoch_stats_tracker.stage_end()
        return test_summary

    
    def test_step(self, inputs, targets, batch_idx=0, total_batches=0):
        self.epoch_stats_tracker.batch_start()
        # Map to available device
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        # Forward propagation, compute loss, get predictions
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss = loss.mean()

        self.epoch_stats_tracker.update(preds=outputs, targets=targets)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        # Print message on console
        self.print_message(index_batch=batch_idx+1,
                            total_batches=total_batches)

    def print_message(self, index_batch, total_batches):
        message = f'{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.epochs} |'
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        if is_rank0():
            metrics = self.epoch_stats_tracker._require_active_stage().current_avgs()
        if metrics is not None:
            train_metrics_message = ''
            index = 0
            for metric_name, metric_value in metrics.items():
                if metric_name in ['batch_time_sec', 'samples_per_sec']:
                    index += 1
                    continue
                train_metrics_message += '{}={:.5f}{} '.format(metric_name, metric_value,
                                                            ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += train_metrics_message
        message += '|'
        h,m,s = convert_to_hms(self.epoch_stats_tracker.get_current_running_time())
        message += ' Epoch time {}:{:02d}:{:02d} |'.format(h,m,s)
        h,m,s = convert_to_hms(self.train_stats_tracker.get_current_running_time())
        message += ' Total time {}:{:02d}:{:02d} |'.format(h,m,s)
        self.logger.print_it_same_line(message)