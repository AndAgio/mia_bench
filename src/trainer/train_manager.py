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
from src.utils.configs import TrainConfigs, ModelConfigs, OptimizerConfigs, SchedulerConfigs
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
        # Simple time-based profiler (uses time.time)
        # `profiler_epoch` stores per-epoch accumulations, `profiler_total` stores totals across training
        self.profiler_epoch: dict[str, float] = {}
        self.profiler_total: dict[str, float] = {}

    
    def reset_configs(self, configs: TrainConfigs):
        self.train_configs = configs
        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)
    

    def setup_folders(self, train_configs: TrainConfigs):
        models_folder = os.path.join(train_configs.ckpts_folder, self.name)
        self.logger.print_it(f"Setting up checkpoints folder to {models_folder}")
        os.makedirs(models_folder, exist_ok=True)
        self.models_folder = models_folder
        self.ckpts_folder = train_configs.ckpts_folder
        resume_folder = os.path.join(train_configs.resume_ckpts_folder, self.name)
        self.logger.print_it(f"Setting up resume folder to {resume_folder}")
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
            self.model = DDP(model, device_ids=[self.local_rank])
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
            assert metric_to_track in list(self.performance_metrics.keys()), f"Metric to track should be among tracked metrics. Found '{metric_to_track}' and {list(self.performance_metrics.keys())}!"
            self.metric_to_track = metric_to_track

    def setup_optimizer(self, opt_cfg: OptimizerConfigs):
        self.logger.print_it('Setting up "{}" optimizer...'.format(opt_cfg.name))
        if opt_cfg.name == 'adam':
            self.optimizer = Adam(params=self.model.parameters(),
                                lr=opt_cfg.lr)
            self.criterion.reduction = 'mean'
        elif opt_cfg.name == 'sgd':
            self.optimizer = SGD(params=self.model.parameters(),
                                lr=opt_cfg.lr,
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
            self.criterion.reduction = 'mean'
        elif opt_cfg.name.split('_')[-1] == 'sam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = SAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=opt_cfg.lr,
                                rho=0.05,
                                adaptive=adaptive,
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'esam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = ESAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=opt_cfg.lr,
                                rho=0.05,
                                beta=1,
                                gamma=0.5,
                                adaptive=adaptive,
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'wsam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = WSAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=opt_cfg.lr,
                                rho=0.05,
                                gamma=0.9,
                                sam_eps=1e-12,
                                adaptive=adaptive,
                                decouple=True,
                                max_norm=None,
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'looksam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = LookSAM(params=self.model.parameters(),
                                    base_optimizer=SGD,
                                    rho=0.05,
                                    k=10,
                                    alpha=0.7,
                                    adaptive=adaptive,
                                    use_gc=False,
                                    perturb_eps=1e-12,
                                    nesterov=opt_cfg.nesterov,
                                    weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'friendlysam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = FriendlySAM(params=self.model.parameters(),
                                        base_optimizer=SGD,
                                        rho=0.05,
                                        sigma=1,
                                        lmbda=0.9,
                                        adaptive=adaptive,
                                        perturb_eps=1e-12,
                                        momentum=opt_cfg.momentum,
                                        nesterov=opt_cfg.nesterov,
                                        weight_decay=opt_cfg.weight_decay)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(opt_cfg.name))

    def setup_lr_scheduler(self, sched_cfg: SchedulerConfigs):
        self.logger.print_it('Setting up "{}" learning rate scheduler...'.format(sched_cfg.name))
        if sched_cfg.name == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif sched_cfg.name == 'warmup_step':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(sched_cfg.epochs/3), gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(sched_cfg.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif sched_cfg.name == 'warmup_exp':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(sched_cfg.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif sched_cfg.name == 'warmup_cosine':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            assert sched_cfg.lr is not None and 0 < sched_cfg.lr < 1
            cycle_steps = math.ceil(sched_cfg.epochs/5)
            warmup_steps = math.ceil(cycle_steps/10)
            max_lr=sched_cfg.lr
            min_lr=max_lr/100
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=cycle_steps, cycle_mult=1.0, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=0.5)
        elif sched_cfg.name == 'step':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(sched_cfg.epochs/3), gamma=0.1)
        elif sched_cfg.name == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        elif sched_cfg.name == 'cosine':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, sched_cfg.epochs)
        else:
            raise ValueError('Learning rate scheduler "{}" not available!'.format(sched_cfg.name))

    def setup_training(self):
        self.logger.print_it('Setting up training...')
        self.setup_loss(loss=self.train_configs.loss)
        self.setup_performance_metrics(metrics=self.train_configs.metrics,
                                        metric_to_track=self.train_configs.metric_to_track)
        self.setup_optimizer(opt_cfg=self.train_configs.optimizer_config)
        self.setup_lr_scheduler(sched_cfg=self.train_configs.scheduler_config)
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
        self.setup_training()
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
        # Reset global profiler totals when starting a new training run
        self.profiler_total = {}

    
    def reset_epoch_stats(self, phase: str = 'train'):
        metrics = {"loss": self.criterion}
        for met, met_fn in self.performance_metrics.items():
            metrics[met] = met_fn
        self.epoch_stats_tracker.stage_begin(stage=phase,
                                            metrics=metrics,
                                            sync_cuda=True,
                                            sync_mps=True)

    # --- Simple profiler helpers (time.time based) ---
    def _profiler_reset_epoch(self):
        self.profiler_epoch = {}

    def _profiler_add(self, label: str, seconds: float):
        self.profiler_epoch[label] = self.profiler_epoch.get(label, 0.0) + float(seconds)
        self.profiler_total[label] = self.profiler_total.get(label, 0.0) + float(seconds)

    def _profiler_log_epoch_summary(self):
        if is_rank0():
            if not self.profiler_epoch:
                return
            lines = [f"Profiling summary for epoch {self.epoch}:"]
            for k, v in sorted(self.profiler_epoch.items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v:.4f}s")
            self.logger.print_it('\n'.join(lines))
    

    def train(self, extra_configs: dict[str, Any] = None, return_best_model: bool = True, return_last_model: bool = False, return_stats: bool = False):
        self.extra_configs = extra_configs
        

        # Checkpoint manager works in both single and DDP
        self.ckpts_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_dir=self.resume_folder, # self.ckpts_folder, #self.train_configs.ckpts_folder,
            scheduler=self.scheduler,
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

        while(self.epoch <= self.train_configs.scheduler_config.epochs):
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

            self.logger.print_it(f"Best epoch so far is {best_epoch}: {'test' if self.run_test else 'train'} {self.metric_to_track} = {new_best:.4f}")

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
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        self._profiler_reset_epoch()
        self.model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.train_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.train_loader))
        self.logger.set_logger_newline()
        
        t_ddp_red0 = time.time()
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        t_ddp_red1 = time.time()
        self._profiler_add('ddp_reduce_current_stage', t_ddp_red1 - t_ddp_red0)
        train_summary = self.epoch_stats_tracker.stage_end()
        t_ddp_red2 = time.time()
        self._profiler_add('stage_end', t_ddp_red2 - t_ddp_red1)
        # Log per-epoch profiling summary
        self._profiler_log_epoch_summary()

        return train_summary

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device (profile this)
        t0 = time.time()
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        t1 = time.time()
        self._profiler_add('to_device', t1 - t0)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        t_comp0 = time.time()
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
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
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        t_comp1 = time.time()
        self._profiler_add('compute', t_comp1 - t_comp0)

        t_up0 = time.time()
        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        t_up1 = time.time()
        self._profiler_add('metrics_update', t_up1 - t_up0)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        t_up2 = time.time()
        self._profiler_add('metrics_batch_end', t_up2 - t_up1)

        # Print message on console (the print itself is profiled inside print_message)
        self.print_message(index_batch=batch_idx+1, total_batches=total_batches)


    def test_epoch(self):
        # reset per-epoch profiler for test stage as well (keeps same epoch bucket)
        self.reset_epoch_stats(phase='test')
        self._profiler_reset_epoch()
        self.model.eval()
        if self.distributed:
            self.test_loader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.test_loader))
        self.logger.set_logger_newline()

        t_ddp_red0 = time.time()
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        t_ddp_red1 = time.time()
        self._profiler_add('ddp_reduce_current_stage', t_ddp_red1 - t_ddp_red0)
        test_summary = self.epoch_stats_tracker.stage_end()
        t_ddp_red2 = time.time()
        self._profiler_add('stage_end', t_ddp_red2 - t_ddp_red1)
        # Log per-epoch profiling summary for test
        self._profiler_log_epoch_summary()
        return test_summary

    
    def test_step(self, inputs, targets, batch_idx=0, total_batches=0):
        self.epoch_stats_tracker.batch_start()
        # Map to available device (profile)
        t0 = time.time()
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
        t1 = time.time()
        self._profiler_add('to_device', t1 - t0)

        t_c0 = time.time()
        # Forward propagation, compute loss, get predictions
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss = loss.mean()
        t_c1 = time.time()
        self._profiler_add('compute', t_c1 - t_c0)

        t_u0 = time.time()
        self.epoch_stats_tracker.update(preds=outputs, targets=targets)
        t_u1 = time.time()
        self._profiler_add('metrics_update', t_u1 - t_u0)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        t_u2 = time.time()
        self._profiler_add('metrics_batch_end', t_u2 - t_u1)

        # Print message on console (profiled inside print_message)
        self.print_message(index_batch=batch_idx+1,
                            total_batches=total_batches)

    def print_message(self, index_batch, total_batches):
        t0 = time.time()
        message = f"{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
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
        current_lr = self.get_current_lr()
        if current_lr is not None:
            if isinstance(current_lr, (list, tuple)):
                message += ' LR=[' + ','.join(f"{x:.2e}" for x in current_lr) + '] |'
            else:
                message += f" LR={current_lr:.2e} |"
        h,m,s = convert_to_hms(self.epoch_stats_tracker.get_current_running_time())
        message += ' Epoch time {}:{:02d}:{:02d} |'.format(h,m,s)
        h,m,s = convert_to_hms(self.train_stats_tracker.get_current_running_time())
        message += ' Total time {}:{:02d}:{:02d} |'.format(h,m,s)
        self.logger.print_it_same_line(message)
        t1 = time.time()
        # record logging duration
        try:
            self._profiler_add('logging', t1 - t0)
        except Exception:
            pass
    
    def get_current_lr(self):
        # Append current learning rate(s)
        current_lr = None
        try:
            opt = getattr(self, 'optimizer', None)
            param_groups = None
            if opt is not None:
                param_groups = getattr(opt, 'param_groups', None)
                # Some SAM-like wrappers expose base_optimizer
                if param_groups is None and hasattr(opt, 'base_optimizer'):
                    param_groups = getattr(opt.base_optimizer, 'param_groups', None)
            if param_groups:
                lrs = [pg.get('lr') for pg in param_groups]
                # if all equal, show single value
                if all(abs(lrs[0] - x) < 1e-16 for x in lrs):
                    current_lr = lrs[0]
                else:
                    current_lr = lrs
        except Exception:
            current_lr = None
        return current_lr