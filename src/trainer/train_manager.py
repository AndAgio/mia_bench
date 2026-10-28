import os
import copy
from typing import Tuple, Union, Callable, Any
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from src.data.helpers import MultiDatasets
from src.models import get_model
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats, asam_param_groups
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts
from src.utils.configs import TrainConfigs, ModelConfigs, OptimizerConfigs, SchedulerConfigs
from src.utils import convert_to_hms
from src.trainer.stats_tracker import TrainStats, EpochStats, StageSummary
from src.trainer.checkpoints import CheckpointManager
from src.trainer.metrics import get_performance_metric_func
from src.trainer.distributed import maybe_init_ddp, is_rank0, ddp_barrier, maybe_cleanup_ddp
from src.utils.log import Loggable, MyLogger


# Default radius of the weight perturbation: the CIFAR-10 values of SAM (Foret et al., 2021) and of ASAM (Kwon et al.,
# 2021) for the adaptive versions, whose perturbation is relative to the weights.
SAM_RHO, ASAM_RHO = 0.05, 0.5


def sam_setting(opt_cfg: OptimizerConfigs, name: str, default: Any) -> Any:
    """Setting of the SAM-like optimizers given on the command line (see SAM_SETTINGS), or the default if not given.
    The defaults are starting points to tune per dataset, model and defense: see the Optimizers section of the README,
    which lists where each one comes from."""
    value = opt_cfg.extra.get(name)
    return default if value is None else value


class TrainManager(Loggable):
    """Train a model on train, validation and test splits, with checkpoints and resuming.

    Defenses subclass it and override setup_loss, train_step or train_epoch.
    """
    def __init__(self, train_configs: TrainConfigs, name: str, logger: MyLogger = None):
        """Set up the checkpoint folders, the device (and DDP) and the random seed."""
        super().__init__(logger=logger)
        self.train_configs = train_configs
        self.name = name

        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)

    
    def reset_configs(self, configs: TrainConfigs):
        """Replace the training configs, and set up the folders, device and seed again."""
        self.train_configs = configs
        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)
    

    def setup_folders(self, train_configs: TrainConfigs):
        """Create the checkpoint and resume folders of this run."""
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
        """Pick the device, start DDP if requested, and seed the random generators."""
        self._setup_distributed_training(distributed=train_configs.distributed,
                                        device=train_configs.device)
        self._setup_seed(seed=train_configs.seed)


    def _setup_distributed_training(self, distributed: bool = False, device: str = 'cpu'):
        """Start DDP if requested, and record the rank, world size and device."""
        self.global_rank, self.world_size, self.local_rank, self.device = maybe_init_ddp(use_ddp=distributed, device=device)
        self.distributed = True if (self.world_size > 1 and distributed) else False

    def _setup_seed(self, seed: int = 12345):
        """Seed Python, NumPy and PyTorch (CPU, CUDA and MPS), and make cuDNN deterministic."""
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
        """Build the model described by the configs."""
        self.setup_model_from_name_and_info(model_name=model_configs.model_name,
                                            im_channels=model_configs.im_channels,
                                            num_classes=model_configs.num_classes,
                                            im_size=model_configs.im_size)


    def setup_model_from_name_and_info(self, model_name: str, im_channels: int, num_classes: int, im_size: Tuple[int, ...]):
        """Build a model by name, on the device and wrapped in DDP if distributed."""
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
        """Train a copy of the given model, on the device and wrapped in DDP if distributed."""
        self.logger.print_it('Setting up model "{}"...'.format(model.name))
        self.model = copy.deepcopy(model)
        # Move model to device
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = self.model.to(self.device)
        self.model_name = model.name
        self.logger.print_it('Model setup done!')

    def get_model_name(self):
        """Return the name of the model being trained."""
        return self.model_name

    def setup_loss(self, loss: Union[str, Callable]):
        """Set the criterion from a name ('crossentropy') or a loss function. Override to use another loss."""
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
        """Set the metrics to log and the one that picks the best checkpoint (the loss if none)."""
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
        """Create the optimizer named in the configs, SAM-like ones included."""
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
            self.optimizer = SAM(params=asam_param_groups(self.model) if adaptive else self.model.parameters(),
                                base_optimizer=SGD,
                                lr=opt_cfg.lr,
                                rho=sam_setting(opt_cfg, 'sam_rho', ASAM_RHO if adaptive else SAM_RHO),
                                adaptive=adaptive,
                                eta=sam_setting(opt_cfg, 'asam_eta', 0.01),
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'esam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = ESAM(params=asam_param_groups(self.model) if adaptive else self.model.parameters(),
                                base_optimizer=SGD,
                                lr=opt_cfg.lr,
                                rho=sam_setting(opt_cfg, 'sam_rho', ASAM_RHO if adaptive else SAM_RHO),
                                beta=sam_setting(opt_cfg, 'esam_beta', 0.6),
                                gamma=sam_setting(opt_cfg, 'esam_gamma', 0.5),
                                adaptive=adaptive,
                                eta=sam_setting(opt_cfg, 'asam_eta', 0.01),
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'wsam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = WSAM(params=asam_param_groups(self.model) if adaptive else self.model.parameters(),
                                base_optimizer=SGD,
                                lr=opt_cfg.lr,
                                rho=sam_setting(opt_cfg, 'sam_rho', ASAM_RHO if adaptive else SAM_RHO),
                                gamma=sam_setting(opt_cfg, 'wsam_gamma', 0.9),
                                sam_eps=1e-12,
                                adaptive=adaptive,
                                eta=sam_setting(opt_cfg, 'asam_eta', 0.01),
                                decouple=True,
                                max_norm=None,
                                momentum=opt_cfg.momentum,
                                nesterov=opt_cfg.nesterov,
                                weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'looksam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = LookSAM(params=asam_param_groups(self.model) if adaptive else self.model.parameters(),
                                    base_optimizer=SGD,
                                    lr=opt_cfg.lr,
                                    rho=sam_setting(opt_cfg, 'sam_rho', ASAM_RHO if adaptive else SAM_RHO),
                                    k=sam_setting(opt_cfg, 'looksam_k', 5),
                                    alpha=sam_setting(opt_cfg, 'looksam_alpha', 0.3),
                                    adaptive=adaptive,
                                    eta=sam_setting(opt_cfg, 'asam_eta', 0.01),
                                    perturb_eps=1e-12,
                                    momentum=opt_cfg.momentum,
                                    nesterov=opt_cfg.nesterov,
                                    weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name.split('_')[-1] == 'friendlysam':
            adaptive = True if opt_cfg.name.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = FriendlySAM(params=asam_param_groups(self.model) if adaptive else self.model.parameters(),
                                        base_optimizer=SGD,
                                        lr=opt_cfg.lr,
                                        rho=sam_setting(opt_cfg, 'sam_rho', ASAM_RHO if adaptive else SAM_RHO),
                                        sigma=sam_setting(opt_cfg, 'friendlysam_sigma', 1.0),
                                        lmbda=sam_setting(opt_cfg, 'friendlysam_lambda', 0.9),
                                        adaptive=adaptive,
                                        eta=sam_setting(opt_cfg, 'asam_eta', 0.01),
                                        perturb_eps=1e-12,
                                        momentum=opt_cfg.momentum,
                                        nesterov=opt_cfg.nesterov,
                                        weight_decay=opt_cfg.weight_decay)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(opt_cfg.name))

    def setup_lr_scheduler(self, sched_cfg: SchedulerConfigs):
        """Create the learning rate scheduler named in the configs."""
        self.logger.print_it('Setting up "{}" learning rate scheduler...'.format(sched_cfg.name))
        if sched_cfg.name == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif sched_cfg.name == 'warmup_step':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            assert sched_cfg.extra['step_size'] < sched_cfg.epochs, f"In step-like schedulers the step size should be smaller than the total number of epochs. Found {sched_cfg.extra['step_size']} and {sched_cfg.epochs}!"
            assert 0 < sched_cfg.extra['step_gamma'] < 1, f"In step-like schedulers the gamma factor should be between 0 and 1. Found {sched_cfg.extra['step_gamma']}!"
            assert sched_cfg.extra['warmup_epochs'] < sched_cfg.epochs - sched_cfg.extra['step_size'], f"Invalid configuration for the number of warmup epochs, as it would not allow for step decay afterward! Warmup epochs: {sched_cfg.extra['warmup_epochs']}, step size: {sched_cfg.extra['step_size']} and total epochs: {sched_cfg.epochs}"
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_cfg.extra['step_size'], gamma=sched_cfg.extra['step_gamma'])
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=sched_cfg.extra['warmup_multiplier'], total_epoch=sched_cfg.extra['warmup_epochs'], after_scheduler=scheduler)
            self.scheduler.step()
        elif sched_cfg.name == 'warmup_exp':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            assert sched_cfg.extra['warmup_epochs'] < sched_cfg.epochs, f"Invalid configuration for the number of warmup epochs! Warmup epochs (found {sched_cfg.extra['warmup_epochs']}) should be less than the total total epochs (found {sched_cfg.epochs})"
            assert 0 < sched_cfg.extra['exp_gamma'] < 1, f"Exponential decay should be between 0 and 1, found {sched_cfg.extra['exp_gamma']} instead!"
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=sched_cfg.extra['exp_gamma'])
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=sched_cfg.extra['warmup_multiplier'], total_epoch=sched_cfg.extra['warmup_epochs'], after_scheduler=scheduler)
            self.scheduler.step()
        elif sched_cfg.name == 'warmup_cosine':
            assert sched_cfg.extra['cycle_step'] < sched_cfg.epochs, f"The number of epochs per cycle in warmup cosine scheduler should be less than the total number of epochs!"
            assert sched_cfg.extra['cycle_gamma'] <= 1, f"The decaying factor per cycle in warmup cosine scheduler should be less than or equal to 1 to avoid lr becoming too large!"
            assert sched_cfg.extra['warmup_epochs'] < sched_cfg.extra['cycle_step'], f"The number of warmup epochs in warmup cosine scheduler should be less than the number of epochs per cycle! Found {sched_cfg.extra['warmup_epochs']} and {sched_cfg.extra['cycle_step']}"
            # Every cycle of cycle_step epochs warms up linearly from cosine_min to the peak learning rate, then anneals
            # it back to cosine_min with a cosine, and the peak decays by cycle_gamma at every restart. As in the other
            # warmup schedulers, the peak is the learning rate times warmup_multiplier.
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                                           first_cycle_steps=sched_cfg.extra['cycle_step'],
                                                           max_lr=self.optimizer.param_groups[0]['lr'] * sched_cfg.extra['warmup_multiplier'],
                                                           min_lr=sched_cfg.extra['cosine_min'],
                                                           warmup_steps=sched_cfg.extra['warmup_epochs'],
                                                           gamma=sched_cfg.extra['cycle_gamma'])
        elif sched_cfg.name == 'step':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            assert sched_cfg.extra['step_size'] < sched_cfg.epochs, f"In step-like schedulers the step size should be smaller than the total number of epochs. Found {sched_cfg.extra['step_size']} and {sched_cfg.epochs}!"
            assert 0 < sched_cfg.extra['step_gamma'] < 1, f"In step-like schedulers the gamma factor should be between 0 and 1. Found {sched_cfg.extra['step_gamma']}!"
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sched_cfg.extra['step_size'], gamma=sched_cfg.extra['step_gamma'])
        elif sched_cfg.name == 'multistep':
            assert all(milestone < sched_cfg.epochs for milestone in sched_cfg.extra['step_milestones']), f"All milestones should be before the final epoch in the multistep lr scheduler!"
            assert 0 < sched_cfg.extra['step_gamma'] < 1, f"In step-like schedulers the gamma factor should be between 0 and 1. Found {sched_cfg.extra['step_gamma']}!"
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=sched_cfg.extra['step_milestones'], gamma=sched_cfg.extra['step_gamma'])
        elif sched_cfg.name == 'exp':
            assert 0 < sched_cfg.extra['exp_gamma'] < 1, f"Exponential decay should be between 0 and 1, found {sched_cfg.extra['exp_gamma']} instead!"
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=sched_cfg.extra['exp_gamma'])
        elif sched_cfg.name == 'cosine':
            assert sched_cfg.epochs is not None and sched_cfg.epochs > 0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=sched_cfg.epochs, eta_min=sched_cfg.extra['cosine_min'])
        else:
            raise ValueError(f"Learning rate scheduler '{sched_cfg.name}' not available!")

    def setup_training(self):
        """Set up the loss, metrics, optimizer and scheduler from the configs."""
        self.logger.print_it('Setting up training...')
        self.setup_loss(loss=self.train_configs.loss)
        self.setup_performance_metrics(metrics=self.train_configs.metrics,
                                        metric_to_track=self.train_configs.metric_to_track)
        self.setup_optimizer(opt_cfg=self.train_configs.optimizer_config)
        self.setup_lr_scheduler(sched_cfg=self.train_configs.scheduler_config)
        self.logger.print_it('Training setup done!')

    def setup_dataloaders_from_multidatasets(self, dataset: MultiDatasets, batch_size: int = 128):
        """Create the loaders of the train split and, when present, of the val and test splits."""
        self.dataset = dataset
        try:
            train_dataset = dataset.get('train')
        except (KeyError, ValueError):
            raise ValueError('At least the train split should be in the dataset!')
        try:
            val_dataset = dataset.get('val')
            self.run_val = True
        except (KeyError, ValueError):
            self.run_val = False
            self.logger.print_it('No validation split found in the dataset. Will only set up train and test dataloaders.')
        try:
            test_dataset = dataset.get('test')
            self.run_test = True
        except (KeyError, ValueError):
            self.run_test = False
            self.logger.print_it(f'No test split found in the dataset. Will only set up train {"and validation " if self.run_val else ""}dataloader{"s" if self.run_val else ""}.')
        
        if self.distributed:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                            pin_memory=True, shuffle=False,
                                            sampler=DistributedSampler(train_dataset,
                                                                        num_replicas=self.world_size,
                                                                        rank=self.global_rank,
                                                                        shuffle=True))
            if self.run_val:
                self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                                pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(val_dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=False))
            if self.run_test:
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                                pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(test_dataset,
                                                                            num_replicas=self.world_size,
                                                                            rank=self.global_rank,
                                                                            shuffle=False))
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            if self.run_val:
                self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if self.run_test:
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def setup_dataloaders(self, dataset: MultiDatasets, batch_size: int = 128):
        """Check that the dataset is a MultiDatasets and create its loaders."""
        assert isinstance(dataset, MultiDatasets), f"Dataset should be of type MultiDatasets. Found {type(dataset)} instead!"
        self.setup_dataloaders_from_multidatasets(dataset=dataset, batch_size=batch_size)

    def initialize_train(self, 
                        dataset: MultiDatasets,
                        model: Union[ModelConfigs,nn.Module],
                        configs: TrainConfigs,
                        ):
        """Prepare a run: data loaders, model (a module or its configs) and training setup."""
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
        # self.validate_and_fix_model_for_dp(dp_config=configs.dp_config)
        self.setup_training()
        # Modifying model, optimizer and loaders for differential privacy if needed
        # self.check_and_set_dp(dp_config=configs.dp_config)
        self.logger.print_it('Training initialization completed!')


    def reset_running_stats(self):
        """Reset the tracking of the best epoch before a run."""
        # Start from the worst possible value so that the first epoch is always saved as best.pth.
        best_record = np.inf if self.metric_to_track in ["loss", "mse", "mae", "rmse"] else -np.inf
        mode_to_track_best = "min" if self.metric_to_track in ["loss", "mse", "mae", "rmse"] else "max"
        stage_to_track_best = "val" if self.run_val else "test" if self.run_test else "train"
        self.train_stats_tracker = TrainStats(best_epoch=0,
                                            best_record=best_record,
                                            metric_to_track_best=self.metric_to_track,
                                            stage_to_track_best=stage_to_track_best,
                                            mode_to_track_best=mode_to_track_best)
        # Reset global profiler totals when starting a new training run
        self.profiler_total = {}

    
    def reset_epoch_stats(self, phase: str = 'train'):
        """Start tracking the loss and metrics of a stage (train, val or test)."""
        metrics = {"loss": self.criterion}
        for met, met_fn in self.performance_metrics.items():
            metrics[met] = met_fn
        self.epoch_stats_tracker.stage_begin(stage=phase,
                                            metrics=metrics,
                                            sync_cuda=True,
                                            sync_mps=True)

    def train(self, extra_configs: dict[str, Any] = None, return_best_model: bool = True, return_last_model: bool = False, return_stats: bool = False):
        """Train for the configured epochs, resuming if possible, and return the best and/or last model."""
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
                self.load_checkpoint_extra(resume_ckpt.get("extra", {}))
            else:
                self.epoch = 1
        else:
            self.epoch = 1
        
        self.train_stats_tracker.start_timer()
        self.best_perf = self.train_stats_tracker.get_best()
        self.epoch_stats_tracker = EpochStats()

        while(self.epoch <= self.train_configs.scheduler_config.epochs):
            self.epoch_stats_tracker.epoch_start()
            self.train_executions()
            self.epoch_stats_tracker.epoch_end()
            self.epoch_stats_tracker.ddp_consolidate_epoch_time(op="max")
            epoch_time = self.epoch_stats_tracker.get_epoch_time()
            # Finalize epoch and checkpoint (rank-0)
            epoch_summary = self.epoch_stats_tracker.finalize_epoch(epoch=self.epoch)
            best_epoch, new_best = self.train_stats_tracker.update_history_and_best(epoch_summary=epoch_summary)

            self.logger.print_it(f"Best epoch so far is {best_epoch}: {'val' if self.run_val else 'test' if self.run_test else 'train'} {self.metric_to_track} = {new_best:.4f}")

            if self.local_rank == 0:
                ckpt = self.ckpts_manager.build_checkpoint(epoch=self.epoch,
                                                        train_stats=self.train_stats_tracker,
                                                        extra=self.get_checkpoint_extra())
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
        self.logger.print_it('Training for "{}" with seed {} completed in: {}:{:02d}:{:02d}'.format(self.model_name, self.seed, h, m, s))

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

    def get_checkpoint_extra(self) -> dict[str, Any]:
        """Return extra training state to store in checkpoints. Override when a defense has some."""
        # Hook for subclasses with additional training state (e.g., auxiliary models) to be stored in checkpoints.
        return {}

    def load_checkpoint_extra(self, extra: dict[str, Any]):
        """Restore the state saved by get_checkpoint_extra when resuming."""
        # Hook for subclasses to restore the state stored by get_checkpoint_extra when resuming.
        pass

    def train_executions(self):
        """Run one epoch: training, validation and test when their splits exist, then a scheduler step."""
        self.train_epoch()
        if self.run_val:
            self.val_epoch()
        if self.run_test:
            self.test_epoch()
        self.scheduler.step()

    def train_epoch(self):
        """Train for one epoch, calling train_step on each batch, and return the epoch summary."""
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        self.model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.train_loader):
            self.train_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.train_loader))
        self.logger.set_logger_newline(console_only=True)
        
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()

        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)

        return train_summary

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        """Run one optimization step on a batch and log its metrics. Override to change how batches are used."""
        # Map to available device (profile this)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
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
            # loss = loss.mean() if loss.numel() > 1 else loss
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            self.optimizer.step()

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)

    def to_be_averaged_loss(self, loss: torch.Tensor) -> bool:
        """Return whether the loss holds one value per sample and must be averaged."""
        try:
            return loss.numel() > 1
        except Exception:
            return isinstance(self.criterion, torch.nn.modules.loss._Loss) and self.criterion.reduction == 'none'

    def val_epoch(self):
        """Evaluate on the validation split and return its summary."""
        # reset per-epoch profiler for test stage as well (keeps same epoch bucket)
        self.reset_epoch_stats(phase='val')
        self.model.eval()
        if self.distributed:
            self.val_loader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.val_loader):
                self.val_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.val_loader))
        self.logger.set_logger_newline(console_only=True)

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        val_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=val_summary)
        self.logger.print_it(f"{message}", file_only=True)
        return val_summary

    def val_step(self, inputs, targets, batch_idx=0, total_batches=0):
        """Evaluate one validation batch, like test_step."""
        self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=total_batches)

    def test_epoch(self):
        """Evaluate on the test split and return its summary."""
        # reset per-epoch profiler for test stage as well (keeps same epoch bucket)
        self.reset_epoch_stats(phase='test')
        self.model.eval()
        if self.distributed:
            self.test_loader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.test_loader):
                self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.test_loader))
        self.logger.set_logger_newline(console_only=True)

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        test_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=test_summary)
        self.logger.print_it(f"{message}", file_only=True)
        return test_summary

    
    def test_step(self, inputs, targets, batch_idx=0, total_batches=0):
        """Evaluate one batch and log its metrics."""
        self.epoch_stats_tracker.batch_start()
        # Map to available device (profile)
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Forward propagation, compute loss, get predictions
        outputs = self.model(inputs)

        self.epoch_stats_tracker.update(preds=outputs, targets=targets)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        # Print message on console (profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                            total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)
    
    def get_current_lr(self):
        """Return the current learning rate, or a list when the parameter groups differ."""
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
    
    def build_message_for_stage_end(self, stage_summary: StageSummary) -> str:
        """Build the log line printed at the end of a stage."""
        message = f"{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        message += ' {}: '.format(stage_summary.stage.upper())
        metrics = {}
        if is_rank0():
            metrics = stage_summary.metrics
        message = self.append_metrics(message, metrics)
        message = self.append_lr(message)
        message = self.append_times(message)
        return message
    
    def build_message_for_batch_end(self, index_batch, total_batches) -> str:
        """Build the progress line printed after each batch."""
        message = f"{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        metrics = {}
        if is_rank0():
            metrics = self.epoch_stats_tracker._require_active_stage().current_avgs()
        message = self.append_metrics(message, metrics)
        message = self.append_lr(message)
        message = self.append_times(message)
        return message
    
    @staticmethod
    def append_metrics(message: str, metrics: dict[str, float]) -> str:
        """Append the metrics, except the timing ones, to a log line."""
        if metrics is not None:
            metrics_message = ''
            index = 0
            for metric_name, metric_value in metrics.items():
                if metric_name in ['batch_time_sec', 'samples_per_sec']:
                    index += 1
                    continue
                metrics_message += '{}={:.5f}{} '.format(metric_name, metric_value,
                                                            ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += metrics_message
        message += '|'
        return message
    
    def append_lr(self, message: str) -> str:
        """Append the current learning rate to a log line."""
        current_lr = self.get_current_lr()
        if current_lr is not None:
            if isinstance(current_lr, (list, tuple)):
                message += ' LR=[' + ','.join(f"{x:.2e}" for x in current_lr) + '] |'
            else:
                message += f" LR={current_lr:.2e} |"
        return message
    
    def append_times(self, message: str) -> str:
        """Append the epoch and total running times to a log line."""
        h,m,s = convert_to_hms(self.epoch_stats_tracker.get_current_running_time())
        message += ' Epoch time {}:{:02d}:{:02d} |'.format(h,m,s)
        h,m,s = convert_to_hms(self.train_stats_tracker.get_current_running_time())
        message += ' Total time {}:{:02d}:{:02d} |'.format(h,m,s)
        return message
