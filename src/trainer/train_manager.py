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
from typing import Dict, Set
from torch.utils.data import Subset
from src.trainer.prop_noise_obfs import *
from src.trainer.prop_noise_obfs import _snapshot_layer_grads, _print_grad_changes, metric_privacy_obfuscation,_select_trainable_layer
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from src.data.multi import MultiDatasets
from src.models import get_model
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.optimizers.schedulers import GradualWarmupScheduler
from src.utils.configs import TrainConfigs, ModelConfigs, OptimizerConfigs, SchedulerConfigs, DPConfigs
from src.utils import convert_to_hms
from src.trainer.stats_tracker import TrainStats, EpochStats, StageSummary
from src.trainer.checkpoints import CheckpointManager
from src.trainer.metrics import get_performance_metric_func
from src.trainer.distributed import maybe_init_ddp, is_rank0, ddp_barrier, maybe_cleanup_ddp
from src.utils.log import Loggable, SmartLogger, DumbLogger
import torch.nn.functional as F


REF_RATIO = 0.1
SEED_SPLIT = 42

def _param_ids_in_module(module: torch.nn.Module) -> Set[int]:
    """Returns Python ids of parameter objects in `module` (including submodules)."""
    return {id(p) for p in module.parameters()}


@torch.no_grad()
def _grad_checksum_excluding(
    model: torch.nn.Module,
    exclude_param_ids: Set[int],
) -> Dict[str, float]:
    """
    Returns a lightweight 'checksum' per parameter grad tensor in model,
    excluding parameters whose id is in exclude_param_ids.

    The checksum is (sum, sum_abs, max_abs, numel) combined into one float tuple-like,
    but stored as a float for quick compare; you can make it a tuple if you prefer.
    """
    ck = {}
    for name, p in model.named_parameters():
        if id(p) in exclude_param_ids:
            continue
        if p.grad is None:
            continue

        g = p.grad.detach()
        # robust-ish fingerprint (still cheap)
        s = float(g.sum().item())
        a = float(g.abs().sum().item())
        m = float(g.abs().max().item())
        n = int(g.numel())
        ck[name] = (s, a, m, n)
    return ck


def _compare_checksums(before: Dict[str, tuple], after: Dict[str, tuple]):
    """
    Prints differences. Returns True if identical.
    """
    ok = True
    bkeys = set(before.keys())
    akeys = set(after.keys())

    if bkeys != akeys:
        ok = False
        print("[DEFENSE CHECK] ❌ Different set of grad tensors present pre/post.")
        print("  only-before:", sorted(bkeys - akeys)[:10])
        print("  only-after :", sorted(akeys - bkeys)[:10])

    for k in sorted(bkeys & akeys):
        if before[k] != after[k]:
            ok = False
            print(f"[DEFENSE CHECK] ❌ Non-target layer grad changed: {k}")
            print(f"  before={before[k]}")
            print(f"  after ={after[k]}")
            # don't spam: show first few
            # break
    return ok


class TrainManager(Loggable):
    def __init__(self, train_configs: TrainConfigs, name: str, logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        self.train_configs = train_configs
        self.name = name
        self.frozen_model = None
        self.ref_loader = None
        self.risk_ema = None          # EMA accumulator tensor (layer-shaped)
        self.risk_beta = 0.95         # EMA smoothing
        self.risk_frac = 0.001        # top 0.1% coords in chosen layer
        self.risk_lam = 0.7           # lambda in (1-lam)*CE + lam*KL
        self.risk_temp = 1.0          # temperature for KL softmax
        self.ref_iter = None
        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)

    
    def reset_configs(self, configs: TrainConfigs):
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
            assert metric_to_track in list(self.performance_metrics.keys()), f'Metric to track should be among tracked metrics. Found "{metric_to_track}" and {list(self.performance_metrics.keys())}!'
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
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                T_0=sched_cfg.extra['cycle_step'],
                                                                                T_mult=sched_cfg.extra['cycle_gamma'],
                                                                                eta_min=sched_cfg.extra['cosine_min'],
                                                                                last_epoch=sched_cfg.epochs)
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
        self.logger.print_it('Setting up training...')
        self.setup_loss(loss=self.train_configs.loss)
        self.setup_performance_metrics(metrics=self.train_configs.metrics,
                                        metric_to_track=self.train_configs.metric_to_track)
        self.setup_optimizer(opt_cfg=self.train_configs.optimizer_config)
        self.setup_lr_scheduler(sched_cfg=self.train_configs.scheduler_config)
        self.logger.print_it('Training setup done!')
    
    def _split_train_and_ref(self, dataset, ref_ratio=0.1, seed=42):
        n = len(dataset)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        n_ref = int(n * ref_ratio)
        ref_idx = indices[:n_ref]
        train_idx = indices[n_ref:]

        train_set = Subset(dataset, train_idx)
        ref_set = Subset(dataset, ref_idx)

        return train_set, ref_set


    def setup_dataloaders_from_multidatasets(self, dataset: MultiDatasets, batch_size: int = 128):
        global SEED_SPLIT, REF_RATIO
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

        if self.name.startswith("shadow_"): #if it is shadow dont take the same seed as the initial one, resulting in exactly the same split
            actual_seed = SEED_SPLIT + 10
        else:
            actual_seed = SEED_SPLIT
        full_train_dataset = train_dataset
        train_set, ref_set = self._split_train_and_ref(
        full_train_dataset,
        ref_ratio=REF_RATIO,
        seed=SEED_SPLIT
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.ref_loader = DataLoader(
            ref_set,
            batch_size=batch_size,   # can be smaller if you want
            shuffle=True,                 # shuffle is OK
            pin_memory=True,
            drop_last=True,
        )

    def _init_ref_iter(self):
        self.ref_iter = iter(self.ref_loader)


    def _next_ref_x(self):
        if not hasattr(self, "ref_iter") or self.ref_iter is None:
            self.ref_iter = iter(self.ref_loader)

        try:
            x, _ = next(self.ref_iter)
        except StopIteration:
            self.ref_iter = iter(self.ref_loader)
            x, _ = next(self.ref_iter)
        return x.to(self.device, non_blocking=True)
    def on_epoch_start(self):
        self._init_ref_iter()

    def setup_dataloaders_from_torch_dataset(self, dataset: Dataset, batch_size: int = 128, split: bool = False):
        self.run_train = True
        global REF_RATIO, SEED_SPLIT

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
            train_dataset = dataset
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
        full_train_dataset = train_dataset
        if self.name.startswith("shadow_"): #if it is shadow dont take the same seed as the initial one, resulting in exactly the same split
            actual_seed = SEED_SPLIT + 10
        else:
            actual_seed = SEED_SPLIT
        train_set, ref_set = self._split_train_and_ref(
        full_train_dataset,
        ref_ratio=REF_RATIO,
        seed=actual_seed
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        self.ref_loader = DataLoader(
            ref_set,
            batch_size=batch_size,   # can be smaller if you want
            shuffle=True,                 # shuffle is OK
            pin_memory=True,
            drop_last=True,
        )

    def setup_dataloaders(self, dataset: Union[MultiDatasets, Dataset], batch_size: int = 128):
        if isinstance(dataset, MultiDatasets):
            self.setup_dataloaders_from_multidatasets(dataset=dataset,
                                                        batch_size=batch_size)
        elif isinstance(dataset, Dataset):
            self.setup_dataloaders_from_torch_dataset(dataset=dataset,
                                                        batch_size=batch_size,
                                                        split=False)

    def check_and_set_dp(self, dp_config: DPConfigs):
        if dp_config.use_dp:
            self.logger.print_it('Differential Privacy with Opacus: updating model, optimizer and data loaders accordingly...')
            
            if dp_config.clip_per_layer:
                # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_grad_norm`.
                n_layers = len(
                    [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
                )
                max_grad_norm = [
                    dp_config.max_grad_norm / np.sqrt(n_layers)
                ] * n_layers
            else:
                max_grad_norm = dp_config.max_grad_norm

            if self.distributed and dp_config.clip_per_layer:
                self.model = DPDDP(self.model)

            privacy_engine = PrivacyEngine()
            clipping = "per_layer" if dp_config.clip_per_layer else "flat"
            if dp_config.grad_sample_mode in ['ghost']:
                self.model, self.optimizer, self.criterion, self.train_loader = privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=dp_config.noise_multiplier,
                    max_grad_norm=max_grad_norm,
                    clipping=clipping,
                    grad_sample_mode=dp_config.grad_sample_mode,
                )
            elif dp_config.grad_sample_mode in ['hook']:
                self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=dp_config.noise_multiplier,
                    max_grad_norm=max_grad_norm,
                    clipping=clipping,
                    grad_sample_mode=dp_config.grad_sample_mode,
                )
            else:
                raise ValueError("Unsupported mode '{dp_config.grad_sample_mode}' for dp_config.grad_sample_mode when using DP!")
            self.logger.print_it('Differential Privacy setup completed!')
            self.model = self.model.to(self.device)
        else:
            pass

    def validate_and_fix_model_for_dp(self, dp_config: DPConfigs):
        if dp_config.use_dp:
            self.logger.print_it('Differential Privacy with Opacus: validating model and fixing it if necessary...')
            if not ModuleValidator.is_valid(self.model):
                self.model = ModuleValidator.fix(self.model)
            self.model = self.model.to(self.device)

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
            self.frozen_model = copy.deepcopy(model).to(self.device)
            self.frozen_model.eval()
            for p in self.frozen_model.parameters():
                p.requires_grad_(False)

        elif isinstance(model, ModelConfigs):
            self.setup_model_from_configs(model_configs=model)
        else:
            raise ValueError('Not recognizing model given!')
        self.validate_and_fix_model_for_dp(dp_config=configs.dp_config)
        self.setup_training()
        # Modifying model, optimizer and loaders for differential privacy if needed
        self.check_and_set_dp(dp_config=configs.dp_config)
        self.logger.print_it('Training initialization completed!')



    def _next_ref_x(self):
        try:
            batch = next(self.ref_iter)
        except StopIteration:
            self.ref_iter = iter(self.ref_loader)
            batch = next(self.ref_iter)
        x_re = batch[0] if isinstance(batch, (tuple, list)) else batch
        return x_re.to(self.device, non_blocking=True)

    def _top_frac_mask(self, scores: torch.Tensor, frac: float) -> torch.Tensor:
        flat = scores.reshape(-1)
        k = max(1, int(flat.numel() * frac))
        # if k is huge, topk can be slow; for your typical penultimate layer it's fine
        vals, idx = torch.topk(flat, k, largest=True, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[idx] = True
        return mask.view_as(scores)

    @torch.no_grad()
    def _update_ema(self, step_scores: torch.Tensor):
        if self.risk_ema is None:
            self.risk_ema = step_scores.detach().clone()
        else:
            b = self.risk_beta
            self.risk_ema.mul_(b).add_(step_scores.detach(), alpha=(1 - b))

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
            lines = [f'Profiling summary for epoch {self.epoch}:']
            for k, v in sorted(self.profiler_epoch.items(), key=lambda x: -x[1]):
                lines.append(f'  {k}: {v:.4f}s')
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
            self.on_epoch_start()
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
            
            self.logger.print_it(
                f"Best epoch so far is {best_epoch}: "
                f"{'test' if self.run_test else 'train'} "
                f"{self.metric_to_track} = {new_best:.4f}"
            )
                        
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
        self.model.train()
       
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.train_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.train_loader))
        self.logger.set_logger_newline(console_only=True)
        
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()

        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)

        return train_summary

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device (profile this)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

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
            self.optimizer.zero_grad()

            # forward on train batch
            outputs = self.model(inputs)
            """
            loss = self.criterion(outputs, targets)
            loss = loss.mean() if loss.numel() > 1 else loss
            loss.backward()
            """
            self.optimizer.zero_grad()

            # --------------------------------------------------
            # 1) NORMAL TRAINING (this is the ONLY loss that updates the model)
            # --------------------------------------------------
            outputs = self.model(inputs)
            loss_ce = self.criterion(outputs, targets)
            loss_ce = loss_ce.mean() if loss_ce.numel() > 1 else loss_ce
            loss_ce.backward()    # <-- only backward that writes to .grad

            # --------------------------------------------------
            # 2) RISK SCORING (NO effect on training grads)
            # --------------------------------------------------
            # get a reference batch
            x_re = self._next_ref_x().to(self.device, non_blocking=True)

            # compute the logits on the "frozen" model i.e. initial model
            with torch.no_grad():
                logits_vn = self.frozen_model(x_re)

            # current model predictions (we need the gradient graph)
            logits_up = self.model(x_re)

            #compute the KL divergence betweens the logits of the initial model and the current model
            T = self.risk_temp  #temperature for KL 
            loss_kl = F.kl_div(
                F.log_softmax(logits_up / T, dim=-1),
                F.softmax(logits_vn / T, dim=-1),
                reduction="batchmean"
            ) * (T ** 2)

            # compute KL gradients WITHOUT touching .grad
            model_for_defense = self.model.module if hasattr(self.model, "module") else self.model
            target_layer = _select_trainable_layer(model_for_defense, which="penultimate", debug=False)
            
            g_risk = torch.autograd.grad(
                loss_kl,
                target_layer.weight,
                retain_graph=False,
                create_graph=False
            )[0]

            # --------------------------------------------------
            # 3) UPDATE RISK EMA
            # --------------------------------------------------
            with torch.no_grad():
                step_scores = g_risk.detach().abs() * target_layer.weight.detach().abs()
            self._update_ema(step_scores)

            # --------------------------------------------------
            # 4) BUILD MASK + OBFUSCATE TRAINING GRADS
            # --------------------------------------------------
            d = 0.5
            b = 50

            weight_mask, stats = select_coords_toprisk_until_weight_l1_le_d_(
                weight=target_layer.weight.detach(),
                risk_scores=self.risk_ema,
                d=d,
                require_at_least_one=True,
            )
           
            weight_mask, stats = select_coords_risk_biased_random_until_weight_l1_le_d_(
            weight=target_layer.weight.detach(),
            risk_scores=self.risk_ema,
            d=d,
            alpha=0.7,          # start <1 to avoid always picking the same coords
            max_draw=None,      # optional speed cap; tune
            require_at_least_one=True,
        )

            bias_mask = weight_mask.any(dim=1)
            grads_before = _snapshot_layer_grads(target_layer)
            metric_privacy_obfuscation(
                model_for_defense,
                d=d,
                b=b,
                which="penultimate",
                clip_scope="masked",
                coord_mask={"weight": weight_mask, "bias": bias_mask},
            )

            # --------------------------------------------------
            # 5) UPDATE
            # --------------------------------------------------
            self.optimizer.step()
            
            #uncomment for prints: verify that only the penultimate layer changes - nothing else and that some of the other parameters actually change
            # print target changes (what you already do)
            #_print_grad_changes(target_layer, grads_before, k_show=5)
            """
            # checksum grads of ALL OTHER layers after defense
            ck_after = _grad_checksum_excluding(model_for_defense, exclude_param_ids=exclude_ids)

            # verify nothing else changed
            ok = _compare_checksums(ck_before, ck_after)
            if ok:
                print("[DEFENSE CHECK] ✅ No other layer grads changed.")
            else:
                print("[DEFENSE CHECK] ❌ Some non-target grads changed (see above).")


            
             
        
            _print_grad_changes(
            target_layer,
            grads_before,
            k_show=5,
            )

            layer = target_layer
            print("WEIGHT  max|w|:", layer.weight.detach().abs().max().item(),
                  "mean|w|:", layer.weight.detach().abs().mean().item())
            print("GRAD    max|g|:", layer.weight.grad.detach().abs().max().item(),
                  "mean|g|:", layer.weight.grad.detach().abs().mean().item())
            """
            #self.optimizer.step()
        t_comp1 = time.time()
        self._profiler_add('compute', t_comp1 - t_comp0)

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)


    def test_epoch(self):
        # reset per-epoch profiler for test stage as well (keeps same epoch bucket)
        self.reset_epoch_stats(phase='test')
        self.model.eval()
        if self.distributed:
            self.test_loader.sampler.set_epoch(self.epoch)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.test_loader))
        self.logger.set_logger_newline(console_only=True)

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        test_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=test_summary)
        self.logger.print_it(f"{message}", file_only=True)
        return test_summary

    
    def test_step(self, inputs, targets, batch_idx=0, total_batches=0):
        self.epoch_stats_tracker.batch_start()
        # Map to available device (profile)
        inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

        # Forward propagation, compute loss, get predictions
        outputs = self.model(inputs)

        self.epoch_stats_tracker.update(preds=outputs, targets=targets)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        # Print message on console (profiled inside print_message)
        self.print_message(index_batch=batch_idx+1,
                            total_batches=total_batches)

    def print_message(self, index_batch, total_batches):
        t0 = time.time()
        message = f'{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |'
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
                message += f' LR={current_lr:.2e} |'
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
    
    def build_message_for_stage_end(self, stage_summary: StageSummary) -> str:
        message = f"{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        message += ' {}: '.format(stage_summary.stage.upper())
        if is_rank0():
            metrics = stage_summary.metrics
        message = self.append_metrics(message, metrics)
        message = self.append_lr(message)
        message = self.append_times(message)
        return message
    
    def build_message_for_batch_end(self, index_batch, total_batches) -> str:
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
        message = self.append_metrics(message, metrics)
        message = self.append_lr(message)
        message = self.append_times(message)
        return message
    
    @staticmethod
    def append_metrics(message: str, metrics: dict[str, float]) -> str:
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
        current_lr = self.get_current_lr()
        if current_lr is not None:
            if isinstance(current_lr, (list, tuple)):
                message += ' LR=[' + ','.join(f"{x:.2e}" for x in current_lr) + '] |'
            else:
                message += f" LR={current_lr:.2e} |"
        return message
    
    def append_times(self, message: str) -> str:
        h,m,s = convert_to_hms(self.epoch_stats_tracker.get_current_running_time())
        message += ' Epoch time {}:{:02d}:{:02d} |'.format(h,m,s)
        h,m,s = convert_to_hms(self.train_stats_tracker.get_current_running_time())
        message += ' Total time {}:{:02d}:{:02d} |'.format(h,m,s)
        return message