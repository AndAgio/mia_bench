import pathlib
import hashlib
import json
# from dataclasses import dataclass, field
from src.utils.variables import DEFAULT_MODELS_FOLDER, DEFAULT_RESUME_CKPTS_FOLDER, DEFAULT_LOG_FOLDER, DEFAULT_DATASETS_FOLDER
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
import torch

Loss = Union[str, Callable[..., Any], torch.nn.Module]


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class OptimizerConfigs:
    name: str = 'sgd'
    lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9
    nesterov: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class DPConfigs:
    use_dp: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    clip_per_layer: bool = False
    grad_sample_mode: str = 'hook'

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class MetricConfigs:
    use_metric: bool = False
    metric_b: float = 1.0
    metric_d: float = 1.0



@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class SchedulerConfigs:
    name: str = 'cosine'
    epochs: int = 100
    extra: Dict[str, Any] = field(default_factory=dict)
    
def build_scheduler_configs_from_settings(settings: Any, mode: str = 'victim') -> SchedulerConfigs:
    assert mode in ['victim', 'attacker'], f"Mode '{mode}' not available to build LR scheduler configurations!"
    name = settings.victim_lr_sched if mode == 'victim' else settings.att_lr_sched
    total_epochs = settings.victim_epochs if mode == 'victim' else settings.att_epochs
    extra = {}
    if name == 'warmup_step':
        extra['step_size'] = settings.victim_lr_step_size  if mode == 'victim' else settings.att_lr_step_size
        extra['step_gamma'] = settings.victim_lr_step_gamma if mode == 'victim' else settings.att_lr_step_gamma
        extra['warmup_multiplier'] = settings.victim_lr_warmup_multiplier if mode == 'victim' else settings.att_lr_warmup_multiplier
        extra['warmup_epochs'] = settings.victim_lr_warmup_epochs if mode == 'victim' else settings.att_lr_warmup_epochs
    elif name == 'warmup_exp':
        extra['exp_gamma'] = settings.victim_lr_exp_gamma if mode == 'victim' else settings.att_lr_exp_gamma
        extra['warmup_multiplier'] = settings.victim_lr_warmup_multiplier if mode == 'victim' else settings.att_lr_warmup_multiplier
        extra['warmup_epochs'] = settings.victim_lr_warmup_epochs if mode == 'victim' else settings.att_lr_warmup_epochs
    elif name == 'warmup_cosine':
        extra['cycle_step'] = settings.victim_lr_cycle_step if mode == 'victim' else settings.att_lr_cycle_step
        extra['cycle_gamma'] = settings.victim_lr_cycle_gamma if mode == 'victim' else settings.att_lr_cycle_gamma
        extra['cosine_min'] = settings.victim_lr_cosine_min if mode == 'victim' else settings.att_lr_cosine_min
    elif name == 'step':
        extra['step_size'] = settings.victim_lr_step_size  if mode == 'victim' else settings.att_lr_step_size
        extra['step_gamma'] = settings.victim_lr_step_gamma if mode == 'victim' else settings.att_lr_step_gamma
    elif name == 'multistep':
        extra['step_milestones'] = settings.victim_lr_step_milestones  if mode == 'victim' else settings.victim_lr_step_milestones
        extra['step_gamma'] = settings.victim_lr_step_gamma if mode == 'victim' else settings.att_lr_step_gamma
    elif name == 'exp':
        extra['exp_gamma'] = settings.victim_lr_exp_gamma if mode == 'victim' else settings.att_lr_exp_gamma
    elif name == 'cosine':
        extra['cosine_min'] = settings.victim_lr_cosine_min if mode == 'victim' else settings.att_lr_cosine_min
    else:
        raise ValueError(f"Learning rate scheduler '{name}' not available!")
    return SchedulerConfigs(name=name,
                            epochs=total_epochs,
                            extra=extra)


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class TrainConfigs:
    # Mandatory arguments (accept either dict or OptimizerConfigs/SchedulerConfigs)
    optimizer_config: Union[OptimizerConfigs, Dict[str, Any]]
    scheduler_config: Union[SchedulerConfigs, Dict[str, Any]]
    # Optional arguments with default values
    dp_config: DPConfigs = field(default_factory=DPConfigs)
    metric_config: MetricConfigs = field(default_factory=MetricConfigs)
    batch_size: Optional[int] = 256
    loss: Optional[Loss] = 'crossentropy'
    metrics: Optional[Tuple[Union[str, Callable[..., Any]], ...]] = ('multi_class_accuracy',)
    metric_to_track: Optional[str] = 'multi_class_accuracy'
    lr_sched: Optional[str] = 'const'
    device: Optional[str] = 'cpu'
    distributed: Optional[bool] = False
    seed: Optional[int] = 12345
    resume: Optional[bool] = True
    ckpts_folder: Optional[pathlib.Path] = DEFAULT_MODELS_FOLDER
    resume_ckpts_folder: Optional[pathlib.Path] = DEFAULT_RESUME_CKPTS_FOLDER

    def __post_init__(self):
        # normalize nested optimizer/scheduler configs
        if isinstance(self.optimizer_config, dict):
            self.optimizer_config = OptimizerConfigs(**self.optimizer_config)
        if isinstance(self.scheduler_config, dict):
            self.scheduler_config = SchedulerConfigs(**self.scheduler_config)

        # normalize metrics to a tuple
        if self.metrics is None:
            self.metrics = tuple()
        elif isinstance(self.metrics, (str, Callable)):
            self.metrics = (self.metrics,)
        elif isinstance(self.metrics, list):
            self.metrics = tuple(self.metrics)

        # metric_to_track must be one of metrics (if provided); otherwise keep as-is
        if self.metric_to_track is None and len(self.metrics) > 0:
            self.metric_to_track = self.metrics[0] if isinstance(self.metrics[0], str) else None

        # loss: allow str, callable, or nn.Module instance — nothing to coerce here
        # but keep a convenience hook: if the user passed a class (not instance) instantiate it without args
        try:
            from inspect import isclass
        except Exception:
            isclass = lambda x: False

        if isclass(self.loss) and not isinstance(self.loss, str):
            try:
                self.loss = self.loss()
            except Exception:
                # if instantiation fails, leave as provided
                pass


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class ModelConfigs:
    # Mandatory arguments
    model_name: str
    im_channels: int
    num_classes: int
    im_size: Tuple[int, ...]


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class LogConfigs:
    # Mandatory arguments
    name: str
    # Optional arguments with default values
    log_folder: pathlib.Path = DEFAULT_LOG_FOLDER
    log_mode: str = 'smart'


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class DatasetConfigs:
    # Mandatory arguments
    name: str
    # Optional arguments with default values
    data_folder: pathlib.Path = DEFAULT_DATASETS_FOLDER
    data_augmentation: bool = True
    im_size: Optional[Tuple[int, ...]] = None
    im_channels: Optional[int] = None
    num_classes: Optional[int] = None
    info: Optional[dict] = None

    def __post_init__(self):
        if self.im_size is None:
            self.im_size = get_im_size_from_name(self.name)
        if self.im_channels is None:
            self.im_channels = get_im_channels_from_name(self.name)
        if self.num_classes is None:
            self.num_classes = get_num_classes_from_name(self.name)
        if self.info is None:
            self.info = get_dataset_info_from_name(self.name)

def get_dataset_info_from_name(dataset: str):
    info = {'im_channels': get_im_channels_from_name(dataset=dataset),
            'im_size': get_im_size_from_name(dataset=dataset),
            'num_classes': get_num_classes_from_name(dataset=dataset),}
    return info

def get_im_size_from_name(dataset: str):
    if dataset == 'cifar10':
        im_size = (32,32)
    elif dataset == 'cifar100':
        im_size = (32,32)
    elif dataset == 'svhn':
        im_size = (32, 32)
    elif dataset == 'fmnist':
        im_size = (28, 28)
    elif dataset == 'cinic10':
        im_size = (32, 32)
    elif dataset == 'imagenet':
        im_size = (224, 224)
    elif dataset == 'tinyimagenet':
        im_size = (64, 64)
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    return im_size

def get_im_channels_from_name(dataset: str):
    if dataset == 'cifar10':
        im_channels = 3
    elif dataset == 'cifar100':
        im_channels = 3
    elif dataset == 'svhn':
        im_channels = 3
    elif dataset == 'fmnist':
        im_channels = 1
    elif dataset == 'cinic10':
        im_channels = 3
    elif dataset == 'imagenet':
        im_channels = 3
    elif dataset == 'tinyimagenet':
        im_channels = 3
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    return im_channels

def get_num_classes_from_name(dataset: str):
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'svhn':
        num_classes = 10
    elif dataset == 'fmnist':
        num_classes = 10
    elif dataset == 'cinic10':
        num_classes = 10
    elif dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'tinyimagenet':
        num_classes = 200
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    return num_classes


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class AuditingDataConfigs:
    # Mandatory arguments
    n_auditing_samples: int
    # Optional arguments with default values
    in_perc: Optional[float] = 0.5
    seed: Optional[int] = 12345


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class ShadowDataConfigs:
    # Mandatory arguments
    n_shadow_datasets: int
    n_samples_per_dataset: int
    mode: str
    # Optional arguments with default values
    test_perc: Optional[float] = 0.5
    seed: Optional[int] = 12345


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class AttackConfigs:
    # Shared arguments
    mode: str = 'offline'

    # RobustMIA
    alpha: Union[float, list[float]] = 0.5
    gamma: float = 1
    random_pop_size: int = 1000
    # LiRA

    # Quantile MIA
    n_quantile: int = 100
    low_quantile: float = 0.01
    high_quantile: float = 0.99 
    use_logscale: bool = False
    use_gaussian: bool = False
    quantile_alpha: float = 0.05

    # Neural MIA
    neural_input_mode: str = 'logit'
    model_layers: list[int] = field(default_factory=lambda: [64, 32])
    model_epochs: int = 10
    model_lr: float = 0.01

    # Attack-R MIA
    r_alpha: float = 0.05
    r_score_type: str = 'loss'  # options: loss, confidence, entropy

    # Attack-P MIA
    p_alpha: float = 0.05
    p_score_type: str = 'loss'  # options: loss, confidence, entropy


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class VictimConfigs:
    hash: str
    dataset: DatasetConfigs
    log: LogConfigs
    model: ModelConfigs
    train: TrainConfigs


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class AttackerConfigs:
    hash: str
    log: LogConfigs
    model: ModelConfigs
    train: TrainConfigs
    audit: AuditingDataConfigs
    shadow: ShadowDataConfigs
    attack: AttackConfigs


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class ExperimentConfigs:
    hash: str
    victim: VictimConfigs
    attacker: AttackerConfigs


def get_relevant_settings(settings: Any, mode: str = 'attacker') -> Dict[str, Any]:
    assert mode in ['attacker', 'victim', 'experiment'], f"Mode '{mode}' to get relevant settings not recognized! Choose between 'attacker', 'victim' or 'experiment'."
    if mode == 'attacker':
        attacker_settings = ['dataset', 'attacker_model', 'attack_mode', 
                            'n_auditing_samples', 'audit_in_perc', 'n_shadows', 'n_samples_per_shadow_dataset', 'shadow_test_perc', 
                            'random_population_size', 'robust_alphas', 'robust_gamma', 
                            'n_quantile', 'low_quantile', 'high_quantile', 'quantile_alpha', 'quantile_use_logscale', 'quantile_use_gaussian',
                            'neural_model_layers', 'neural_model_epochs', 'neural_model_lr',
                            'r_alpha']
        relevant_settings = {k: v for k, v in vars(settings).items() if k.startswith('att_') or k in attacker_settings}
    elif mode == 'victim':
        victim_settings = ['dataset', 'victim_model', 'data_augmentation', 'perf_metrics', 'perf_metric_to_track']
        relevant_settings = {k: v for k, v in vars(settings).items() if k.startswith('victim_') or k in victim_settings}
    else:
        exclude_keys = ["resume", "device"]
        relevant_settings = {k: str(v) if isinstance(v, pathlib.PosixPath) else v for k, v in vars(settings).items() if k not in exclude_keys}
    return relevant_settings

def get_hash_from_settings(settings: Any, mode: str = 'attacker') -> str:
    relevant_settings = get_relevant_settings(settings, mode=mode)
    # Convert settings dict to a JSON string with sorted keys to ensure consistent ordering
    settings_str = json.dumps(relevant_settings, sort_keys=True)
    # Create a MD5 hash of the settings string
    hash_object = hashlib.md5(settings_str.encode())
    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()

def generate_configs_from_settings(settings: Any) -> ExperimentConfigs:
    exp_hash = get_hash_from_settings(settings, mode='experiment')
    victim_hash = get_hash_from_settings(settings, mode='victim')
    attacker_hash = get_hash_from_settings(settings, mode='attacker')
    exp_log_folder = settings.out_folder/'experiments'/exp_hash/'logs'
    victim_ckpts_folder = settings.out_folder/'victims'/victim_hash/'ckpts'
    victim_resume_ckpts_folder = settings.out_folder/'victims'/victim_hash/'resume_ckpts'
    attacker_ckpts_folder = settings.out_folder/'attackers'/attacker_hash/'ckpts'
    attacker_resume_ckpts_folder = settings.out_folder/'attackers'/attacker_hash/'resume_ckpts'
    victim_dp_config = DPConfigs(use_dp=settings.victim_use_dp,
                                noise_multiplier=settings.victim_dp_noise_multiplier,
                                max_grad_norm=settings.victim_dp_max_grad_norm,
                                clip_per_layer=settings.victim_dp_clip_per_layer,
                                grad_sample_mode=settings.victim_dp_grad_sample_mode)

    victim_metric_config=MetricConfigs(use_metric=settings.use_metric,
                                        metric_d =settings.metric_d,
                                        metric_b = settings.metric_b)


    victim_dataset_configs = DatasetConfigs(name=settings.dataset,
                                            data_folder=settings.datasets_folder,
                                            data_augmentation=settings.data_augmentation)
    victim_model_configs = ModelConfigs(model_name=settings.victim_model,
                                        im_channels=victim_dataset_configs.im_channels,
                                        num_classes=victim_dataset_configs.num_classes,
                                        im_size=victim_dataset_configs.im_size)
    victim_optimizer_configs = OptimizerConfigs(name=settings.victim_optimizer,
                                                lr=settings.victim_lr,
                                                weight_decay=settings.victim_weight_decay,
                                                momentum=settings.victim_momentum,
                                                nesterov=settings.victim_nesterov,)
    victim_scheduler_configs = build_scheduler_configs_from_settings(settings, mode='victim')
    victim_train_configs = TrainConfigs(optimizer_config=victim_optimizer_configs,
                                        scheduler_config=victim_scheduler_configs,
                                        dp_config=victim_dp_config,
                                        metric_config=victim_metric_config,
                                        batch_size=settings.victim_batch_size,
                                        device=settings.device,
                                        distributed=settings.distributed,
                                        seed=settings.victim_seed,
                                        resume=settings.resume,
                                        ckpts_folder=victim_ckpts_folder,
                                        resume_ckpts_folder=victim_resume_ckpts_folder)

    victim_log_configs = LogConfigs(name='victim',
                                    log_folder=exp_log_folder,
                                    log_mode='smart')
    victim_configs = VictimConfigs(hash=victim_hash,
                                    dataset=victim_dataset_configs,
                                    log=victim_log_configs,
                                    model=victim_model_configs,
                                    train=victim_train_configs)

    attacker_dp_config = DPConfigs(use_dp=settings.att_use_dp,
                                noise_multiplier=settings.att_dp_noise_multiplier,
                                max_grad_norm=settings.att_dp_max_grad_norm,
                                clip_per_layer=settings.att_dp_clip_per_layer,
                                grad_sample_mode=settings.att_dp_grad_sample_mode)
    attacker_log_configs = LogConfigs(name='attacker',
                                    log_folder=exp_log_folder,
                                    log_mode='smart')
    attacker_model_configs = ModelConfigs(model_name=settings.att_model,
                                            im_channels=victim_dataset_configs.im_channels,
                                            num_classes=victim_dataset_configs.num_classes,
                                            im_size=victim_dataset_configs.im_size)
    attacker_optimizer_configs = OptimizerConfigs(name=settings.att_optimizer,
                                                    lr=settings.att_lr,
                                                    weight_decay=settings.att_weight_decay,
                                                    momentum=settings.att_momentum,
                                                    nesterov=settings.att_nesterov,)
    attacker_scheduler_configs = build_scheduler_configs_from_settings(settings, mode='attacker')
    attacker_train_configs = TrainConfigs(optimizer_config=attacker_optimizer_configs,
                                            scheduler_config=attacker_scheduler_configs,
                                            dp_config=attacker_dp_config,
                                            batch_size=settings.att_batch_size,
                                            loss=settings.att_loss,
                                            metrics=settings.perf_metrics,
                                            metric_to_track=settings.perf_metric_to_track,
                                            device=settings.device,
                                            distributed=settings.distributed,
                                            seed=settings.att_seed,
                                            resume=settings.resume,
                                            ckpts_folder=attacker_ckpts_folder,
                                            resume_ckpts_folder=attacker_resume_ckpts_folder)
    attacker_auditing_configs = AuditingDataConfigs(n_auditing_samples=settings.n_auditing_samples,
                                                in_perc=settings.audit_in_perc,
                                                seed=settings.att_seed)
    attacker_shadow_configs = ShadowDataConfigs(n_shadow_datasets=settings.n_shadows,
                                                n_samples_per_dataset=settings.n_samples_per_shadow_dataset,
                                                mode='online',
                                                test_perc=settings.shadow_test_perc,
                                                seed=settings.att_seed)

    if settings.attack_mode in ['online_robust', 'offline_robust', 'on_robust', 'off_robust']:
        robust_mode = 'online' if settings.attack_mode in ['online_robust', 'on_robust'] else 'offline'
        attack_configs = AttackConfigs(mode=robust_mode,
                                        alpha=settings.robust_alphas,
                                        gamma=settings.robust_gamma,
                                        random_pop_size=settings.random_population_size)
        attacker_shadow_configs.mode = robust_mode
    elif settings.attack_mode == 'lira':
        attack_configs = AttackConfigs(mode='online')
    elif settings.attack_mode == 'quantile':
        attacker_train_configs.metric_to_track = "quantile_coverage"
        attacker_train_configs.metrics = ["quantile_coverage"]
        attack_configs = AttackConfigs(mode='offline',
                                        n_quantile=settings.n_quantile,
                                        low_quantile=settings.low_quantile,
                                        high_quantile=settings.high_quantile,
                                        use_logscale=settings.quantile_use_logscale,
                                        use_gaussian=settings.quantile_use_gaussian,
                                        quantile_alpha=settings.quantile_alpha)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['neural_feat', 'neural_prob', 'neural_logit']:
        neural_input_mode = settings.attack_mode.split('_')[-1]
        attack_configs = AttackConfigs(mode='online',
                                        neural_input_mode=neural_input_mode,
                                        model_layers=settings.neural_model_layers,
                                        model_epochs=settings.neural_model_epochs,
                                        model_lr=settings.neural_model_lr)
    elif settings.attack_mode in ['rmia_loss', 'rmia_confidence', 'rmia_entropy']:
        score_type = settings.attack_mode.split('_')[-1]
        attack_configs = AttackConfigs(mode='offline',
                                        r_alpha=settings.r_alpha,
                                        r_score_type=score_type)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['pmia_loss', 'pmia_confidence', 'pmia_entropy']:
        score_type = settings.attack_mode.split('_')[-1]
        attack_configs = AttackConfigs(mode='offline',
                                        p_alpha=settings.p_alpha,
                                        p_score_type=score_type)
        attacker_shadow_configs.n_shadow_datasets = 1
        attacker_shadow_configs.mode = 'offline'
        attacker_shadow_configs.test_perc = 1.0
    else:
        raise ValueError('Attack mode "{}" not recognized!'.format(settings.attack_mode))
    attacker_configs = AttackerConfigs(hash=attacker_hash,
                                        log=attacker_log_configs,
                                        model=attacker_model_configs,
                                        train=attacker_train_configs,
                                        audit=attacker_auditing_configs,
                                        shadow=attacker_shadow_configs,
                                        attack=attack_configs)
    experiment_configs = ExperimentConfigs(hash=exp_hash,
                                            victim=victim_configs,
                                            attacker=attacker_configs)
    return experiment_configs
