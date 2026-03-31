import pathlib
import hashlib
import json
# from dataclasses import dataclass, field
from src.utils.variables import DEFAULT_MODELS_FOLDER, DEFAULT_RESUME_CKPTS_FOLDER, DEFAULT_LOG_FOLDER, DEFAULT_DATASETS_FOLDER
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field, ValidationError, TypeAdapter
from typing import Optional, Tuple, Union, Callable, List, Dict, Any, Annotated, Literal

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
class SchedulerConfigs:
    name: str = 'cosine'
    epochs: int = 100
    extra: Dict[str, Any] = field(default_factory=dict)
    
def build_scheduler_configs_from_settings(settings: Any, mode: str = 'defender') -> SchedulerConfigs:
    assert mode in ['defender', 'attacker'], f"Mode '{mode}' not available to build LR scheduler configurations!"
    name = settings.defender_lr_sched if mode == 'defender' else settings.att_lr_sched
    total_epochs = settings.defender_epochs if mode == 'defender' else settings.att_epochs
    extra = {}
    if name == 'warmup_step':
        extra['step_size'] = settings.defender_lr_step_size  if mode == 'defender' else settings.att_lr_step_size
        extra['step_gamma'] = settings.defender_lr_step_gamma if mode == 'defender' else settings.att_lr_step_gamma
        extra['warmup_multiplier'] = settings.defender_lr_warmup_multiplier if mode == 'defender' else settings.att_lr_warmup_multiplier
        extra['warmup_epochs'] = settings.defender_lr_warmup_epochs if mode == 'defender' else settings.att_lr_warmup_epochs
    elif name == 'warmup_exp':
        extra['exp_gamma'] = settings.defender_lr_exp_gamma if mode == 'defender' else settings.att_lr_exp_gamma
        extra['warmup_multiplier'] = settings.defender_lr_warmup_multiplier if mode == 'defender' else settings.att_lr_warmup_multiplier
        extra['warmup_epochs'] = settings.defender_lr_warmup_epochs if mode == 'defender' else settings.att_lr_warmup_epochs
    elif name == 'warmup_cosine':
        extra['cycle_step'] = settings.defender_lr_cycle_step if mode == 'defender' else settings.att_lr_cycle_step
        extra['cycle_gamma'] = settings.defender_lr_cycle_gamma if mode == 'defender' else settings.att_lr_cycle_gamma
        extra['cosine_min'] = settings.defender_lr_cosine_min if mode == 'defender' else settings.att_lr_cosine_min
    elif name == 'step':
        extra['step_size'] = settings.defender_lr_step_size  if mode == 'defender' else settings.att_lr_step_size
        extra['step_gamma'] = settings.defender_lr_step_gamma if mode == 'defender' else settings.att_lr_step_gamma
    elif name == 'multistep':
        extra['step_milestones'] = settings.defender_lr_step_milestones  if mode == 'defender' else settings.defender_lr_step_milestones
        extra['step_gamma'] = settings.defender_lr_step_gamma if mode == 'defender' else settings.att_lr_step_gamma
    elif name == 'exp':
        extra['exp_gamma'] = settings.defender_lr_exp_gamma if mode == 'defender' else settings.att_lr_exp_gamma
    elif name == 'cosine':
        extra['cosine_min'] = settings.defender_lr_cosine_min if mode == 'defender' else settings.att_lr_cosine_min
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
    # dp_config: DPConfigs = field(default_factory=DPConfigs)
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
    seed: int = 12345
    val_split: float = 0.2

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


# @dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
# class AttackConfigs:
#     # Shared arguments
#     mode: str = 'offline'

#     # RobustMIA
#     alpha: Union[float, list[float]] = 0.5
#     gamma: float = 1
#     random_pop_size: int = 1000
#     # LiRA

#     # Quantile MIA
#     n_quantile: int = 100
#     low_quantile: float = 0.01
#     high_quantile: float = 0.99 
#     use_logscale: bool = False
#     use_gaussian: bool = False
#     quantile_alpha: float = 0.05

#     # Neural MIA
#     neural_input_mode: str = 'logit'
#     model_layers: list[int] = field(default_factory=lambda: [64, 32])
#     model_epochs: int = 10
#     model_lr: float = 0.01

#     # Attack-R MIA
#     r_alpha: float = 0.05
#     r_score_type: str = 'loss'  # options: loss, confidence, entropy

#     # Attack-P MIA
#     p_alpha: float = 0.05
#     p_score_type: str = 'loss'  # options: loss, confidence, entropy

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class RobustMiaConfigs:
    strategy: Literal["robust"] = "robust"
    mode: str = 'offline'
    alpha: Union[float, list[float]] = 0.5
    gamma: float = 1
    random_pop_size: int = 1000


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class LiraMiaConfigs:
    strategy: Literal["lira"] = "lira"
    mode: str = 'online'


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class QuantileMiaConfigs:
    strategy: Literal["quantile"] = "quantile"
    mode: str = 'offline'
    n_quantile: int = 100
    low_quantile: float = 0.01
    high_quantile: float = 0.99 
    use_logscale: bool = False
    use_gaussian: bool = False
    quantile_alpha: float = 0.05


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class NeuralMiaConfigs:
    strategy: Literal["neural"] = "neural"
    mode: str = 'offline'
    neural_input_mode: str = 'logit'
    model_layers: list[int] = field(default_factory=lambda: [64, 32])
    model_epochs: int = 10
    model_lr: float = 0.01

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class AttackRMiaConfigs:
    strategy: Literal["attack_r"] = "attack_r"
    mode: str = 'offline'
    r_alpha: float = 0.05
    r_score_type: str = 'loss'  # options: loss, confidence, entropy


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class AttackPMiaConfigs:
    strategy: Literal["attack_p"] = "attack_p"
    mode: str = 'offline'
    p_alpha: float = 0.05
    p_score_type: str = 'loss'  # options: loss, confidence, entropy


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class QebaConfigs:
    reduction_mode: str = 'spatial'  # only for qeba variants
    reduction_factor: int = 8  # only for qeba variants

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class BoundaryMiaConfigs:
    n_queries: int = 1000
    mode: str = 'hsj' # options: hsj, hopskipjump, hopskip, hop, qeba, qeba-spatial, qeba-dct, qeba-pca, qeba-custom
    norm: Literal["l2", "linf"] = "l2"
    qeba: Optional[QebaConfigs] = None

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class SupervisedBoundaryAttackConfig:
    strategy: Literal["sba"] = "sba"
    mode: str = 'offline'
    boundary: BoundaryMiaConfigs = field(default_factory=BoundaryMiaConfigs)

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class UnsupervisedBoundaryAttackConfig:
    strategy: Literal["uba"] = "uba"
    mode: str = 'offline'
    boundary: BoundaryMiaConfigs = field(default_factory=BoundaryMiaConfigs)
    quantile: float = 0.95

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class NoiseRobustnessAttackConfig:
    strategy: Literal["noise_robustness"] = "noise_robustness"
    mode: str = 'offline'
    n_queries: int = 1000
    sigmas: Union[float, list[float]] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2])

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class TransferAttackConfig:
    strategy: Literal["transfer"] = "transfer"
    mode: str = 'offline'
    feature_mode: str = 'loss' # options: loss, max_confidence, entropy

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class OsloAttackConfig:
    strategy: Literal["oslo"] = "oslo"
    mode: str = 'offline'
    n_models: int = 10 # number of shadow models to train for the attack
    same_arch: bool = True # whether to use the same architecture for all shadow models or not
    source_models_ratio: float = 0.75 # ratio of shadow models to use as source models for the attack (the rest will be used as validation models to define the attack decision boundary)
    K: int = 10 # number of attack sub-procedures
    N: int = 1000 # number of attack iterations per sub-procedure
    max_epsilon: float = 4/255 # maximum perturbation for the attack
    ga_mode: str = 'difgsm' # algorithm for gradient ascent step: options are 'difgsm', 'mifgsm', 'tifgsm', 'tmifgsm'
    threshold: float = 0.01 # decision threshold for the attack

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class DHAttackConfig:
    strategy: Literal["dh"] = "dh"
    mode: str = 'offline'
    n_models: int = 10 # number of shadow models to train for the attack
    n_queries: int = 1000 # number of queries to perform inference
    fixed_input_mode: str = 'white' # options: 'white', 'black'

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class YoqoAttackConfig:
    strategy: Literal["yoqo"] = "yoqo"
    mode: str = 'online'
    adv_opt_max_iter: int = 50 # maximum number of iterations for the adversarial optimization procedure
    adv_opt_lr: float = 0.01 # learning rate for the adversarial optimization procedure
    adv_opt_loss_threshold: float = 6 # loss threshold for early stopping of the adversarial optimization procedure
    alpha: float = 2 # weight for the out-shadow-models loss in the adversarial optimization procedure
    gamma: float = 5 # weight for the MSE loss between the adversarial example and the original sample in the adversarial optimization procedure

AttackConfigs = Annotated[
    Union[RobustMiaConfigs, LiraMiaConfigs, QuantileMiaConfigs, NeuralMiaConfigs, AttackRMiaConfigs, AttackPMiaConfigs,
        SupervisedBoundaryAttackConfig, UnsupervisedBoundaryAttackConfig, NoiseRobustnessAttackConfig, TransferAttackConfig,
        OsloAttackConfig, DHAttackConfig, YoqoAttackConfig],
    Field(discriminator="strategy")
]


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class NoDefenseConfigs:
    """No-op defense."""
    strategy: Literal["none"] = "none"

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class DPDefenseConfigs:
    strategy: Literal["dp"] = "dp"
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    clip_per_layer: bool = False
    grad_sample_mode: str = 'hook'

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class MemGuardDefenseConfigs:
    strategy: Literal["mem_guard"] = "mem_guard"
    budget: float = 10.0
    shadow_attacker_model_layers: list[int] = field(default_factory=lambda: [64, 32])
    shadow_attacker_model_epochs: int = 30
    shadow_attacker_model_lr: float = 0.01

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class RelaxLossDefenseConfigs:
    strategy: Literal["relax_loss"] = "relax_loss"
    relax_alpha: float = 0.5

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class AdvRegDefenseConfigs:
    strategy: Literal["adv_reg"] = "adv_reg"
    shadow_attacker_model_layers: list[int] = field(default_factory=lambda: [64, 32])
    adv_lambda: float = 1.0
    shadow_attacker_k: int = 1

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class MixupDefenseConfigs:
    strategy: Literal["mixup"] = "mixup"
    alpha: float = 1.0

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class HampDefenseConfigs:
    strategy: Literal["hamp"] = "hamp"
    mode: str = 'full'  # options: 'train_only', 'test_only', 'full'
    gamma: float = 0.5
    alpha: float = 0.001

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class SelenaDefenseConfigs:
    strategy: Literal["selena"] = "selena"
    K: int = 25
    L: int = 10

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class MistDefenseConfigs:
    strategy: Literal["mist"] = "mist"
    num_submodels: int = 5
    split_method: str = 'random'  # options: 'random', 'stratified'
    submodel_epochs: int = 5
    lmbd: float = 4
    mix_up: bool = False
    alpha_mixup: float = 0.0

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class WeightedSmoothingDefenseConfigs:
    strategy: Literal["weighted_smoothing"] = "weighted_smoothing"
    sigma_noise: float = 0.1
    warmup_epochs: int = 5

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class PurifierDefenseConfigs:
    strategy: Literal["purifier"] = "purifier"
    reformer_latent_dim: int = 16
    reformer_hidden_dim: int = 128
    reformer_epochs: int = 20
    reformer_lr: float = 0.01
    reformer_batch_size: int = 256
    reformer_lambda: float = 1.0
    pindex_size: int = 1000
    swap_threshold: float = 0.01

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class MmdDefenseConfigs:
    strategy: Literal["mmd"] = "mmd"
    lmbd: float = 1.0
    use_mixup: bool = False
    mixup_alpha: float = 0.0

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class LdlDefenseConfigs:
    strategy: Literal["ldl"] = "ldl"
    n_queries: int = 100
    noise_type: str = 'bernoulli' # options: 'bernoulli', 'gaussian'
    noise_scale: float = 0.1

@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class DataAugmentationDefenseConfigs:
    strategy: Literal["data_augmentation"] = "data_augmentation"
    horizontal_flip: bool = False
    rotation: int = 10
    random_crop: int = 32
    jitter_brightness: float = 0.2
    jitter_hue: float = 0.2
    perspective_distortion_scale: float = 0.2


# One-of: only the selected strategy's fields are validated/available
DefenseConfigs = Annotated[
    Union[NoDefenseConfigs, DPDefenseConfigs, MemGuardDefenseConfigs, RelaxLossDefenseConfigs, 
        AdvRegDefenseConfigs, MixupDefenseConfigs, HampDefenseConfigs, SelenaDefenseConfigs, 
        MistDefenseConfigs, WeightedSmoothingDefenseConfigs, PurifierDefenseConfigs, MmdDefenseConfigs,
        LdlDefenseConfigs,
        DataAugmentationDefenseConfigs],
    Field(discriminator="strategy")
]


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class DefenderConfigs:
    hash: str
    dataset: DatasetConfigs
    log: LogConfigs
    model: ModelConfigs
    train: TrainConfigs
    defense: DefenseConfigs


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
    defender: DefenderConfigs
    attacker: AttackerConfigs


def get_relevant_settings(settings: Any, mode: str = 'attacker') -> Dict[str, Any]:
    assert mode in ['attacker', 'defender', 'experiment'], f"Mode '{mode}' to get relevant settings not recognized! Choose between 'attacker', 'defender' or 'experiment'."
    if mode == 'attacker':
        attacker_settings = ['dataset', 'val_split', 'attacker_model', 'attack_mode', 
                            'n_auditing_samples', 'audit_in_perc', 'n_shadows', 'n_samples_per_shadow_dataset', 'shadow_test_perc', 
                            'random_population_size', 'robust_alphas', 'robust_gamma', 
                            'n_quantile', 'low_quantile', 'high_quantile', 'quantile_alpha', 'quantile_use_logscale', 'quantile_use_gaussian',
                            'neural_model_layers', 'neural_model_epochs', 'neural_model_lr',
                            'r_alpha']
        relevant_settings = {k: v for k, v in vars(settings).items() if k.startswith('att_') or k in attacker_settings}
        if settings.attack_mode == 'quantile':
            relevant_settings.pop('n_shadows')
    elif mode == 'defender':
        defender_settings = ['dataset', 'val_split', 'defender_model', 'data_augmentation', 'perf_metrics', 'perf_metric_to_track']
        relevant_settings = {k: v for k, v in vars(settings).items() if k.startswith('defender_') or k in defender_settings}
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
    defender_hash = get_hash_from_settings(settings, mode='defender')
    attacker_hash = get_hash_from_settings(settings, mode='attacker')
    exp_log_folder = settings.out_folder/'experiments'/exp_hash/'logs'
    defender_ckpts_folder = settings.out_folder/'defenders'/defender_hash/'ckpts'
    defender_resume_ckpts_folder = settings.out_folder/'defenders'/defender_hash/'resume_ckpts'
    attacker_ckpts_folder = settings.out_folder/'attackers'/attacker_hash/'ckpts'
    attacker_resume_ckpts_folder = settings.out_folder/'attackers'/attacker_hash/'resume_ckpts'
    # defender_dp_config = DPConfigs(use_dp=settings.defender_use_dp,
    #                             noise_multiplier=settings.defender_dp_noise_multiplier,
    #                             max_grad_norm=settings.defender_dp_max_grad_norm,
    #                             clip_per_layer=settings.defender_dp_clip_per_layer,
    #                             grad_sample_mode=settings.defender_dp_grad_sample_mode)
    defender_dataset_configs = DatasetConfigs(name=settings.dataset,
                                            data_folder=settings.datasets_folder,
                                            data_augmentation=settings.data_augmentation,
                                            seed=settings.defender_seed,
                                            val_split=settings.val_split)
    defender_model_configs = ModelConfigs(model_name=settings.defender_model,
                                        im_channels=defender_dataset_configs.im_channels,
                                        num_classes=defender_dataset_configs.num_classes,
                                        im_size=defender_dataset_configs.im_size)
    defender_optimizer_configs = OptimizerConfigs(name=settings.defender_optimizer,
                                                lr=settings.defender_lr,
                                                weight_decay=settings.defender_weight_decay,
                                                momentum=settings.defender_momentum,
                                                nesterov=settings.defender_nesterov,)
    defender_scheduler_configs = build_scheduler_configs_from_settings(settings, mode='defender')
    defender_train_configs = TrainConfigs(optimizer_config=defender_optimizer_configs,
                                        scheduler_config=defender_scheduler_configs,
                                        # dp_config=defender_dp_config,
                                        batch_size=settings.defender_batch_size,
                                        device=settings.device,
                                        distributed=settings.distributed,
                                        seed=settings.defender_seed,
                                        resume=settings.resume,
                                        ckpts_folder=defender_ckpts_folder,
                                        resume_ckpts_folder=defender_resume_ckpts_folder)
    defender_log_configs = LogConfigs(name='defender',
                                    log_folder=exp_log_folder,
                                    log_mode='smart')

    if settings.defense_mode in ['no', 'none', 'vanilla']:
        defender_defense_configs = NoDefenseConfigs()
    elif settings.defense_mode == 'dp':
        defender_defense_configs = DPDefenseConfigs(noise_multiplier=settings.defender_dp_noise_multiplier,
                                                    max_grad_norm=settings.defender_dp_max_grad_norm,
                                                    clip_per_layer=settings.defender_dp_clip_per_layer,
                                                    grad_sample_mode=settings.defender_dp_grad_sample_mode)
    elif settings.defense_mode in ['mem_guard', 'memguard', 'mem-guard']:
        defender_defense_configs = MemGuardDefenseConfigs(shadow_attacker_model_layers=settings.defender_mem_guard_shadow_model_layers,
                                                        shadow_attacker_model_epochs=settings.defender_mem_guard_shadow_model_epochs,
                                                        shadow_attacker_model_lr=settings.defender_mem_guard_shadow_model_lr,
                                                        budget=settings.defender_mem_guard_budget)
    elif settings.defense_mode in ['relax_loss', 'relaxloss', 'relax-loss']:
        defender_defense_configs = RelaxLossDefenseConfigs(relax_alpha=settings.defender_relax_loss_alpha)
    elif settings.defense_mode in ['adv_reg', 'advreg', 'adv-reg']:
        defender_defense_configs = AdvRegDefenseConfigs(shadow_attacker_model_layers=settings.defender_adv_reg_shadow_attacker_model_layers,
                                                        adv_lambda=settings.defender_adv_reg_lambda,
                                                        shadow_attacker_k=settings.defender_adv_reg_shadow_attacker_k)
    elif settings.defense_mode in ['mixup']:
        defender_defense_configs = MixupDefenseConfigs(alpha=settings.defender_mixup_alpha)
    elif settings.defense_mode in ['hamp_train', 'hamp_test', 'hamp_full', 'hamp']:
        if settings.defense_mode == 'hamp_train':
            hamp_mode = 'train_only'
        elif settings.defense_mode == 'hamp_test':
            hamp_mode = 'test_only'
        elif settings.defense_mode in ['hamp_full', 'hamp']:
            hamp_mode = 'full'
        else:
            raise ValueError('Hamp defense mode "{}" not recognized!'.format(settings.defense_mode))
        defender_defense_configs = HampDefenseConfigs(mode=hamp_mode,
                                                    gamma=settings.defender_hamp_gamma,
                                                    alpha=settings.defender_hamp_alpha)
    elif settings.defense_mode in ['selena']:
        defender_defense_configs = SelenaDefenseConfigs(K=settings.defender_selena_K,
                                                        L=settings.defender_selena_L)
    elif settings.defense_mode in ['mist', 'mist_mixup', 'mist-mixup']:
        mixup = settings.defense_mode in ['mist_mixup', 'mist-mixup']
        defender_defense_configs = MistDefenseConfigs(num_submodels=settings.defender_mist_num_submodels,
                                                    split_method=settings.defender_mist_split_method,
                                                    submodel_epochs=settings.defender_mist_submodel_epochs,
                                                    lmbd=settings.defender_mist_lambda,
                                                    mixup=mixup,
                                                    alpha_mixup=settings.defender_mixup_alpha if mixup else 0.0)
    elif settings.defense_mode in ['weighted_smoothing', 'weighted-smoothing', 'weighted_smooth', 'weighted-smooth', 'weightedsmoothing', 'weightedsmooth', 'ws']:
        defender_defense_configs = WeightedSmoothingDefenseConfigs(sigma_noise=settings.defender_weighted_smoothing_sigma_noise,
                                                                warmup_epochs=settings.defender_weighted_smoothing_warmup_epochs)
    elif settings.defense_mode in ['purifier']:
        defender_defense_configs = PurifierDefenseConfigs(reformer_latent_dim=settings.defender_purifier_reformer_latent_dim,
                                                        reformer_hidden_dim=settings.defender_purifier_reformer_hidden_dim,
                                                        reformer_epochs=settings.defender_purifier_reformer_epochs,
                                                        reformer_lr=settings.defender_purifier_reformer_lr,
                                                        reformer_batch_size=settings.defender_purifier_reformer_batch_size,
                                                        reformer_lambda=settings.defender_purifier_reformer_lambda,
                                                        pindex_size=settings.defender_purifier_pindex_size,
                                                        swap_threshold=settings.defender_purifier_swap_threshold)
    elif settings.defense_mode in ['mmd', 'mmd_mixup', 'mmd-mixup']:
        defender_defense_configs = MmdDefenseConfigs(lmbd=settings.defender_mmd_lambda,
                                                    use_mixup=True if settings.defense_mode in ['mmd_mixup', 'mmd-mixup'] else False,
                                                    mixup_alpha=settings.defender_mixup_alpha)
    elif settings.defense_mode in ['ldl']:
        if settings.dataset in ["cifar10", "cifar100", "svhn", "fmnist", "cinic10", "imagenet", "tinyimagenet"]:
            noise_type = "normal"
        elif settings.dataset in []:
            noise_type = "bernoulli"
        defender_defense_configs = LdlDefenseConfigs(n_queries=settings.defender_ldl_n_queries,
                                                    noise_type=noise_type,
                                                    noise_scale=settings.defender_ldl_noise_scale)
    elif settings.defense_mode in ['data_augmentation', 'data-augmentation', 'dataaug', 'data-aug']:
        defender_defense_configs = DataAugmentationDefenseConfigs(horizontal_flip=settings.defender_augment_horizontal_flip,
                                                                rotation=settings.defender_augment_rotation,
                                                                random_crop=settings.defender_augment_random_crop,
                                                                jitter_brightness=settings.defender_augment_jitter_brightness,
                                                                jitter_hue=settings.defender_augment_jitter_hue,
                                                                perspective_distortion_scale=settings.defender_augment_perspective_distortion_scale)
    else:
        raise ValueError('Defense mode "{}" not recognized!'.format(settings.defense_mode))
    defender_configs = DefenderConfigs(hash=defender_hash,
                                    dataset=defender_dataset_configs,
                                    log=defender_log_configs,
                                    model=defender_model_configs,
                                    train=defender_train_configs,
                                    defense=defender_defense_configs)

    # attacker_dp_config = DPConfigs(use_dp=settings.att_use_dp,
    #                             noise_multiplier=settings.att_dp_noise_multiplier,
    #                             max_grad_norm=settings.att_dp_max_grad_norm,
    #                             clip_per_layer=settings.att_dp_clip_per_layer,
    #                             grad_sample_mode=settings.att_dp_grad_sample_mode)
    attacker_log_configs = LogConfigs(name='attacker',
                                    log_folder=exp_log_folder,
                                    log_mode='smart')
    attacker_model_configs = ModelConfigs(model_name=settings.att_model,
                                            im_channels=defender_dataset_configs.im_channels,
                                            num_classes=defender_dataset_configs.num_classes,
                                            im_size=defender_dataset_configs.im_size)
    attacker_optimizer_configs = OptimizerConfigs(name=settings.att_optimizer,
                                                    lr=settings.att_lr,
                                                    weight_decay=settings.att_weight_decay,
                                                    momentum=settings.att_momentum,
                                                    nesterov=settings.att_nesterov,)
    attacker_scheduler_configs = build_scheduler_configs_from_settings(settings, mode='attacker')
    attacker_train_configs = TrainConfigs(optimizer_config=attacker_optimizer_configs,
                                            scheduler_config=attacker_scheduler_configs,
                                            # dp_config=attacker_dp_config,
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
        attack_configs = RobustMiaConfigs(mode=robust_mode,
                                        alpha=settings.robust_alphas,
                                        gamma=settings.robust_gamma,
                                        random_pop_size=settings.random_population_size)
        attacker_shadow_configs.mode = robust_mode
    elif settings.attack_mode == 'lira':
        attack_configs = LiraMiaConfigs(mode='online')
    elif settings.attack_mode == 'quantile':
        attacker_train_configs.metric_to_track = "quantile_coverage"
        attacker_train_configs.metrics = ["quantile_coverage"]
        attack_configs = QuantileMiaConfigs(mode='offline',
                                            n_quantile=settings.n_quantile,
                                            low_quantile=settings.low_quantile,
                                            high_quantile=settings.high_quantile,
                                            use_logscale=settings.quantile_use_logscale,
                                            use_gaussian=settings.quantile_use_gaussian,
                                            quantile_alpha=settings.quantile_alpha)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['neural_feat', 'neural_prob', 'neural_logit']:
        neural_input_mode = settings.attack_mode.split('_')[-1]
        attack_configs = NeuralMiaConfigs(mode='online',
                                        neural_input_mode=neural_input_mode,
                                        model_layers=settings.neural_model_layers,
                                        model_epochs=settings.neural_model_epochs,
                                        model_lr=settings.neural_model_lr)
    elif settings.attack_mode in ['rmia_loss', 'rmia_confidence', 'rmia_entropy']:
        score_type = settings.attack_mode.split('_')[-1]
        attack_configs = AttackRMiaConfigs(mode='offline',
                                        r_alpha=settings.r_alpha,
                                        r_score_type=score_type)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['pmia_loss', 'pmia_confidence', 'pmia_entropy']:
        score_type = settings.attack_mode.split('_')[-1]
        attack_configs = AttackPMiaConfigs(mode='offline',
                                        p_alpha=settings.p_alpha,
                                        p_score_type=score_type)
        attacker_shadow_configs.n_shadow_datasets = 1
        attacker_shadow_configs.mode = 'offline'
        attacker_shadow_configs.test_perc = 1.0
    elif settings.attack_mode in ['sba', 'sba_hopskipjump', 'sba_hsj', 'sba_hopskip', 'sba_hop', 'sba_qeba', 'sba_qeba-spatial', 'sba_qeba-dct', 'sba_qeba-pca', 'sba_qeba-custom']:
        if settings.attack_mode in ['sba', 'sba_hopskipjump', 'sba_hsj', 'sba_hopskip', 'sba_hop']:
            bound_mode = 'hop_skip_jump'
        elif settings.attack_mode in ['sba_qeba', 'sba_qeba-spatial', 'sba_qeba-dct', 'sba_qeba-pca', 'sba_qeba-custom']:
            bound_mode = 'qeba'
            qeba_reduction_mode = settings.attack_mode.split('-')[-1] if '-' in settings.attack_mode else 'spatial'
            qeba_configs = QebaConfigs(reduction_mode=qeba_reduction_mode,
                                        reduction_factor=settings.bound_qeba_reduction_factor)
        else:
            raise ValueError('Supervised Boundary Attack mode "{}" not recognized!'.format(settings.attack_mode))
        boundary_configs = BoundaryMiaConfigs(n_queries=settings.bound_n_queries,
                                            mode=bound_mode,
                                            norm=settings.bound_norm,
                                            qeba=qeba_configs if bound_mode == 'qeba' else None)
        attack_configs = SupervisedBoundaryAttackConfig(boundary=boundary_configs,
                                                        mode='offline')
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['uba', 'uba_hopskipjump', 'uba_hsj', 'uba_hopskip', 'uba_hop', 'uba_qeba', 'uba_qeba-spatial', 'uba_qeba-dct', 'uba_qeba-pca', 'uba_qeba-custom']:
        if settings.attack_mode in ['uba', 'uba_hopskipjump', 'uba_hsj', 'uba_hopskip', 'uba_hop']:
            bound_mode = 'hop_skip_jump'
        elif settings.attack_mode in ['uba_qeba', 'uba_qeba-spatial', 'uba_qeba-dct', 'uba_qeba-pca', 'uba_qeba-custom']:
            bound_mode = 'qeba'
            qeba_reduction_mode = settings.attack_mode.split('-')[-1] if '-' in settings.attack_mode else 'spatial'
            qeba_configs = QebaConfigs(reduction_mode=qeba_reduction_mode,
                                        reduction_factor=settings.bound_qeba_reduction_factor)
        else:
            raise ValueError('Unsupervised Boundary Attack mode "{}" not recognized!'.format(settings.attack_mode))
        boundary_configs = BoundaryMiaConfigs(n_queries=settings.bound_n_queries,
                                        mode=bound_mode,
                                        norm=settings.bound_norm,
                                        qeba=qeba_configs if bound_mode == 'qeba' else None)
        attack_configs = UnsupervisedBoundaryAttackConfig(boundary=boundary_configs,
                                                        mode='offline',
                                                        quantile=settings.bound_quantile)
        attacker_shadow_configs.mode = 'offline'
    
    elif settings.attack_mode in ['noise_robust', 'noise_robustness', 'noise_rob', 'nr']:
        attack_configs = NoiseRobustnessAttackConfig(n_queries=settings.noise_robust_n_queries,
                                                    sigmas=settings.noise_robust_sigmas)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['transfer_loss', 'transfer_confidence', 'transfer_entropy']:
        feature_mode = settings.attack_mode.split('_')[-1]
        attack_configs = TransferAttackConfig(feature_mode=feature_mode)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['oslo', 'oslo_difgsm', 'oslo_mifgsm', 'oslo_tifgsm', 'oslo_tmifgsm']:
        ga_mode = settings.attack_mode.split('_')[-1] if '_' in settings.attack_mode else 'difgsm'
        attack_configs = OsloAttackConfig(n_models=settings.oslo_n_models,
                                        same_arch=settings.oslo_same_arch,
                                        source_models_ratio=settings.oslo_source_models_ratio,
                                        K=settings.oslo_K,
                                        N=settings.oslo_N,
                                        max_epsilon=settings.oslo_max_epsilon,
                                        ga_mode=ga_mode,
                                        threshold=settings.oslo_threshold)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['dh', 'dh_white', 'dh_black', 'dh_random', 'dh-attack', 'dh-attack_white', 'dh-attack_black', 'dh-attack_random']:
        fixed_input_mode = settings.attack_mode.split('_')[-1] if '_' in settings.attack_mode else 'white'
        attack_configs = DHAttackConfig(n_models=settings.dh_n_models,
                                        n_queries=settings.dh_n_queries,
                                        fixed_input_mode=fixed_input_mode)
        attacker_shadow_configs.mode = 'offline'
    elif settings.attack_mode in ['online_yoqo', 'offline_yoqo', 'on_yoqo', 'off_yoqo', 'yoqo']:
        yoqo_mode = 'online' if settings.attack_mode in ['online_yoqo', 'on_yoqo'] else 'offline'
        attack_configs = YoqoAttackConfig(mode=yoqo_mode,
                                        adv_opt_max_iter=settings.yoqo_adv_opt_max_iter,
                                        adv_opt_lr=settings.yoqo_adv_opt_lr,
                                        adv_opt_loss_threshold=settings.yoqo_adv_opt_loss_threshold,
                                        alpha=settings.yoqo_alpha,
                                        gamma=settings.yoqo_gamma)
        attacker_shadow_configs.mode = yoqo_mode
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
                                            defender=defender_configs,
                                            attacker=attacker_configs)
    return experiment_configs
