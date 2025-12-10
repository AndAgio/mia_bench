import pathlib
from dataclasses import dataclass
from src.utils.variables import DEFAULT_MODELS_FOLDER, DEFAULT_RESUME_CKPTS_FOLDER, DEFAULT_LOG_FOLDER, DEFAULT_DATASETS_FOLDER
from pydantic import validate_arguments
from typing import Optional, Tuple, Union, Callable


@validate_arguments
@dataclass
class TrainConfigs:
    # Mandatory arguments
    optimizer: str
    lr: float
    epochs: int
    batch_size: int
    # Optional arguments with default values
    loss: Optional[Union[str, Callable]] = 'crossentropy'
    lr_sched: Optional[str] = 'const'
    device: Optional[str] = 'cpu'
    seed: Optional[int] = 12345
    distributed: Optional[bool] = False
    ckpts_folder: Optional[pathlib.Path] = DEFAULT_MODELS_FOLDER
    resume_ckpts_folder: Optional[pathlib.Path] = DEFAULT_RESUME_CKPTS_FOLDER


@validate_arguments
@dataclass
class ModelConfigs:
    # Mandatory arguments
    model_name: str
    im_channels: int
    num_classes: int
    im_size: Tuple[int, ...]


@validate_arguments
@dataclass
class LogConfigs:
    # Mandatory arguments
    name: str
    # Optional arguments with default values
    log_folder: pathlib.Path = DEFAULT_LOG_FOLDER
    log_mode: str = 'smart'


@validate_arguments
@dataclass
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
    elif dataset == 'imagenet':
        im_size = (224, 224)
    elif dataset == 'tiny_imagenet':
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
    elif dataset == 'imagenet':
        im_channels = 3
    elif dataset == 'tiny_imagenet':
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
    elif dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    return num_classes


@validate_arguments
@dataclass
class AuditingDataConfigs:
    # Mandatory arguments
    n_auditing_samples: int
    # Optional arguments with default values
    in_perc: Optional[float] = 0.5
    seed: Optional[int] = 12345


@validate_arguments
@dataclass
class ShadowDataConfigs:
    # Mandatory arguments
    n_shadow_datasets: int
    n_samples_per_dataset: int
    mode: str
    # Optional arguments with default values
    test_perc: Optional[float] = 0.5
    seed: Optional[int] = 12345


@validate_arguments
@dataclass
class AttackConfigs:
    # Optional arguments with default values

    # RMIA
    mode: str = 'offline'
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
