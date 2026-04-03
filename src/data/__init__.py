import os
import numpy as np
import copy
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
from torchvision.datasets import FashionMNIST
from .cinic import Cinic10
from .imagenet import ImageNet
from .imagenet1k import ImageNet1K
from .tinyimagenet import TinyImageNet
from .gtsrb import GTSRB
from .purchase import Purchase
from .texas import Texas
from .news import News
from .helpers import (
    MultiDatasets,
    MySubset,
    MyOriginalIndexSubset,
    IndexedDataset,
    MergedIndexedDataset,
    SampleMetricTracker,
    build_merged_dataset,
    split_merged_dataset_into_chunks,
    get_stable_index_from_item,
    get_metric_for_sample,
    subset_by_original_indices,
    sample_subset_by_available_original_indices,
)

from .helpers import MyConcatDataset

from src.utils.variables import DEFAULT_DATASETS_FOLDER
from src.utils.configs import get_dataset_info_from_name


def get_dataset_mean_std(dataset: str):
    if dataset == 'cifar10':
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'svhn':
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
    elif dataset == 'fmnist':
        mean = [0.2861]
        std = [0.3530]
    elif dataset == 'cinic10':
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
    elif dataset in ['imagenet', 'imagenet1k']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == 'tinyimagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == 'gtsrb':
        mean = [0.3403, 0.3121, 0.3214]
        std = [0.2724, 0.2608, 0.2669]
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    return mean, std


def import_dataset_by_name(dataset: str, datasets_folder: str = DEFAULT_DATASETS_FOLDER, augment: bool = False, logger: callable = None):
    """Import and return the appropriate dataset based on the provided name."""
    printer_func = print if logger is None else logger.print_it
    printer_func('Gathering dataset "{}". This may take a while...'.format(dataset))
    # Image Preprocessing
    if dataset in ['cifar10', 'cifar100']:
        mean, std = get_dataset_mean_std(dataset)
        # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        # Setup train transforms
        train_transform = transforms.Compose([])
        if augment:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.RandomRotation(10))
            train_transform.transforms.append(transforms.ColorJitter(brightness=0.2, hue=0.1))
            train_transform.transforms.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize(mean=mean, 
                                                                std=std))
        # Setup test transforms
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize(mean=mean, 
                                                                std=std)])
    elif dataset == 'svhn':
        mean, std = get_dataset_mean_std(dataset)
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    elif dataset == 'fmnist':
        mean, std = get_dataset_mean_std(dataset)
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    elif dataset == 'cinic10':
        mean, std = get_dataset_mean_std(dataset)
        train_transform = transforms.Compose([])
        if augment:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.RandomRotation(10))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize(mean=mean, 
                                                                std=std))
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize(mean=mean, 
                                                                std=std)])
    elif dataset in ['imagenet', 'imagenet1k']:
        mean, std = get_dataset_mean_std(dataset)
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.RandomResizedCrop(224))
        if augment:
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        test_transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize])
    elif dataset == 'tinyimagenet':
        mean, std = get_dataset_mean_std(dataset)
        normalize = transforms.Normalize(mean=mean, std=std)
        # Setup train transforms for Tiny ImageNet
        train_transform = transforms.Compose([])
        if augment:
            train_transform.transforms.append(transforms.RandomCrop(64, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        # Setup test transforms for Tiny ImageNet
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif dataset == 'gtsrb':
        mean, std = get_dataset_mean_std(dataset)
        normalize = transforms.Normalize(mean=mean, std=std)
        # Setup train transforms for GTSRB
        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.Resize((32, 32)))
        if augment:
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.RandomRotation(10))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        # Setup test transforms for GTSRB
        test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), normalize])
    elif dataset == 'purchase':
        train_transform = None
        test_transform = None
    elif dataset == 'texas':
        train_transform = None
        test_transform = None
    elif dataset == 'news':
        train_transform = None
        test_transform = None
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))

    # Load the appropriate train and test datasets
    if dataset == 'cifar10':
        root = os.path.join(datasets_folder, 'cifar10')
        train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        root = os.path.join(datasets_folder, 'cifar100')
        train_dataset = CIFAR100(root=root, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        root = os.path.join(datasets_folder, 'svhn')
        train_dataset = SVHN(root=root, split='train', download=True, transform=train_transform)
        train_dataset.targets = train_dataset.labels
        test_dataset = SVHN(root=root, split='test', download=True, transform=test_transform)
        test_dataset.targets = test_dataset.labels
    elif dataset == 'fmnist':
        root = os.path.join(datasets_folder, 'fmnist')
        train_dataset = FashionMNIST(root=root, train=True, download=True, transform=train_transform)
        test_dataset = FashionMNIST(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'cinic10':
        root = os.path.join(datasets_folder, 'cinic10')
        train_dataset = Cinic10(root=root, train=True, download=True, transform=train_transform)
        test_dataset = Cinic10(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'imagenet':
        root = os.path.join(datasets_folder, 'imagenet')
        train_dataset = ImageNet(root=root, split='train', download=True, transform=train_transform)
        test_dataset = ImageNet(root=root, split='val', download=True, transform=test_transform)
    elif dataset == 'imagenet1k':
        root = os.path.join(datasets_folder, 'imagenet1k')
        train_dataset = ImageNet1K(root=root, split='train', download=True, transform=train_transform)
        test_dataset = ImageNet1K(root=root, split='val', download=True, transform=test_transform)
    elif dataset == 'tinyimagenet':
        root = os.path.join(datasets_folder, 'tinyimagenet')
        train_dataset = TinyImageNet(root=root, train=True, transform=train_transform, download=True)
        test_dataset = TinyImageNet(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'purchase':
        root = os.path.join(datasets_folder, 'purchase')
        train_dataset = Purchase(root=root, train=True, transform=train_transform, download=True)
        test_dataset = Purchase(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'texas':
        root = os.path.join(datasets_folder, 'texas')
        train_dataset = Texas(root=root, train=True, transform=train_transform, download=True)
        test_dataset = Texas(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'news':
        root = os.path.join(datasets_folder, 'news')
        train_dataset = News(root=root, train=True, transform=train_transform, download=True)
        test_dataset = News(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'gtsrb':
        root = os.path.join(datasets_folder, 'gtsrb')
        train_dataset = GTSRB(root=root, train=True, transform=train_transform, download=True)
        test_dataset = GTSRB(root=root, train=False, transform=test_transform, download=True)
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    # Merge the train and test datasets into a single dataset with a global index space, while preserving original indices
    merged_dataset = build_merged_dataset([train_dataset, test_dataset],
                                            dataset_names=['train', 'test'])
    logger.print_it(f"Loaded dataset '{dataset}' with {len(train_dataset)} training samples and {len(test_dataset)} testing samples. Merged dataset has {len(merged_dataset)} samples in total.")
    return merged_dataset

def get_defender_datas(dataset: str, datasets_folder: str = DEFAULT_DATASETS_FOLDER, def_split: float = 0.5, att_split: float = 0.5, seed: int= 12345, augment: bool = False, logger: callable = None):
    """Return dataset splits packed into a `MultiDatasets` object.

    Parameters:
    - val_split: if >0 and <=1 treated as fraction of training set to use as validation; if >=1 treated as absolute number of samples.
    - val_indices: explicit list of indices (w.r.t. original training set ordering) to use as validation. If provided, overrides `val_split`.
    - val_seed: seed used when sampling a validation split randomly (deterministic if `val_shuffle` is True). If `None` and `rng` is provided, the external RNG will be used.
    - val_shuffle: whether to shuffle training indices before taking a split.
    - rng: optional `random.Random`-like instance (with `.shuffle(list)` and `.randint(a,b)`) to be used for sampling. If not provided, a `random.Random(val_seed)` instance is used when `val_seed` is provided, otherwise the module-level `random` is used.
    - report_seed: if True, the function prints either the explicit `val_seed` (when used) or a small sample integer from the used RNG so that you can compare RNG states across scripts.

    The function preserves original training indices for `val` by creating `torch.utils.data.Subset`
    instances whose `.indices` attribute corresponds to the indices in the original full training set. This
    allows mapping back to precomputed per-sample statistics (e.g., memorization scores).
    """
    merged_dataset = import_dataset_by_name(dataset=dataset, datasets_folder=datasets_folder, augment=augment, logger=logger)
    # The merged dataset allows us to keep track of original indices across train/test splits, which is crucial for mapping back to precomputed per-sample statistics (e.g., memorization scores) that are typically computed on the original training set.
    # Now split the merged dataset between defender and attacker
    percentage_for_defender = def_split
    percentage_for_attacker = att_split
    # 5b) By fractions
    chunks_frac, chunks_frac_ids = split_merged_dataset_into_chunks(
        merged_dataset=merged_dataset,
        chunk_fractions=[percentage_for_defender*0.75, percentage_for_attacker*0.1, percentage_for_defender*0.15,
                        percentage_for_attacker],
        sampling_weights=[1.0 for _ in range(len(merged_dataset))],  # uniform sampling
        seed=seed
    )

    printer_func = print if logger is None else logger.print_it
    printer_func(f"Defender dataset split:")
    printer_func(f"  - Defender training set: {len(chunks_frac[0])} samples where first 10 indices are {chunks_frac_ids[0][:10]}")
    printer_func(f"  - Defender validation set: {len(chunks_frac[1])} samples where first 10 indices are {chunks_frac_ids[1][:10]}")
    printer_func(f"  - Defender testing set: {len(chunks_frac[2])} samples where first 10 indices are {chunks_frac_ids[2][:10]}")
    printer_func(f"Attacker dataset split:")
    printer_func(f"  - Attacker dataset: {len(chunks_frac[3])} samples where first 10 indices are {chunks_frac_ids[3][:10]}")

    defender_train = chunks_frac[0]
    defender_val = chunks_frac[1]
    defender_test = chunks_frac[2]
    data = MultiDatasets()
    data.add(defender_train, 'train')
    data.add(defender_val, 'val')
    data.add(defender_test, 'test')
    info = get_dataset_info_from_name(dataset=dataset)
    data.add_info(info)
    return data


def get_attacker_datas(dataset: str, datasets_folder: str = DEFAULT_DATASETS_FOLDER, def_split: float = 0.5, att_split: float = 0.5, seed: int= 12345, augment: bool = False, logger: callable = None):
    """Return dataset splits packed into a `MultiDatasets` object.

    Parameters:
    - val_split: if >0 and <=1 treated as fraction of training set to use as validation; if >=1 treated as absolute number of samples.
    - val_indices: explicit list of indices (w.r.t. original training set ordering) to use as validation. If provided, overrides `val_split`.
    - val_seed: seed used when sampling a validation split randomly (deterministic if `val_shuffle` is True). If `None` and `rng` is provided, the external RNG will be used.
    - val_shuffle: whether to shuffle training indices before taking a split.
    - rng: optional `random.Random`-like instance (with `.shuffle(list)` and `.randint(a,b)`) to be used for sampling. If not provided, a `random.Random(val_seed)` instance is used when `val_seed` is provided, otherwise the module-level `random` is used.
    - report_seed: if True, the function prints either the explicit `val_seed` (when used) or a small sample integer from the used RNG so that you can compare RNG states across scripts.

    The function preserves original training indices for `val` by creating `torch.utils.data.Subset`
    instances whose `.indices` attribute corresponds to the indices in the original full training set. This
    allows mapping back to precomputed per-sample statistics (e.g., memorization scores).
    """
    merged_dataset = import_dataset_by_name(dataset=dataset, datasets_folder=datasets_folder, augment=augment, logger=logger)
    # The merged dataset allows us to keep track of original indices across train/test splits, which is crucial for mapping back to precomputed per-sample statistics (e.g., memorization scores) that are typically computed on the original training set.
    # Now split the merged dataset between defender and attacker
    percentage_for_defender = def_split
    percentage_for_attacker = att_split
    # 5b) By fractions
    chunks_frac, chunks_frac_ids = split_merged_dataset_into_chunks(
        merged_dataset=merged_dataset,
        chunk_fractions=[percentage_for_defender*0.75, percentage_for_attacker*0.1, percentage_for_defender*0.15,
                        percentage_for_attacker],
        sampling_weights=[1.0 for _ in range(len(merged_dataset))],  # uniform sampling
        seed=seed
    )

    printer_func = print if logger is None else logger.print_it
    printer_func(f"Defender dataset split:")
    printer_func(f"  - Defender training set: {len(chunks_frac[0])} samples where first 10 indices are {chunks_frac_ids[0][:10]}")
    printer_func(f"  - Defender validation set: {len(chunks_frac[1])} samples where first 10 indices are {chunks_frac_ids[1][:10]}")
    printer_func(f"  - Defender testing set: {len(chunks_frac[2])} samples where first 10 indices are {chunks_frac_ids[2][:10]}")
    printer_func(f"Attacker dataset split:")
    printer_func(f"  - Attacker dataset: {len(chunks_frac[3])} samples where first 10 indices are {chunks_frac_ids[3][:10]}")

    attacker_data = chunks_frac[3]
    data = MultiDatasets()
    data.add(attacker_data, 'all')
    info = get_dataset_info_from_name(dataset=dataset)
    data.add_info(info)
    return data


    # # Optionally split the training set into train/val while preserving original indices
    # val_created = False
    # if val_split is not None and val_split == 0:
    #     logger.print_it("val_split is set to 0, skipping creation of validation split...")
    # elif val_split is not None and (0 < val_split < 1):
    #     n_train = len(train_dataset)
    #     full_indices = list(range(n_train))
    #     k = int(n_train * val_split)
    #     rng = np.random.default_rng(seed=seed)
    #     val_indices = rng.choice(full_indices, k, replace=False).tolist()
    #     val_idx = sorted(val_indices)
    #     # logger.print_it(f"Dataset getter: val_seed={seed}; val_n={len(val_idx)}; val_indices_sample={val_idx[:10]}")
        
    #     train_idx = [i for i in full_indices if i not in set(val_idx)]

    #     # Create separate dataset instances (deepcopy) so we can use different transforms
    #     train_ds_copy = copy.deepcopy(train_dataset)
    #     val_ds_copy = copy.deepcopy(train_dataset)
    #     # Ensure train keeps augmentation and val uses the test transforms
    #     try:
    #         train_ds_copy.transform = train_transform
    #         val_ds_copy.transform = test_transform
    #     except NameError:
    #         # In case transforms aren't available on the dataset object, ignore
    #         pass

    #     # Create Subsets that preserve original indices in `.indices`
    #     train_dataset = Subset(train_ds_copy, train_idx)
    #     val_dataset = Subset(val_ds_copy, val_idx)
    #     val_created = True
    # elif val_split is not None and (val_split >= 1 or val_split < 0):
    #     raise ValueError(f"Invalid val_split={val_split}. Must be >0 and <=1 for fraction or >=1 for absolute number of samples.")
    # else:
    #     raise ValueError(f"Invalid val_split={val_split}. Must be a positive number or None.")

    # info = get_dataset_info_from_name(dataset=dataset)

    # if val_created:
    #     printer_func('Gathered dataset "{}": Training samples = {} '
    #                     '& Validation samples = {} & Testing samples = {}'.format(dataset,
    #                                                                                 len(train_dataset),
    #                                                                                 len(val_dataset),
    #                                                                                 len(test_dataset)))
    # else:
    #     printer_func('Gathered dataset "{}": Training samples = {} '
    #                     '& Testing samples = {}'.format(dataset,
    #                                                     len(train_dataset),
    #                                                     len(test_dataset)))

    # data = MultiDatasets()
    # data.add(train_dataset, 'train')
    # if val_created:
    #     data.add(val_dataset, 'val')
    # data.add(test_dataset, 'test')
    # data.add_info(info)
    # data.wrap(IndexedDataset)  # Wrap all datasets to preserve original indices
    # return data

def wrap_datasets_with_indices(data: MultiDatasets):
    """Wrap all datasets in `data` with `IndexedDataset` to preserve original indices."""
    return data.wrap(IndexedDataset)

def wrap_dataset_with_indices(dataset: Dataset):
    """Wrap a single dataset with `IndexedDataset` to preserve original indices."""
    return IndexedDataset(dataset)