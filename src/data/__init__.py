import os
import pathlib
import sys
from torchvision.transforms import transforms

from .synthetic import Synthetic
from .subclass_synthetic import SubclassSynthetic
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
from torchvision.datasets import FashionMNIST
from .imagenet import ImageNet
from .tiny_imagenet import TinyImageNet
from .wrapper import DatasetWrapper
from .multi import MultiDatasets


from src.utils.variables import DEFAULT_DATASETS_FOLDER


def get_dataset(dataset: str, datasets_folder: str = DEFAULT_DATASETS_FOLDER, augment: bool = False):
    print('Gathering dataset "{}". This may take a while...'.format(dataset))
    # Image Preprocessing
    if dataset in ['cifar10', 'cifar100']:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        # Setup train transforms
        train_transform = transforms.Compose([])
        if augment:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        # Setup test transforms
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    elif dataset == 'svhn':
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    elif dataset == 'fmnist':
        mean = [0.2861]
        std = [0.3530]
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
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
    elif dataset == 'tiny_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
        # Setup train transforms for Tiny ImageNet
        train_transform = transforms.Compose([])
        if augment:
            train_transform.transforms.append(transforms.RandomCrop(64, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        # Setup test transforms for Tiny ImageNet
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))

    # Load the appropriate train and test datasets
    if dataset == 'cifar10':
        num_classes = 10
        im_size = (32,32)
        im_channels = 3
        root = os.path.join(datasets_folder, 'cifar10')
        train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        num_classes = 100
        im_size = (32,32)
        im_channels = 3
        root = os.path.join(datasets_folder, 'cifar100')
        train_dataset = CIFAR100(root=root, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(root=root, train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        im_channels = 3
        im_size = (32, 32)
        num_classes = 10
        root = os.path.join(datasets_folder, 'svhn')
        train_dataset = SVHN(root=root, split='train', download=True, transform=train_transform)
        train_dataset.targets = train_dataset.labels
        test_dataset = SVHN(root=root, split='test', download=True, transform=test_transform)
        test_dataset.targets = test_dataset.labels
    elif dataset == 'fmnist':
        im_channels = 1
        im_size = (28, 28)
        num_classes = 10
        root = os.path.join(datasets_folder, 'fmnist')
        train_dataset = FashionMNIST(root=root, train=True, download=True, transform=train_transform)
        test_dataset = FashionMNIST(root=root, train=False, download=True, transform=test_transform)
    elif dataset == 'imagenet':
        im_channels = 3
        im_size = (224, 224)
        num_classes = 1000
        root = os.path.join(datasets_folder, 'imagenet')
        train_dataset = ImageNet(root=root, split='train', download=True, transform=train_transform)
        test_dataset = ImageNet(root=root, split='val', download=True, transform=test_transform)
    elif dataset == 'tiny_imagenet':
        im_channels = 3
        im_size = (64, 64)
        num_classes = 200
        root = os.path.join(datasets_folder, 'tiny_imagenet')
        train_dataset = TinyImageNet(root=root, train=True, transform=train_transform)
        test_dataset = TinyImageNet(root=root, train=False, transform=test_transform)
    else:
        raise ValueError('Dataset "{}" is not available!'.format(dataset))
    info = {'im_channels': im_channels,
            'im_size': im_size,
            'num_classes': num_classes,}
    print('Gathered dataset "{}":\tTraining samples = {} '
                        '& Testing samples = {}'.format(dataset,
                                                        len(train_dataset),
                                                        len(test_dataset)))
    
    data = MultiDatasets()
    data.add(train_dataset, 'train')
    data.add(test_dataset, 'test')
    data.add_info(info)
    return data