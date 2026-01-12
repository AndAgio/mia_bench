import os
from src.data import CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageNet, TinyImageNet, ImageNet1K
from src.utils import gather_settings


def main():
    settings = gather_settings()
    print('Downloading CIFAR10...')
    root = os.path.join(settings.datasets_folder, 'cifar10')
    CIFAR10(root=root, train=True, transform=None, download=True)
    CIFAR10(root=root, train=False, transform=None, download=True)
    print('Downloading CIFAR100...')
    root = os.path.join(settings.datasets_folder, 'cifar100')
    CIFAR100(root=root, train=True, transform=None, download=True)
    CIFAR100(root=root, train=False, transform=None, download=True)
    print('Downloading SVHN...')
    root = os.path.join(settings.datasets_folder, 'svhn')
    SVHN(root=root, split='train', download=True, transform=None)
    SVHN(root=root, split='test', download=True, transform=None)
    print('Downloading FMNIST...')
    root = os.path.join(settings.datasets_folder, 'fmnist')
    FashionMNIST(root=root, train=True, download=True, transform=None)
    FashionMNIST(root=root, train=False, download=True, transform=None)
    print('Downloading TinyImageNet...')
    root = os.path.join(settings.datasets_folder, 'tiny_imagenet')
    TinyImageNet(root=root, train=True, transform=None)
    TinyImageNet(root=root, train=False, transform=None)
    print('Downloading ImageNet...')
    root = os.path.join(settings.datasets_folder, 'imagenet')
    ImageNet(root=root, split='train', download=True, transform=None)
    ImageNet(root=root, split='val', download=True, transform=None)
    print('Downloading ImageNet1K...')
    root = os.path.join(settings.datasets_folder, 'imagenet1k')
    ImageNet1K(root=root, split='train', download=True, transform=None)
    ImageNet1K(root=root, split='val', download=True, transform=None)
    print('Done!')

if __name__=='__main__':
    main()