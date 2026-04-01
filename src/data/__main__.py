import os
import pathlib
import sys
PATH_REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(PATH_REPO))
from . import CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageNet, TinyImageNet, ImageNet1K, Purchase, Texas, News, GTSRB
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
    print('Downloading GTSRB...')
    root = os.path.join(settings.datasets_folder, 'gtsrb')
    GTSRB(root=root, train=True, download=True, transform=None)
    GTSRB(root=root, train=False, download=True, transform=None)
    print('Downloading TinyImageNet...')
    root = os.path.join(settings.datasets_folder, 'tinyimagenet')
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
    print('Downloading Purchase100...')
    root = os.path.join(settings.datasets_folder, 'purchase')
    Purchase(root=root, train=True, transform=None, download=True)
    Purchase(root=root, train=False, transform=None, download=True)
    print('Downloading Texas100...')
    root = os.path.join(settings.datasets_folder, 'texas')
    Texas(root=root, train=True, transform=None, download=True)
    Texas(root=root, train=False, transform=None, download=True)
    print('Downloading News20...')
    root = os.path.join(settings.datasets_folder, 'news')
    News(root=root, train=True, transform=None, download=True)
    News(root=root, train=False, transform=None, download=True)
    print('Done!')

if __name__=='__main__':
    main()