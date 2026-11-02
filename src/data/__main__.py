"""Download the benchmark's datasets: python -m src.data [names...] [--datasets_folder FOLDER].

Without names, every dataset is downloaded, ImageNet included.
"""
import argparse
import os
import pathlib
import sys
PATH_REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(PATH_REPO))
from . import CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageNet, TinyImageNet, ImageNet1K, Purchase, Texas, News, GTSRB
from src.utils.variables import DEFAULT_DATASETS_FOLDER

TRAIN_TEST = [{'train': True}, {'train': False}]
# For each dataset folder under the datasets folder: the printed name, the dataset class, the splits and the options
# that make instantiating the class download the split.
DOWNLOADS = {
    'cifar10': ('CIFAR10', CIFAR10, TRAIN_TEST, {'download': True}),
    'cifar100': ('CIFAR100', CIFAR100, TRAIN_TEST, {'download': True}),
    'svhn': ('SVHN', SVHN, [{'split': 'train'}, {'split': 'test'}], {'download': True}),
    'fmnist': ('FMNIST', FashionMNIST, TRAIN_TEST, {'download': True}),
    'gtsrb': ('GTSRB', GTSRB, TRAIN_TEST, {'download': True}),
    'tinyimagenet': ('TinyImageNet', TinyImageNet, TRAIN_TEST, {}),
    'imagenet': ('ImageNet', ImageNet, [{'split': 'train'}, {'split': 'val'}], {'download': True}),
    'imagenet1k': ('ImageNet1K', ImageNet1K, [{'split': 'train'}, {'split': 'val'}], {'download': True}),
    'purchase': ('Purchase100', Purchase, TRAIN_TEST, {'download': True}),
    'texas': ('Texas100', Texas, TRAIN_TEST, {'download': True}),
    'news': ('News20', News, TRAIN_TEST, {'download': True}),
}


def main(argv=None):
    """Download the datasets named on the command line, or all of them, into the datasets folder."""
    parser = argparse.ArgumentParser(description='Download the datasets of the benchmark.')
    parser.add_argument('datasets', nargs='*', metavar='dataset',
                        help=f'Datasets to download, among: {", ".join(DOWNLOADS)}. Default: all of them')
    parser.add_argument('--datasets_folder', type=pathlib.Path, default=DEFAULT_DATASETS_FOLDER)
    args = parser.parse_args(argv)
    unknown = [name for name in args.datasets if name not in DOWNLOADS]
    if unknown:
        parser.error(f'unknown datasets {unknown}; choose among {list(DOWNLOADS)}')
    for name in args.datasets or DOWNLOADS:
        printed_name, dataset_class, splits, options = DOWNLOADS[name]
        print(f'Downloading {printed_name}...')
        for split in splits:
            dataset_class(root=os.path.join(args.datasets_folder, name), transform=None, **split, **options)
    print('Done!')


if __name__ == '__main__':
    main()
