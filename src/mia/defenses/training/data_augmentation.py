
from typing import Union
import torch
from torchvision import transforms
from src.utils.configs import DefenderConfigs, DataAugmentationDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender
from src.data.helpers import AugmentWrappedDataset, MultiDatasets

# TODO: Enable attacker to also use data augmentation.
# Issue URL: https://github.com/AndAgio/mia_bench/issues/48
# assignees: AndAgio


# TODO: Enable data augmentation to be used together with other defenses, such as adversarial regularization or label smoothing.
# Issue URL: https://github.com/AndAgio/mia_bench/issues/47
# assignees: AndAgio


class DataAugmentationDefender(BaseDefender):
    """Trains with data augmentation: random crops, flips, rotations and erasing, or feature masking on tabular data.

    Documentation: docs/defenses/data_augmentation.md.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select data augmentation and keep its settings."""
        assert isinstance(defender_configs.defense, DataAugmentationDefenseConfigs), f"DataAugmentationDefender can only be used with DataAugmentationDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'data_augmentation_defender'
        self.data_augmentation_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        """Train on augmented training samples and return the best model, with the stats if return_stats."""
        self.logger.print_it(f'DataAugmentation Defender: training defender model with augmentation configs: {self.data_augmentation_configs}...')
        # Train on a copy of the splits whose training split is augmented. The defender's own splits stay clean, so that
        # what is measured on them after training (e.g., the training accuracy in run.py) uses the samples themselves.
        clean_dataset = self.dataset
        self.dataset = MultiDatasets()
        for split in clean_dataset.get_ids():
            self.dataset.add(clean_dataset.get(split), split)
        self.dataset.add_info(clean_dataset.get_info())
        augmented_dataset = AugmentWrappedDataset(base_dataset=clean_dataset.get('train'),
                                                extra_transform=self.get_augmentations())
        self.dataset.update(dataset=augmented_dataset,
                            id='train')
        try:
            return super().train_model(train_configs=train_configs, return_stats=return_stats)
        finally:
            self.dataset = clean_dataset

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        """Return the trained model unchanged: data augmentation only changes training."""
        self.logger.print_it('DataAugmentation Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model

    def get_augmentations(self):
        """Return the random transforms that the configs enable, applied in turn to each normalized training image."""
        if self.dataset_configs.im_size is None:
            return self.get_feature_augmentations()
        augmentation_transforms = transforms.Compose([])
        crop_padding = self.data_augmentation_configs.crop_padding
        assert crop_padding is None or crop_padding >= 0, \
            f"Invalid crop padding {crop_padding} for data augmentation defense!"
        if crop_padding:
            crop = transforms.RandomCrop(self.dataset_configs.im_size, padding=crop_padding)
            augmentation_transforms.transforms.append(crop)
        assert self.data_augmentation_configs.horizontal_flip >= 0 and self.data_augmentation_configs.horizontal_flip <= 1, f"Invalid horizontal flip probability {self.data_augmentation_configs.horizontal_flip} for data augmentation defense!"
        if self.data_augmentation_configs.horizontal_flip > 0:
            augmentation_transforms.transforms.append(transforms.RandomHorizontalFlip(p=self.data_augmentation_configs.horizontal_flip))
        if self.data_augmentation_configs.rotation != 0:
            augmentation_transforms.transforms.append(transforms.RandomRotation(self.data_augmentation_configs.rotation))
        if self.data_augmentation_configs.erase_prob > 0:
            augmentation_transforms.transforms.append(transforms.RandomErasing(p=self.data_augmentation_configs.erase_prob))
        return augmentation_transforms

    def get_feature_augmentations(self):
        """Return the random transforms of the feature vectors of tabular datasets: flips or noise, then masking."""
        self.logger.print_it('DataAugmentation Defender: tabular dataset, so the image transforms are skipped.')
        configs = self.data_augmentation_configs
        assert 0 <= configs.feature_mask_prob <= 1, f"Invalid feature mask probability {configs.feature_mask_prob}!"
        augmentation_transforms = transforms.Compose([])
        if configs.feature_flip_prob:
            augmentation_transforms.transforms.append(FeatureFlip(p=configs.feature_flip_prob))
        if configs.feature_noise_std:
            augmentation_transforms.transforms.append(FeatureNoise(std=configs.feature_noise_std))
        # Masking comes last, so that masked features are 0.
        if configs.feature_mask_prob > 0:
            augmentation_transforms.transforms.append(FeatureMask(p=configs.feature_mask_prob))
        return augmentation_transforms


class FeatureMask:
    """Sets each feature of a sample to 0 with probability p."""
    def __init__(self, p: float):
        """Keep the probability of masking each feature."""
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return x with each feature set to 0 with probability p."""
        return x * (torch.rand_like(x) >= self.p)

    def __repr__(self):
        """Show the transform and its probability, as torchvision transforms do."""
        return f'{type(self).__name__}(p={self.p})'


class FeatureFlip:
    """Flips each binary feature of a sample with probability p."""
    def __init__(self, p: float):
        """Keep the probability of flipping each feature."""
        assert 0 <= p <= 1, f"Invalid feature flip probability {p}!"
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return x with each feature replaced by 1 - x with probability p."""
        return torch.where(torch.rand_like(x) < self.p, 1 - x, x)

    def __repr__(self):
        """Show the transform and its probability, as torchvision transforms do."""
        return f'{type(self).__name__}(p={self.p})'


class FeatureNoise:
    """Adds Gaussian noise of standard deviation std to each feature of a sample."""
    def __init__(self, std: float):
        """Keep the standard deviation of the noise."""
        assert std >= 0, f"Invalid feature noise standard deviation {std}!"
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return x plus independent Gaussian noise on each feature."""
        return x + self.std * torch.randn_like(x)

    def __repr__(self):
        """Show the transform and its standard deviation, as torchvision transforms do."""
        return f'{type(self).__name__}(std={self.std})'
