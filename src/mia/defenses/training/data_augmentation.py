
from typing import Union
import torch
from torchvision import transforms
from src.utils.configs import DefenderConfigs, DataAugmentationDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender
from src.data.helpers import AugmentWrappedDataset

# TODO: Enable attacker to also use data augmentation.
# Issue URL: https://github.com/AndAgio/mia_bench/issues/48
# assignees: AndAgio


# TODO: Enable data augmentation to be used together with other defenses, such as adversarial regularization or label smoothing.
# Issue URL: https://github.com/AndAgio/mia_bench/issues/47
# assignees: AndAgio


class DataAugmentationDefender(BaseDefender):
    # Implementation of Data Augmentation as a Defense approach.
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, DataAugmentationDefenseConfigs), f"DataAugmentationDefender can only be used with DataAugmentationDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'data_augmentation_defender'
        self.data_augmentation_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f'DataAugmentation Defender: training defender model with augmentation configs: {self.data_augmentation_configs}...')
        non_augmented_dataset = self.dataset.get('train')
        augmented_dataset = AugmentWrappedDataset(base_dataset=non_augmented_dataset,
                                                extra_transform=self.get_augmentations())
        self.dataset.update(dataset=augmented_dataset,
                            id='train')
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        self.logger.print_it('DataAugmentation Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model

    def get_augmentations(self):
        augmentation_transforms = transforms.Compose([])
        assert self.data_augmentation_configs.horizontal_flip >= 0 and self.data_augmentation_configs.horizontal_flip <= 1, f"Invalid horizontal flip probability {self.data_augmentation_configs.horizontal_flip} for data augmentation defense!"
        if self.data_augmentation_configs.horizontal_flip > 0:
            augmentation_transforms.transforms.append(transforms.RandomHorizontalFlip(p=self.data_augmentation_configs.horizontal_flip))
        if self.data_augmentation_configs.rotation != 0:
            augmentation_transforms.transforms.append(transforms.RandomRotation(self.data_augmentation_configs.rotation))
        if self.data_augmentation_configs.erase_prob > 0:
            augmentation_transforms.transforms.append(transforms.RandomErasing(p=self.data_augmentation_configs.erase_prob))
        return augmentation_transforms
