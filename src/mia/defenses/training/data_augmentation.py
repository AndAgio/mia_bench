
from typing import Union
import torch
from src.utils.configs import DefenderConfigs, DataAugmentationDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender


class DataAugmentationDefender(BaseDefender):
    # Implementation of Data Augmentation as a Defense approach.
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, DataAugmentationDefenseConfigs), f"DataAugmentationDefender can only be used with DataAugmentationDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'data_augmentation_defender'
        self.data_augmentation_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f'DataAugmentation Defender: training defender model with relaxed loss and alpha {self.data_augmentation_configs.relax_alpha}...')
        self.dataset = get_dataset(dataset=self.dataset_configs.name,
                                datasets_folder=self.dataset_configs.data_folder,
                                val_split=self.dataset_configs.val_split,
                                seed=self.dataset_configs.seed,
                                augment=self.dataset_configs.data_augmentation,
                                logger=self.logger)
        super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        self.logger.print_it('DataAugmentation Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model

