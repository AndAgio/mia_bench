from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, MixupDefenseConfigs
from src.mia.defenses.base import BaseDefender


class MixupDefender(BaseDefender):
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MixupDefenseConfigs), f"MixupDefender can only be used with MixupDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mixup_defender'
        self.mixup_configs = defender_configs.defense
        # TODO: Implement the mixup optimization component.
        # assignees: AndAgio

    def train_model(self, train_configs, return_stats: bool = False):
        raise NotImplementedError("MixupDefender: train_model is not implemented yet.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("MixupDefender: defend_model is not implemented yet.")