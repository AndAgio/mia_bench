from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, MmdDefenseConfigs
from src.mia.defenses.base import BaseDefender


class MmdDefender(BaseDefender):
    # Implementation of training time optimization component of MMD defense from "Membership Inference Attacks and Defenses in Classification Models" (https://dl.acm.org/doi/pdf/10.1145/3422337.3447836).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MmdDefenseConfigs), f"MmdDefender can only be used with MmdDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mmd_defender'
        self.mmd_configs = defender_configs.defense
        # TODO: Implement the training time optimization component of MMD defense.
        # assignees: AndAgio

    def train_model(self, train_configs, return_stats: bool = False):
        raise NotImplementedError("MmdDefender: train_model is not implemented yet.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("MmdDefender: defend_model is not implemented yet.")