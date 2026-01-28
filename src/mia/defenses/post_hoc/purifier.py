from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, PurifierDefenseConfigs
from src.mia.defenses.base import BaseDefender


class PurifierDefender(BaseDefender):
    # Implementation of running time component of Purifier defense from "Defending Model Inversion and Membership Inference Attacks via Prediction Purification" (https://arxiv.org/pdf/2005.03915).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, PurifierDefenseConfigs), f"PurifierDefender can only be used with PurifierDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'purifier_defender'
        self.purifier_configs = defender_configs.defense
        # TODO: Implement the Purifier defense.
        # assignees: AndAgio

    def train_model(self, train_configs, return_stats: bool = False):
        raise NotImplementedError("PurifierDefender: train_model is not implemented yet.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("PurifierDefender: defend_model is not implemented yet.")