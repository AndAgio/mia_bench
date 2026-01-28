from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, SelenaDefenseConfigs
from src.mia.defenses.base import BaseDefender


class SelenaDefender(BaseDefender):
    # Implementation of training time optimization component of Selena defense from "Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture" (https://www.usenix.org/system/files/sec22-tang.pdf).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, SelenaDefenseConfigs), f"SelenaDefender can only be used with SelenaDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'selena_defender'
        self.selena_configs = defender_configs.defense
        # TODO: Implement the Selena defense.
        # assignees: AndAgio

    def train_model(self, train_configs, return_stats: bool = False):
        raise NotImplementedError("SelenaDefender: train_model is not implemented yet.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("SelenaDefender: defend_model is not implemented yet.")