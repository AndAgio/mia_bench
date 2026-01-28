from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, HampDefenseConfigs
from src.mia.defenses.base import BaseDefender


class HampDefender(BaseDefender):
    # Implementation of running time component of HAMP defense from "Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction" (https://arxiv.org/pdf/2307.01610).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, HampDefenseConfigs), f"HampDefender can only be used with HampDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'hamp_defender'
        self.hamp_configs = defender_configs.defense
        # TODO: Implement the full HAMP defense.
        # assignees: AndAgio

    def train_model(self, train_configs, return_stats: bool = False):
        raise NotImplementedError("HampDefender: train_model is not implemented yet.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("HampDefender: defend_model is not implemented yet.")