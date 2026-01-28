from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, MistDefenseConfigs
from src.mia.defenses.base import BaseDefender


class MistDefender(BaseDefender):
    # Implementation of training time optimization component of MIST defense from "MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training" (https://www.usenix.org/system/files/usenixsecurity24-li-jiacheng.pdf).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MistDefenseConfigs), f"MistDefender can only be used with MistDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mist_defender'
        self.mist_configs = defender_configs.defense
        # TODO: Implement the MIST defense.
        # assignees: AndAgio

    def train_model(self, train_configs, return_stats: bool = False):
        raise NotImplementedError("MistDefender: train_model is not implemented yet.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("MistDefender: defend_model is not implemented yet.")