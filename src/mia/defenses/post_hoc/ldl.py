from typing import Union
import time
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, LdlDefenseConfigs
from src.mia.defenses.base import BaseDefender



class LdlDefender(BaseDefender):
    # Implementation of LDL defense from "LDL: A Defense for Label-Based Membership Inference Attacks" (https://dl.acm.org/doi/pdf/10.1145/3579856.3582821).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, LdlDefenseConfigs), f"LdlDefender can only be used with LDLDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'ldl_defender'
        self.ldl_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it("LdlDefender: No training required for LDL defense, training with standard procedure.")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('LdlDefender: building defense layer...')
        self.defended_model = LdlModel(victim=self.trained_model,
                                n_queries=self.ldl_configs.n_queries,
                                noise_type=self.ldl_configs.noise_type,
                                noise_scale=self.ldl_configs.noise_scale
                                ).to(device)
        print('LdlDefender: defense layer built, moving to device {}...'.format(device))
        self.logger.print_it('LdlDefender: finished building defense layer!')
        return self.defended_model

    @staticmethod
    def get_device(dev_str: str = 'cpu'):
        # Set appropriate devices
        if torch.cuda.is_available() and dev_str != 'cpu':
            dev_str = 'cuda:{}'.format(dev_str)
            device = torch.device(dev_str)
        elif torch.backends.mps.is_available() and dev_str != 'cpu':
            dev_str = 'mps'
            device = torch.device(dev_str)
        else:
            device = torch.device('cpu')
        return device


class LdlModel(torch.nn.Module):
    # Implementation of the LDL defense layer.
    def __init__(self, victim: torch.nn.Module, n_queries: int = 100, noise_type: str = 'normal', noise_scale: float = 0.1):
        super().__init__()
        self.victim = victim.eval()
        self.n_queries = n_queries
        assert noise_type in ['normal', 'bernoulli'], f"noise_type must be one of ['normal', 'bernoulli'], got {noise_type}!"
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy_output = self.victim(x)
        logits_matrix = torch.zeros((self.n_queries, *dummy_output.shape), device=x.device)
        for i in range(self.n_queries):
            noisy_x = self.add_noise(x)
            logits_matrix[i] = self.victim(noisy_x)
        avg_logits = logits_matrix.mean(dim=0)
        return avg_logits
    
    @torch.no_grad()
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "normal":
            noise = torch.normal(mean=0, std=self.noise_scale, size=x.shape, device=x.device)
            x_noisy = x + noise
        elif self.noise_type == "bernoulli":
            x_noisy = torch.zeros_like(x, device=x.device)
            for i in range(x.shape[0]):
                x_temp=x[i:i+1].detach().cpu().numpy().astype(np.bool)
                noise = np.random.binomial(1, self.noise_scale, [100, x_temp.shape[-1]])
                x_sampled = np.tile(np.copy(x_temp), (100, 1))
                x_noisy[i] = torch.tensor(np.invert(x_temp, out=x_sampled, where=noise.astype(np.bool)).astype(np.int32), device=x.device)
        return x_noisy
