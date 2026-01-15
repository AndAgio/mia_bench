from typing import Union
import torch
from data.multi import MultiDatasets
from mia.base_mia import BaseMIA
from utils.configs import AttackConfigs, AuditingDataConfigs, ModelConfigs, ShadowDataConfigs
from utils.log import DumbLogger, SmartLogger


class NeuralMIA(BaseMIA):
    # Implementation of Neural Network-based MIA (https://ieeexplore.ieee.org/document/7958568)
    def __init__(self, 
                victim_model: torch.nn.Module,
                victim_dataset: MultiDatasets,
                audit_configs: AuditingDataConfigs,
                attack_configs: AttackConfigs,
                shadow_configs: ShadowDataConfigs, 
                model_configs: ModelConfigs,
                logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(victim_model=victim_model, victim_dataset=victim_dataset, audit_configs=audit_configs, attack_configs=attack_configs, logger=logger)
        self.logger.print_it(f'Working with neural network-based MIA!')
        raise NotImplementedError('Neural network-based MIA method is not yet implemented!')

        # TODO: Implement neural network-based MIA method (https://ieeexplore.ieee.org/document/7958568).
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/17
        # assignees: AndAgio