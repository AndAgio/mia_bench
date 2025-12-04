from typing import Callable, Union
import torch
from src.data.multi import MultiDatasets
from .auditing_data_manager import AuditingDatasetManager
from src.utils.configs import AuditingDataConfigs
from src.utils.log import Loggable, SmartLogger, DumbLogger


class BaseMIA(Loggable):
    # TODO: Implement attacker functionalities.
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/1
    # assignee: AndAgio
    def __init__(self, victim_model: torch.nn.Module, victim_dataset: MultiDatasets, audit_configs: AuditingDataConfigs, logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        self.logger.print_it('Setting up and MIA attacker. First thing to do is sampling the auditing dataset...')
        self.victim_model = victim_model
        self.victim_dataset = victim_dataset
        self.seed = audit_configs.seed
        self.audit_manager = AuditingDatasetManager(original_datasets=victim_dataset,
                                                    configs=audit_configs,
                                                    logger=logger)

    def optimize(self):
        raise NotImplementedError('MIA should implement the method to optimize it!')
    
    def measure_effectiveness(self):
        raise NotImplementedError('MIA should implement the method to measure its effectiveness!')