from typing import Union

import torch
from src.data import get_dataset
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.utils.configs import TrainConfigs, DefenderConfigs
from src.utils.log import Loggable, get_logger_from_configs


class BaseDefender(Loggable):
    def __init__(self, defender_configs: DefenderConfigs):
        self.name = 'base_defender'
        logger=get_logger_from_configs(defender_configs.log)
        super().__init__(logger=logger)
        self.defender_hash = defender_configs.hash
        self.dataset_configs = defender_configs.dataset
        self.model_configs = defender_configs.model
        self.dataset = get_dataset(dataset=self.dataset_configs.name,
                                datasets_folder=self.dataset_configs.data_folder,
                                augment=self.dataset_configs.data_augmentation,
                                logger=self.logger)
        self.untrained_model = get_model(model_name=self.model_configs.model_name,
                            im_channels=self.model_configs.im_channels,
                            num_classes=self.model_configs.num_classes,
                            im_size=self.model_configs.im_size,
                            logger=self.logger)

    def get_dataset(self):
        return self.dataset
    
    def train(self, train_configs: TrainConfigs, return_stats: bool = False):
        train_manager = TrainManager(train_configs=train_configs,
                                    name=self.name,
                                    logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        configs=train_configs)
        # TODO: Refactor return of stats for victim and train manager.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/21
        # assignees: AndAgio
        if return_stats:
            self.trained_model, train_stats = train_manager.train(return_best_model=True,
                                                        return_last_model=False,
                                                        return_stats=return_stats)
        else:
            self.trained_model = train_manager.train(return_best_model=True,
                                            return_last_model=False,
                                            return_stats=return_stats)
        if return_stats:
            return self.trained_model, train_stats
        else:
            return self.trained_model
    
    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        raise NotImplementedError("defend_model method must be implemented in defense classes!")
        
    def get_defended_model(self) -> torch.nn.Module:
        assert hasattr(self, 'defended_model'), "Model has not been optimized for defense yet!"
        return self.defended_model
    
    def get_undefended_model(self) -> torch.nn.Module:
        assert hasattr(self, 'trained_model'), "Model has not been trained yet!"
        return self.trained_model
    
    def get_untrained_model(self) -> torch.nn.Module:
        assert hasattr(self, 'untrained_model'), "Model has not been initialized yet!"
        return self.untrained_model