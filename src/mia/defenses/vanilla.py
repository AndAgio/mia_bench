from typing import Union
from src.data import get_dataset
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.utils.configs import TrainConfigs, DefenderConfigs, NoDefenseConfigs
from src.utils.log import Loggable, get_logger_from_configs


class VanillaVictim(Loggable):
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, NoDefenseConfigs), f"VanillaVictim can only be used with NoDefenseConfigs, got {type(defender_configs.defense)}"
        logger=get_logger_from_configs(defender_configs.log)
        super().__init__(logger=logger)
        self.defender_hash = defender_configs.hash
        self.dataset_configs = defender_configs.dataset
        self.model_configs = defender_configs.model
        self.dataset = get_dataset(dataset=self.dataset_configs.name,
                                datasets_folder=self.dataset_configs.data_folder,
                                augment=self.dataset_configs.data_augmentation,
                                logger=self.logger)
        self.model = get_model(model_name=self.model_configs.model_name,
                            im_channels=self.model_configs.im_channels,
                            num_classes=self.model_configs.num_classes,
                            im_size=self.model_configs.im_size,
                            logger=self.logger)

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model
    
    def optimize(self, train_configs: TrainConfigs, return_stats: bool = False):
        train_manager = TrainManager(train_configs=train_configs,
                                    name='vanilla_defender',
                                    logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.model,
                                        configs=train_configs)
        # TODO: Refactor return of stats for victim and train manager.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/21
        # assignees: AndAgio
        if return_stats:
            self.model, train_stats = train_manager.train(return_best_model=True,
                                                        return_last_model=False,
                                                        return_stats=return_stats)
        else:
            self.model = train_manager.train(return_best_model=True,
                                            return_last_model=False,
                                            return_stats=return_stats)
        if return_stats:
            return self.model, train_stats
        else:
            return self.model
