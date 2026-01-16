from typing import Union
from src.data import get_dataset
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.utils.configs import ExperimentConfigs, TrainConfigs, VictimConfigs
from src.utils.log import Loggable, get_logger_from_configs


class Victim(Loggable):
    def __init__(self, victim_configs: VictimConfigs):
        logger=get_logger_from_configs(victim_configs.log)
        super().__init__(logger=logger)
        self.victim_hash = victim_configs.hash
        self.dataset_configs = victim_configs.dataset
        self.model_configs = victim_configs.model
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
    
    def train_model(self, train_configs: TrainConfigs):
        train_manager = TrainManager(train_configs=train_configs,
                                    name='victim',
                                    logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.model,
                                        configs=train_configs)
        self.model = train_manager.train(return_best_model=True,
                                        return_last_model=False,
                                        return_stats=False)
        return self.model
