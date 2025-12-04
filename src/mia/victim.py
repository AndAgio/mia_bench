from typing import Union
from src.data import get_dataset
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.utils.configs import TrainConfigs, ModelConfigs, DatasetConfigs, LogConfigs
from src.utils.log import get_logger

from src.utils.log import Loggable, SmartLogger, DumbLogger


class Victim(Loggable):
    def __init__(self, dataset_configs: DatasetConfigs,
                model_configs: ModelConfigs,
                logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        self.dataset_configs = dataset_configs
        self.model_configs = model_configs
        self.dataset = get_dataset(dataset=dataset_configs.name,
                                datasets_folder=dataset_configs.data_folder,
                                augment=dataset_configs.data_augmentation,
                                logger=self.logger)
        self.model = get_model(model_name=model_configs.model_name,
                            im_channels=model_configs.im_channels,
                            num_classes=model_configs.num_classes,
                            im_size=model_configs.im_size,
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
        self.model = train_manager.train(return_model=True)
        return self.model
