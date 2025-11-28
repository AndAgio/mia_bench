from src.data import get_dataset
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.trainer.utils import TrainConfigs
from src.utils.variables import DEFAULT_LOG_FOLDER, DEFAULT_MODELS_FOLDER, DEFAULT_RESUME_CKPTS_FOLDER, DEFAULT_DATASETS_FOLDER


class Victim():
    def __init__(self, dataset: str, model_name: str, datasets_folder: str = DEFAULT_DATASETS_FOLDER, data_augmentation: bool = False):
        self.dataset = get_dataset(dataset=dataset, datasets_folder=datasets_folder, augment=data_augmentation)
        self.model = get_model(model_name=model_name, dataset_info=self.dataset.get_info())

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model
    
    def train_model(self, train_configs: TrainConfigs, log_folder: str = DEFAULT_LOG_FOLDER, logging_mode: str = 'smart'):
        train_manager = TrainManager(train_configs=train_configs,
                                    name='victim',
                                    log_folder=log_folder,
                                    logging_mode=logging_mode)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.model,
                                        configs=train_configs)
        self.model = train_manager.train(return_model=True)
