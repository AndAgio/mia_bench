import torch
from src.trainer.train_manager import TrainManager
from src.mia.helpers.shadow_data_manager import ShadowDatasetsManager
from src.mia.helpers.shadow_models_manager import ShadowModelsManager
from src.data.helpers import MultiDatasets
from src.mia.helpers.auditing_data_manager import AuditingDatasetManager
from src.utils.configs import ShadowDataConfigs, TrainConfigs, ModelConfigs
from src.utils.log import Loggable, MyLogger, get_logger


class ShadowManager(Loggable):
    def __init__(self, shadow_data: ShadowDatasetsManager = None, shadow_models: ShadowModelsManager = None, logger: MyLogger = None):
        super().__init__(logger=logger)
        if shadow_data is not None and shadow_models is not None:
            assert shadow_data.get_num_dataset() == shadow_models.get_num_models(), f"The number of shadow datasets and models should be the same!"
        self.shadow_data = shadow_data
        self.shadow_models = shadow_models

    def sample_shadow_datasets(self, attacker_data_distribution: MultiDatasets, auditing_dataset: AuditingDatasetManager, shadow_configs: ShadowDataConfigs, attacker_hash: str):
        if self.shadow_models is not None:
            assert shadow_configs.n_shadow_datasets == self.shadow_models.get_num_models(), f"Number of shadow datasets you're trying to sample does not match the number of shadow models already built!"
        self.shadow_data = ShadowDatasetsManager(attacker_data_distribution=attacker_data_distribution,
                                                auditing_dataset=auditing_dataset,
                                                shadow_configs=shadow_configs,
                                                attacker_hash=attacker_hash,
                                                logger=self.logger)

    def build_shadow_models(self, n_models: int, model_configs: ModelConfigs, same_model_arch: bool=True):
        if self.shadow_data is not None:
            assert n_models == self.shadow_data.get_num_dataset(), f"Number of shadow models you're trying to build does not match the number of shadow datasets already sampled!"
        self.shadow_models = ShadowModelsManager(n_models=n_models,
                                                model_configs=model_configs,
                                                same_model_arch=same_model_arch,
                                                logger=self.logger)

    def train_single_model(self, id: int, train_configs: TrainConfigs, labels_mode: str = 'original'):
        assert self.shadow_models.check_id(id), f"Invalid ID for shadow model to be trained with id {id}"
        assert self.shadow_data.check_id(id), f"Invalid ID for shadow dataset to be used to train the shadow model with id {id}"
        assert labels_mode in ['original', 'mia'], f"Invalid label mode found when trying to train shadow model with ID {id} and mode {labels_mode}!"
        logger = get_logger(name='{} shadow {}'.format(self.logger.name, id),
                            log_folder=self.logger.get_folder(),
                            mode=self.logger.get_mode())
        train_manager = TrainManager(train_configs=train_configs,
                                    name='shadow_{}'.format(id),
                                    logger=logger)
        dataset = self.shadow_data.get(index=id, labels=labels_mode)
        wrapped_dataset = MultiDatasets([dataset], ids=['train'])
        train_manager.initialize_train(dataset=wrapped_dataset,
                                        model=self.shadow_models.get(index=id),
                                        configs=train_configs)
        model = train_manager.train(return_best_model=False,
                                    return_last_model=True,
                                    return_stats=False)
        self.shadow_models.update(index=id,
                                model=model)
        self.logger.print_it(f"Finished training shadow model with ID {id} on the corresponding dataset!")

    def train_all(self, train_configs: TrainConfigs, labels_mode: str = 'original'):
        assert self.shadow_data.get_all_ids() == self.shadow_models.get_all_ids(), f"Shadow models and dataset indices do not correspond one to one!"
        for id in self.shadow_data.get_all_ids():
            self.train_single_model(id=id,
                                    train_configs=train_configs,
                                    labels_mode=labels_mode)
        self.logger.print_it('Finished training all shadow models on all shadow datasets!')

    def get_all_models(self):
        return self.shadow_models.get_all()
    
    def get_all_datasets(self, labels: str = 'original'):
        return self.shadow_data.get_all(labels=labels)
    
    def get_model(self, index: int):
        return self.shadow_models.get(index=index)
    
    def update_model(self, index: int, model: torch.nn.Module):
        self.shadow_models.update(index=index, model=model)
    
    def get_dataset(self, index: int, labels: str = 'original'):
        return self.shadow_data.get(index=index,
                                    labels=labels)
    
    def get_all_model_indeces(self):
        return self.shadow_models.get_all_ids()
    
    def get_all_dataset_indices(self):
        return self.shadow_data.get_all_ids()
    
    def get_n_models(self):
        return self.shadow_models.get_num_models()
    
    def get_n_datasets(self):
        return self.shadow_data.get_num_dataset()
    

    def get_all_in_dataset_for_sample_id(self, id: int, labels: str = 'original'):
        return self.shadow_data.get_shadow_datasets_containing_sample_id(id=id,
                                                                        labels=labels)
    
    def find_all_in_dataset_indices_for_sample_id(self, id: int):
        return self.shadow_data.find_shadow_datasets_containing_sample_id(id=id)
    
    def sample_random_population_indices(self, num_data: int = None):
        return self.shadow_data.sample_random_indices(num_data=num_data)
    
    def get_random_population(self, indices: dict = None, num_data: int = None, labels: str = 'original'):
        return self.shadow_data.get_random_population(indices=indices,
                                                    num_data=num_data,
                                                    labels=labels)
    
    def get_all_samples_ids(self):
        return self.shadow_data.get_all_samples_ids()
    
    def get_all_samples_in_all_shadow_datasets(self, labels: str = 'original'):
        return self.shadow_data.get_all_samples_in_all_shadow_datasets(labels=labels)
    
    def sample_outside_shadow_dataset(self, index: int, num_data: int, labels: str = 'original'):
        return self.shadow_data.sample_outside_shadow_dataset(index=index, num_data=num_data, labels=labels)
    
    def train_single_model_on_given_dataset(self, id: int, train_configs: TrainConfigs, dataset: MultiDatasets):
        assert self.shadow_models.check_id(id), f"Invalid ID for shadow model to be trained with id {id}"
        assert isinstance(dataset, MultiDatasets), f"Expected dataset to be of type MultiDatasets, got {type(dataset)} instead when trying to train shadow model with ID {id} on given dataset!"
        logger = get_logger(name='{} shadow {}'.format(self.logger.name, id),
                            log_folder=self.logger.get_folder(),
                            mode=self.logger.get_mode())
        train_manager = TrainManager(train_configs=train_configs,
                                    name='shadow_{}'.format(id),
                                    logger=logger)
        train_manager.initialize_train(dataset=dataset,
                                        model=self.shadow_models.get(index=id),
                                        configs=train_configs)
        model = train_manager.train(return_best_model=False,
                                    return_last_model=True,
                                    return_stats=False)
        self.shadow_models.update(index=id,
                                model=model)
        self.logger.print_it(f"Finished training shadow model with ID {id} on the corresponding dataset!")

    def add_dataset_replicas(self, index: int, n_replicas: int):
        self.shadow_data.replicate_dataset(index=index, n_replicas=n_replicas)