import torch
from typing import Union
from src.models import get_model
from src.utils.log import Loggable, SmartLogger, DumbLogger
from src.utils.configs import ModelConfigs
from typing import Union

class ShadowModelsManager(Loggable):
    def __init__(self, n_models: int, model_configs: ModelConfigs, logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        assert 0 < n_models < 101, f"Invalid number of models should be between 1 and 100: {n_models}"
        self.n_models = n_models
        self.models = {i: get_model(model_name=model_configs.model_name,
                                    im_channels=model_configs.im_channels,
                                    num_classes=model_configs.num_classes,
                                    im_size=model_configs.im_size,
                                    logger=self.logger) for i in range(self.n_models)}

    def add(self, index: int, model: Union[torch.nn.Module, ModelConfigs]):
        assert not self.check_id(index), f"ID for shadow model to be added is already in use!"
        if isinstance(model, torch.nn.Module):
            self.models[index] = model
            self.n_models += 1
        elif isinstance(model, str):
            self.models[index] = get_model(model_name=model.model_name,
                                        im_channels=model.im_channels,
                                        num_classes=model.num_classes,
                                        im_size=model.im_size,
                                        logger=self.logger)
            self.n_models += 1
        else:
            raise ValueError('Did not recognize correctly the type of the given model!')

    def update(self, index: int, model: torch.nn.Module):
        assert self.check_id(index), f"Invalid ID for shadow model to be updated with id {index}"
        self.models[index] = model

    def remove(self, index: int):
        self.models.pop(index)
        self.n_models -= 1

    def get(self, index: int):
        assert self.check_id(index), f"ID '{index}' not in datasets managed by {self}!"
        return self.models[index]
    
    def get_all(self):
        return self.models

    def get_num_models(self):
        return self.n_models

    def get_all_ids(self):
        return list(self.models.keys())
    
    def check_id(self, index: int):
        return index in list(self.models.keys())
