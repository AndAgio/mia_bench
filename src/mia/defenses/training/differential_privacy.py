
from typing import Union
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from src.data.multi import MultiDatasets
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from src.utils.configs import DefenderConfigs, DPDefenseConfigs, TrainConfigs, ModelConfigs 
from src.trainer.train_manager import TrainManager
from src.data import get_dataset
from src.models import get_model
from src.utils.log import Loggable, get_logger_from_configs


class DifferentialPrivacyDefender(Loggable):
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, DPDefenseConfigs), f"DifferentialPrivacyDefender can only be used with DPDefenseConfigs, got {type(defender_configs.defense)}"
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
        self.dp_configs = defender_configs.defense

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model
    
    def optimize(self, train_configs: TrainConfigs, return_stats: bool = False):
        train_manager = DifferentialPrivacyTrainManager(train_configs=train_configs,
                                                        name='dp_defender',
                                                        logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.model,
                                        configs=train_configs,
                                        dp_configs=self.dp_configs)
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
        

class DifferentialPrivacyTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)

    def check_and_set_dp(self, dp_config: DPDefenseConfigs):
        self.logger.print_it('Differential Privacy with Opacus: updating model, optimizer and data loaders accordingly...')
        
        if dp_config.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                dp_config.max_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            max_grad_norm = dp_config.max_grad_norm

        if self.distributed and dp_config.clip_per_layer:
            self.model = DPDDP(self.model)

        privacy_engine = PrivacyEngine()
        clipping = "per_layer" if dp_config.clip_per_layer else "flat"
        if dp_config.grad_sample_mode in ['ghost']:
            self.model, self.optimizer, self.criterion, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=dp_config.noise_multiplier,
                max_grad_norm=max_grad_norm,
                clipping=clipping,
                grad_sample_mode=dp_config.grad_sample_mode,
            )
        elif dp_config.grad_sample_mode in ['hook']:
            self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=dp_config.noise_multiplier,
                max_grad_norm=max_grad_norm,
                clipping=clipping,
                grad_sample_mode=dp_config.grad_sample_mode,
            )
        else:
            raise ValueError(f"Unsupported mode '{dp_config.grad_sample_mode}' for dp_config.grad_sample_mode when using DP!")
        self.logger.print_it('Differential Privacy setup completed!')
        self.model = self.model.to(self.device)

    def validate_and_fix_model_for_dp(self):
        self.logger.print_it('Differential Privacy with Opacus: validating model and fixing it if necessary...')
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)
        self.model = self.model.to(self.device)

    def initialize_train(self, 
                        dataset: Union[MultiDatasets, Dataset],
                        model: Union[ModelConfigs,nn.Module],
                        configs: TrainConfigs,
                        dp_configs: DPDefenseConfigs,
                        ):
        self.logger.print_it('Initializing training...')
        if not configs.__eq__(self.train_configs):
            self.logger.print_it('Found different training configurations in the initialize_train method. Resetting the trainer configs...')
            self.reset_configs(configs)

        self.setup_dataloaders(dataset=dataset,
                                batch_size=self.train_configs.batch_size)
        if isinstance(model, nn.Module):
            self.set_model(model)
        elif isinstance(model, ModelConfigs):
            self.setup_model_from_configs(model_configs=model)
        else:
            raise ValueError('Not recognizing model given!')
        self.validate_and_fix_model_for_dp()
        self.setup_training()
        # Modifying model, optimizer and loaders for differential privacy if needed
        self.check_and_set_dp(dp_config=dp_configs)
        self.logger.print_it('Training initialization completed!')