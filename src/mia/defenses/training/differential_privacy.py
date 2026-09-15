
from typing import Union
import torch
from torch.utils.data import Dataset
import numpy as np
from src.data.helpers import MultiDatasets
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from src.utils.configs import DefenderConfigs, DPDefenseConfigs, TrainConfigs, ModelConfigs 
from src.trainer.train_manager import TrainManager
from src.mia.defenses.base import BaseDefender


class DifferentialPrivacyDefender(BaseDefender):
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, DPDefenseConfigs), f"DifferentialPrivacyDefender can only be used with DPDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'dp_defender'
        self.dp_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        train_manager = DifferentialPrivacyTrainManager(train_configs=train_configs,
                                                        name=self.name,
                                                        logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        configs=train_configs,
                                        dp_configs=self.dp_configs)
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
        self.logger.print_it('Differential Privacy Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model

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
            private_objects = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=dp_config.noise_multiplier,
                max_grad_norm=max_grad_norm,
                clipping=clipping,
                grad_sample_mode=dp_config.grad_sample_mode,
            )
            # Opacus 1.5+ returns a wrapped criterion for ghost clipping, while
            # older compatible releases return the usual three-item tuple.
            if len(private_objects) == 4:
                self.model, self.optimizer, self.criterion, self.train_loader = private_objects
            elif len(private_objects) == 3:
                self.model, self.optimizer, self.train_loader = private_objects
                self.logger.print_it(
                    'Opacus did not return a ghost-clipping criterion; '
                    'continuing with the configured criterion.'
                )
            else:
                raise RuntimeError(
                    f"Unexpected number of values returned by Opacus make_private(): "
                    f"{len(private_objects)}"
                )
        elif dp_config.grad_sample_mode in ['hooks']:
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
                        model: Union[ModelConfigs,torch.nn.Module],
                        configs: TrainConfigs,
                        dp_configs: DPDefenseConfigs,
                        ):
        self.logger.print_it('Initializing training...')
        if not configs.__eq__(self.train_configs):
            self.logger.print_it('Found different training configurations in the initialize_train method. Resetting the trainer configs...')
            self.reset_configs(configs)

        self.setup_dataloaders(dataset=dataset,
                                batch_size=self.train_configs.batch_size)
        if isinstance(model, torch.nn.Module):
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
