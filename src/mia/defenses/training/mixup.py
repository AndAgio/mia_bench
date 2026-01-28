from typing import Union
import numpy as np
import torch
from src.trainer.train_manager import TrainManager
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.utils.configs import DefenderConfigs, MixupDefenseConfigs
from src.mia.defenses.base import BaseDefender


class MixupDefender(BaseDefender):
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MixupDefenseConfigs), f"MixupDefender can only be used with MixupDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mixup_defender'
        self.mixup_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it(f'Mixup Defender: training defender model with alpha {self.mixup_configs.alpha}...')
        train_manager = MixupTrainManager(train_configs=train_configs,
                                                name=self.name,
                                                logger=self.logger,
                                                mixup_configs=self.mixup_configs)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        configs=train_configs)
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
        self.logger.info("MixupDefender: defend_model does not modify the model at inference time.")
        self.defended_model = self.trained_model
        return self.defended_model


class MixupTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, mixup_configs: MixupDefenseConfigs = None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.mixup_configs = mixup_configs

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        mixed_inputs, targets_a, targets_b, lmbd = mixup_data(inputs, targets, alpha=self.mixup_configs.alpha)
        mixed_inputs = mixed_inputs.to(self.device, non_blocking=True)
        targets_a = targets_a.to(self.device, non_blocking=True)
        targets_b = targets_b.to(self.device, non_blocking=True)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(mixed_inputs, targets_a, targets_b, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                outputs = self.model(mixed_inputs)
                loss = lmbd * self.criterion(outputs, targets_a) + (1 - lmbd) * self.criterion(outputs, targets_b)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            self.optimizer.step(closure, mixed_inputs, targets_a, targets_b)
            self.optimizer.zero_grad()
            loss, outputs = self.optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            self.optimizer.zero_grad()
            outputs = self.model(mixed_inputs)
            loss = lmbd * self.criterion(outputs, targets_a) + (1 - lmbd) * self.criterion(outputs, targets_b)
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            self.optimizer.step()

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)



def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
