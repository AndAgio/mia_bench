
from typing import Union
import torch
from src.utils.configs import DefenderConfigs, RelaxLossDefenseConfigs, TrainConfigs 
from src.trainer.train_manager import TrainManager
from src.mia.defenses.base import BaseDefender
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats


class RelaxLossDefender(BaseDefender):
    # Implementation of "RelaxLoss: Defending Membership Inference Attacks without Losing Utility" (https://arxiv.org/pdf/2207.05801).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, RelaxLossDefenseConfigs), f"RelaxLossDefender can only be used with RelaxLossDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'relax_loss_defender'
        self.relax_loss_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f'RelaxLoss Defender: training defender model with relaxed loss and alpha {self.relax_loss_configs.relax_alpha}...')
        train_manager = RelaxLossTrainManager(train_configs=train_configs,
                                                name=self.name,
                                                logger=self.logger,
                                                relax_alpha=self.relax_loss_configs.relax_alpha)
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
        self.logger.print_it('RelaxLoss Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model


class RelaxLossTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, relax_alpha: float = 0.5):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.relax_alpha = relax_alpha

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device (profile this)
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(inputs, targets, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                relaxed_loss = self.relax_loss(loss=loss, outputs=outputs, targets=targets)
                if mean:
                    relaxed_loss = relaxed_loss.mean()
                if backward:
                    relaxed_loss.backward()
                return relaxed_loss, outputs
            self.optimizer.step(closure, inputs, targets)
            self.optimizer.zero_grad()
            relaxed_loss, outputs = self.optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            relaxed_loss = self.relax_loss(loss=loss, outputs=outputs, targets=targets)
            relaxed_loss.backward()
            self.optimizer.step()

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)


    def relax_loss(self, loss: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if loss >= self.relax_alpha:
            relaxed_loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
        else:
            if self.train_stats_tracker.current_epoch % 2 == 0:
                # just negate the loss to do grad ascent
                relaxed_loss = -loss
            else:
                # flatten probs, compute loss wrt soft labels
                with torch.no_grad():
                    prob_gt = torch.nn.functional.softmax(outputs, dim=1)[torch.arange(targets.size(0)), targets]
                    prob_ngt = (1.0 - prob_gt) / (outputs.size(1) - 1)
                    onehot = torch.nn.functional.one_hot(targets, num_classes=outputs.size(1))
                    soft_labels = onehot * prob_gt.unsqueeze(-1).repeat(1, outputs.size(1))\
                        + (1 - onehot) * prob_ngt.unsqueeze(-1).repeat(1, outputs.size(1))
                relaxed_loss = self.criterion(outputs, soft_labels)
        return relaxed_loss