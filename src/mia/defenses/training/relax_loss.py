
from typing import Union
import torch
import torch.nn.functional as F
from src.utils.configs import DefenderConfigs, RelaxLossDefenseConfigs, TrainConfigs
from src.trainer.train_manager import TrainManager
from src.mia.defenses.base import BaseDefender
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats


class RelaxLossDefender(BaseDefender):
    # Implementation of "RelaxLoss: Defending Membership Inference Attacks without Losing Utility" (https://arxiv.org/pdf/2207.05801),
    # following Algorithm 1 and App. B.3 of the paper. See RelaxLossTrainManager.relax_loss for the differences with the
    # official code (https://github.com/DingfanChen/RelaxLoss).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, RelaxLossDefenseConfigs), f"RelaxLossDefender can only be used with RelaxLossDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'relax_loss_defender'
        self.relax_loss_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f'RelaxLoss Defender: training defender model with relaxed loss (alpha={self.relax_loss_configs.relax_alpha}, '
                             f'upper={self.relax_loss_configs.relax_upper}, posterior flattening on '
                             f'{"incorrect predictions" if self.relax_loss_configs.flatten_incorrect_only else "all samples"})...')
        train_manager = RelaxLossTrainManager(train_configs=train_configs,
                                                name=self.name,
                                                logger=self.logger,
                                                relax_alpha=self.relax_loss_configs.relax_alpha,
                                                relax_upper=self.relax_loss_configs.relax_upper,
                                                flatten_incorrect_only=self.relax_loss_configs.flatten_incorrect_only)
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
    def __init__(self, train_configs, name: str, logger=None, relax_alpha: float = 0.5, relax_upper: float = 1.0, flatten_incorrect_only: bool = True):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        assert relax_alpha > 0, f"RelaxLoss alpha should be positive, found {relax_alpha}!"
        assert 0 < relax_upper <= 1, f"RelaxLoss upper confidence should be in (0, 1], found {relax_upper}!"
        self.relax_alpha = relax_alpha
        self.relax_upper = relax_upper
        self.flatten_incorrect_only = flatten_incorrect_only

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device (profile this)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            # The branch of Algorithm 1 is decided once per step, by the first call, which all of them make at the current
            # weights on the whole batch. The later calls run at perturbed weights and, with ESAM, on the selected samples
            # only, whose loss is typically higher than the one of the batch: deciding again there would often switch to
            # gradient descent while the batch loss is below alpha.
            branch = None
            def closure(inputs, targets, mean=True, backward=True, run_stats=True):
                nonlocal branch
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                outputs = self.model(inputs)
                if branch is None:
                    branch = self.relax_branch(outputs=outputs, targets=targets)
                relaxed_loss = self.relax_loss(outputs=outputs, targets=targets, branch=branch)
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
            relaxed_loss = self.relax_loss(outputs=outputs, targets=targets).mean()
            relaxed_loss.backward()
            self.optimizer.step()

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)


    def relax_branch(self, outputs: torch.Tensor, targets: torch.Tensor) -> str:
        # Branch of Algorithm 1 for a batch with these outputs: gradient descent while the batch loss is not below the
        # target loss, otherwise gradient ascent on even epochs and posterior flattening on odd ones.
        if F.cross_entropy(outputs.detach(), targets).item() >= self.relax_alpha:
            return 'descent'
        return 'ascent' if self.epoch % 2 == 0 else 'flatten'

    def relax_loss(self, outputs: torch.Tensor, targets: torch.Tensor, branch: str = None) -> torch.Tensor:
        # Algorithm 1 of the paper, with the per-modality techniques of its App. B.3. Returns per-sample terms whose
        # batch mean is the objective of the current step, so that optimizers working on per-sample losses (e.g., ESAM)
        # can be used as well. The cross-entropy is computed here instead of with self.criterion, whose reduction is
        # changed by setup_optimizer depending on the optimizer in use. The branch of Algorithm 1 is the one of these
        # outputs (see relax_branch), unless given.
        # Following the paper, this differs from the official code in that: epochs are counted from 1 (the official code
        # starts with a gradient ascent epoch), the soft labels are constants (the official code does not detach them),
        # and, for image data, the flattened samples do not also get gradient ascent (as in the official code).
        loss = F.cross_entropy(outputs, targets, reduction='none')
        if branch is None:
            branch = self.relax_branch(outputs=outputs, targets=targets)
        if branch == 'descent':
            return loss
        if branch == 'ascent':
            return -loss
        # Posterior flattening on odd epochs: keep the ground-truth score and spread the rest evenly over the other
        # classes (Sec. 4.2), with the ground-truth score clamped to relax_upper (App. B.3, non-image data).
        with torch.no_grad():
            num_classes = outputs.size(1)
            prob_gt = F.softmax(outputs, dim=1).gather(1, targets.unsqueeze(1)).clamp(max=self.relax_upper)
            onehot = F.one_hot(targets, num_classes=num_classes).to(outputs.dtype)
            soft_targets = onehot * prob_gt + (1 - onehot) * (1 - prob_gt) / (num_classes - 1)
        soft_loss = F.cross_entropy(outputs, soft_targets, reduction='none')
        if self.flatten_incorrect_only:
            # Image data (App. B.3): flattening is restricted to incorrect predictions. The paper does not say what correct
            # predictions get: as in the official code, they get gradient ascent, the other step of Algorithm 1 below alpha.
            incorrect = (outputs.argmax(dim=1) != targets).to(outputs.dtype)
            return incorrect * soft_loss - (1 - incorrect) * loss
        return soft_loss
