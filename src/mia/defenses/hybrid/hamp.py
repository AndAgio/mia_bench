from typing import Union, Callable
import numpy as np
import torch
from src.trainer.train_manager import TrainManager
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.utils.configs import DefenderConfigs, HampDefenseConfigs
from src.mia.defenses.base import BaseDefender


class HampDefender(BaseDefender):
    # Implementation of running time component of HAMP defense from "Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction" (https://arxiv.org/pdf/2307.01610).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, HampDefenseConfigs), f"Hamp Defender can only be used with HampDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'hamp_defender'
        self.hamp_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it(f"Hamp Defender: selected mode {self.hamp_configs.mode}")
        if self.hamp_configs.mode in ['train_only', 'full']:
            self.logger.print_it(f'Hamp Defender: training defender model with alpha {self.hamp_configs.alpha}...')
            train_manager = HampTrainManager(train_configs=train_configs,
                                            name=self.name,
                                            logger=self.logger,
                                            hamp_configs=self.hamp_configs,
                                            num_classes=self.dataset_configs.num_classes)
        elif self.hamp_configs.mode == 'test_only':
            self.logger.print_it("Hamp Defender: test_only mode selected, training with standard loss and labels.")
            train_manager = TrainManager(train_configs=train_configs,
                                        name=self.name,
                                        logger=self.logger)
        else:
            raise ValueError(f"Hamp Defender: Unknown mode {self.hamp_configs.mode}!")
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
        if self.hamp_configs.mode in ['test_only', 'full']:
            self.logger.print_it(f'Hamp Defender: adding defense layer on outputs using entropy trick...')
            if isinstance(device, str):
                device = self.get_device(dev_str=device)
            self.defended_model = HampTestWrapper(model=self.trained_model).to(device)
        elif self.hamp_configs.mode == 'train_only':
            self.logger.print_it("Hamp Defender: train_only mode selected, defend_model does not modify the model at inference time.")
            self.defended_model = self.trained_model
        else:
            raise ValueError(f"Hamp Defender: Unknown mode {self.hamp_configs.mode}!")
        return self.defended_model
    


class HampTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, hamp_configs: HampDefenseConfigs = None, num_classes: int = None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.hamp_configs = hamp_configs
        assert num_classes is not None, "HampTrainManager: num_classes must be provided."
        self.num_classes = num_classes

    def setup_loss(self, loss: Union[str, Callable]):
        self.criterion = HampLoss(gamma=self.hamp_configs.gamma, alpha=self.hamp_configs.alpha)

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        original_targets = targets.clone()
        targets = self.get_soft_labels(original_labels=original_targets)
        inputs= inputs.to(self.device)
        targets = targets.to(self.device)
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
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            self.optimizer.step(closure, inputs, targets)
            self.optimizer.zero_grad()
            loss, outputs = self.optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            outputs = self.model(inputs)
        
        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)

    @torch.no_grad()
    def get_soft_labels(self, original_labels):
        uniform_preds = torch.ones(self.num_classes, device=original_labels.device) / self.num_classes
        logp = torch.log(uniform_preds)
        highest_entropy = torch.sum( -uniform_preds * logp , dim=0 )
        entropy_threshold = highest_entropy*self.hamp_configs.gamma
        reduced_prob = 0.001
        true_target = 0
        preds = torch.zeros(self.num_classes, device=original_labels.device)
        preds[true_target]  = 1.
        while(True):
            preds[true_target] -= reduced_prob
            preds[:true_target] += reduced_prob/(self.num_classes-1) 
            preds[true_target+1:] += reduced_prob/(self.num_classes-1) 
            if(torch.sum(-preds * torch.log(preds), dim=0) >= entropy_threshold):
                break
        top1 = preds[true_target]
        uniform_non_top1 = preds[true_target+1]
        new_soft_label = torch.zeros((original_labels.shape[0], self.num_classes), device=original_labels.device) 
        for i in range( original_labels.shape[0] ):
            new_soft_label[i][original_labels[i]] = top1
            others = [j for j in range(self.num_classes) if j != original_labels[i]]
            new_soft_label[i][others] = uniform_non_top1
        return new_soft_label


class HampLoss(torch.nn.Module):
    def __init__(self, gamma, alpha):
        super(HampLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = 'mean'

    def forward(self, outputs, targets):
        entropy = torch.distributions.Categorical(probs = torch.nn.functional.softmax(outputs, dim=1)).entropy()
        loss1 = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(outputs, dim=1), targets, reduction='batchmean') / targets.size(1) # average over support size to make it equivalent to original implementation
        loss2 = -1 * self.alpha * torch.mean(entropy)
        loss = loss1 + loss2
        return loss


class HampTestWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(HampTestWrapper, self).__init__()
        self.model = model


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        random_inputs = torch.rand(*inputs.shape, device=inputs.device)
        original_outputs = self.model(inputs)
        with torch.no_grad():
            random_outputs = self.model(random_inputs)
            original_output_sorted = torch.sort(original_outputs, dim=1)
            random_output_sorted = torch.sort(random_outputs, dim=1)
            new_outputs = torch.zeros_like(original_outputs)
            for i in range(original_outputs.size(0)):
                for j in range(original_outputs.size(1)):
                    new_outputs[i][torch.where(original_outputs[i]==original_output_sorted.values[i][j])] = random_output_sorted.values[i%random_outputs.size(0)][j]
        return new_outputs