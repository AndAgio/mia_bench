from typing import Union
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.helpers import SubsampledDataset
from src.utils.configs import DefenderConfigs, MistDefenseConfigs
from src.mia.defenses.base import BaseDefender
from src.trainer.train_manager import TrainManager
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.mia.defenses.training.mixup import mixup_data


class MistDefender(BaseDefender):
    """MIST: membership-invariant subspace training, with sub-models trained on subsets of the data.

    Paper: "MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training",
    https://arxiv.org/abs/2311.00919. Documentation: docs/defenses/mist.md.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select MIST and keep its settings."""
        assert isinstance(defender_configs.defense, MistDefenseConfigs), f"MistDefender can only be used with MistDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mist_defender'
        self.mist_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        """Train with MIST's sub-models and return the best model, with the stats if return_stats."""
        self.logger.print_it(f'Mist Defender: training defender model...')
        train_manager = MistTrainManager(train_configs=train_configs,
                                                name=self.name,
                                                logger=self.logger,
                                                mist_configs=self.mist_configs)
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
        """Return the trained model unchanged: MIST only changes training."""
        self.logger.print_it("Mist Defender: defend_model does not modify the model at inference time.")
        self.defended_model = self.trained_model
        return self.defended_model


class MistTrainManager(TrainManager):
    """TrainManager whose epochs are MIST's: train sub-models on subsets, align them, then average them."""
    def __init__(self, train_configs, name: str, logger: callable, mist_configs: MistDefenseConfigs):
        """Set up the training as TrainManager does, and keep MIST's settings."""
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.mist_configs = mist_configs
        self.train_labels = None

    def clone_model(self):
        """Copy the main model into each sub-model."""
        self.sub_models = {i: copy.deepcopy(self.model) for i in range(self.mist_configs.num_submodels)}
        self.logger.print_it(f"MIST TrainManager: cloned {self.mist_configs.num_submodels} sub-models for MIST training.")

    def clone_optimizer(self):
        """Copy the main optimizer, with its state, for each sub-model."""
        self.sub_optimizers = {}
        for i in range(self.mist_configs.num_submodels):
            model = self.sub_models[i]
            optimizer = self.clone_optimizer_for_model(self.optimizer, self.model, model)
            self.sub_optimizers[i] = optimizer
        self.logger.print_it(f"MIST TrainManager: cloned optimizers for {self.mist_configs.num_submodels} sub-models.")

    @staticmethod
    def clone_optimizer_for_model(old_opt, old_model, new_model):
        """Return a copy of old_opt that updates new_model, a copy of old_model, with no tensor shared with old_opt."""
        # The memo replaces every parameter of old_model with the one at the same position in new_model, wherever it
        # appears: in the parameter groups, also when they do not follow the order of the model (e.g. the weights and
        # biases groups of the adaptive SAM variants), and in the base optimizer of the SAM variants.
        new_opt = copy.deepcopy(old_opt, memo={id(p): q for p, q in zip(old_model.parameters(), new_model.parameters())})
        # The copy leaves out the state of the optimizers for the other losses of the SAM variants (see for_objective),
        # so the whole state (e.g. the momentum buffers) is loaded again. It is loaded from a copy, since loading keeps
        # the tensors that are already on the right device, which old_opt and new_opt would then share.
        new_opt.load_state_dict(copy.deepcopy(old_opt.state_dict()))
        return new_opt

    def clone_criterion(self):
        """Copy the training criterion for each sub-model."""
        self.sub_criteria = {}
        for i in range(self.mist_configs.num_submodels):
            criterion = copy.deepcopy(self.criterion)
            self.sub_criteria[i] = criterion
        self.logger.print_it(f"MIST TrainManager: cloned criteria for {self.mist_configs.num_submodels} sub-models.")

    def clone_everything_for_submodels(self):
        """Copy the model, the optimizer and the criterion for each sub-model."""
        # The submodels have no LR scheduler of their own: their optimizers take the learning rate of the main one,
        # which the main scheduler sets once per (global) epoch, so every step of an epoch, in both phases, uses the
        # configured learning rate of that epoch.
        self.clone_model()
        self.clone_optimizer()
        self.clone_criterion()

    def split_data(self):
        """Partition the training data into one subset per sub-model, with the configured split method."""
        if self.mist_configs.split_method == 'random':
            self.split_data_random()
        elif self.mist_configs.split_method == 'stratified':
            self.split_data_stratified()
        else:
            raise ValueError(f"MIST TrainManager: unknown split method {self.mist_configs.split_method}!")

    def split_data_random(self):
        """Partition the training data at random into subsets of equal size, the last one taking the remainder."""
        try:
            all_indices = self.train_loader.dataset.get_indices(mode='current', to_torch=True)
        except AttributeError:
            all_indices = torch.arange(len(self.train_loader.dataset))
        total_size = len(all_indices)
        indices = all_indices[torch.randperm(total_size).to(dtype=torch.long, device=all_indices.device)]
        subset_size = total_size // self.mist_configs.num_submodels
        self.submodel_datasets = {}
        for i in range(self.mist_configs.num_submodels):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size if i < self.mist_configs.num_submodels - 1 else total_size
            subset_indices = indices[start_idx:end_idx]
            subset_dataset = SubsampledDataset.from_positions(self.train_loader.dataset, subset_indices.tolist())
            self.submodel_datasets[i] = subset_dataset
        self.logger.print_it(f"MIST TrainManager: split training data into {self.mist_configs.num_submodels} subsets for sub-models.")

    def split_data_stratified(self):
        """Partition the training data so that each subset has the same share of every class."""
        # The labels are read once, since reading them loads every sample.
        if self.train_labels is None:
            try:
                self.train_labels = np.asarray(self.train_loader.dataset.get_all_targets(to_torch=True))
            except AttributeError:
                self.train_labels = np.asarray(self.train_loader.dataset.targets)
        y = self.train_labels
        classes = np.unique(y)
        # A new partition at every epoch, as in Algorithm 1 of the paper (and as the random split does).
        rng = np.random.default_rng([self.seed, self.epoch])
        parts = [list() for _ in range(self.mist_configs.num_submodels)]
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            splits = np.array_split(idx, self.mist_configs.num_submodels)
            for i in range(self.mist_configs.num_submodels):
                parts[i].extend(splits[i].tolist())
        indices = [np.array(sorted(p), dtype=int) for p in parts]
        self.submodel_datasets = {}
        for i in range(self.mist_configs.num_submodels):
            subset_indices = indices[i]
            subset_dataset = SubsampledDataset.from_positions(self.train_loader.dataset, subset_indices.tolist())
            self.submodel_datasets[i] = subset_dataset
        self.logger.print_it(f"MIST TrainManager: stratified split of training data into {self.mist_configs.num_submodels} subsets for sub-models.")

    def get_submodel_dataset(self, submodel_index: int) -> torch.utils.data.Dataset:
        """Return the subset of the training data of a sub-model."""
        return self.submodel_datasets[submodel_index]

    def average_submodels(self):
        """Set the main model to the average of the sub-models' parameters and buffers."""
        with torch.no_grad():
            sd_list = [m.state_dict() for _, m in self.sub_models.items()]
            keys = sd_list[0].keys()
            # Integer buffers (the num_batches_tracked of BatchNorm) are averaged with integer division, keeping their type.
            avg_params = {k: sum(sd[k] for sd in sd_list) / len(sd_list) if sd_list[0][k].is_floating_point()
                          else sum(sd[k] for sd in sd_list) // len(sd_list) for k in keys}
            self.model.load_state_dict(avg_params)
        self.logger.print_it("MIST TrainManager: averaged sub-models to update the main model.")

    def average_submodel_optimizers(self):
        """Set the main optimizer's state to the average of the sub-models' optimizer states."""
        # As the parameters are averaged into the main model, so that the submodels of the next epoch start from it
        # (e.g. with the momentum built so far) and checkpoints include it.
        state_dicts = [optimizer.state_dict() for _, optimizer in self.sub_optimizers.items()]
        self.optimizer.load_state_dict(self.average_optimizer_states(state_dicts))
        self.logger.print_it("MIST TrainManager: averaged sub-model optimizer states into the main optimizer.")

    @staticmethod
    def average_optimizer_states(states: list):
        """Return the average of optimizer state dicts of the same structure."""
        # Entry by entry: floating point tensors (e.g. momentum buffers) are averaged, the rest (e.g. step counters and
        # the parameter groups, the same for all the submodels) is taken from the first submodel.
        first = states[0]
        if isinstance(first, dict):
            return {k: MistTrainManager.average_optimizer_states([s[k] for s in states if isinstance(s, dict) and k in s])
                    for k in first}
        if isinstance(first, torch.Tensor) and first.is_floating_point() and \
                all(isinstance(s, torch.Tensor) and s.shape == first.shape for s in states):
            return sum(states) / len(states)
        return first

    @torch.no_grad()
    def avg_probs_other_models(self, exclude: int, inputs: torch.Tensor) -> torch.Tensor:
        """Return the probabilities of the inputs averaged over the phase-1 sub-models other than exclude."""
        tot, cnt = 0.0, 0
        for j, model_j in self.reference_models.items():
            if j == exclude:
                continue
            logits = model_j(inputs)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            tot = tot + probs
            cnt += 1
        return tot / cnt

    def train_epoch(self):
        """Run one global epoch of MIST (Algorithm 1) and return its training stats."""
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        # MIST training procedure
        self.clone_everything_for_submodels()
        self.split_data()
        self.train_submodels()
        self.optimize_difference_of_submodels()
        self.average_submodels()
        self.average_submodel_optimizers()

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)
        return train_summary

    def train_submodels(self):
        """Phase 1: train each sub-model on its subset with the training loss."""
        for submodel_index in range(self.mist_configs.num_submodels):
            self.logger.print_it(f"MIST TrainManager: training sub-model {submodel_index+1}/{self.mist_configs.num_submodels}...")
            model = self.sub_models[submodel_index]
            model.train()
            optimizer = self.sub_optimizers[submodel_index]
            criterion = self.sub_criteria[submodel_index]
            # Setup data loader for this submodel
            submodel_dataset = self.get_submodel_dataset(submodel_index=submodel_index)
            submodel_loader = DataLoader(submodel_dataset, batch_size=self.train_configs.batch_size, shuffle=True, num_workers=0)
            for epoch in range(self.mist_configs.submodel_epochs):
                for batch_idx, (inputs, targets, _, _) in enumerate(submodel_loader):
                    model, optimizer, criterion = self.train_step_submodel(model, submodel_index, optimizer, criterion, inputs, targets, epoch, batch_idx=batch_idx, total_batches=len(submodel_loader))
            self.logger.set_logger_newline(console_only=True)
            self.sub_models[submodel_index] = model
            self.logger.print_it(f"MIST TrainManager: finished training sub-model {submodel_index+1}/{self.mist_configs.num_submodels}!")
        self.logger.print_it("MIST TrainManager: finished training all sub-models.")

    def train_step_submodel(self, model, model_index, optimizer, criterion,inputs, targets, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
        """Run one phase-1 step on a batch, with mixup in mist_mixup mode."""
        if self.mist_configs.mix_up:
            return self.train_step_submodel_with_mixup(model, model_index, optimizer, criterion, inputs, targets, epoch, batch_idx, total_batches)
        else:
            return self.train_step_submodel_standard(model, model_index, optimizer, criterion, inputs, targets, epoch, batch_idx, total_batches)
        
    def train_step_submodel_with_mixup(self, model, model_index, optimizer, criterion,inputs, targets, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
        """Run one phase-1 step on a mixed-up batch, and record the stats on the clean batch."""
        self.epoch_stats_tracker.batch_start()
        # Map to available device
        mixed_inputs, targets_a, targets_b, lmbd = mixup_data(inputs, targets, alpha=self.mist_configs.alpha_mixup)
        mixed_inputs = mixed_inputs.to(self.device)
        targets_a = targets_a.to(self.device)
        targets_b = targets_b.to(self.device)

        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(mixed_inputs, targets_a, targets_b, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(model)
                else:
                    disable_running_stats(model)
                outputs = model(mixed_inputs)
                loss = lmbd * criterion(outputs, targets_a) + (1 - lmbd) * criterion(outputs, targets_b)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            optimizer.step(closure, mixed_inputs, targets_a, targets_b)
            optimizer.zero_grad()
            loss, outputs = optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = lmbd * criterion(outputs, targets_a) + (1 - lmbd) * criterion(outputs, targets_b)
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            optimizer.step()

        # Train stats on the clean inputs, as in the Mixup defense. The submodel is in train mode, so freeze the BN
        # running stats, which this pass would otherwise update with the clean batch.
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        disable_running_stats(model)
        with torch.no_grad():
            clean_outputs = model(inputs)
        enable_running_stats(model)
        self.epoch_stats_tracker.update(preds=clean_outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        message = f"{self.device.type.upper()}:{self.local_rank} |"
        message += f" GLOBAL-EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        message += f" SUB-MODEL: {model_index+1}/{self.mist_configs.num_submodels} |"
        message += f" SUB-EPOCH: {epoch+1}/{self.mist_configs.submodel_epochs} |"
        message += f" TRAINING PHASE |"
        bar_length = 10
        progress = float(batch_idx) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        message += 'CE loss={:.5f} |'.format(loss.item())
        message = self.append_lr(message)
        message = self.append_times(message)
        self.logger.print_it_same_line(message, console_only=True)

        return model, optimizer, criterion

    def train_step_submodel_standard(self, model, model_index, optimizer, criterion,inputs, targets, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
        """Run one phase-1 step on a batch with the training loss."""
        # Map to available device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(inputs, targets, criterion, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(model)
                else:
                    disable_running_stats(model)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            optimizer.step(closure, inputs, targets, criterion)
            optimizer.zero_grad()
            loss, outputs = optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss = loss.mean() if loss.numel() > 1 else loss
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            optimizer.step()

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        message = f"{self.device.type.upper()}:{self.local_rank} |"
        message += f" GLOBAL-EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        message += f" SUB-MODEL: {model_index+1}/{self.mist_configs.num_submodels} |"
        message += f" SUB-EPOCH: {epoch+1}/{self.mist_configs.submodel_epochs} |"
        message += f" TRAINING PHASE |"
        bar_length = 10
        progress = float(batch_idx) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        message += 'CE loss={:.5f} |'.format(loss.item())
        message = self.append_lr(message)
        message = self.append_times(message)
        self.logger.print_it_same_line(message, console_only=True)

        return model, optimizer, criterion

    
    def optimize_difference_of_submodels(self):
        """Phase 2: move each sub-model to lower its cross-difference loss against the phase-1 sub-models."""
        # Each submodel keeps the optimizer of the training phase, so that the momentum is shared by the two losses
        # (the SAM variants keep their own state for this loss, see optimize_difference_step).
        # As in Algorithm 1 of the paper, every submodel is aligned to the other submodels as left by the training phase,
        # not to those already updated by this phase, so the alignment does not depend on the order of the submodels.
        self.reference_models = {i: copy.deepcopy(model).eval() for i, model in self.sub_models.items()}
        # Optimize each submodel to minimize the cross-difference loss from the average of the other submodels.
        for submodel_index in range(self.mist_configs.num_submodels):
            self.logger.print_it(f"MIST TrainManager: optimizing difference for sub-model {submodel_index+1}/{self.mist_configs.num_submodels}...")
            model = self.sub_models[submodel_index]
            model.train()
            optimizer = self.sub_optimizers[submodel_index]
            # Element-wise, so that SAM-like optimizers can get per-sample losses (ESAM selects samples by them).
            criterion = torch.nn.L1Loss(reduction="none")
            # Setup data loader for this submodel
            submodel_dataset = self.get_submodel_dataset(submodel_index=submodel_index)
            submodel_loader = DataLoader(submodel_dataset, batch_size=self.train_configs.batch_size, shuffle=True, num_workers=0)
            for epoch in range(self.mist_configs.submodel_epochs):
                for batch_idx, (inputs, targets, _, _) in enumerate(submodel_loader):
                    model, optimizer, criterion = self.optimize_difference_step(model, submodel_index, optimizer, criterion, inputs, targets, epoch, batch_idx=batch_idx, total_batches=len(submodel_loader))
            self.logger.set_logger_newline(console_only=True)
            self.sub_models[submodel_index] = model
            self.logger.print_it(f"MIST TrainManager: finished optimizing difference for sub-model {submodel_index+1}/{self.mist_configs.num_submodels}!")
        # Free the memory of the snapshots until the next alignment phase.
        self.reference_models = None

    def cross_difference_loss(self, criterion, outputs, targets, other_probs) -> torch.Tensor:
        """Return lambda times MIST's cross-difference loss (Eq. 3) of each sample.

        It is the L1 distance between the sub-model's probability of the true class and other_probs, the average
        probability of the true class under the other sub-models.
        """
        probs = torch.nn.functional.softmax(outputs, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return self.mist_configs.lmbd * criterion(probs, other_probs)

    def optimize_difference_step(self, model, model_index, optimizer, criterion, inputs, targets, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
        """Run one phase-2 step on a batch with the cross-difference loss."""
        # Map to available device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        # The other sub-models are not updated by this step, so their probabilities of the true class are computed once.
        other_probs = self.avg_probs_other_models(exclude=model_index, inputs=inputs).gather(1, targets.unsqueeze(1)).squeeze(1)
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(inputs, targets, other_probs, criterion, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(model)
                else:
                    disable_running_stats(model)
                outputs = model(inputs)
                loss = self.cross_difference_loss(criterion, outputs, targets, other_probs)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            # The alignment loss is minimized by an optimizer sharing parameters, momentum and learning rate with the
            # one training the sub-model, but with its own state for this loss.
            alignment_optimizer = optimizer.for_objective('alignment')
            alignment_optimizer.step(closure, inputs, targets, other_probs, criterion)
            alignment_optimizer.zero_grad()
            loss, outputs = alignment_optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.cross_difference_loss(criterion, outputs, targets, other_probs)
            # loss = loss.mean() if loss.numel() > 1 else loss
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            optimizer.step()
        
        message = f"{self.device.type.upper()}:{self.local_rank} |"
        message += f" GLOBAL-EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        message += f" SUB-MODEL: {model_index+1}/{self.mist_configs.num_submodels} |"
        message += f" SUB-EPOCH: {epoch+1}/{self.mist_configs.submodel_epochs} |"
        message += f" ALIGNMENT PHASE |"
        bar_length = 10
        progress = float(batch_idx) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        message += 'MAE loss={:.5f} |'.format(loss.item())
        message = self.append_lr(message)
        message = self.append_times(message)
        self.logger.print_it_same_line(message, console_only=True)

        return model, optimizer, criterion
