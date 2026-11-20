from typing import Union
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.data.helpers import MultiDatasets, DatasetSplitter
from src.utils.configs import DefenderConfigs, MmdDefenseConfigs, TrainConfigs
from src.utils.variables import MMD_CODE_MIN_ACCURACY_GAP, MMD_REFERENCE_FRACTION_OF_VAL_SPLIT
from src.mia.defenses.base import BaseDefender
from src.mia.defenses.training.mixup import mixup_data
from src.trainer.train_manager import TrainManager


class MmdDefender(BaseDefender):
    """MMD defense: penalizes the distance between the model's outputs on training data and on reference non-members.

    Paper: "Membership Inference Attacks and Defenses in Classification Models",
    https://arxiv.org/abs/2002.12062. Documentation: docs/defenses/mmd.md.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select MMD and keep its settings."""
        assert isinstance(defender_configs.defense, MmdDefenseConfigs), f"MmdDefender can only be used with MmdDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mmd_defender'
        self.mmd_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        """Train with the MMD loss and return the best model, with the stats if return_stats."""
        self.logger.print_it(f'MMD Defender: training defender model with MMD loss and lambda {self.mmd_configs.lmbd}...')
        train_manager = MmdTrainManager(train_configs=train_configs,
                                        name=self.name,
                                        logger=self.logger,
                                        mmd_configs=self.mmd_configs)
        # As for AdvReg, the reference non-members of the MMD loss must not be the samples that select the best
        # epoch, so val is split into two disjoint stratified parts. The reference part comes last, so that every
        # class of val keeps at least one reference sample.
        selection_dataset, reference_dataset = DatasetSplitter.split_stratified(
            self.dataset.get('val'),
            proportions=[1 - MMD_REFERENCE_FRACTION_OF_VAL_SPLIT, MMD_REFERENCE_FRACTION_OF_VAL_SPLIT],
            seed=self.dataset_configs.seed)
        self.logger.print_it(f'MMD Defender: split val into {len(selection_dataset)} samples for checkpoint selection '
                             f'and {len(reference_dataset)} reference samples for the MMD loss.')
        train_dataset = MultiDatasets()
        train_dataset.add(self.dataset.get('train'), 'train')
        train_dataset.add(selection_dataset, 'val')
        train_dataset.add(self.dataset.get('test'), 'test')
        train_dataset.add_info(self.dataset.get_info())
        train_manager.initialize_train(dataset=train_dataset,
                                        model=self.untrained_model,
                                        configs=train_configs,
                                        reference_dataset=reference_dataset)
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
        """Return the trained model unchanged: MMD only changes training."""
        self.logger.print_it('MMD Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model
    

class MmdTrainManager(TrainManager):
    """TrainManager that follows each epoch with an MMD pass when the model overfits, as in the authors' code."""
    def __init__(self, train_configs, name: str, logger=None, mmd_configs: MmdDefenseConfigs = None):
        """Set up the training as TrainManager does, and keep MMD's settings."""
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.mmd_configs = mmd_configs
        self.reference_dataset = None
        self._reference_index_dataset = None
        self._reference_indices_by_label = {}

    def initialize_train(self, dataset: MultiDatasets, model, configs: TrainConfigs, reference_dataset):
        """Set up the training as TrainManager does, with the reference non-members of the MMD loss."""
        super().initialize_train(dataset=dataset, model=model, configs=configs)
        self.reference_dataset = reference_dataset
        sampler = DistributedSampler(reference_dataset, num_replicas=self.world_size, rank=self.global_rank,
                                     shuffle=False) if self.distributed else None
        self.reference_loader = DataLoader(reference_dataset, batch_size=self.train_loader.batch_size,
                                           shuffle=False, sampler=sampler)

    def train_executions(self):
        """Run a training epoch, an MMD pass if the accuracy gap calls for it, then the validation and test epochs."""
        assert self.run_val, "MMD training requires a validation split to select the best epoch."
        assert self.reference_dataset is not None, "MMD training requires reference non-members (reference_dataset)."
        self.train_epoch()
        train_acc = self.get_train_accuracy()
        val_acc = self.get_reference_accuracy()
        if abs(train_acc - val_acc) >= MMD_CODE_MIN_ACCURACY_GAP:
            self.logger.print_it(f"Training accuracy: {train_acc:.4f}, Reference accuracy: {val_acc:.4f}. The gap "
                                 f"between training and reference accuracy is {abs(train_acc - val_acc):.4f}, "
                                 f"which is large enough...")
            if self.mmd_configs.lmbd >= 1e-5 and self.epoch < self.train_configs.scheduler_config.epochs:
                self.logger.print_it(f"Other conditions for training with MMD loss are met (lambda {self.mmd_configs.lmbd} >= 1e-5 and epoch {self.epoch} < max epochs {self.train_configs.scheduler_config.epochs}), we will run another epoch with MMD loss.")
                self.train_with_mmd_distance()
            else:
                self.logger.print_it(f"Other conditions for training with MMD loss are not met (lambda {self.mmd_configs.lmbd} < 1e-5 or epoch {self.epoch} >= max epochs {self.train_configs.scheduler_config.epochs}), we will NOT run another epoch with MMD loss.")
        else:
            self.logger.print_it(f"Training accuracy: {train_acc:.4f}, Reference accuracy: {val_acc:.4f}. The gap "
                                 f"between training and reference accuracy is {abs(train_acc - val_acc):.4f}, "
                                 f"which is small enough, we will NOT run another epoch with MMD loss.")
        self.val_epoch()
        if self.run_test:
            self.test_epoch()
        self.scheduler.step()
    
    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        """Take one step on the batch, with mixup if configured, and record clean-input metrics."""
        if self.mmd_configs.use_mixup:
            mixed_inputs, targets_a, targets_b, lmbd = mixup_data(inputs, targets, alpha=self.mmd_configs.mixup_alpha)
            mixed_inputs = mixed_inputs.to(self.device)
            targets_a = targets_a.to(self.device)
            targets_b = targets_b.to(self.device)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.epoch_stats_tracker.batch_start()
        if self.mmd_configs.use_mixup:
            self.train_step_mixup(mixed_inputs, targets_a, targets_b, lmbd)
            # Clean-input metrics must not update BN a second time after the mixed training pass.
            disable_running_stats(self.model)
            try:
                with torch.no_grad():
                    outputs = self.model(inputs)
            finally:
                enable_running_stats(self.model)
        else:
            # Standard training already has clean predictions; reuse them for metrics.
            outputs = self.train_step_standard(inputs, targets)

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)

    def train_step_mixup(self, mixed_inputs, targets_a, targets_b, lmbd):
        """Take one optimizer step on the mixup loss of the mixed batch."""
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

    def train_step_standard(self, inputs, targets):
        """Take one optimizer step on the cross-entropy of the batch and return its detached outputs."""
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
            enable_running_stats(self.model)
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            self.optimizer.step()
        return outputs.detach()

    @torch.no_grad()
    def get_train_accuracy(self):
        """Return the model's accuracy on the training split, in eval mode and over every rank."""
        self.model.eval()
        cumulative_correct = 0
        cumulative_total = 0
        local_offset = 0
        for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.train_loader):
            self.logger.print_it_same_line(f"Computing training accuracy for batch {batch_idx+1}/{len(self.train_loader)}...", console_only=True)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            correct, total = self._batch_accuracy_counts(outputs, targets, self.train_loader, local_offset)
            cumulative_correct += correct
            cumulative_total += total
            local_offset += targets.size(0)
        self.logger.set_logger_newline(console_only=True)
        return self._global_accuracy(cumulative_correct, cumulative_total)

    @torch.no_grad()
    def get_reference_accuracy(self):
        """Return the model's accuracy on the reference non-members, in eval mode and over every rank."""
        self.model.eval()
        cumulative_correct = 0
        cumulative_total = 0
        local_offset = 0
        for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.reference_loader):
            self.logger.print_it_same_line(f"Computing reference accuracy for batch {batch_idx+1}/"
                                           f"{len(self.reference_loader)}...", console_only=True)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            correct, total = self._batch_accuracy_counts(outputs, targets, self.reference_loader, local_offset)
            cumulative_correct += correct
            cumulative_total += total
            local_offset += targets.size(0)
        self.logger.set_logger_newline(console_only=True)
        return self._global_accuracy(cumulative_correct, cumulative_total)

    @staticmethod
    def _batch_accuracy_counts(outputs, targets, loader, local_offset):
        """Return the correct and total counts of a batch, leaving out the copies a DistributedSampler pads with."""
        valid_count = targets.size(0)
        sampler = loader.sampler
        if isinstance(sampler, DistributedSampler) and not sampler.drop_last:
            # DistributedSampler pads with repeated samples. Keep equal forward counts across ranks,
            # but exclude these copies from the accuracy used to decide whether to regularize.
            real_samples = max(0, (len(loader.dataset) - 1 - sampler.rank) // sampler.num_replicas + 1)
            valid_count = min(valid_count, max(0, real_samples - local_offset))
        correct = (outputs.argmax(dim=-1)[:valid_count] == targets[:valid_count]).sum().item()
        return correct, valid_count

    def _global_accuracy(self, correct, total):
        """Return the accuracy of the counts summed over every rank."""
        if self.distributed:
            counts = torch.tensor([correct, total], dtype=torch.long, device=self.device)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            correct, total = counts.tolist()
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def get_indices_by_label(dataset):
        """Return a dict from each label to the local indices of the dataset's samples with it."""
        try:
            targets = dataset.get_all_targets(to_torch=True)
        except AttributeError:
            try:
                targets = torch.tensor(dataset.targets)
            except AttributeError:
                targets = [dataset[i][1] for i in range(len(dataset))]
        indices_by_label = {}
        for index, label in enumerate(targets):
            label = label.item() if torch.is_tensor(label) else label
            indices_by_label.setdefault(label, []).append(index)
        return indices_by_label

    @staticmethod
    def get_subset_by_label(dataset, target_label):
        """Return the Subset of the dataset's samples with the given label."""
        indices = MmdTrainManager.get_indices_by_label(dataset).get(target_label, [])
        return Subset(dataset, indices)

    def train_with_mmd_distance(self):
        """Run the MMD pass: for each training batch, one step on the MMD between its outputs and the reference's."""
        self.reset_epoch_stats(phase='mmd')
        training_data = self.dataset.get('train')
        sampler = DistributedSampler(training_data, num_replicas=self.world_size,
                                     rank=self.global_rank, shuffle=False) if self.distributed else None
        train_loader_in_order = DataLoader(training_data, batch_size=self.train_loader.batch_size,
                                          shuffle=False, sampler=sampler)
        reference_data = self.reference_dataset
        # Cache local indices in the reference set's current order. Rebuild if the set is replaced.
        if self._reference_index_dataset is not reference_data:
            self._reference_indices_by_label = self.get_indices_by_label(reference_data)
            self._reference_index_dataset = reference_data
        # Match the reference: deterministic outputs and frozen BN buffers, with gradients enabled.
        self.model.eval()
        enable_running_stats(self.model)
        
        for loss_index, (train_images, train_labels, original_indices, resampled_indices) in enumerate(train_loader_in_order):
            self.epoch_stats_tracker.batch_start()
            batch_num = train_labels.size()[0]
            self.optimizer.zero_grad()

            unique_labels = torch.unique(train_labels)
            missing_labels = [label.item() for label in unique_labels
                              if not self._reference_indices_by_label.get(label.item())]
            missing = bool(missing_labels)
            if self.distributed:
                # All ranks must fail together rather than leaving peers waiting in DDP backward.
                missing_flag = torch.tensor(int(missing), device=self.device)
                dist.all_reduce(missing_flag, op=dist.ReduceOp.MAX)
                missing = bool(missing_flag.item())
            if missing:
                detail = str(missing_labels[0]) if missing_labels else 'on another distributed rank'
                raise ValueError(f"MMD reference set has no samples for training class {detail}; "
                                 "provide a validation split containing every training class.")
            subsets_val = []
            for label in unique_labels:
                all_val_with_matching_class = Subset(reference_data,
                                                     self._reference_indices_by_label.get(label.item(), []))
                freq = torch.count_nonzero(train_labels == label).item()

                #TODO: Refactor also MMD to avoid creating Subset datasets and instead use the custom dataset classes defined in data helpers.
                #Issue URL: https://github.com/AndAgio/mia_bench/issues/45
                # assignees: AndAgio

                available = len(all_val_with_matching_class)
                if available < freq:
                    sampled_indices = torch.randint(available, (freq,))
                else:
                    sampled_indices = torch.randperm(available)[:freq]
                subset_val = Subset(all_val_with_matching_class, sampled_indices)
                subsets_val.append(subset_val)
            sampled_val = ConcatDataset(subsets_val)
            valid_images, valid_labels, _, _ = next(iter(DataLoader(sampled_val, batch_size=self.train_loader.batch_size, shuffle=False)))

            train_images = train_images.to(self.device)
            train_labels = train_labels.to(self.device)
            train_labels = train_labels.view(batch_num,1)
            valid_images = valid_images.to(self.device)
            valid_labels = valid_labels.to(self.device)
            valid_labels = valid_labels.view(batch_num,1)

            if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
                # The MMD term gets the same sharpness-aware step as the training loss, through an optimizer sharing
                # parameters, momentum and learning rate with the main one, but with its own state for this loss.
                def closure(train_images, valid_images, mean=True, backward=True, run_stats=True):
                    if run_stats:
                        enable_running_stats(self.model)
                    else:
                        disable_running_stats(self.model)
                    train_outputs = torch.nn.functional.softmax(self.model(train_images), dim=1)
                    # As in the paper, only the training outputs are optimized, not the reference ones.
                    with torch.no_grad():
                        valid_outputs = torch.nn.functional.softmax(self.model(valid_images), dim=1)
                    loss = mix_rbf_mmd2(train_outputs, valid_outputs, sigma_list=[1]) * self.mmd_configs.lmbd
                    if backward:
                        loss.backward()
                    return loss, train_outputs
                mmd_optimizer = self.optimizer.for_objective('mmd')
                try:
                    mmd_optimizer.step(closure, train_images, valid_images)
                finally:
                    enable_running_stats(self.model)
                mmd_optimizer.zero_grad()
                mmd_loss, _ = mmd_optimizer.get_first_closure_outputs()
            else:
                outputs = self.model(train_images)
                all_train_outputs = torch.nn.functional.softmax(outputs,dim=1)
                with torch.no_grad():
                    all_valid_outputs = torch.nn.functional.softmax(self.model(valid_images), dim=1)
                mmd_loss = mix_rbf_mmd2(all_train_outputs,all_valid_outputs,sigma_list=[1])*self.mmd_configs.lmbd
                mmd_loss.backward()
                self.optimizer.step()
    
            self.epoch_stats_tracker.update(preds=mmd_loss.detach(), targets=None,
                                           metrics={'mmd_loss': lambda loss, _: loss})
            self.epoch_stats_tracker.batch_end(batch_size=batch_num)
            message = self.build_message_for_batch_end_mmd(loss_index+1, len(train_loader_in_order), mmd_loss.item())
            self.logger.print_it_same_line(message, console_only=True)
        self.logger.set_logger_newline(console_only=True)

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        mmd_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=mmd_summary)
        self.logger.print_it(f"{message}", file_only=True)

    def build_message_for_batch_end_mmd(self, index_batch, total_batches, mmd_loss) -> str:
        """Return the console line of an MMD batch: epoch, progress bar, MMD loss, learning rate and times."""
        message = f"{self.device.type.upper()}:{self.local_rank} | EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        message += f"MMD Loss: {mmd_loss:.4f} |"
        message = self.append_lr(message)
        message = self.append_times(message)
        return message

def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    """Return the squared MMD between samples X and Y under a sum of Gaussian kernels (code from MMD-GAN)."""
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def _mix_rbf_kernel(X, Y, sigma_list):
    """Return the kernel blocks K_XX, K_XY, K_YY of a sum of Gaussian kernels, and the number of kernels."""
    assert(X.size(0) == Y.size(0))
    m = X.size(0)
    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)
    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    """Return the squared MMD from the kernel blocks, biased (diagonals kept) or unbiased."""
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2
