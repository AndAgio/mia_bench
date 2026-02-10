import time
from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, MmdDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender
from src.mia.defenses.training.mixup import mixup_data
from src.trainer.train_manager import TrainManager


class MmdDefender(BaseDefender):
    # Implementation of training time optimization component of MMD defense from "Membership Inference Attacks and Defenses in Classification Models" (https://dl.acm.org/doi/pdf/10.1145/3422337.3447836).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MmdDefenseConfigs), f"MmdDefender can only be used with MmdDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mmd_defender'
        self.mmd_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f'MMD Defender: training defender model with MMD loss and lambda {self.mmd_configs.lmbd}...')
        train_manager = MmdTrainManager(train_configs=train_configs,
                                        name=self.name,
                                        logger=self.logger,
                                        mmd_configs=self.mmd_configs)
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
        self.logger.print_it('MMD Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model
    

class MmdTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, mmd_configs: MmdDefenseConfigs = None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.mmd_configs = mmd_configs

    def train_executions(self):
        assert self.run_val, "MMD training requires a validation set to compute the MMD loss. Please set run_val to True and provide a validation dataset!"
        self.train_epoch()
        train_acc = self.get_train_accuracy()
        val_acc = self.get_val_accuracy()
        if abs(train_acc - val_acc) > 0.03:
            self.logger.print_it(f"Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}. The gap between training and validation accuracy is {abs(train_acc - val_acc):.4f}, which is large enough...")
            if self.mmd_configs.lmbd > 1e-5 and self.epoch < self.train_configs.scheduler_config.epochs:
                self.logger.print_it(f"Other conditions for training with MMD loss are met (lambda {self.mmd_configs.lmbd} > 1e-5 and epoch {self.epoch} < max epochs {self.train_configs.scheduler_config.epochs}), we will run another epoch with MMD loss.")
                self.train_with_mmd_distance()
            else:
                self.logger.print_it(f"Other conditions for training with MMD loss are not met (lambda {self.mmd_configs.lmbd} <= 1e-5 or epoch {self.epoch} >= max epochs {self.train_configs.scheduler_config.epochs}), we will NOT run another epoch with MMD loss.")
        else:
            self.logger.print_it(f"Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}. The gap between training and validation accuracy is {abs(train_acc - val_acc):.4f}, which is small enough, we will NOT run another epoch with MMD loss.")
        self.val_epoch()
        if self.run_test:
            self.test_epoch()
        self.scheduler.step()
    
    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        if self.mmd_configs.use_mixup:
            mixed_inputs, targets_a, targets_b, lmbd = mixup_data(inputs, targets, alpha=self.mmd_configs.mixup_alpha)
            mixed_inputs = mixed_inputs.to(self.device)
            targets_a = targets_a.to(self.device)
            targets_b = targets_b.to(self.device)
        else:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        self.epoch_stats_tracker.batch_start()
        if self.mmd_configs.use_mixup:
            self.train_step_mixup(mixed_inputs, targets_a, targets_b, lmbd)
        else:
            self.train_step_standard(inputs, targets)
        
        with torch.no_grad():
            outputs = self.model(inputs)

        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)

    def train_step_mixup(self, inputs, targets_a, targets_b, lmbd):
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        mixed_inputs = inputs
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
            loss = loss.mean() if self.to_be_averaged_loss(loss) else loss
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def get_train_accuracy(self):
        self.model.eval()
        cumulative_correct = 0
        cumulative_total = 0
        for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.train_loader):
            self.logger.print_it_same_line(f"Computing training accuracy for batch {batch_idx+1}/{len(self.train_loader)}...", console_only=True)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            cumulative_correct += (outputs.argmax(dim=-1) == targets).sum().item()
            cumulative_total += targets.size(0)
        self.logger.set_logger_newline(console_only=True)
        accuracy = cumulative_correct / cumulative_total if cumulative_total > 0 else 0.0
        return accuracy

    @torch.no_grad()
    def get_val_accuracy(self):
        self.model.eval()
        cumulative_correct = 0
        cumulative_total = 0
        for batch_idx, (inputs, targets, original_indices, resampled_indices) in enumerate(self.val_loader):
            self.logger.print_it_same_line(f"Computing validation accuracy for batch {batch_idx+1}/{len(self.val_loader)}...", console_only=True)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            cumulative_correct += (outputs.argmax(dim=-1) == targets).sum().item()
            cumulative_total += targets.size(0)
        self.logger.set_logger_newline(console_only=True)
        accuracy = cumulative_correct / cumulative_total if cumulative_total > 0 else 0.0
        return accuracy
    
    @staticmethod
    def get_subset_by_label(dataset, target_label):
        try:
            targets = dataset.get_all_targets(to_torch=True)
            indices = (targets == target_label).nonzero(as_tuple=True)[0].tolist()
        except AttributeError:
            try:
                targets = torch.tensor(dataset.targets)
                indices = (targets == target_label).nonzero(as_tuple=True)[0].tolist()
            except AttributeError:
                indices = [i for i in range(len(dataset)) if dataset[i][1] == target_label]
        return Subset(dataset, indices)

    def train_with_mmd_distance(self):
        self.reset_epoch_stats(phase='mmd')
        training_data = self.dataset.get('train')
        train_loader_in_order = DataLoader(training_data, batch_size=self.train_loader.batch_size, shuffle=False)
        validation_data = self.dataset.get('val')
        
        for loss_index, (train_images, train_labels, original_indices, resampled_indices) in enumerate(train_loader_in_order):
            batch_num = train_labels.size()[0]
            self.optimizer.zero_grad()

            unique_labels = torch.unique(train_labels)
            subsets_val = []
            for label in unique_labels:
                all_val_with_matching_class = MmdTrainManager.get_subset_by_label(validation_data, label.item())
                freq = torch.count_nonzero(train_labels == label).item()
                subset_val = Subset(all_val_with_matching_class, torch.randperm(len(all_val_with_matching_class))[:freq])
                subsets_val.append(subset_val)
            sampled_val = ConcatDataset(subsets_val)
            valid_images, valid_labels, _, _ = next(iter(DataLoader(sampled_val, batch_size=self.train_loader.batch_size, shuffle=False)))

            train_images = train_images.to(self.device)
            train_labels = train_labels.to(self.device)
            outputs = self.model(train_images)
            all_train_outputs = torch.nn.functional.softmax(outputs,dim=1)
            #all_train_outputs = all_train_outputs.view(-1,num_classes)
            train_labels = train_labels.view(batch_num,1)

            valid_images = valid_images.to(self.device)
            valid_labels = valid_labels.to(self.device)
            outputs = self.model(valid_images)
            all_valid_outputs = torch.nn.functional.softmax(outputs,dim=1)
            all_valid_outputs = (all_valid_outputs).detach_()
            valid_labels = valid_labels.view(batch_num,1)

            mmd_loss = mix_rbf_mmd2(all_train_outputs,all_valid_outputs,sigma_list=[1])*self.mmd_configs.lmbd            
            mmd_loss.backward()
            self.optimizer.step()
    
            message = self.build_message_for_batch_end_mmd(loss_index+1, len(train_loader_in_order), mmd_loss.item())
            self.logger.print_it_same_line(message, console_only=True)
        self.logger.set_logger_newline(console_only=True)

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        mmd_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=mmd_summary)
        self.logger.print_it(f"{message}", file_only=True)

    def build_message_for_batch_end_mmd(self, index_batch, total_batches, mmd_loss) -> str:
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
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def _mix_rbf_kernel(X, Y, sigma_list):
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