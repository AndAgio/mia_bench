import time
from typing import Union
import torch
from torch.utils.data import DataLoader
from src.data.helpers import IndexedDataset
from src.trainer.train_manager import TrainManager
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.utils.configs import DefenderConfigs, WeightedSmoothingDefenseConfigs
from src.mia.defenses.base import BaseDefender


class WeightedSmoothingDefender(BaseDefender):
    # Implementation of Weighted Smoothing defense from the paper: "Mitigating Membership Inference Attacks via Weighted Smoothing" (https://dl.acm.org/doi/pdf/10.1145/3627106.3627189).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, WeightedSmoothingDefenseConfigs), f"WeightedSmoothingDefender can only be used with WeightedSmoothingDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'weighted_smoothing_defender'
        self.weighted_smoothing_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it(f'Weighted Smoothing Defender: training defender model...')
        train_manager = WeightedSmoothingTrainManager(train_configs=train_configs,
                                                name=self.name,
                                                logger=self.logger,
                                                weighted_smoothing_configs=self.weighted_smoothing_configs)
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
        self.logger.info("Weighted Smoothing Defender: defend_model does not modify the model at inference time.")
        self.defended_model = self.trained_model
        return self.defended_model


class WeightedSmoothingTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, weighted_smoothing_configs: WeightedSmoothingDefenseConfigs = None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.weighted_smoothing_configs = weighted_smoothing_configs

    def train_epoch(self):
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        self.model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        if self.epoch <= self.weighted_smoothing_configs.warmup_epochs:
            for batch_idx, (inputs, targets, _, _) in enumerate(self.train_loader):
                # Standard training for the first warmup_epochs
                self.train_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.train_loader))
        else:
            # Weighted Smoothing training after warmup_epochs
            # Compute weights based on class frequencies
            self.indexed_train_loader = DataLoader(self.train_loader.dataset, 
                                                batch_size=self.train_loader.batch_size,
                                                shuffle=True)
            self.compute_weights()
            self.normalize_weights()
            for batch_idx, (inputs, targets, _, indices) in enumerate(self.indexed_train_loader):
                # Noise injection based on weights training
                self.train_step_with_weighted_smoothing(inputs, targets, indices, batch_idx=batch_idx, total_batches=len(self.train_loader))

        self.logger.set_logger_newline(console_only=True)
        
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()

        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)

        return train_summary

    def compute_weights(self):
        start = time.time()
        assert isinstance(self.train_loader.dataset, IndexedDataset), "Train loader dataset must be an IndexedDataset!"
        data_loader = DataLoader(self.train_loader.dataset, batch_size=self.train_loader.batch_size, shuffle=False)
        self.weights = torch.zeros(len(self.train_loader.dataset))
        self.logger.print_it(f"Computing weights for all training samples based on Mentr. This may take a while...")
        for batch_index, (inputs, targets, _, sample_indices) in enumerate(data_loader):
            self.logger.print_it_same_line(f"Computing weights for batch {batch_index+1}/{len(data_loader)}...", console_only=True)
            with torch.no_grad():
                outputs = self.model(inputs.to(self.device))
                probs = torch.softmax(outputs, dim=1)
            m_entr = self.mentr(probs, targets, from_logits=True)  # shape (batch_size,)
            self.weights[sample_indices] = m_entr.cpu()
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Computed weights for all training samples in {time.time() - start:.2f} seconds.")
    
    def normalize_weights(self):
        start = time.time()
        assert isinstance(self.train_loader.dataset, IndexedDataset), "Train loader dataset must be an IndexedDataset!"
        all_targets = self.train_loader.dataset.get_all_targets(to_torch=True)
        # all_targets = torch.tensor(self.train_loader.dataset.ds.targets)
        classes = torch.unique(all_targets, return_counts=False)
        self.logger.print_it(f"Normalizing weights per class. This may take a while...")
        for cls in classes:
            cls_indices = (all_targets == cls).nonzero(as_tuple=True)[0]
            cls_weights = self.weights[cls_indices]
            if torch.sum(cls_weights) > 0:
                avg_cls_weight = torch.mean(cls_weights)
                std_cls_weight = torch.std(cls_weights)
                normalized_cls_weights = 1 - (cls_weights - avg_cls_weight) / (std_cls_weight + 1e-12)  # standardization
                self.weights[cls_indices] = normalized_cls_weights
        self.logger.print_it(f"Normalized weights for all training samples in {time.time() - start:.2f} seconds.")

    @staticmethod
    def mentr(preds: torch.Tensor,                # logits or probabilities, shape (N, C)
            labels: torch.LongTensor,           # shape (N,)
            from_logits: bool = True,
            eps: float = 1e-12,
        ) -> torch.Tensor:
        """
        Computes Mentr(x, l) per sample:
        M = -(1 - f_l) * log(f_l) - sum_{i != l} f_i * log(1 - f_i)

        preds: if from_logits=True, supply logits (will softmax internally).
        Returns: tensor of shape (N,) if reduction='none' else scalar.
        """
        if from_logits:
            probs = torch.softmax(preds, dim=1)
        else:
            probs = preds

        # numerical stability
        probs = probs.clamp(min=eps, max=1.0 - eps)

        N = probs.size(0)
        idx = torch.arange(N, device=probs.device)
        f_l = probs[idx, labels]                     # shape (N,)
        log_f_l = torch.log(f_l)                     # shape (N,)
        log_one_minus = torch.log1p(-probs)          # log(1 - p) in a stable way

        # sum over all classes of f_i * log(1 - f_i)
        s_all = torch.sum(probs * log_one_minus, dim=1)  # shape (N,)
        # exclude the label class from the sum:
        s_excl = s_all - (f_l * log_one_minus[idx, labels])

        term1 = -(1.0 - f_l) * log_f_l                # shape (N,)
        term2 = - s_excl                              # shape (N,)
        m_entr = term1 + term2                        # shape (N,)

        return m_entr

    def train_step_with_weighted_smoothing(self, inputs, targets, inputs_indices, batch_idx=0, total_batches=0):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        batch_size = inputs.shape[0]
        batch_weights = self.weights[inputs_indices].to(self.device)
        gaussian_noise = torch.normal(mean=0.0, std=self.weighted_smoothing_configs.sigma_noise, size=(batch_size, 1), device=self.device)

        self.epoch_stats_tracker.batch_start()
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(inputs, targets, batch_weights, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                outputs = self.model(inputs) + batch_weights.unsqueeze(1) * gaussian_noise
                loss = self.criterion(outputs, targets)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            self.optimizer.step(closure, inputs, targets, batch_weights)
            self.optimizer.zero_grad()
            loss, outputs = self.optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            self.optimizer.zero_grad()
            outputs = self.model(inputs) + batch_weights.unsqueeze(1) * gaussian_noise
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        
        self.epoch_stats_tracker.update(preds=outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)
