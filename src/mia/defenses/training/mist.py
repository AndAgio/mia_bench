from typing import Union
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.configs import DefenderConfigs, MistDefenseConfigs
from src.mia.defenses.base import BaseDefender
from src.trainer.train_manager import TrainManager
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.mia.defenses.training.mixup import mixup_data


class MistDefender(BaseDefender):
    # Implementation of training time optimization component of MIST defense from "MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training" (https://www.usenix.org/system/files/usenixsecurity24-li-jiacheng.pdf).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MistDefenseConfigs), f"MistDefender can only be used with MistDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mist_defender'
        self.mist_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
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
        self.logger.print_it("Mist Defender: defend_model does not modify the model at inference time.")
        self.defended_model = self.trained_model
        return self.defended_model


class MistTrainManager(TrainManager):
    # Custom TrainManager for MIST defense training procedure.
    def __init__(self, train_configs, name: str, logger: callable, mist_configs: MistDefenseConfigs):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.mist_configs = mist_configs

    def clone_model(self):
        # Clone the model to create the submodels used in MIST.
        self.sub_models = {i: copy.deepcopy(self.model) for i in range(self.mist_configs.num_submodels)}
        self.logger.print_it(f"MIST TrainManager: cloned {self.mist_configs.num_submodels} sub-models for MIST training.")

    def clone_optimizer(self):
        # Clone the optimizer for each submodel.
        self.sub_optimizers = {}
        for i in range(self.mist_configs.num_submodels):
            model = self.sub_models[i]
            optimizer = self.clone_optimizer_for_model(self.optimizer, model)
            self.sub_optimizers[i] = optimizer
        self.logger.print_it(f"MIST TrainManager: cloned optimizers for {self.mist_configs.num_submodels} sub-models.")

    @staticmethod
    def clone_optimizer_for_model(old_opt, new_model):
        # 1) make new optimizer with same param-group hyperparams
        old_sd = old_opt.state_dict()
        # replicate param_groups structure but with new model params in the same order
        new_param_iter = iter(new_model.parameters())
        new_param_groups = []
        for old_pg in old_sd['param_groups']:
            pg = {k: v for k, v in old_pg.items() if k != 'params'}
            pg['params'] = [next(new_param_iter) for _ in range(len(old_pg['params']))]
            new_param_groups.append(pg)
        # new_opt = type(old_opt)(new_param_groups)
        new_opt = copy.deepcopy(old_opt)
        new_opt.param_groups = new_param_groups

        # 2) remap parameter ids in state_dict
        new_sd = new_opt.state_dict()
        old_ids = [p for pg in old_sd['param_groups'] for p in pg['params']]
        new_ids = [p for pg in new_sd['param_groups'] for p in pg['params']]
        if len(old_ids) != len(new_ids):
            raise ValueError("Parameter counts differ; cannot remap optimizer state.")
        id_map = dict(zip(old_ids, new_ids))

        remapped_state = {}
        for old_id, state in old_sd['state'].items():
            new_id = id_map.get(old_id)
            if new_id is None:
                continue
            # optional: check shapes of tensor buffers before accepting
            remapped_state[new_id] = state

        # replace param group param ids
        remapped_pgs = []
        for pg in old_sd['param_groups']:
            pg_copy = {k: v for k, v in pg.items()}
            pg_copy['params'] = [id_map[p] for p in pg['params']]
            remapped_pgs.append(pg_copy)

        new_sd_mapped = {'state': remapped_state, 'param_groups': remapped_pgs}

        # 3) load and move tensors to device
        new_opt.load_state_dict(new_sd_mapped)
        device = next(new_model.parameters()).device
        for state in new_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        return new_opt

    def clone_criterion(self):
        # Clone the criterion for each submodel.
        self.sub_criteria = {}
        for i in range(self.mist_configs.num_submodels):
            criterion = copy.deepcopy(self.criterion)
            self.sub_criteria[i] = criterion
        self.logger.print_it(f"MIST TrainManager: cloned criteria for {self.mist_configs.num_submodels} sub-models.")

    @staticmethod
    def clone_scheduler_for_optimizer(old_sched, new_opt):
        # 1) create new scheduler instance (use same constructor args from your config)
        new_sched = copy.deepcopy(old_sched)
        new_sched.optimizer = new_opt
        # 2) copy and adapt state dict
        sd = old_sched.state_dict()
        # adapt lists that are per-param-group
        if 'base_lrs' in sd:
            if len(sd['base_lrs']) != len(new_opt.param_groups):
                # fallback: use current lr of new optimizer param groups
                sd['base_lrs'] = [pg.get('lr', 0.0) for pg in new_opt.param_groups]
        if 'last_lr' in sd:
            if len(sd['last_lr']) != len(new_opt.param_groups):
                sd['last_lr'] = [pg.get('lr', 0.0) for pg in new_opt.param_groups]
        # move any tensor state to device
        device = MistTrainManager.get_device_from_optimizer(new_opt)
        for k, v in sd.items():
            if isinstance(v, list):
                sd[k] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
            elif isinstance(v, torch.Tensor) and device is not None:
                sd[k] = v.to(device)
        # 3) load
        new_sched.load_state_dict(sd)
        return new_sched

    @staticmethod
    def get_device_from_optimizer(opt) -> torch.device:
        devices = set()
        for pg in opt.param_groups:
            for p in pg.get("params", []):
                # p is usually a torch.nn.Parameter (has .device)
                try:
                    devices.add(p.device)
                except Exception:
                    continue
        if not devices:
            return torch.device("cpu")  # fallback
        if len(devices) > 1:
            # optional: logger.warning("Optimizer parameters on multiple devices; using first.")
            return next(iter(devices))
        return devices.pop()
    
    def clone_lr_scheduler(self):
        # Clone the LR scheduler for each submodel.
        self.sub_lr_schedulers = {}
        for i in range(self.mist_configs.num_submodels):
            optimizer = self.sub_optimizers[i]
            lr_scheduler = self.clone_scheduler_for_optimizer(self.scheduler, optimizer)
            self.sub_lr_schedulers[i] = lr_scheduler
        self.logger.print_it(f"MIST TrainManager: cloned LR schedulers for {self.mist_configs.num_submodels} sub-models.")
    
    def clone_everything_for_submodels(self):
        # Clone model, optimizer, criterion, and LR scheduler for each submodel.
        self.clone_model()
        self.clone_optimizer()
        self.clone_criterion()
        self.clone_lr_scheduler()

    def split_data(self):
        # Split the training data into subsets for each submodel.
        if self.mist_configs.split_method == 'random':
            self.split_data_random()
        elif self.mist_configs.split_method == 'stratified':
            self.split_data_stratified()
        else:
            raise ValueError(f"MIST TrainManager: unknown split method {self.mist_configs.split_method}!")

    def split_data_random(self):
        # Split the training data into subsets for each submodel.
        try:
            all_indices = self.train_loader.dataset.get_indices(to_torch=True)
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
            subset_dataset = torch.utils.data.Subset(self.train_loader.dataset, subset_indices)
            self.submodel_datasets[i] = subset_dataset
        self.logger.print_it(f"MIST TrainManager: split training data into {self.mist_configs.num_submodels} subsets for sub-models.")

    def split_data_stratified(self):
        try:
            y = self.train_loader.dataset.get_all_targets()
        except AttributeError:
            y = self.train_loader.dataset.targets
        classes = np.unique(y)
        rng = np.random.default_rng(self.seed)
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
            subset_dataset = torch.utils.data.Subset(self.train_loader.dataset, subset_indices)
            self.submodel_datasets[i] = subset_dataset
        self.logger.print_it(f"MIST TrainManager: stratified split of training data into {self.mist_configs.num_submodels} subsets for sub-models.")

    def get_submodel_dataset(self, submodel_index: int) -> torch.utils.data.Dataset:
        # Get the dataset for a specific submodel.
        return self.submodel_datasets[submodel_index]

    def average_submodels(self):
        # Average the parameters of the submodels to update the main model.
        with torch.no_grad():
            sd_list = [m.state_dict() for _, m in self.sub_models.items()]
            keys = sd_list[0].keys()
            avg_params = {k: sum(sd[k] for sd in sd_list) / len(sd_list) for k in keys}
            self.model.load_state_dict(avg_params)
        self.logger.print_it("MIST TrainManager: averaged sub-models to update the main model.")

    @torch.no_grad()
    def avg_probs_other_models(self, exclude: int, inputs: torch.Tensor) -> torch.Tensor:
        tot, cnt = 0.0, 0
        for j, model_j in self.sub_models.items():
            if j == exclude:
                continue
            model_j.eval()
            logits = model_j(inputs)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            tot = tot + probs
            cnt += 1
        return tot / cnt
    
    def reset_submodel_optimizers(self):
        for submodel_index in range(self.mist_configs.num_submodels):
            optimizer = self.clone_optimizer_for_model(self.optimizer, self.sub_models[submodel_index])
            self.sub_optimizers[submodel_index] = optimizer
        self.logger.print_it(f"MIST TrainManager: reset optimizer state for all sub-models.")
    
    def reset_submodel_lr_schedulers(self):
        for submodel_index in range(self.mist_configs.num_submodels):
            self.reset_submodel_lr_scheduler(submodel_index=submodel_index)
        self.logger.print_it(f"MIST TrainManager: reset lr scheduler state for all sub-models.")
    
    def reset_submodel_lr_scheduler(self, submodel_index: int):
        # Reset the LR scheduler state for a specific submodel.
        lr_scheduler = copy.deepcopy(self.scheduler)
        lr_scheduler.load_state_dict(self.scheduler.state_dict())
        optimizer = self.sub_optimizers[submodel_index]
        lr_scheduler.optimizer = optimizer
        self.sub_lr_schedulers[submodel_index] = lr_scheduler

    def train_epoch(self):
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        # MIST training procedure
        self.clone_everything_for_submodels()
        self.split_data()
        self.train_submodels()
        self.optimize_difference_of_submodels()
        self.average_submodels()

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()
        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)
        return train_summary

    def train_submodels(self):
        # Train each submodel on its respective data subset.
        for submodel_index in range(self.mist_configs.num_submodels):
            self.logger.print_it(f"MIST TrainManager: training sub-model {submodel_index+1}/{self.mist_configs.num_submodels}...")
            model = self.sub_models[submodel_index]
            model.train()
            optimizer = self.sub_optimizers[submodel_index]
            criterion = self.sub_criteria[submodel_index]
            lr_scheduler = self.sub_lr_schedulers[submodel_index]
            # Setup data loader for this submodel
            submodel_dataset = self.get_submodel_dataset(submodel_index=submodel_index)
            submodel_loader = DataLoader(submodel_dataset, batch_size=self.train_configs.batch_size, shuffle=True, num_workers=1)
            for epoch in range(self.mist_configs.submodel_epochs):
                for batch_idx, (inputs, targets, _, _) in enumerate(submodel_loader):
                    model, optimizer, criterion = self.train_step_submodel(model, submodel_index, optimizer, criterion, inputs, targets, epoch, batch_idx=batch_idx, total_batches=len(submodel_loader))
                lr_scheduler.step()
            self.logger.set_logger_newline(console_only=True)
            self.sub_models[submodel_index] = model
            self.logger.print_it(f"MIST TrainManager: finished training sub-model {submodel_index+1}/{self.mist_configs.num_submodels}!")
        self.logger.print_it("MIST TrainManager: finished training all sub-models.")

    def train_step_submodel(self, model, model_index, optimizer, criterion,inputs, targets, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
        if self.mist_configs.mix_up:
            return self.train_step_submodel_with_mixup(model, model_index, optimizer, criterion, inputs, targets, epoch, batch_idx, total_batches)
        else:
            return self.train_step_submodel_standard(model, model_index, optimizer, criterion, inputs, targets, epoch, batch_idx, total_batches)
        
    def train_step_submodel_with_mixup(self, model, model_index, optimizer, criterion,inputs, targets, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
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
        # Map to available device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
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
        self.reset_submodel_optimizers()
        self.reset_submodel_lr_schedulers()
        # Optimize each submodel to maximize difference from average of other submodels.
        for submodel_index in range(self.mist_configs.num_submodels):
            self.logger.print_it(f"MIST TrainManager: optimizing difference for sub-model {submodel_index+1}/{self.mist_configs.num_submodels}...")
            model = self.sub_models[submodel_index]
            model.train()
            optimizer = self.sub_optimizers[submodel_index]
            criterion = torch.nn.L1Loss(reduction="mean")
            lr_scheduler = self.sub_lr_schedulers[submodel_index]
            # Setup data loader for this submodel
            submodel_dataset = self.get_submodel_dataset(submodel_index=submodel_index)
            submodel_loader = DataLoader(submodel_dataset, batch_size=self.train_configs.batch_size, shuffle=True, num_workers=1)
            for epoch in range(self.mist_configs.submodel_epochs):
                for batch_idx, (inputs, targets, _, _) in enumerate(submodel_loader):
                    model, optimizer, criterion = self.optimize_difference_step(model, submodel_index, optimizer, criterion, inputs, epoch, batch_idx=batch_idx, total_batches=len(submodel_loader))
                lr_scheduler.step()
            self.logger.set_logger_newline(console_only=True)
            self.sub_models[submodel_index] = model
            self.logger.print_it(f"MIST TrainManager: finished optimizing difference for sub-model {submodel_index+1}/{self.mist_configs.num_submodels}!")
    
    def optimize_difference_step(self, model, model_index, optimizer, criterion, inputs, epoch, batch_idx, total_batches) -> Union[torch.nn.Module, torch.optim.Optimizer, callable]:
        # Map to available device
        inputs = inputs.to(self.device)
        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(inputs, criterion, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(model)
                else:
                    disable_running_stats(model)
                outputs = model(inputs)
                targets = self.avg_probs_other_models(exclude=model_index, inputs=inputs)
                loss = self.mist_configs.lmbd * criterion(outputs, targets)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            optimizer.step(closure, inputs, criterion)
            optimizer.zero_grad()
            loss, outputs = optimizer.get_first_closure_outputs()
        else:
            # Forward propagation, compute loss, get predictions (no GradScaler/AMP)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = self.avg_probs_other_models(exclude=model_index, inputs=inputs)
            loss = self.mist_configs.lmbd * criterion(outputs, targets)
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
