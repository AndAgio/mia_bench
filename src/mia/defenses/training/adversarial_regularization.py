
from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from src.data.helpers import MultiDatasets
from src.utils.configs import DefenderConfigs, AdvRegDefenseConfigs, TrainConfigs, ModelConfigs
from src.trainer.train_manager import TrainManager
from src.mia.defenses.base import BaseDefender
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats


class AdvRegDefender(BaseDefender):
    # Implementation of "RelaxLoss: Defending Membership Inference Attacks without Losing Utility" (https://arxiv.org/pdf/2207.05801).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, AdvRegDefenseConfigs), f"AdvRegDefender can only be used with AdvRegDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'adv_reg_defender'
        self.adv_reg_configs = defender_configs.defense

    def train(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f'AdvReg Defender: training defender model with adversarial regularization and lambda {self.adv_reg_configs.adv_lambda}...')
        train_manager = AdvRegTrainManager(train_configs=train_configs,
                                            name=self.name,
                                            logger=self.logger,
                                            adv_reg_configs=self.adv_reg_configs)
        # Setting up attacker model
        in_dim = self.dataset_configs.num_classes  # input: logits/probs of classifier
        self.attacker_model = AttackNet(in_dim=in_dim,
                                        hidden=self.adv_reg_configs.shadow_attacker_model_layers)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        attacker_model=self.attacker_model,
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
        self.logger.print_it('AdvReg Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model


class AdvRegTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, adv_reg_configs: AdvRegDefenseConfigs = None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.adv_reg_configs = adv_reg_configs

    def initialize_train(self, 
                        dataset: Union[MultiDatasets, Dataset],
                        model: Union[ModelConfigs,torch.nn.Module],
                        attacker_model: torch.nn.Module,
                        configs: TrainConfigs,
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
        # self.validate_and_fix_model_for_dp(dp_config=configs.dp_config)
        self.setup_training()
        # Modifying model, optimizer and loaders for differential privacy if needed
        # self.check_and_set_dp(dp_config=configs.dp_config)

        self.logger.print_it('AdvReg Defender: Setting up adversarial regularization components...')
        self.attacker_model = attacker_model.to(self.device)
        # Setting up optimizers
        self.attacker_optimizer = torch.optim.Adam(self.attacker_model.parameters(),
                                                lr=0.0001)
        self.attacker_criterion = torch.nn.MSELoss()
        assert hasattr(self, 'train_loader') and hasattr(self, 'test_loader'), f"Data loaders not found when initializing adversarial regularization training!"
        # self.heldout_loader = self.test_loader
        held_sampler = RandomSampler(self.test_loader.dataset,
                                    replacement=True,
                                    num_samples=len(self.train_loader) * self.train_configs.batch_size)
        self.heldout_loader = DataLoader(self.test_loader.dataset,
                                        batch_size=self.train_configs.batch_size,
                                        sampler=held_sampler,
                                        drop_last=True)

        self.logger.print_it('Training initialization completed!')

    def train_epoch(self):
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        self.model.train()
        self.attacker_model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
            self.heldout_loader.sampler.set_epoch(self.epoch)
        for batch_idx, ((train_inputs, train_targets), (heldout_inputs, heldout_targets)) in enumerate(zip(self.train_loader, self.heldout_loader)):
            self.train_step(train_inputs, train_targets, heldout_inputs, heldout_targets, batch_idx=batch_idx, total_batches=len(self.train_loader))
        self.logger.set_logger_newline(console_only=True)
        
        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()

        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)

        return train_summary

    def train_step(self, train_inputs, train_targets, heldout_inputs, heldout_targets, batch_idx=0, total_batches=0):
        # Map to available device (profile this)
        train_inputs = train_inputs.to(self.device, non_blocking=True)
        train_targets = train_targets.to(self.device, non_blocking=True)
        heldout_inputs = heldout_inputs.to(self.device, non_blocking=True)
        heldout_targets = heldout_targets.to(self.device, non_blocking=True)

        self.epoch_stats_tracker.batch_start()

        # 1) Forward through classifier
        train_outputs = self.model(train_inputs)
        heldout_outputs = self.model(heldout_inputs)
        # 2) k attacker steps (maximize attack success on member vs non-member)
        for _ in range(self.adv_reg_configs.shadow_attacker_k):
            self.attacker_optimizer.zero_grad()
            atk_in  = torch.cat([train_outputs, heldout_outputs], dim=0)
            atk_lab = torch.cat([torch.ones(len(train_outputs),), torch.zeros(len(heldout_outputs),)], dim=0).to(self.device)
            atk_pred = self.attacker_model(atk_in)
            atk_loss = self.attacker_criterion(atk_pred, atk_lab)
            atk_loss.backward()
            self.attacker_optimizer.step()

        # Compute loss and predictions (profile compute: forward + backward + optimizer)
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(train_inputs, train_targets, heldout_inputs, heldout_targets, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                train_outputs = self.model(train_inputs)
                heldout_outputs = self.model(heldout_inputs)
                task_loss = self.criterion(train_outputs, train_targets)
                with torch.no_grad():  # adversary is treated as fixed opponent
                    atk_m_tr  = self.attacker_model(train_outputs)
                    atk_m_h   = self.attacker_model(heldout_outputs)
                    atk_gain  = self.attacker_criterion(atk_m_tr, torch.ones_like(atk_m_tr)) + self.attacker_criterion(atk_m_h, torch.zeros_like(atk_m_h))
                loss = task_loss - self.adv_reg_configs.adv_lambda * atk_gain
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, train_outputs
    
            self.optimizer.step(closure, train_inputs, train_targets, heldout_inputs, heldout_targets)
            self.optimizer.zero_grad()
            loss, train_outputs = self.optimizer.get_first_closure_outputs()
        else:
            # 3) 1 defender step (minimize task loss - λ * attack gain)
            self.optimizer.zero_grad()
            train_outputs = self.model(train_inputs)
            heldout_outputs = self.model(heldout_inputs)
            task_loss = self.criterion(train_outputs, train_targets)

            with torch.no_grad():  # adversary is treated as fixed opponent
                atk_m_tr  = self.attacker_model(train_outputs)
                atk_m_h   = self.attacker_model(heldout_outputs)
                atk_gain  = self.attacker_criterion(atk_m_tr, torch.ones_like(atk_m_tr)) + self.attacker_criterion(atk_m_h, torch.zeros_like(atk_m_h))

            loss = task_loss - self.adv_reg_configs.adv_lambda * atk_gain
            loss.backward()
            self.optimizer.step()

        self.epoch_stats_tracker.update(preds=train_outputs, targets=train_targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=train_targets.size(0))
        
        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)


class AttackNet(torch.nn.Module):
    """Simple MLP attacker: takes sorted logits (or probs) → membership logit."""
    def __init__(self, in_dim: int, hidden=(64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(inplace=True)]
            d = h
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, return_logits=False):
        out = self.net(x).squeeze(-1)   # (B,)
        return out if return_logits else torch.sigmoid(out)
