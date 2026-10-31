
from contextlib import contextmanager
from typing import Any, Union
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from src.data.helpers import MultiDatasets, DatasetSplitter
from src.utils.configs import DefenderConfigs, AdvRegDefenseConfigs, TrainConfigs, ModelConfigs
from src.utils.variables import ADV_REG_REFERENCE_FRACTION_OF_VAL_SPLIT
from src.trainer.train_manager import TrainManager
from src.mia.defenses.base import BaseDefender
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats


class AdvRegDefender(BaseDefender):
    """Adversarial regularization: trains the model against an inference attack that it learns alongside.

    Paper: "Machine Learning with Membership Privacy using Adversarial Regularization",
    https://arxiv.org/abs/1807.05852. Documentation: docs/defenses/adv_reg.md.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select AdvReg and keep its settings."""
        assert isinstance(defender_configs.defense, AdvRegDefenseConfigs), f"AdvRegDefender can only be used with AdvRegDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'adv_reg_defender'
        self.adv_reg_configs = defender_configs.defense

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        """Train with adversarial regularization and return the best model, with the stats if return_stats."""
        self.logger.print_it(f'AdvReg Defender: training defender model with adversarial regularization, lambda {self.adv_reg_configs.adv_lambda} and {self.adv_reg_configs.penalty} penalty...')
        train_manager = AdvRegTrainManager(train_configs=train_configs,
                                            name=self.name,
                                            logger=self.logger,
                                            adv_reg_configs=self.adv_reg_configs)
        # Setting up attacker model, which reads the softmax probabilities and the one-hot label
        num_classes = self.dataset_configs.num_classes
        if self.adv_reg_configs.shadow_attacker_model_layers is None:
            self.attacker_model = PaperAttackNet(num_classes=num_classes)
        else:
            self.attacker_model = AttackNet(in_dim=2 * num_classes,
                                            hidden=self.adv_reg_configs.shadow_attacker_model_layers)
        # The reference set D' must not overlap with the audit non-members (drawn from test), nor with the
        # samples used to select the best checkpoint, since the classifier is optimized against D'.
        # Hence, val is split into two disjoint stratified parts: one for checkpoint selection and one for D'.
        selection_dataset, reference_dataset = DatasetSplitter.split_stratified(self.dataset.get('val'),
                                                                                proportions=[1 - ADV_REG_REFERENCE_FRACTION_OF_VAL_SPLIT,
                                                                                            ADV_REG_REFERENCE_FRACTION_OF_VAL_SPLIT],
                                                                                seed=self.dataset_configs.seed)
        self.logger.print_it(f'AdvReg Defender: split val into {len(selection_dataset)} samples for checkpoint selection and {len(reference_dataset)} samples for the reference set.')
        train_dataset = MultiDatasets()
        train_dataset.add(self.dataset.get('train'), 'train')
        train_dataset.add(selection_dataset, 'val')
        train_dataset.add(self.dataset.get('test'), 'test')
        train_dataset.add_info(self.dataset.get_info())
        train_manager.initialize_train(dataset=train_dataset,
                                        model=self.untrained_model,
                                        attacker_model=self.attacker_model,
                                        reference_dataset=reference_dataset,
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
        """Return the trained model unchanged: AdvReg only changes training."""
        self.logger.print_it('AdvReg Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model


@contextmanager
def frozen_bn_running_stats(model: torch.nn.Module):
    """Run forward passes in train mode without updating the BatchNorm running statistics.

    Unlike disable_running_stats/enable_running_stats, the exact momentum and batch counters are restored,
    so this can be nested within the BatchNorm handling of SAM-like optimizers.
    """
    bns = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    states = [(bn.momentum, bn.num_batches_tracked.clone() if bn.num_batches_tracked is not None else None) for bn in bns]
    try:
        for bn in bns:
            bn.momentum = 0.0
        yield
    finally:
        for bn, (momentum, num_batches_tracked) in zip(bns, states):
            bn.momentum = momentum
            if num_batches_tracked is not None:
                bn.num_batches_tracked.copy_(num_batches_tracked)


class AdvRegTrainManager(TrainManager):
    """TrainManager that alternates attack model and classifier steps, as in Algorithm 1 of AdvReg."""
    def __init__(self, train_configs, name: str, logger=None, adv_reg_configs: AdvRegDefenseConfigs = None):
        """Set up the training as TrainManager does, and keep AdvReg's settings."""
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.adv_reg_configs = adv_reg_configs

    def initialize_train(self,
                        dataset: Union[MultiDatasets, Dataset],
                        model: Union[ModelConfigs,torch.nn.Module],
                        attacker_model: torch.nn.Module,
                        reference_dataset: Dataset,
                        configs: TrainConfigs,
                        ):
        """Set up the classifier as TrainManager does, then the attack model, its optimizer and its batch loaders."""
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
        if self.distributed:
            # Averaging the attacker gradients across ranks keeps a single attacker, as for the classifier.
            self.attacker_model = DDP(self.attacker_model, device_ids=[self.local_rank])
        # Setting up optimizers
        self.attacker_optimizer = torch.optim.Adam(self.attacker_model.parameters(),
                                                lr=self.adv_reg_configs.attacker_lr)
        self.attacker_criterion = torch.nn.BCEWithLogitsLoss()
        assert hasattr(self, 'train_loader') and hasattr(self, 'val_loader'), f"Data loaders not found when initializing adversarial regularization training!"
        # As in Algorithm 1 of the paper, each of the k attacker steps draws fresh mini-batches from the training set D and
        # the reference set D'. The classifier step draws one more reference mini-batch, used only for BatchNorm statistics.
        # Samplers share a generator that is reseeded every epoch (see train_epoch).
        self.attacker_batches_generator = torch.Generator()
        n_steps = len(self.train_loader)
        k = self.adv_reg_configs.shadow_attacker_k
        self.attacker_member_loader = self._build_sampling_loader(dataset=dataset.get('train'),
                                                                n_batches=k * n_steps)
        self.reference_loader = self._build_sampling_loader(dataset=reference_dataset,
                                                            n_batches=(k + 1) * n_steps)

        self.logger.print_it('Training initialization completed!')

    def _build_sampling_loader(self, dataset: Dataset, n_batches: int) -> DataLoader:
        """Return a loader of n_batches full batches drawn with replacement, using the shared attacker generator."""
        sampler = RandomSampler(dataset,
                                replacement=True,
                                num_samples=n_batches * self.train_configs.batch_size,
                                generator=self.attacker_batches_generator)
        return DataLoader(dataset,
                        batch_size=self.train_configs.batch_size,
                        sampler=sampler,
                        drop_last=True)

    def _attacker_module(self) -> torch.nn.Module:
        """Return the attack model, unwrapped from DistributedDataParallel if needed."""
        return getattr(self.attacker_model, 'module', self.attacker_model)

    def get_checkpoint_extra(self) -> dict[str, Any]:
        """Return the attack model and its optimizer state, to save in each checkpoint."""
        return {'attacker_state': self._attacker_module().state_dict(),
                'attacker_optimizer_state': self.attacker_optimizer.state_dict()}

    def load_checkpoint_extra(self, extra: dict[str, Any]):
        """Restore the attack model and its optimizer from a checkpoint, or refuse a checkpoint without them."""
        # Resuming without the attacker would train the classifier against a fresh attacker, and a checkpoint without it
        # may come from an incompatible implementation, so its models must not be reused.
        if 'attacker_state' not in extra:
            raise RuntimeError(f"AdvReg checkpoint in {self.resume_folder} has no attacker state and cannot be resumed. "
                               f"Delete this folder and the matching adv_reg_defender folder under ckpts, then retrain.")
        self._attacker_module().load_state_dict(extra['attacker_state'])
        self.attacker_optimizer.load_state_dict(extra['attacker_optimizer_state'])
        self.logger.print_it('AdvReg Defender: attacker state restored from the checkpoint.')

    def train_epoch(self):
        """Run one epoch: for each training batch, the attack model steps and then one classifier step."""
        # reset epoch stats and per-epoch profiler
        self.reset_epoch_stats(phase='train')
        self.model.train()
        self.attacker_model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)
        # Seeding per epoch and rank keeps the attacker batches reproducible on resume and distinct across ranks.
        self.attacker_batches_generator.manual_seed(self.seed + self.epoch * self.world_size + self.global_rank)
        member_iter = iter(self.attacker_member_loader)
        reference_iter = iter(self.reference_loader)
        for batch_idx, (inputs, targets, _, _) in enumerate(self.train_loader):
            self.train_step(inputs, targets, member_iter, reference_iter, batch_idx=batch_idx, total_batches=len(self.train_loader))
        self.logger.set_logger_newline(console_only=True)

        self.epoch_stats_tracker.ddp_reduce_current_stage()
        train_summary = self.epoch_stats_tracker.stage_end()

        message = self.build_message_for_stage_end(stage_summary=train_summary)
        self.logger.print_it(f"{message}", file_only=True)

        return train_summary

    @staticmethod
    def attack_features(outputs, targets):
        """Return the attack model's input: the softmax probabilities followed by the one-hot label."""
        # The attack model is h(x, y, f(x)): it sees the prediction vector and the label. Softmax probabilities are used
        # instead of logits, since shifting all logits of a sample by a constant would change what the attacker sees
        # without changing the predictions, letting the classifier fool the attacker without reducing the leakage.
        return torch.cat([F.softmax(outputs, dim=1),
                          F.one_hot(targets.long(), num_classes=outputs.size(1)).to(outputs.dtype)], dim=1)

    def attacker_steps(self, member_iter, reference_iter):
        """Train the attack model for k steps, each on a fresh batch of members and one of reference samples."""
        for _ in range(self.adv_reg_configs.shadow_attacker_k):
            member_inputs, member_targets, _, _ = next(member_iter)
            reference_inputs, reference_targets, _, _ = next(reference_iter)
            inputs = torch.cat([member_inputs, reference_inputs], dim=0).to(self.device)
            targets = torch.cat([member_targets, reference_targets], dim=0).to(self.device)
            # Members and references go through a single forward pass to share the BatchNorm batch statistics: with
            # separate passes, the statistics alone would tell the classifier which batch holds members.
            # These passes are not classifier training steps, so they leave the BatchNorm running statistics untouched.
            with torch.no_grad(), frozen_bn_running_stats(self.model):
                outputs = self.model(inputs)
            atk_in = self.attack_features(outputs, targets)
            atk_lab = torch.cat([torch.ones(len(member_inputs)),
                                 torch.zeros(len(reference_inputs))]).to(self.device)
            self.attacker_optimizer.zero_grad()
            atk_pred = self.attacker_model(atk_in, return_logits=True)
            atk_loss = self.attacker_criterion(atk_pred, atk_lab)
            atk_loss.backward()
            self.attacker_optimizer.step()

    def adversarial_gain(self, outputs, targets):
        """Return the attack log-loss -log h(x, y, f(x)) of each member, with gradients for the classifier only."""
        # The attack model's parameters are frozen instead of using torch.no_grad, which would also cut the gradient
        # back to the classifier. The unwrapped model keeps DDP from expecting an attack model backward pass.
        attacker = self._attacker_module()
        requires_grad = [parameter.requires_grad
                         for parameter in attacker.parameters()]
        try:
            for parameter in attacker.parameters():
                parameter.requires_grad_(False)
            atk_m_tr = attacker(self.attack_features(outputs, targets), return_logits=True)
            return F.binary_cross_entropy_with_logits(atk_m_tr, torch.ones_like(atk_m_tr), reduction='none')
        finally:
            for parameter, enabled in zip(attacker.parameters(), requires_grad):
                parameter.requires_grad_(enabled)

    def classifier_loss(self, inputs, targets, reference_inputs):
        """Return the per-sample loss l(f(x), y) + λ·h of members, or + λ·log h with the paper's penalty."""
        # Reference samples join the forward pass only to share the BatchNorm batch statistics with the members, so that
        # outputs are normalized as the ones the attacker is trained on. They do not enter the loss.
        outputs = self.model(torch.cat([inputs, reference_inputs], dim=0))[:len(inputs)]
        task_loss = self.criterion(outputs, targets)
        log_h = -self.adversarial_gain(outputs, targets)
        # λ·log h has no lower bound: the classifier can keep lowering it by giving up on the task.
        penalty = log_h if self.adv_reg_configs.penalty == 'log' else torch.exp(log_h)
        loss = task_loss + self.adv_reg_configs.adv_lambda * penalty
        return loss, outputs

    def train_step(self, inputs, targets, member_iter, reference_iter, batch_idx=0, total_batches=0):
        """Run the attack model steps, then one classifier step on the batch, with any optimizer, and record stats."""
        self.epoch_stats_tracker.batch_start()

        # 1) k attacker steps (maximize attack success on member vs non-member)
        self.attacker_steps(member_iter, reference_iter)

        # 2) 1 defender step (minimize task loss + λ·penalty(h) on members)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        reference_inputs = next(reference_iter)[0].to(self.device)
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # SAM-like optimizers use a closure that handles two forward/backward passes.
            def closure(inputs, targets, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                loss, outputs = self.classifier_loss(inputs, targets, reference_inputs)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs

            self.optimizer.step(closure, inputs, targets)
            self.optimizer.zero_grad()
            loss, train_outputs = self.optimizer.get_first_closure_outputs()
        else:
            self.optimizer.zero_grad()
            loss, train_outputs = self.classifier_loss(inputs, targets, reference_inputs)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

        self.epoch_stats_tracker.update(preds=train_outputs, targets=targets, extras=self.extra_configs)
        self.epoch_stats_tracker.batch_end(batch_size=targets.size(0))

        # Print message on console (the print itself is profiled inside print_message)
        message = self.build_message_for_batch_end(index_batch=batch_idx+1,
                                                total_batches=total_batches)
        self.logger.print_it_same_line(message, console_only=True)


class PaperAttackNet(torch.nn.Module):
    """Attack model of the paper (Fig. 3): a branch for the prediction vector, one for the label, then a common one."""
    def __init__(self, num_classes: int):
        """Build the three branches, with weights drawn from N(0, 0.01) and zero biases (Sec. 5.3)."""
        super().__init__()
        self.num_classes = num_classes
        self.prediction = torch.nn.Sequential(torch.nn.Linear(num_classes, 1024), torch.nn.ReLU(),
                                              torch.nn.Linear(1024, 512), torch.nn.ReLU(),
                                              torch.nn.Linear(512, 64), torch.nn.ReLU())
        self.label = torch.nn.Sequential(torch.nn.Linear(num_classes, 512), torch.nn.ReLU(),
                                         torch.nn.Linear(512, 64), torch.nn.ReLU())
        self.combine = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(),
                                           torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                           torch.nn.Linear(64, 1))
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.01)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, return_logits=False):
        """Return the membership logit, or probability, of features made of the prediction vector and one-hot label."""
        prediction, label = x[:, :self.num_classes], x[:, self.num_classes:]
        out = self.combine(torch.cat([self.prediction(prediction), self.label(label)], dim=1)).squeeze(-1)
        return out if return_logits else torch.sigmoid(out)


class AttackNet(torch.nn.Module):
    """Simple MLP attacker: takes softmax probabilities and one-hot label → membership logit."""
    def __init__(self, in_dim: int, hidden=(64, 32)):
        """Build the MLP with the given hidden layer sizes and ReLU activations."""
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(inplace=True)]
            d = h
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, return_logits=False):
        """Return the membership logit, or probability, of each row of features."""
        out = self.net(x).squeeze(-1)   # (B,)
        return out if return_logits else torch.sigmoid(out)
