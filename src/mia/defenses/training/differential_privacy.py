
from typing import Any, Optional, Union
import torch
from torch.utils.data import Dataset
import numpy as np
from src.data.helpers import MultiDatasets
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.optimizers import DPOptimizer, DPOptimizerFastGradientClipping
from opacus.optimizers.ddp_perlayeroptimizer import DistributedPerLayerOptimizer
from src.utils.configs import DefenderConfigs, DPDefenseConfigs, TrainConfigs, ModelConfigs, OptimizerConfigs
from src.utils.variables import DP_SAT_PAPER_RHO
from src.trainer.train_manager import TrainManager, sam_setting
from src.mia.defenses.base import BaseDefender
from src.optimizers import SAM, ESAM, WSAM, LookSAM, FriendlySAM


class DifferentialPrivacyDefender(BaseDefender):
    """Trains with DP-SGD, using Opacus, or with DP-SAT when the optimizer is SAM.

    Paper: "Deep Learning with Differential Privacy",
    https://arxiv.org/abs/1607.00133. DP-SAT: "Differentially Private Sharpness-Aware Training",
    https://arxiv.org/abs/2306.05651. Documentation: docs/defenses/dp.md.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select DP-SGD and keep its settings."""
        assert isinstance(defender_configs.defense, DPDefenseConfigs), f"DifferentialPrivacyDefender can only be used with DPDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'dp_defender'
        self.dp_configs = defender_configs.defense
        self.privacy_budget = None

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        """Train with DP-SGD and return the best model, with the stats if return_stats."""
        train_manager = DifferentialPrivacyTrainManager(train_configs=train_configs,
                                                        name=self.name,
                                                        logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        configs=train_configs,
                                        dp_configs=self.dp_configs)
        if return_stats:
            self.trained_model, train_stats = train_manager.train(return_best_model=True,
                                                        return_last_model=False,
                                                        return_stats=return_stats)
        else:
            self.trained_model = train_manager.train(return_best_model=True,
                                            return_last_model=False,
                                            return_stats=return_stats)
        self.privacy_budget = {'epsilon': train_manager.epsilon, 'delta': self.dp_configs.delta}
        if return_stats:
            return self.trained_model, train_stats
        else:
            return self.trained_model

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        """Return the trained model unchanged: DP-SGD only changes training."""
        self.logger.print_it('Differential Privacy Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model

    def get_privacy_budget(self) -> Optional[dict]:
        """Return the epsilon that all the epochs of DP-SGD spent, with its delta, once the model is trained."""
        return self.privacy_budget


class DifferentialPrivacyTrainManager(TrainManager):
    """TrainManager whose training is DP-SGD: Opacus clips and noises per-sample gradients, and keeps the accounting.

    With SAM, the training is DP-SAT: each step computes the gradient at weights moved by rho along the previous
    step's noisy gradient, and updates the weights it started from.
    """
    # DP-SAT's radius, the previous noisy gradient and the perturbation of the current step: off until the optimizer
    # is SAM.
    sat_rho, sat_direction, sat_perturbation = None, None, None

    def __init__(self, train_configs, name: str, logger=None):
        """Set up the training as TrainManager does."""
        super().__init__(train_configs=train_configs, name=name, logger=logger)

    def train(self, extra_configs: dict[str, Any] = None, return_best_model: bool = True,
              return_last_model: bool = False, return_stats: bool = False):
        """Train as TrainManager does, then log the epsilon that all the epochs spent at delta."""
        result = super().train(extra_configs=extra_configs, return_best_model=return_best_model,
                               return_last_model=return_last_model, return_stats=return_stats)
        # The best model is picked among the models of all the epochs, so the guarantee is the one of the whole run.
        self.epsilon = self.privacy_engine.get_epsilon(self.dp_configs.delta)
        self.logger.print_it(f'DP-SGD: the training spent epsilon = {self.epsilon:.4f} '
                             f'at delta = {self.dp_configs.delta:g}.')
        return result

    def reset_epoch_stats(self, phase: str = 'train'):
        """Start tracking a stage as TrainManager does, with the loss of the criterion before Opacus wrapped it."""
        # With ghost clipping, self.criterion is Opacus' wrapper, whose output is not a loss value.
        criterion, self.criterion = self.criterion, self.metrics_criterion
        try:
            super().reset_epoch_stats(phase=phase)
        finally:
            self.criterion = criterion

    def get_checkpoint_extra(self) -> dict[str, Any]:
        """Return the privacy accountant's state, and DP-SAT's last noisy gradient, to save in each checkpoint."""
        extra = {'privacy_accountant': self.privacy_engine.accountant.state_dict()}
        if self.sat_direction is not None:
            extra['sat_direction'] = [direction.cpu() for direction in self.sat_direction]
        return extra

    def load_checkpoint_extra(self, extra: dict[str, Any]):
        """Restore the privacy accountant when resuming, so that epsilon also counts the steps before the resume."""
        if self.sat_rho is not None and 'sat_direction' in extra:
            self.sat_direction = [direction.to(p.device) for direction, p in zip(extra['sat_direction'],
                                                                                 self.optimizer.params)]
        if 'privacy_accountant' in extra:
            self.privacy_engine.accountant.load_state_dict(extra['privacy_accountant'])
            return
        # Checkpoints from before the accounting was saved: every epoch took the same number of steps, with the same
        # noise multiplier and sample rate, so the history is rebuilt from the epochs already done.
        steps = (self.epoch - 1) * len(self.train_loader)
        self.privacy_engine.accountant.history = [(self.optimizer.noise_multiplier, self.train_loader.sample_rate,
                                                    steps)]
        self.logger.print_it(f'DP-SGD: the checkpoint has no privacy accounting, '
                             f'rebuilt from its {self.epoch - 1} epochs.')

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        """Run one DP-SGD step on a batch, including an empty Poisson batch, and record its stats."""
        if self.sat_rho is not None:
            self.perturb_weights()
        self.private_step(inputs, targets, batch_idx=batch_idx, total_batches=total_batches)
        if self.device.type == 'mps':
            # Poisson batches change size at every step, and MPS caches the per-sample gradients' memory per size:
            # without this, the cache grows until the machine swaps (docs/defenses/dp.md). CUDA and CPU skip it.
            torch.mps.empty_cache()

    def private_step(self, inputs, targets, batch_idx=0, total_batches=0):
        """Run the DP-SGD step of train_step, on a batch or on an empty Poisson batch."""
        if targets.size(0) != 0:
            return super().train_step(inputs, targets, batch_idx=batch_idx, total_batches=total_batches)

        # An empty Poisson draw is a real DP-SGD step: the summed data gradient
        # is zero, but noise, optimizer state and the accountant still advance.
        # Initialize the gradient inputs expected by Opacus directly. No model
        # or criterion call is needed, so this works for backbones that cannot
        # forward an empty tensor and does not modify their buffers or RNG state.
        if not isinstance(self.optimizer, DPOptimizer):
            raise TypeError("Empty DP batches require an Opacus DPOptimizer")
        if isinstance(self.optimizer, DistributedPerLayerOptimizer):
            # This optimizer adds noise inside backward hooks, not step(). Do
            # not silently perform an unnoised/unaccounted update for this mode.
            raise NotImplementedError(
                "Empty Poisson batches with distributed per-layer clipping are not supported; "
                "use distributed flat clipping or single-process per-layer clipping."
            )
        self.epoch_stats_tracker.batch_start()
        self.optimizer.zero_grad()
        for parameter in self.optimizer.params:
            if isinstance(self.optimizer, DPOptimizerFastGradientClipping):
                # Ghost clipping consumes the already clipped summed gradient.
                parameter.grad = torch.zeros_like(parameter)
            else:
                # Hooks clipping consumes per-sample gradients. Zero samples
                # means an empty tensor, not an invented training example.
                parameter.grad_sample = parameter.new_empty((0, *parameter.shape))
        self.optimizer.step()
        self.epoch_stats_tracker.batch_end(batch_size=0)
        # Empty batches have no loss/accuracy observations to add to the meters.
        self.logger.print_it("DP-SGD: completed optimizer/accounting step for an empty Poisson batch.")

    @torch.no_grad()
    def perturb_weights(self):
        """DP-SAT's ascent: move the weights by rho along the previous step's noisy gradient, not at the first step."""
        if self.sat_direction is None:
            return
        norm = torch.norm(torch.stack([direction.norm(p=2) for direction in self.sat_direction]), p=2)
        self.sat_perturbation = [direction * (self.sat_rho / (norm + 1e-12)) for direction in self.sat_direction]
        for p, perturbation in zip(self.optimizer.params, self.sat_perturbation):
            p.add_(perturbation)

    @torch.no_grad()
    def restore_weights(self, optimizer, args, kwargs):
        """DP-SAT's descent, run before the base optimizer's step: move the weights back, so that the noisy gradient
        computed at the perturbed weights updates the original ones, and keep that gradient for the next ascent."""
        if self.sat_perturbation is not None:
            for p, perturbation in zip(self.optimizer.params, self.sat_perturbation):
                p.sub_(perturbation)
            self.sat_perturbation = None
        # The noisy gradient is already released, so the next ascent costs no privacy: it is post-processing.
        self.sat_direction = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                              for p in self.optimizer.params]

    def setup_optimizer(self, opt_cfg: OptimizerConfigs):
        """Create the optimizer as TrainManager does, with SAM's base SGD alone for DP-SAT. Refuse the other SAM-like
        optimizers and the adaptive SAM, which have no private version here."""
        super().setup_optimizer(opt_cfg=opt_cfg)
        if type(self.optimizer) is SAM and not self.optimizer.defaults['adaptive']:
            self.sat_rho = sam_setting(opt_cfg, 'sam_rho', DP_SAT_PAPER_RHO)
            # Opacus privatizes the base SGD, and train_step and restore_weights do the perturbation.
            self.optimizer = self.optimizer.base_optimizer
            self.criterion.reduction = 'mean'
            self.logger.print_it(f'DP-SAT: SAM runs as DP-SAT, with rho = {self.sat_rho:g}.')
        elif type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            raise ValueError(f"The dp defense runs SAM as DP-SAT, and has no private version of "
                             f"{opt_cfg.name}: choose sgd, adam or sam.")

    def check_and_set_dp(self, dp_config: DPDefenseConfigs):
        """Make the model, optimizer and training loader private with Opacus, and keep the privacy engine."""
        self.logger.print_it('Differential Privacy with Opacus: updating model, optimizer and data loaders accordingly...')
        if dp_config.grad_sample_mode not in ['ghost', 'hooks', 'functorch']:
            raise ValueError(f"Unsupported mode '{dp_config.grad_sample_mode}' for dp_config.grad_sample_mode "
                             f"when using DP!")

        if dp_config.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                dp_config.max_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            max_grad_norm = dp_config.max_grad_norm

        if self.distributed and dp_config.clip_per_layer:
            self.model = DPDDP(self.model)

        self.privacy_engine = PrivacyEngine()
        clipping = "per_layer" if dp_config.clip_per_layer else "flat"
        # The loss to log and track, kept before ghost clipping wraps self.criterion.
        self.metrics_criterion = self.criterion
        options = dict(module=self.model, optimizer=self.optimizer, data_loader=self.train_loader,
                       max_grad_norm=max_grad_norm, clipping=clipping, grad_sample_mode=dp_config.grad_sample_mode)
        if dp_config.grad_sample_mode in ['ghost']:
            options['criterion'] = self.criterion
        if dp_config.target_epsilon is None:
            private_objects = self.privacy_engine.make_private(noise_multiplier=dp_config.noise_multiplier, **options)
        else:
            epochs = self.train_configs.scheduler_config.epochs
            private_objects = self.privacy_engine.make_private_with_epsilon(target_epsilon=dp_config.target_epsilon,
                                                                            target_delta=dp_config.delta,
                                                                            epochs=epochs, **options)
        if dp_config.grad_sample_mode in ['ghost']:
            # Ghost clipping requires Opacus' wrapped criterion to perform its
            # two backward passes and populate the optimizer's gradient state.
            if len(private_objects) == 4:
                self.model, self.optimizer, self.criterion, self.train_loader = private_objects
            elif len(private_objects) == 3:
                raise RuntimeError(
                    "This Opacus version did not return the loss wrapper required "
                    "for ghost clipping. Use --defender_dp_grad_sample_mode hooks "
                    "or install a compatible Opacus release."
                )
            else:
                raise RuntimeError(
                    f"Unexpected number of values returned by Opacus make_private(): "
                    f"{len(private_objects)}"
                )
        else:
            self.model, self.optimizer, self.train_loader = private_objects
        if self.sat_rho is not None:
            # Opacus' optimizers all end their step with the base optimizer's, once the noisy gradient is in p.grad.
            self.optimizer.original_optimizer.register_step_pre_hook(self.restore_weights)
        self.logger.print_it(f'Differential Privacy setup completed, '
                             f'with noise multiplier {self.optimizer.noise_multiplier:.4f}!')
        self.model = self.model.to(self.device)

    def validate_and_fix_model_for_dp(self):
        """Replace the layers that Opacus cannot train privately, such as BatchNorm with GroupNorm, if any."""
        self.logger.print_it('Differential Privacy with Opacus: validating model and fixing it if necessary...')
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)
        self.model = self.model.to(self.device)

    def initialize_train(self,
                        dataset: Union[MultiDatasets, Dataset],
                        model: Union[ModelConfigs,torch.nn.Module],
                        configs: TrainConfigs,
                        dp_configs: DPDefenseConfigs,
                        ):
        """Set up the training as TrainManager does, then make it private with Opacus."""
        self.logger.print_it('Initializing training...')
        if not configs.__eq__(self.train_configs):
            self.logger.print_it('Found different training configurations in the initialize_train method. Resetting the trainer configs...')
            self.reset_configs(configs)

        self.dp_configs = dp_configs
        self.setup_dataloaders(dataset=dataset,
                                batch_size=self.train_configs.batch_size)
        if isinstance(model, torch.nn.Module):
            self.set_model(model)
        elif isinstance(model, ModelConfigs):
            self.setup_model_from_configs(model_configs=model)
        else:
            raise ValueError('Not recognizing model given!')
        self.validate_and_fix_model_for_dp()
        self.setup_training()
        # Modifying model, optimizer and loaders for differential privacy if needed
        self.check_and_set_dp(dp_config=dp_configs)
        self.logger.print_it('Training initialization completed!')
