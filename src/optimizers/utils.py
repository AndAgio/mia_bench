import contextlib
import copy
from collections import defaultdict
import torch
import torch.nn as nn


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class BaseOptimizerStateMixin:
    """For optimizers wrapping a base optimizer (SAM and its variants), whose state (e.g. the SGD momentum
    buffers) lives in the base optimizer: checkpoints include it, and loading one keeps the parameter groups
    shared between wrapper and base optimizer, so that learning rate schedulers keep reaching the base optimizer."""

    def for_objective(self, name: str):
        """Optimizer for another loss that the same model minimizes in separate steps (e.g. the MMD term of the MMD
        defense). It shares parameters, base optimizer (momentum, learning rate, weight decay) and settings with this
        one, but keeps its own state, since what some variants keep across steps holds for a single loss (e.g. the
        gradient average of F-SAM or the direction reused by LookSAM)."""
        objectives = self.__dict__.setdefault('_objectives', {})
        if name not in objectives:
            objective_optimizer = copy.copy(self)
            objective_optimizer.state = defaultdict(dict)
            objectives[name] = objective_optimizer
        return objectives[name]

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        objectives = self.__dict__.get('_objectives', {})
        if objectives:
            state_dict['objectives'] = {name: super(BaseOptimizerStateMixin, optimizer).state_dict()['state']
                                        for name, optimizer in objectives.items()}
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        base_optimizer_state = state_dict.pop('base_optimizer', None)
        objectives_state = state_dict.pop('objectives', {})
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        # Checkpoints saved before the base optimizer state was included do not have it.
        if base_optimizer_state is not None:
            self.base_optimizer.load_state_dict(base_optimizer_state)
            self.base_optimizer.param_groups = self.param_groups
        for name, objective_state in objectives_state.items():
            super(BaseOptimizerStateMixin, self.for_objective(name)).load_state_dict(
                {'state': objective_state, 'param_groups': state_dict['param_groups']})
        for objective_optimizer in self.__dict__.get('_objectives', {}).values():
            objective_optimizer.param_groups = self.param_groups

    def __getstate__(self) -> dict:
        # torch only keeps defaults, state and param_groups, which loses the base optimizer and the wrapper's own
        # settings (e.g. k or beta), so copies (e.g. the sub-model optimizers of MIST) could not step. Keep them too,
        # leaving out private attributes, callables bound to this instance (e.g. the step wrapper installed by
        # learning rate schedulers) and the outputs of the last step.
        state = super().__getstate__()
        for name, value in self.__dict__.items():
            if not name.startswith('_') and not callable(value) and name not in state and name != 'to_return':
                state[name] = value
        return state


def whether_to_sync(model, sync=False):
    if not sync:
        return model.no_sync()
    else:
        return contextlib.ExitStack()
    

def get_global_gradient_norm(param_groups, device: torch.device) -> torch.Tensor:
    """Get global gradient norm."""
    norms: list[torch.Tensor] = []
    for group in param_groups or []:
        params: list[torch.Tensor] = group.get('params', []) or []
        adaptive: bool = group.get('adaptive', False)
        for p in params:
            if p.grad is not None:
                norm = ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(device)
                norms.append(norm)

    if not norms:
        return torch.tensor(0.0, device=device)

    return torch.norm(torch.stack(norms), p=2)


def centralize_gradient(grad: torch.Tensor, gc_conv_only: bool = False) -> None:
    """Gradient Centralization (GC).

    Args:
        grad (torch.Tensor): Gradient tensor.
        gc_conv_only (bool): If False, apply GC to both convolutional and fully connected layers; if True, apply only
            to convolutional layers.
    """
    size: int = grad.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        grad.add_(-grad.mean(dim=tuple(range(1, size)), keepdim=True))


