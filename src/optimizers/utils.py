import contextlib
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

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        base_optimizer_state = state_dict.pop('base_optimizer', None)
        super().load_state_dict(state_dict)
        # Checkpoints saved before the base optimizer state was included do not have it.
        if base_optimizer_state is not None:
            self.base_optimizer.load_state_dict(base_optimizer_state)
        self.base_optimizer.param_groups = self.param_groups


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


