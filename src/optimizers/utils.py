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


