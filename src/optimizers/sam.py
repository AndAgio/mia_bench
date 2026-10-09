import torch
from .utils import BaseOptimizerStateMixin, asam_scale
from typing import Callable


class SAM(BaseOptimizerStateMixin, torch.optim.Optimizer):
    # Sharpness-Aware Minimization for Efficiently Improving Generalization.
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, eta=0.01, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert eta >= 0.0, f"Invalid eta, should be non-negative: {eta}"
        # print('Adaptive set to {}'.format(adaptive))

        defaults = dict(rho=rho, adaptive=adaptive, eta=eta, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = asam_scale(p, group) ** 2 * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Callable, *args):
        '''
        Expects closure to be, with args its positional arguments (inputs, targets, ...):
        def closure(inputs, targets, mean=True, backward=True):
            loss = self.criterion(self.model(inputs), targets)
            if mean:
                loss = loss.mean()
            if backward:
                loss.backward()
            return loss
        '''
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        loss, outputs = closure(*args, mean=True, backward=True, run_stats=True)
        self.to_return = loss, outputs
        self.first_step(zero_grad=True)
        closure(*args, mean=True, backward=True, run_stats=False)
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (asam_scale(p, group) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
            )
        return norm

    def get_first_closure_outputs(self):
        return self.to_return
