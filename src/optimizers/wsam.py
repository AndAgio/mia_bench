import torch
import torch.distributed as dist
from typing import Callable

from .utils import disable_running_stats, enable_running_stats


class WSAM(torch.optim.Optimizer):
    # Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term.
    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        gamma=0.9,
        sam_eps=1e-12,
        adaptive=False,
        decouple=True,
        max_norm=None,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        # print('Adaptive set to {}'.format(adaptive))

        # self.model = model
        self.decouple = decouple
        self.max_norm = max_norm
        alpha = gamma / (1 - gamma)
        defaults = dict(rho=rho, alpha=alpha, sam_eps=sam_eps, adaptive=adaptive, **kwargs)
        super(WSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["sam_eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w, alpha=1.0)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                if torch.distributed.is_initialized():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.param_groups, self.max_norm)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["grad"] = p.grad.detach().clone()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.distributed.is_initialized():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                p.add_(self.state[p]["e_w"], alpha=-1.0)  # get back to "w" from "w + e(w)"

        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.param_groups, self.max_norm)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self.decouple:
                    p.grad.mul_(group["alpha"]).add_(self.state[p]["grad"], alpha=1.0 - group["alpha"])
                else:
                    self.state[p]["sharpness"] = p.grad.detach().clone() - self.state[p]["grad"]
                    p.grad.mul_(0.0).add_(self.state[p]["grad"], alpha=1.0)

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if self.decouple:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.add_(self.state[p]["sharpness"], alpha=-group["lr"] * group["alpha"])

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Callable, inputs: torch.Tensor, targets: torch.Tensor):
        '''
        Expects closure to be:
        def closure(inputs, targets, mean=True, backward=True):
            loss = self.criterion(self.model(inputs), targets)
            if mean:
                loss = loss.mean()
            if backward:
                loss.backward()
            return loss
        '''
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        loss, outputs = closure(inputs, targets, mean=True, backward=True, run_stats=True)
        self.to_return = loss, outputs
        self.first_step(zero_grad=True)
        closure(inputs, targets, mean=True, backward=True, run_stats=False)
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
    
    def get_first_closure_outputs(self):
        return self.to_return
