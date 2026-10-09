import torch
import random
import math
from .utils import BaseOptimizerStateMixin, asam_scale
from typing import Callable


class ESAM(BaseOptimizerStateMixin, torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,eta=0.01,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert eta >= 0.0, f"Invalid eta, should be non-negative: {eta}"
        assert 0.0 < beta <= 1.0, f"Invalid beta, should be in (0, 1]: {beta}"
        assert 0.0 < gamma <= 1.0, f"Invalid gamma, should be in (0, 1]: {gamma}"
        # print('Adaptive set to {}'.format(adaptive))
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho, beta=beta, gamma=gamma, adaptive=adaptive, eta=eta, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        for group in self.param_groups:
            group["rho"] = rho
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                #original sam 
                # e_w = p.grad * scale.to(p)
                # asam 
                e_w = asam_scale(p, group) ** 2 * p.grad * scale.to(p)
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w



        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure: Callable, *args):
        '''
        Expects closure to be, with args its positional arguments (inputs, targets, ...), where every tensor holds
        one entry per sample, so that the selected samples can be taken from all of them:
        def closure(inputs, targets, mean=True, backward=True):
            loss = self.criterion(self.model(inputs), targets)
            if mean:
                loss = loss.mean()
            if backward:
                loss.backward()
            return loss
        '''
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        loss, outputs = closure(*args, mean=False, backward=False, run_stats=True)
        l_before = loss.clone().detach()
        self.to_return = loss.mean(), outputs
        loss.mean().backward()
        self.first_step(zero_grad=True)
        if l_before.dim() == 0:
            # A loss of the whole batch (e.g. MMD, a function of the set of outputs) has no per-sample values to
            # select samples by: all of them are used, as in SAM.
            selected = args
        else:
            with torch.no_grad():
                # At the perturbed weights: BatchNorm running statistics are only updated by the clean pass, as in SAM.
                l_after, _ = closure(*args, mean=False, backward=False, run_stats=False)
                instance_sharpness = l_after-l_before
                # Sharpness-sensitive data selection: the gamma * |B| samples whose loss increases the most (at least
                # one). The official code keeps the samples above the k-th largest value instead, which is one sample
                # fewer, or none at all for small batches or ties.
                position = max(1, math.ceil(len(instance_sharpness) * self.gamma))
                indices = torch.topk(instance_sharpness, position).indices
            selected = [arg[indices] if isinstance(arg, torch.Tensor) else arg for arg in args]
        closure(*selected, mean=True, backward=True, run_stats=False)
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        (asam_scale(p, group) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
                )
        return norm
    
    def get_first_closure_outputs(self):
        return self.to_return
