import torch
from .utils import BaseOptimizerStateMixin, get_global_gradient_norm
from typing import Callable


class FriendlySAM(BaseOptimizerStateMixin, torch.optim.Optimizer):
    # Friendly Sharpness-Aware Minimization.
    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        sigma: float = 1.0,
        lmbda: float = 0.9,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert sigma >= 0.0, f"Invalid sigma, should be non-negative: {sigma}"
        assert 0.0 <= lmbda <= 1.0, f"Invalid lmbda, should be in [0, 1]: {lmbda}"
        assert perturb_eps >= 0.0, f"Invalid perturb_eps, should be non-negative: {perturb_eps}"
        # print('Adaptive set to {}'.format(adaptive))

        self.perturb_eps = perturb_eps

        defaults = {'rho': rho, 'sigma': sigma, 'lmbda': lmbda, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'FriendlySAM'

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Algorithm 1 of the paper: m_t = lmbda * m_{t-1} + (1 - lmbda) * g_t from the raw gradient, with
                # m_{-1} = 0, then perturbation along d_t = g_t - sigma * m_t. The official code starts from m = g_0
                # and subtracts m_{t-1}, which gives the same direction with sigma = 1 once the start has faded.
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(grad)
                momentum = state['momentum']
                momentum.lerp_(grad, weight=1.0 - group['lmbda'])
                grad.sub_(momentum, alpha=group['sigma'])

        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.state[p]['old_p'] = p.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data = self.state[p]['old_p']

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

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

    def get_first_closure_outputs(self):
        return self.to_return
