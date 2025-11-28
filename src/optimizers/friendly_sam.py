import torch
from .utils import get_global_gradient_norm


class FriendlySAM(torch.optim.Optimizer):
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
        assert lmbda >= 0.0, f"Invalid lmbda, should be non-negative: {lmbda}"
        assert perturb_eps >= 0.0, f"Invalid sigma, should be non-negative: {perturb_eps}"
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

                if 'momentum' not in state:
                    state['momentum'] = grad.clone()
                else:
                    momentum = state['momentum']

                    grad.sub_(momentum, alpha=group['sigma'])
                    momentum.lerp_(grad, weight=1.0 - group['lmbda'])

        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad

                self.state[p]['old_p'] = p.clone()
                self.state[f'old_grad_p_{i}']['old_grad_p'] = grad.clone()

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
    def step(self, closure = None):
        assert closure is not None, "Friendly Sharpness Aware Minimization requires closure, but it was not provided"

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()

        self.second_step()

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups