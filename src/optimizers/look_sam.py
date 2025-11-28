import torch
from .utils import centralize_gradient, get_global_gradient_norm
from typing import Callable


class LookSAM(torch.optim.Optimizer):
    # An Expeditiously Adaptive Parameter-Free Learner.
    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.1,
        k: int = 10,
        alpha: float = 0.7,
        adaptive: bool = False,
        use_gc: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert k > 0.0, f"Invalid k, should be positive: {k}"
        assert 0. < alpha < 1.0, f"Invalid alpha, should be between 0 and 1: {alpha}"
        assert perturb_eps >= 0.0, f"Invalid perturb_eps, should be non-negative: {perturb_eps}"
        # print('Adaptive set to {}'.format(adaptive))

        self.k = k
        self.alpha = alpha
        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        defaults = {'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'LookSAM'

    def init_group(self, group, **kwargs) -> None:
        pass

    def get_step(self):
        return (
            self.param_groups[0]['step']
            if 'step' in self.param_groups[0]
            else next(iter(self.base_optimizer.state.values()))['step'] if self.base_optimizer.state else 0
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        if self.get_step() % self.k != 0:
            return

        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad
                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                self.state[p]['old_p'] = p.clone()
                self.state[f'old_grad_p_{i}']['old_grad_p'] = grad.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        step = self.get_step()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad
                grad_norm = grad.norm(p=2)

                if step % self.k == 0:
                    old_grad_p = self.state[f'old_grad_p_{i}']['old_grad_p']

                    g_grad_norm = old_grad_p / old_grad_p.norm(p=2)
                    g_s_grad_norm = grad / grad_norm

                    self.state[f'gv_{i}']['gv'] = torch.sub(
                        grad, grad_norm * torch.sum(g_grad_norm * g_s_grad_norm) * g_grad_norm
                    )
                else:
                    gv = self.state[f'gv_{i}']['gv']
                    grad.add_(grad_norm / (gv.norm(p=2) + 1e-8) * gv, alpha=self.alpha)

                p.data = self.state[p]['old_p']

        self.base_optimizer.step()

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

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def get_first_closure_outputs(self):
        return self.to_return

