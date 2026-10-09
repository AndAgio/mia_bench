import torch
from .utils import BaseOptimizerStateMixin, get_global_gradient_norm
from typing import Callable


class LookSAM(BaseOptimizerStateMixin, torch.optim.Optimizer):
    # Towards Efficient and Scalable Sharpness-Aware Minimization (Liu et al., CVPR 2022), Algorithm 1.
    # Every k steps a SAM step, which also keeps the component g_v of the SAM gradient orthogonal to the gradient;
    # on the other steps a single forward-backward pass, whose gradient g is updated with g + alpha * |g| / |g_v| * g_v.
    # Projection and norms are taken over the whole model, as in the paper.
    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        k: int = 5,
        alpha: float = 0.3,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ):
        # k = 5 is the value the paper recommends. The paper only tunes alpha for ViTs with rho = 1.0 (best: 0.7);
        # with rho = 0.05, alpha = 0.7 makes the reused component several times larger than SAM's own, and ResNet-18
        # on CIFAR-10 barely trains, while alpha = 0.3 behaves close to SAM.
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert isinstance(k, int) and k > 0, f"Invalid k, should be a positive integer: {k}"
        assert alpha >= 0.0, f"Invalid alpha, should be non-negative: {alpha}"
        assert perturb_eps >= 0.0, f"Invalid perturb_eps, should be non-negative: {perturb_eps}"

        self.k = k
        self.alpha = alpha
        self.perturb_eps = perturb_eps

        defaults = {'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return 'LookSAM'

    def get_step(self) -> int:
        # The step counter lives in the optimizer state, so that it is saved in checkpoints and the optimizers of
        # other objectives (see for_objective) count their own steps.
        return self.state['looksam'].get('step', 0)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        device = self.param_groups[0]['params'][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device).add_(self.perturb_eps)

        for group in self.param_groups:
            scale = group['rho'] / grad_norm

            for p in group['params']:
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.clone()
                self.state[p]['old_grad_p'] = p.grad.clone()

                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        # p.grad holds the SAM gradient g_s: keep g_v = g_s - (g . g_s / |g|^2) g for the following steps.
        params = [p for group in self.param_groups for p in group['params']
                  if p.grad is not None and 'old_grad_p' in self.state[p]]
        g_dot_g_s = sum((self.state[p]['old_grad_p'] * p.grad).sum() for p in params)
        g_norm_sq = sum(self.state[p]['old_grad_p'].pow(2).sum() for p in params)
        projection = g_dot_g_s / (g_norm_sq + self.perturb_eps)

        for p in params:
            self.state[p]['gv'] = p.grad - projection * self.state[p]['old_grad_p']
            p.data = self.state[p]['old_p']
            del self.state[p]['old_p'], self.state[p]['old_grad_p']

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def reuse_step(self, zero_grad: bool = False) -> None:
        # p.grad holds the gradient g: add the g_v of the last SAM step, rescaled to alpha times the norm of g.
        params = [p for group in self.param_groups for p in group['params']
                  if p.grad is not None and 'gv' in self.state[p]]
        g_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in params]), p=2)
        gv_norm = torch.norm(torch.stack([self.state[p]['gv'].norm(p=2) for p in params]), p=2)
        scale = self.alpha * g_norm / (gv_norm + self.perturb_eps)

        for p in params:
            p.grad.add_(scale * self.state[p]['gv'])

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
        step = self.get_step()
        # The paper counts steps from 1 and takes a SAM step when t % k == 0, which leaves g_v undefined for the
        # first k - 1 steps: counting from 0, the first step is a SAM step.
        if step % self.k == 0:
            self.first_step(zero_grad=True)
            closure(*args, mean=True, backward=True, run_stats=False)
            self.second_step()
        else:
            self.reuse_step()
        self.state['looksam']['step'] = step + 1

    def get_first_closure_outputs(self):
        return self.to_return
