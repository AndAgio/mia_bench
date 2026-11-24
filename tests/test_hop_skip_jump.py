"""HopSkipJump against a linear model, whose boundary distances are exact: L2 and linf distances, query budget."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.label_only.helpers.hop_skip_jump import HopSkipJump


class Silent:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Linear(torch.nn.Module):
    """Two classes split by the hyperplane w . x + b = 0, on inputs of shape (1, 4, 4)."""
    def __init__(self, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = torch.randn(16, generator=generator)
        self.b = 0.1
        self.seen = []

    def margin(self, x):
        return x.flatten(1) @ self.w + self.b

    def forward(self, x):
        self.seen.append(x.detach().clone())
        z = self.margin(x)[:, None]
        return torch.cat([z, -z], dim=1)


def samples(model, n=6, seed=1):
    """Inputs in [0.2, 0.8] and their labels under the model (so all are classified correctly)."""
    x = 0.2 + 0.6 * torch.rand(n, 1, 4, 4, generator=torch.Generator().manual_seed(seed))
    return x, (model.margin(x) < 0).long()


class HopSkipJumpTest(unittest.TestCase):
    def test_l2_distance_matches_the_linear_model(self):
        torch.manual_seed(0)
        model = Linear()
        x, y = samples(model)
        _, distances, _ = HopSkipJump(norm='l2', max_queries=3000, logger=Silent()).run(
            inputs=x, labels=y, model=model, targeted=False, device=torch.device('cpu'))
        exact = model.margin(x).abs() / model.w.norm()
        torch.testing.assert_close(distances.float(), exact, rtol=0.05, atol=1e-4)

    def test_linf_distance_matches_the_linear_model(self):
        torch.manual_seed(0)
        model = Linear()
        x, y = samples(model)
        _, distances, _ = HopSkipJump(norm='linf', max_queries=5000, logger=Silent()).run(
            inputs=x, labels=y, model=model, targeted=False, device=torch.device('cpu'))
        exact = model.margin(x).abs() / model.w.abs().sum()
        torch.testing.assert_close(distances.float(), exact, rtol=0.1, atol=1e-4)

    def test_no_sample_gets_more_queries_than_the_budget_and_each_point_is_adversarial(self):
        torch.manual_seed(0)
        model = Linear()
        x, y = samples(model)
        for budget in (150, 400, 1000):
            with self.subTest(budget=budget):
                adversarials, _, queries = HopSkipJump(norm='l2', max_queries=budget, logger=Silent()).run(
                    inputs=x, labels=y, model=model, targeted=False, device=torch.device('cpu'))
                self.assertLessEqual(max(queries), budget)
                self.assertTrue((model(adversarials).argmax(1) != y).all())

if __name__ == '__main__':
    unittest.main()
