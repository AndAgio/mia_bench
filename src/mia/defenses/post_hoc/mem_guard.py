from typing import Union
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from src.data.helpers import FixedLabelDataset
from src.utils.configs import DefenderConfigs, MemGuardDefenseConfigs
from src.mia.defenses.base import BaseDefender


BATCH_SIZE = 256


class MemGuardDefender(BaseDefender):
    # Implementation of MemGuard defense from "MemGuard: Defending Against Black-Box Membership Inference Attacks via Adversarial Examples" (https://dl.acm.org/doi/pdf/10.1145/3319535.3363201).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, MemGuardDefenseConfigs), f"MemGuardDefender can only be used with MemGuardDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'mem_guard_defender'
        self.mem_guard_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it("MemGuardDefender: No training required for MemGuard defense, training with standard procedure.")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('MemGuard Defender: fitting shadow attack model...')
        shadow_model = self.fit_shadow_attack_model(device=device)
        self.logger.print_it('MemGuard Defender: building defense layer...')
        self.defended_model = MemGuard(victim=self.trained_model,
                                        membership_model=shadow_model,
                                        use_sorted=True,
                                        c1=1.0,
                                        c2=10.0,
                                        c3_init=0.1,
                                        c3_max=1e5,
                                        max_iter=300,
                                        step_size=10.0,
                                        randomize_mix=0.,
                                        apply_expected_budget=True,
                                        budget_l1=self.mem_guard_configs.budget).to(device)
        return self.defended_model

    def _sort_with_index(self, t: torch.Tensor):
        sort_idx = torch.argsort(t, dim=1)
        back_idx = torch.zeros_like(sort_idx)
        for i in range(t.size(0)):
            back_idx[i, sort_idx[i]] = torch.arange(t.size(1), device=t.device)
        t_sorted = torch.gather(t, 1, sort_idx)
        return t_sorted, sort_idx, back_idx

    def _prep(self, logits, probs, labels):
        x = logits if self.use_logits else probs
        if self.use_sorted:
            x, _, _ = self._sort_with_index(x)
        return x, labels.float()

    def fit_shadow_attack_model(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        shadow_attacker_model = AttackNet(
            in_dim=self.model_configs.num_classes,
            hidden=self.mem_guard_configs.shadow_attacker_model_layers
        ).to(device)
        attack_data = ConcatDataset([FixedLabelDataset(self.dataset.get('train'),
                                                    fixed_label=1),
                                    FixedLabelDataset(self.dataset.get('test'),
                                                    fixed_label=0),])
        features = self.get_model_outputs(model=self.trained_model,
                                        data=attack_data,
                                        device=device)
        labels = torch.cat([torch.ones(len(self.dataset.get('train'))),
                            torch.zeros(len(self.dataset.get('test')))], dim=0)
        data_loader = DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(shadow_attacker_model.parameters(), lr=self.mem_guard_configs.shadow_attacker_model_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        shadow_attacker_model.train()
        for epoch in range(self.mem_guard_configs.shadow_attacker_model_epochs):
            run_loss = 0.0
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = shadow_attacker_model(features, return_logits=True)
                loss = criterion(outputs, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
            self.logger.print_it(f"MemGuard Defender: Training shadow attack model -> epoch {epoch+1}/{self.mem_guard_configs.shadow_attacker_model_epochs}, attack Loss: {run_loss/len(data_loader):.4f}")
        shadow_attacker_model.eval()
        return shadow_attacker_model


    def get_model_outputs(self, model: torch.nn.Module, data: torch.utils.data.Dataset, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model.eval()
        dummy_data, _ = data[0]
        dummy_data = dummy_data.unsqueeze(0)
        dummy_feature = MemGuardDefender.get_model_out(model=model,
                                                data=dummy_data,
                                                device=device)
        feature_shape = dummy_feature.shape[1]
        features = torch.zeros((len(data), feature_shape))
        dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
        current_index = 0
        for enumerate_index, (batch_data, _) in enumerate(dataloader):
            # self.logger.print_it_same_line(f"Processing batch {enumerate_index}/{len(dataloader)} for attacking model dataset construction...")
            batch_size = batch_data.size(0)
            batch_features = MemGuardDefender.get_model_out(model=model,
                                                        data=batch_data,
                                                        device=device)
            features[current_index:current_index+batch_size, :] = batch_features
            current_index += batch_size
        return features
    
    @staticmethod
    def get_model_out(model: torch.nn.Module, data: torch.Tensor, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = MemGuardDefender.get_device(dev_str=device)
        model.eval()
        with torch.no_grad():
            output = torch.nn.functional.softmax(model(data.to(device)), dim=1).cpu()
        return output

    @staticmethod
    def get_device(dev_str: str = 'cpu'):
        # Set appropriate devices
        if torch.cuda.is_available() and dev_str != 'cpu':
            dev_str = 'cuda:{}'.format(dev_str)
            device = torch.device(dev_str)
        elif torch.backends.mps.is_available() and dev_str != 'cpu':
            dev_str = 'mps'
            device = torch.device(dev_str)
        else:
            device = torch.device('cpu')
        return device



class AttackNet(torch.nn.Module):
    """Simple MLP attacker: takes sorted logits (or probs) → membership logit."""
    def __init__(self, in_dim: int, hidden=(64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(inplace=True)]
            d = h
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, return_logits=False):
        out = self.net(x).squeeze(-1)   # (B,)
        return out if return_logits else torch.sigmoid(out)


class MemGuard(torch.nn.Module):
    """
    MemGuard defense (single class).
    - Wraps a victim (logits model) and a membership model (attacker/defense).
    - Internally runs an optimization to craft per-query defended outputs with:
        L_opt = c1 * |membership_logit|        (push sigmoid -> 0.5)
              + c2 * max(0, max_k!=y z_k - z_y)  (preserve original top-1 label in logits)
              + c3 * ||softmax(z') - softmax(z)||_1  (small utility change in prob space)
      using L2-normalized gradient steps in *logit* space; escalates c3 if constraints not met.
    - Then applies the crafted output with probability p = min(1, budget / L1) to meet an expected L1 budget.
    """

    def __init__(
        self,
        victim,                 # Victim
        membership_model,       # attacker/defense model expecting (sorted) logits
        *,
        use_sorted: bool = True,
        c1: float = 1.0,
        c2: float = 10.0,
        c3_init: float = 0.1,
        c3_max: float = 1e5,
        max_iter: int = 300,
        step_size: float = 0.1,
        randomize_mix: float = 0.01,    # 0 disables
        apply_expected_budget: bool = True,
        budget_l1: float = 0.10
    ):
        super().__init__()
        self.victim = victim.eval()
        self.M = membership_model.eval()
        self.use_sorted = use_sorted

        # Optimization (old "phase I") hyperparameters
        self.c1, self.c2, self.c3_init, self.c3_max = c1, c2, c3_init, c3_max
        self.max_iter, self.step_size = max_iter, step_size
        self.randomize_mix = randomize_mix

        # Application (old "phase II") settings
        self.apply_expected_budget = apply_expected_budget
        self.budget_l1 = budget_l1

    # ---------- helpers ----------
    @staticmethod
    def _sort_with_index(t: torch.Tensor):
        sort_idx = torch.argsort(t, dim=1)       # ascending
        back_idx = torch.zeros_like(sort_idx)
        for i in range(t.size(0)):
            back_idx[i, sort_idx[i]] = torch.arange(t.size(1), device=t.device)
        t_sorted = torch.gather(t, 1, sort_idx)
        return t_sorted, sort_idx, back_idx

    @staticmethod
    def _to_sorted(t: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(t, 1, sort_idx)

    @staticmethod
    def _to_original(sorted_t: torch.Tensor, back_idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(sorted_t, 1, back_idx)

    @staticmethod
    def _mix_uniform(q: torch.Tensor, mix: float) -> torch.Tensor:
        if mix <= 0: return q
        B, K = q.shape
        u = torch.full((B, K), 1.0 / K, device=q.device, dtype=q.dtype)
        q2 = (1 - mix) * q + mix * u
        return (q2.clamp_min(0.0) / q2.sum(dim=1, keepdim=True))

    # ---------- the optimization (formerly "Phase I") ----------
    def _optimize_confidence(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given victim logits (B, K) in original order, return:
            q_adv: defended probabilities (B, K) in original order
            l1:    per-sample L1 distance in probability space ||q_adv - q_orig||_1
        """
        B, K = logits.shape
        y_orig = logits.argmax(dim=1)
        q_orig = torch.nn.functional.softmax(logits, dim=1)

        # Optional sorting for membership model input
        if self.use_sorted:
            logits_s, sort_idx, back_idx = self._sort_with_index(logits)
            # y in sorted space = position of original argmax in sorted order
            y_sorted = torch.gather(back_idx, 1, y_orig.view(-1,1)).squeeze(1)
            to_back = lambda z: self._to_original(z, back_idx)
            origin = logits_s.clone().detach()
        else:
            logits_s = origin = logits.clone().detach()
            y_sorted = y_orig.clone()
            to_back = lambda z: z

        label_mask = torch.nn.functional.one_hot(y_sorted, num_classes=K).float()
        with torch.no_grad():
            s0 = torch.sigmoid(self.M(logits_s))   # initial membership prob on sorted logits

        # Early-exit samples already at ~0.5
        near = (s0 - 0.5).abs() <= 1e-5
        sample_f = origin.clone().detach().requires_grad_(True)
        last_probs = torch.nn.functional.softmax(sample_f.detach(), dim=1)

        c3 = self.c3_init
        outer_tries = 0
        improved = False

        while True:
            it = 0
            with torch.enable_grad():
                while it < self.max_iter:
                    # Loss components:
                    out_logits = self.M(sample_f, return_logits=True).view(-1)      # L_opt part 1
                    loss1 = out_logits.abs().mean()

                    correct = (sample_f * label_mask).sum(dim=1, keepdim=True)      # L_opt part 2
                    wrong  = sample_f.masked_fill(label_mask.bool(), float('-inf')).max(dim=1, keepdim=True).values
                    loss2 = torch.relu(wrong - correct).mean()

                    prob_now  = torch.nn.functional.softmax(sample_f, dim=1)                           # L_opt part 3
                    prob_orig = torch.nn.functional.softmax(origin, dim=1)
                    loss3 = (prob_now - prob_orig).abs().sum(dim=1).mean()

                    loss = self.c1*loss1 + self.c2*loss2 + c3*loss3

                    # L2-normalized gradient step in logit space
                    if sample_f.grad is not None:
                        sample_f.grad.zero_()
                    loss.backward()
                    g = sample_f.grad
                    gnorm = torch.norm(g.view(B, -1), dim=1, keepdim=True).clamp_min(1e-12)
                    with torch.no_grad():
                        sample_f -= self.step_size * (g / gnorm)
                    sample_f.requires_grad_(True)

                    # Stop when (label preserved) & (attacker crosses 0.5)
                    with torch.no_grad():
                        s_now = torch.sigmoid(self.M(sample_f)).view(-1)
                        y_now = sample_f.argmax(dim=1)
                        ok_label = (y_now == y_sorted)
                        crossed  = ((s_now - 0.5) * (s0 - 0.5) <= 0)
                        if (ok_label & crossed).all():
                            last_probs = torch.nn.functional.softmax(sample_f, dim=1).detach()
                            improved = True
                            break
                    it += 1

            if improved: break
            c3 *= 10.0; outer_tries += 1
            if c3 > self.c3_max or outer_tries > 10:
                with torch.no_grad():
                    last_probs = torch.nn.functional.softmax(sample_f, dim=1).detach()
                break

        # Optional randomization on simplex (sorted space)
        if self.randomize_mix > 0:
            last_probs = self._mix_uniform(last_probs, self.randomize_mix)

        # Map back to original class order; compute L1 distance
        q_adv = to_back(last_probs)
        l1 = (q_adv - q_orig).abs().sum(dim=1)
        return q_adv, l1

    # ---------- public API ----------
    @torch.no_grad()
    def defend(self, x: torch.Tensor) -> torch.Tensor:
        """Return defended probability vectors for batch x."""
        if hasattr(self.victim, 'predict_logits'):
            logits = self.victim.predict_logits(x)     # (B, K)
        else:
            logits = self.victim(x)                    # (B, K)
        q_orig = torch.nn.functional.softmax(logits, dim=1)
        q_adv, l1 = self._optimize_confidence(logits)

        if not self.apply_expected_budget:
            return q_adv

        # Expected-budget application: p = min(1, budget / l1)
        p = torch.zeros_like(l1)
        nz = (l1 > 1e-12)
        p[nz] = (self.budget_l1 / l1[nz]).clamp(max=1.0)
        u = torch.rand_like(p)
        use = (u < p).float().view(-1, 1)
        return use * q_adv + (1 - use) * q_orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("MemGuard defense should not be used in training mode!")
        return self.defend(x)