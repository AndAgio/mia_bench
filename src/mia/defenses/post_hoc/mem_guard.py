from typing import Union
import time
import hashlib
import math
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
from src.data.helpers import ConstantLabelDataset 
from src.utils.configs import DefenderConfigs, MemGuardDefenseConfigs
from src.mia.defenses.base import BaseDefender


BATCH_SIZE = 64


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
        shadow_model = self.fit_shadow_attacker_model(device=device)
        self.logger.print_it('MemGuard Defender: building defense layer...')
        self.defended_model = MemGuard(victim=self.trained_model,
                                        membership_model=shadow_model,
                                        use_sorted=True,
                                        c1=1.0,
                                        c2=10.0,
                                        c3_init=0.1,
                                        c3_max=1e5,
                                        max_iter=300,
                                        step_size=0.1,
                                        randomize_mix=0.,
                                        apply_expected_budget=True,
                                        budget_l1=self.mem_guard_configs.budget,
                                        randomness_quantization=self.mem_guard_configs.randomness_quantization).to(device).eval()
        self.logger.print_it('MemGuard Defender: finished building defense layer!')
        return self.defended_model

    def fit_shadow_attacker_model(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        shadow_attacker_model = AttackNet(
            in_dim=self.model_configs.num_classes,
            hidden=self.mem_guard_configs.shadow_attacker_model_layers
        ).to(device)
        members = self.dataset.get('train')
        non_members = self.dataset.get('val')
        if len(members) == 0 or len(non_members) == 0:
            raise ValueError('MemGuard requires non-empty train and validation splits.')
        # Match the balanced membership prior used by the authors. Subsample
        # each pool with the benchmark's seeded torch RNG, never the audit set.
        n = min(len(members), len(non_members))
        members = Subset(members, torch.randperm(len(members))[:n].tolist())
        non_members = Subset(non_members, torch.randperm(len(non_members))[:n].tolist())
        attack_data = ConcatDataset([ConstantLabelDataset(members,
                                                    constant_label=1),
                                    ConstantLabelDataset(non_members,
                                                    constant_label=0),])
        features = self.get_model_outputs(model=self.trained_model,
                                        data=attack_data,
                                        device=device)
        labels = torch.cat([torch.ones(len(members)),
                            torch.zeros(len(non_members))], dim=0)
        data_loader = DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.SGD(shadow_attacker_model.parameters(), lr=self.mem_guard_configs.shadow_attacker_model_lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        shadow_attacker_model.train()
        start_time = time.time()
        self.logger.print_it(f"MemGuard Defender: Training shadow attack model for {self.mem_guard_configs.shadow_attacker_model_epochs} epochs with LR {self.mem_guard_configs.shadow_attacker_model_lr}. This may take a while...")
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
            self.logger.print_it_same_line(f"MemGuard Defender: Training shadow attack model -> epoch {epoch+1}/{self.mem_guard_configs.shadow_attacker_model_epochs}, attack Loss: {run_loss/len(data_loader):.4f}", console_only=True)
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"MemGuard Defender: Finished training shadow attack model in {time.time() - start_time:.2f} seconds.")
        shadow_attacker_model.eval()
        return shadow_attacker_model


    def get_model_outputs(self, model: torch.nn.Module, data: torch.utils.data.Dataset, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model.eval()
        dummy_data, _, _, _ = data[0]
        dummy_data = dummy_data.unsqueeze(0)
        dummy_feature = MemGuardDefender.get_model_out(model=model,
                                                data=dummy_data,
                                                device=device)
        feature_shape = dummy_feature.shape[1]
        features = torch.zeros((len(data), feature_shape))
        dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
        current_index = 0
        for enumerate_index, (batch_data, _, _, _) in enumerate(dataloader):
            self.logger.print_it_same_line(f"Processing batch {enumerate_index}/{len(dataloader)} for attacking model dataset construction...", console_only=True)
            batch_size = batch_data.size(0)
            batch_features = MemGuardDefender.get_model_out(model=model,
                                                        data=batch_data,
                                                        device=device)
            features[current_index:current_index+batch_size, :] = batch_features
            current_index += batch_size
        self.logger.set_logger_newline(console_only=True)
        return features
    
    @staticmethod
    def get_model_out(model: torch.nn.Module, data: torch.Tensor, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = MemGuardDefender.get_device(dev_str=device)
        model.eval()
        with torch.no_grad():
            output = torch.nn.functional.softmax(model(data.to(device)), dim=1).cpu()
        return output.sort(dim=1).values

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
    """Simple MLP attacker: takes sorted probabilities → membership logit."""
    def __init__(self, in_dim: int, hidden=(256, 128, 64)):
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
    - Membership features are probabilities, sorted when use_sorted=True.
    - forward returns log probabilities; defend returns probabilities.
    - Internally runs an optimization to craft per-query defended outputs with:
        L_opt = c1 * |membership_logit|        (push sigmoid -> 0.5)
              + c2 * max(0, max_k!=y z_k - z_y)  (preserve original top-1 label in logits)
              + c3 * ||softmax(z') - softmax(z)||_1  (small utility change in prob space)
      using L2-normalized gradient steps in *logit* space; increases c3 after successful attempts to reduce distortion.
    - Phase II rejects outputs farther from the membership boundary, then
      applies the candidate with probability min(1, budget / L1).
    - A hash of the quantized query seeds one-time randomness for repeat queries.
    """

    def __init__(
        self,
        victim,                 # Victim
        membership_model,       # attacker expecting (sorted) probabilities
        *,
        use_sorted: bool = True,
        c1: float = 1.0,
        c2: float = 10.0,
        c3_init: float = 0.1,
        c3_max: float = 1e5,
        max_iter: int = 300,
        step_size: float = 0.1,
        randomize_mix: float = 0.0,     # optional extension; disabled for paper fidelity
        apply_expected_budget: bool = True,
        budget_l1: float = 0.10,
        randomness_quantization: float = 1e-3
    ):
        super().__init__()
        if not (math.isfinite(c3_init) and math.isfinite(c3_max) and 0 < c3_init <= c3_max):
            raise ValueError('MemGuard requires 0 < c3_init <= c3_max, both finite.')
        if max_iter < 1 or not math.isfinite(step_size) or step_size <= 0:
            raise ValueError('MemGuard requires max_iter >= 1 and a positive finite step_size.')
        if not math.isfinite(budget_l1) or budget_l1 < 0:
            raise ValueError('MemGuard requires a non-negative finite L1 budget.')
        if not math.isfinite(randomness_quantization) or randomness_quantization <= 0:
            raise ValueError('MemGuard requires a positive finite randomness_quantization.')
        if not 0 <= randomize_mix < 1:
            raise ValueError('MemGuard requires 0 <= randomize_mix < 1.')
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
        self.randomness_quantization = randomness_quantization

    # ---------- helpers ----------
    def _membership_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Consume probability features in the ordering selected by the caller."""
        return self.M(features, return_logits=True).view(-1)

    def _one_time_uniform(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Hash each quantized query, independently of batch order and global RNG.

        The paper does not specify a quantization width. The default is 1e-3
        in model-input units; callers can choose a width appropriate to their data.
        """
        # Transfer first, then cast: MPS can attempt the dtype conversion on
        # the source device when .to(device='cpu', dtype=float64) is combined.
        cpu_query = x.detach().cpu()
        quantized = (cpu_query.to(dtype=torch.float64)
                     / self.randomness_quantization).round().to(torch.int64)
        draws = []
        for row in quantized:
            digest = hashlib.sha256()
            digest.update(str(tuple(row.shape)).encode('ascii'))
            digest.update(row.numpy().astype('<i8', copy=False).tobytes())
            seed = int.from_bytes(digest.digest(), 'big')
            draws.append(random.Random(seed).random())
        return torch.tensor(draws, device=like.device, dtype=like.dtype)

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
        original = logits.detach()
        y_original = original.argmax(dim=1)
        q_original = original.softmax(dim=1)
        if self.use_sorted:
            # The authors sort once, optimize in that fixed ordering, then undo
            # the permutation. Do not re-sort intermediate probability vectors.
            sort_index = original.argsort(dim=1, stable=True)
            back_index = sort_index.argsort(dim=1)
            origin = original.gather(1, sort_index)
            labels = back_index.gather(1, y_original[:, None]).squeeze(1)
        else:
            origin = original
            back_index = None
            labels = y_original
        q_orig = origin.softmax(dim=1)
        with torch.no_grad():
            initial_scores = self._membership_logits(q_orig)

        # Each row searches larger distortion weights only after success. An
        # unsuccessful round ends that row's search and retains its last success.
        searching = initial_scores.sigmoid().sub(0.5).abs() > 1e-5
        last_success = q_orig.clone()
        ever_succeeded = torch.zeros_like(searching)
        label_mask = torch.nn.functional.one_hot(labels, num_classes=origin.size(1)).bool()
        c3 = self.c3_init
        while searching.any() and c3 <= self.c3_max:
            sample_f = origin.clone()
            round_success = torch.zeros_like(searching)
            # Algorithm 1 starts i at 1 and continues while i < max_iter.
            for _ in range(self.max_iter - 1):
                active = (searching & ~round_success).nonzero(as_tuple=True)[0]
                if active.numel() == 0:
                    break
                with torch.enable_grad():
                    z = sample_f[active].detach().requires_grad_(True)
                    probs = z.softmax(dim=1)
                    scores = self._membership_logits(probs)
                    correct = z.gather(1, labels[active, None]).squeeze(1)
                    wrong = z.masked_fill(label_mask[active], float('-inf')).max(dim=1).values
                    distortion = (probs - q_orig[active]).abs().sum(dim=1)
                    loss = (self.c1 * scores.abs() + self.c2 * torch.relu(wrong - correct)
                            + c3 * distortion).sum()
                    gradient, = torch.autograd.grad(loss, z)
                with torch.no_grad():
                    norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    updated = z - self.step_size * gradient / norm
                    sample_f[active] = updated
                    candidate = updated.softmax(dim=1)
                    current_scores = self._membership_logits(candidate)
                    crossed = current_scores * initial_scores[active] <= 0
                    success = (updated.argmax(dim=1) == labels[active]) & crossed
                    accepted = active[success]
                    last_success[accepted] = candidate[success]
                    round_success[accepted] = True
                    ever_succeeded[accepted] = True
            searching &= round_success
            c3 *= 10.0

        q_adv = last_success if back_index is None else last_success.gather(1, back_index)
        # Reordering the softmax reduction can change rounding. Failed and
        # boundary rows must retain the exact original confidence vector.
        q_adv = torch.where(ever_succeeded[:, None], q_adv, q_original)
        q_orig = q_original
        y_orig = y_original
        # Mix successful perturbations only, preserving exact clean fallbacks.
        changed = (q_adv != q_orig).any(dim=1)
        if self.randomize_mix > 0:
            q_adv[changed] = self._mix_uniform(q_adv[changed], self.randomize_mix)
        # Uniform mixing can introduce numerical ties. Always preserve top-1.
        valid = q_adv.argmax(dim=1) == y_orig
        q_adv = torch.where(valid[:, None], q_adv, q_orig)
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

        # Equation 23: crossing can overshoot the boundary, so reject a candidate
        # whose membership probability is no closer to 0.5 than the clean one.
        clean_features = q_orig.sort(dim=1).values if self.use_sorted else q_orig
        adv_features = q_adv.sort(dim=1).values if self.use_sorted else q_adv
        clean_distance = (self._membership_logits(clean_features).sigmoid() - 0.5).abs()
        adv_distance = (self._membership_logits(adv_features).sigmoid() - 0.5).abs()
        p = torch.zeros_like(l1)
        nz = (l1 > 1e-12) & (adv_distance < clean_distance)
        p[nz] = (self.budget_l1 / l1[nz]).clamp(max=1.0)
        u = self._one_time_uniform(x, p)
        return torch.where((u < p)[:, None], q_adv, q_orig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("MemGuard defense should not be used in training mode!")
        # Log probabilities satisfy the repository logits contract: softmax
        # reconstructs the defended confidence vector without a second softmax.
        probabilities = self.defend(x)
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
