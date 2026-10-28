from typing import List, Optional, Union
import torch
from src.data import get_dataset_mean_std
from src.utils.configs import DefenderConfigs, LdlDefenseConfigs
from src.mia.defenses.base import BaseDefender, DefendedModel



class LdlDefender(BaseDefender):
    """LDL: answers each query with the logits averaged over noisy copies of it, and leaves training unchanged.

    Paper: "LDL: A Defense for Label-Based Membership Inference Attacks",
    https://arxiv.org/abs/2212.01688. Documentation: docs/defenses/ldl.md.
    """
    inference_only = True

    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select LDL and keep its settings."""
        assert isinstance(defender_configs.defense, LdlDefenseConfigs), f"LdlDefender can only be used with LdlDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'ldl_defender'
        self.ldl_configs = defender_configs.defense
        # Caps the samples per victim pass: inference without gradients fits wherever a training batch does.
        self.max_batch_size = defender_configs.train.batch_size

    def train_model(self, train_configs, return_stats: bool = False):
        """Train normally, since LDL only changes how the trained model answers queries."""
        self.logger.print_it("LdlDefender: No training required for LDL defense, training with standard procedure.")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        """Wrap the trained model with LdlModel, which adds noise in pixel units to normalized images."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('LdlDefender: building defense layer...')
        # Image inputs are normalized per channel; tabular ones (no image size) are not.
        input_std = None if self.dataset_configs.im_size is None else get_dataset_mean_std(self.dataset_configs.name)[1]
        self.defended_model = LdlModel(model=self.trained_model,
                                n_queries=self.ldl_configs.n_queries,
                                noise_type=self.ldl_configs.noise_type,
                                noise_scale=self.ldl_configs.noise_scale,
                                flip_prob=self.ldl_configs.flip_prob,
                                input_std=input_std,
                                max_batch_size=self.max_batch_size
                                ).to(device)
        self.logger.print_it('LdlDefender: finished building defense layer!')
        return self.defended_model


class LdlModel(DefendedModel):
    """LDL's defense layer: answers each query with the logits averaged over noisy copies of it."""
    def __init__(self, model: torch.nn.Module, n_queries: int = 100, noise_type: str = 'normal', noise_scale: float = 0.02,
                 flip_prob: float = 0.0166, input_std: Optional[List[float]] = None, max_batch_size: int = 256):
        """Wrap the model; the noise type and scale, and the input normalization, decide the noise."""
        super().__init__(model)
        self.n_queries = n_queries
        assert noise_type in ['normal', 'bernoulli'], f"noise_type must be one of ['normal', 'bernoulli'], got {noise_type}!"
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.flip_prob = flip_prob
        # Per-channel std of the input normalization (None for unnormalized inputs), to keep noise_scale in pixel units.
        self.register_buffer('input_std', None if input_std is None else torch.tensor(input_std).view(1, -1, 1, 1), persistent=False)
        self.max_batch_size = max_batch_size

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits averaged over n_queries noisy copies of each input."""
        # Gradients are tracked only when the input requires them, e.g. for an adaptive attack averaging them over the noise.
        # Otherwise the n_queries passes would keep their activations for a backward pass that never comes.
        with torch.set_grad_enabled(torch.is_grad_enabled() and x.requires_grad):
            # Noisy copies go through the victim together, as in the authors' code, up to max_batch_size samples per pass.
            copies_per_pass = max(1, self.max_batch_size // max(1, x.shape[0]))
            logits_sum = 0
            for start in range(0, self.n_queries, copies_per_pass):
                n_copies = min(copies_per_pass, self.n_queries - start)
                copies = self.add_noise(x.repeat(n_copies, *[1] * (x.dim() - 1)))
                logits_sum = logits_sum + self.model(copies).view(n_copies, x.shape[0], -1).sum(dim=0)
            return logits_sum / self.n_queries

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Return a noisy copy of the batch: Gaussian noise, or bit flips."""
        if self.noise_type == "normal":
            # noise_scale is a std in pixel units, as in the authors' code: dividing by the normalization std moves it
            # to the inputs' units.
            noise = torch.randn_like(x) * self.noise_scale
            if self.input_std is not None:
                noise = noise / self.input_std
            x_noisy = x + noise
        elif self.noise_type == "bernoulli":
            # Flip each feature with probability flip_prob, as the paper does for binary features (Sec. 6.3). Flipping as x -> 1 - x
            # also covers non-binary inputs, e.g. attack queries between binary points, which then get the same noise as nearby binary points.
            flip = torch.rand_like(x) < self.flip_prob
            x_noisy = torch.where(flip, 1 - x, x)
        return x_noisy
