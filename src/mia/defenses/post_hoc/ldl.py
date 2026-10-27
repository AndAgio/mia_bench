from typing import List, Optional, Union
import torch
from src.data import get_dataset_mean_std
from src.utils.configs import DefenderConfigs, LdlDefenseConfigs
from src.mia.defenses.base import BaseDefender



class LdlDefender(BaseDefender):
    """
    LDL defense from "LDL: A Defense for Label-Based Membership Inference Attacks" (Rajabi et al., AsiaCCS 2023,
    https://dl.acm.org/doi/pdf/10.1145/3579856.3582821). Use the arXiv version (https://arxiv.org/abs/2212.01688),
    which has the appendix.

    Where the paper is silent or the benchmark needs more, this implementation chooses as follows.

    - Output: the logits averaged over n_queries noisy copies of the input (Alg. 1), not only the label. The paper
      releases only their argmax (a label-only setting). Label-only attacks take the argmax, so they see exactly the
      paper's defense. Score-based attacks read the averaged logits, so against them LDL is a smoothing defense, which
      the paper does not evaluate: report those results as an extension. Returning the logits lets every attack run
      against LDL.

    - Noise: Gaussian for image datasets, and for news, whose TF-IDF features are continuous. noise_scale is its std in
      pixel units of [0, 1] images, as in the paper and the authors' code (0.02 by default, their value). The inputs here
      are normalized, so the noise is divided by the per-channel normalization std. News has no paper value, and its
      noise_scale is in TF-IDF units. Bit flips with probability flip_prob for the binary Purchase and Texas features, as
      in the paper (Sec. 6.3). Unset, flip_prob takes the paper's value for the dataset (Table 1).

    - Bit flips on non-binary inputs: a flipped feature becomes 1 - x, which is the paper's flip on 0/1 values. Label-only
      attacks query points between binary inputs (noise robustness, HopSkipJump). These then get the same noise as the
      binary points nearby, instead of a different noise that would make the attacked defense differ from the evaluated one.

    - Gradients: tracked only when the input requires them, so an adaptive attack can average input gradients over the
      noise. Plain queries keep no graph.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, LdlDefenseConfigs), f"LdlDefender can only be used with LDLDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'ldl_defender'
        self.ldl_configs = defender_configs.defense
        # Caps the samples per victim pass: inference without gradients fits wherever a training batch does.
        self.max_batch_size = defender_configs.train.batch_size

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it("LdlDefender: No training required for LDL defense, training with standard procedure.")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('LdlDefender: building defense layer...')
        # Image inputs are normalized per channel; tabular ones (no image size) are not.
        input_std = None if self.dataset_configs.im_size is None else get_dataset_mean_std(self.dataset_configs.name)[1]
        self.defended_model = LdlModel(victim=self.trained_model,
                                n_queries=self.ldl_configs.n_queries,
                                noise_type=self.ldl_configs.noise_type,
                                noise_scale=self.ldl_configs.noise_scale,
                                flip_prob=self.ldl_configs.flip_prob,
                                input_std=input_std,
                                max_batch_size=self.max_batch_size
                                ).to(device)
        print('LdlDefender: defense layer built, moving to device {}...'.format(device))
        self.logger.print_it('LdlDefender: finished building defense layer!')
        return self.defended_model

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


class LdlModel(torch.nn.Module):
    # Implementation of the LDL defense layer.
    def __init__(self, victim: torch.nn.Module, n_queries: int = 100, noise_type: str = 'normal', noise_scale: float = 0.02,
                 flip_prob: float = 0.0166, input_std: Optional[List[float]] = None, max_batch_size: int = 256):
        super().__init__()
        self.victim = victim.eval()
        self.n_queries = n_queries
        assert noise_type in ['normal', 'bernoulli'], f"noise_type must be one of ['normal', 'bernoulli'], got {noise_type}!"
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.flip_prob = flip_prob
        # Per-channel std of the input normalization (None for unnormalized inputs), to keep noise_scale in pixel units.
        self.register_buffer('input_std', None if input_std is None else torch.tensor(input_std).view(1, -1, 1, 1), persistent=False)
        self.max_batch_size = max_batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradients are tracked only when the input requires them, e.g. for an adaptive attack averaging them over the noise.
        # Otherwise the n_queries passes would keep their activations for a backward pass that never comes.
        with torch.set_grad_enabled(torch.is_grad_enabled() and x.requires_grad):
            # Noisy copies go through the victim together, as in the authors' code, up to max_batch_size samples per pass.
            copies_per_pass = max(1, self.max_batch_size // max(1, x.shape[0]))
            logits_sum = 0
            for start in range(0, self.n_queries, copies_per_pass):
                n_copies = min(copies_per_pass, self.n_queries - start)
                copies = self.add_noise(x.repeat(n_copies, *[1] * (x.dim() - 1)))
                logits_sum = logits_sum + self.victim(copies).view(n_copies, x.shape[0], -1).sum(dim=0)
            return logits_sum / self.n_queries

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "normal":
            # noise_scale is a std in pixel units, as in the paper: dividing by the normalization std moves it to the inputs' units.
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
