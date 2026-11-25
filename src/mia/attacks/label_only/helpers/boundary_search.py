"""The boundary search of the boundary attacks: HopSkipJump or QEBA, on pixels for images."""
from typing import Optional, Tuple

import torch

from src.mia.attacks.label_only.helpers.hop_skip_jump import HopSkipJump
from src.mia.attacks.label_only.helpers.pixels import PixelInputModel
from src.mia.attacks.label_only.helpers.qeba import Qeba


def boundary_distances(model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, boundary_configs,
                       normalization: Optional[Tuple[torch.Tensor, torch.Tensor]],
                       qeba_basis: Optional[torch.Tensor] = None, logger=None) -> torch.Tensor:
    """Return each input's distance to the model's decision boundary around its label, shaped (N,).

    An input the model does not give its label has distance 0. Images, normalised as the datasets hold them, are
    searched as pixels in [0, 1] when `normalization` (mean and std, shaped (1, C, 1, 1)) is given, so every query is
    a valid image and the distances are in pixels. Tabular inputs are searched as they are, with no bounds.
    """
    model.eval()
    pixels = normalization is not None
    if pixels:
        mean, std = (statistic.to(inputs.device) for statistic in normalization)
        model, inputs = PixelInputModel(model, mean, std), (inputs * std + mean).clamp(0, 1)
    if boundary_configs.mode == 'hop_skip_jump':
        search = HopSkipJump(norm=boundary_configs.norm, max_queries=boundary_configs.n_queries, clamping=pixels,
                             logger=logger)
    elif boundary_configs.mode == 'qeba':
        search = Qeba(norm=boundary_configs.norm, max_queries=boundary_configs.n_queries, clamping=pixels,
                      logger=logger, variant=boundary_configs.qeba.reduction_mode,
                      reduction_factor=boundary_configs.qeba.reduction_factor, basis=qeba_basis)
    else:
        raise ValueError(f"Boundary search {boundary_configs.mode} not recognized.")
    _, distances, _ = search.run(model=model, inputs=inputs, labels=labels, targeted=False, device=inputs.device)
    return distances
