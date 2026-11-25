"""Attacking images as pixels in [0, 1], while the datasets hold them normalised."""
from typing import Optional, Tuple

import torch

from src.data import get_dataset_mean_std


def pixel_normalization(dataset_configs) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return the dataset's per-channel mean and std, shaped (1, C, 1, 1), or None for tabular data."""
    if dataset_configs.im_size is None:
        return None
    mean, std = get_dataset_mean_std(dataset_configs.name)
    return torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1)


class PixelInputModel(torch.nn.Module):
    """A model that takes images as pixels in [0, 1] and normalises them before calling the wrapped model."""
    def __init__(self, model: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor):
        """Wrap the model, with the dataset's per-channel mean and std, shaped (1, C, 1, 1)."""
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the wrapped model's output on the normalised pixels."""
        return self.model((pixels - self.mean) / self.std)


class PixelView:
    """A read-only view of an image dataset that returns each image as pixels in [0, 1]."""
    def __init__(self, dataset, mean: torch.Tensor, std: torch.Tensor):
        """Wrap the dataset, with the per-channel mean and std, shaped (C, 1, 1)."""
        self.dataset, self.mean, self.std = dataset, mean, std

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return image `index` as pixels in [0, 1]."""
        return (self.dataset[index][0] * self.std + self.mean).clamp(0, 1)
