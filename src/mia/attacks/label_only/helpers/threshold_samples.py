"""The shadow model's members and non-members on which label-only attacks choose their threshold."""
from typing import Optional

import numpy as np

from src.data.helpers import SubsampledDataset
from src.mia.helpers.shadow_manager import ShadowManager


def balanced_threshold_samples(shadow_manager: ShadowManager, per_side: Optional[int], seed: int):
    """Return as many members as non-members of shadow model 0, at most `per_side` of each (None: no cap).

    Both sides stay the same size, as in the paper's equal-sized sets, so with less data outside the shadow dataset
    both take what it has. Returns the members, the non-members and their number.
    """
    members = shadow_manager.get_dataset(index=0, labels='original')
    n_outside = len(shadow_manager.shadow_data.indices_outside_shadow_dataset(index=0))
    n_per_side = min(len(members), n_outside) if per_side is None else min(per_side, len(members), n_outside)
    positions = np.random.default_rng(seed).choice(len(members), n_per_side, replace=False)
    members = SubsampledDataset.from_positions(members, sorted(positions.tolist()))
    non_members = shadow_manager.sample_outside_shadow_dataset(index=0, num_data=n_per_side, labels='original')
    return members, non_members, n_per_side
