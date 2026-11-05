import numpy as np
from src.data.helpers import MultiDatasets, SubsampledDataset
from src.utils.log import Loggable, MyLogger


class CalibrationDatasetManager(Loggable):
    """Holds out known non-members of the target from the attacker's share, to calibrate decision thresholds.

    The samples come from the attacker's share, which the target never trains on. The shadow datasets treat them
    like the auditing samples (in none of them offline, in exactly half online), so that they are scored the same way
    as the auditing non-members.
    """
    def __init__(self, attacker_datasets: MultiDatasets, n_samples: int, seed: int, logger: MyLogger = None):
        """Sample n_samples of the attacker's samples, the same ones whichever attack runs."""
        super().__init__(logger=logger)
        pool = attacker_datasets.get('all')
        if not 0 < n_samples < len(pool):
            raise ValueError(f'The calibration set must hold between 1 and {len(pool) - 1} of the attacker\'s '
                             f'{len(pool)} samples, got {n_samples}.')
        # Its own random stream: the shadow datasets draw from the same pool with the same seed.
        rng = np.random.default_rng(seed=[seed, 1])
        self.ids = rng.choice(pool.get_indices(mode='original'), n_samples, replace=False).tolist()
        self.dataset = SubsampledDataset(pool, self.ids, strict=True)
        self.logger.print_it(f'Held out {n_samples} of the attacker\'s samples as known non-members, to calibrate '
                             f'decision thresholds.')

    def get(self):
        """Return the calibration samples, with their class labels."""
        return self.dataset

    def get_ids(self):
        """Return the calibration samples' original indices, in the order get() yields them."""
        return list(self.ids)
