"""Utilities for handling victim models and datasets.

This module defines the `Victim` class which bundles dataset and model creation
from configuration dataclasses and provides a small convenience wrapper to run
training with `TrainManager`.
"""

from typing import Union
from src.data import get_dataset
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.utils.configs import ExperimentConfigs, TrainConfigs, VictimConfigs
from src.utils.log import Loggable, get_logger_from_configs


class Victim(Loggable):
    """Encapsulates a victim model and its dataset for experiments.

    The `Victim` class is a thin wrapper that constructs and stores a dataset
    and a model according to the provided `VictimConfigs`. It also provides
    convenience accessors and a training helper that uses `TrainManager`.

    Attributes
    ----------
    victim_hash : str
        Unique hash identifier sourced from the provided `VictimConfigs`.
    dataset_configs : DatasetConfigs
        Dataset configuration dataclass used to build the dataset.
    model_configs : ModelConfigs
        Model configuration dataclass used to instantiate the model.
    dataset : MultiDatasets
        Dataset object returned by `get_dataset`, containing 'train' and 'test' splits.
    model : torch.nn.Module
        Instantiated model returned by `get_model`.
    logger : SmartLogger | DumbLogger
        Logger instance created from `VictimConfigs.log`.

    Examples
    --------
    >>> victim = Victim(victim_configs)
    >>> dataset = victim.get_dataset()
    >>> model = victim.get_model()

    """
    def __init__(self, victim_configs: VictimConfigs):
        """Create a `Victim` instance from `VictimConfigs`.

        Parameters
        ----------
        victim_configs : VictimConfigs
            Configuration dataclass that contains dataset, model and logging settings.

        Notes
        -----
        The dataset and model are constructed eagerly during initialization
        using `get_dataset` and `get_model`. No training is performed here.
        """
        logger=get_logger_from_configs(victim_configs.log)
        super().__init__(logger=logger)
        self.victim_hash = victim_configs.hash
        self.dataset_configs = victim_configs.dataset
        self.model_configs = victim_configs.model
        self.dataset = get_dataset(dataset=self.dataset_configs.name,
                                datasets_folder=self.dataset_configs.data_folder,
                                augment=self.dataset_configs.data_augmentation,
                                logger=self.logger)
        self.model = get_model(model_name=self.model_configs.model_name,
                            im_channels=self.model_configs.im_channels,
                            num_classes=self.model_configs.num_classes,
                            im_size=self.model_configs.im_size,
                            logger=self.logger)

    def get_dataset(self):
        """Return the dataset object associated with this `Victim`.

        Returns
        -------
        MultiDatasets
            The dataset wrapper instance created during construction. The
            wrapper contains at least the 'train' and 'test' splits and
            provides access through `get('train')` / `get('test')` and
            exposes dataset metadata added by `add_info()`.

        Notes
        -----
        The returned object is the same instance stored on `self.dataset` and
        is not copied.
        """
        return self.dataset

    def get_model(self):
        """Return the model instance associated with this `Victim`.

        Returns
        -------
        torch.nn.Module
            The instantiated PyTorch model constructed from `VictimConfigs.model`.

        Notes
        -----
        This method does not move the model to any device; callers that need
        device placement should call `model.to(device)` themselves or rely on
        `TrainManager` to move the model when initializing training.
        """
        return self.model
    
    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        """Train the victim's model using the provided `TrainConfigs`.

        This convenience method constructs a `TrainManager` using
        `train_configs` and initializes it with the Victim's dataset and
        model. It then runs the full training loop and updates `self.model`
        with the trained (best) model.

        Parameters
        ----------
        train_configs : TrainConfigs
            Training configuration that controls optimizer, scheduler, DP
            settings, batch size, device and checkpoint folders.
        return_stats : bool, optional
            If True, return a tuple `(trained_model, train_stats)` where
            `train_stats` is a `TrainStats` object containing aggregate
            statistics for the training run. If False, only the trained model
            instance is returned. Default: False.

        Returns
        -------
        torch.nn.Module or (torch.nn.Module, TrainStats)
            The trained model, optionally accompanied by training statistics.

        Raises
        ------
        Any exception thrown by `TrainManager.initialize_train` or
        `TrainManager.train` (e.g., missing 'train' split, invalid configs)
        will propagate to the caller.

        Notes
        -----
        The method returns the *best* model according to checkpointing logic
        inside `TrainManager.train`. The in-memory `self.model` is updated
        with the returned model (or remains as-is if `TrainManager.train`
        returns the same model object).
        """
        train_manager = TrainManager(train_configs=train_configs,
                                    name='victim',
                                    logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.model,
                                        configs=train_configs)
        # TODO: Refactor return of stats for victim and train manager.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/21
        # assignees: AndAgio
        if return_stats:
            self.model, train_stats = train_manager.train(return_best_model=True,
                                                        return_last_model=False,
                                                        return_stats=return_stats)
        else:
            self.model = train_manager.train(return_best_model=True,
                                            return_last_model=False,
                                            return_stats=return_stats)
        if return_stats:
            return self.model, train_stats
        else:
            return self.model
