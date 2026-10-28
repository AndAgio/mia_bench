import hashlib
from typing import List, Union

import torch
from src.data import get_defender_datas
from src.models import get_model
from src.trainer.train_manager import TrainManager
from src.utils.configs import TrainConfigs, DefenderConfigs
from src.utils.log import Loggable, get_logger_from_configs


class BaseDefender(Loggable):
    """Base class of the defenses: loads the defender's data and model, trains it and applies the defense."""
    inference_only = False
    """Whether the defense acts only at inference. If so, the trained model and its internal features are undefended."""
    def __init__(self, defender_configs: DefenderConfigs):
        """Load the defender's datasets and build the untrained model from the configs."""
        self.name = 'base_defender'
        logger=get_logger_from_configs(defender_configs.log)
        super().__init__(logger=logger)
        self.defender_hash = defender_configs.hash
        self.dataset_configs = defender_configs.dataset.base
        self.model_configs = defender_configs.model
        self.dataset = get_defender_datas(dataset=self.dataset_configs.name,
                                datasets_folder=self.dataset_configs.data_folder,
                                def_split=self.dataset_configs.def_split,
                                att_split=self.dataset_configs.att_split,
                                seed=self.dataset_configs.seed,
                                logger=self.logger)
        self.untrained_model = get_model(model_name=self.model_configs.model_name,
                            im_channels=self.model_configs.im_channels,
                            num_classes=self.model_configs.num_classes,
                            im_size=self.model_configs.im_size,
                            logger=self.logger)

    def get_dataset(self):
        """Return the defender's datasets (train, validation and test splits)."""
        return self.dataset
    
    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        """Train the model on the train split and return the best checkpoint, with the stats if return_stats."""
        train_manager = TrainManager(train_configs=train_configs,
                                    name=self.name,
                                    logger=self.logger)
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        configs=train_configs)
        # TODO: Refactor return of stats for victim and train manager.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/21
        # assignees: AndAgio
        if return_stats:
            self.trained_model, train_stats = train_manager.train(return_best_model=True,
                                                        return_last_model=False,
                                                        return_stats=return_stats)
        else:
            self.trained_model = train_manager.train(return_best_model=True,
                                            return_last_model=False,
                                            return_stats=return_stats)
        if return_stats:
            return self.trained_model, train_stats
        else:
            return self.trained_model
    
    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        """Return the model that answers queries: the trained model, or a DefendedModel around it."""
        raise NotImplementedError("defend_model method must be implemented in defense classes!")
        
    def get_defended_model(self) -> torch.nn.Module:
        """Return the model built by defend_model."""
        assert hasattr(self, 'defended_model'), "Model has not been optimized for defense yet!"
        return self.defended_model
    
    def get_undefended_model(self) -> torch.nn.Module:
        """Return the trained model, without the inference-time defense."""
        assert hasattr(self, 'trained_model'), "Model has not been trained yet!"
        return self.trained_model
    
    def get_untrained_model(self) -> torch.nn.Module:
        """Return the model before training."""
        assert hasattr(self, 'untrained_model'), "Model has not been initialized yet!"
        return self.untrained_model
    
    @staticmethod
    def get_device(dev_str: str = 'cpu'):
        """Map a device string to a torch device: a CUDA index, MPS when CUDA is missing, or CPU."""
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


class DefendedModel(torch.nn.Module):
    """A trained model with an inference-time defense: what the attacks query.

    Subclasses implement defend. The model starts in evaluation mode and refuses queries in training mode.
    """
    def __init__(self, model: torch.nn.Module):
        """Wrap the trained model, in evaluation mode."""
        super().__init__()
        self.model = model
        self.eval()

    def defend(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the defended logits of a batch of queries."""
        raise NotImplementedError('defend must be implemented in defended models!')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Answer a batch of queries with the defended logits."""
        if self.training:
            raise RuntimeError(f'{type(self).__name__} only answers queries: call .eval() before querying it.')
        return self.defend(inputs)

    def feature(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the wrapped model's internal features, which the inference-time defense does not change."""
        return self.model.feature(inputs)

    @staticmethod
    def probabilities_to_logits(probabilities: torch.Tensor) -> torch.Tensor:
        """Return logits whose softmax gives back the probabilities exactly: their logarithm."""
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()

    @staticmethod
    def query_digests(inputs: torch.Tensor, quantization: float) -> List[bytes]:
        """Hash each query, rounded to multiples of quantization, to seed its randomness.

        A repeated query gets the same randomness, whatever batch it comes in and whatever the global RNG state.
        Queries closer to each other than the quantization width usually share it.
        """
        # Transfer first, then cast: MPS can attempt the dtype conversion on the source device when
        # .to(device='cpu', dtype=float64) is combined.
        quantized = (inputs.detach().cpu().to(dtype=torch.float64) / quantization).round().to(torch.int64)
        digests = []
        for row in quantized:
            digest = hashlib.sha256()
            digest.update(str(tuple(row.shape)).encode('ascii'))
            digest.update(row.numpy().astype('<i8', copy=False).tobytes())
            digests.append(digest.digest())
        return digests
