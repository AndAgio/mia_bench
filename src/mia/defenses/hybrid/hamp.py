from typing import List, Optional, Union, Callable
import torch
from src.data import get_dataset_mean_std
from src.trainer.train_manager import TrainManager
from src.utils.configs import DefenderConfigs, HampDefenseConfigs
from src.utils.variables import HAMP_PAPER_PARAMETERS, HAMP_FALLBACK_DATASET
from src.mia.defenses.base import BaseDefender


class HampDefender(BaseDefender):
    """
    HAMP defense from "Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less
    Confident Prediction" (Chen and Pattabiraman, NDSS 2024, https://arxiv.org/pdf/2307.01610).

    Where the paper is silent or the benchmark needs more, this implementation chooses as follows.

    - KL scaling: the authors' code averages the KL divergence over the classes as well as the samples (F.kl_div's default
      reduction), while Eq. 6 sums it over the classes. This implementation follows the code, with which the paper's
      gamma and alpha (Table III, the defaults here) were tuned. Following Eq. 6 instead would make the KL term
      num_classes times larger, weakening the entropy regularizer by as much and changing the step size.

    - Gamma and alpha: unset, they take the paper's values for the dataset (Table III, HAMP_PAPER_PARAMETERS), and
      CIFAR-10's for datasets the paper does not evaluate. Other values are logged as a warning, since results with them
      are not the paper's configuration.

    - Random samples for the testing-time defense: for image datasets, pixels uniform in [0, 1] and then normalized like
      the model inputs (the paper samples pixels in [0, 255], Sec. IV-A). For the binary Purchase and Texas features,
      0 or 1 with equal probability, as in the paper (Sec. III-D). News has no paper value: its TF-IDF features lie in
      [0, 1], so they are drawn uniformly from [0, 1].

    - Soft labels for gamma close to 1: Eq. 4 asks for an entropy of at least gamma times the maximum with the true class
      most likely, which gamma = 1 cannot satisfy. The search keeps the true class most likely and says so in the log.

    - Validation and test loss: the cross-entropy on the true labels, as in the authors' code and the benchmark's reports.
      The training loss is the HAMP objective (Eq. 7) on the soft labels, so the two are not comparable.
    """
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, HampDefenseConfigs), f"Hamp Defender can only be used with HampDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'hamp_defender'
        self.hamp_configs = defender_configs.defense

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it(f"Hamp Defender: selected mode {self.hamp_configs.mode}")
        if self.hamp_configs.mode in ['train_only', 'full']:
            self.logger.print_it(f'Hamp Defender: training defender model with alpha {self.hamp_configs.alpha} and gamma {self.hamp_configs.gamma}...')
            self.warn_if_not_paper_parameters()
            train_manager = HampTrainManager(train_configs=train_configs,
                                            name=self.name,
                                            logger=self.logger,
                                            hamp_configs=self.hamp_configs,
                                            num_classes=self.dataset_configs.num_classes)
        elif self.hamp_configs.mode == 'test_only':
            self.logger.print_it("Hamp Defender: test_only mode selected, training with standard loss and labels.")
            train_manager = TrainManager(train_configs=train_configs,
                                        name=self.name,
                                        logger=self.logger)
        else:
            raise ValueError(f"Hamp Defender: Unknown mode {self.hamp_configs.mode}!")
        train_manager.initialize_train(dataset=self.dataset,
                                        model=self.untrained_model,
                                        configs=train_configs)
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

    def warn_if_not_paper_parameters(self):
        dataset = self.dataset_configs.name
        chosen = (self.hamp_configs.gamma, self.hamp_configs.alpha)
        if dataset not in HAMP_PAPER_PARAMETERS:
            self.logger.print_it(f"Hamp Defender [WARNING]: the paper has no gamma and alpha for {dataset}; unset values take {HAMP_FALLBACK_DATASET}'s {HAMP_PAPER_PARAMETERS[HAMP_FALLBACK_DATASET]}. Using (gamma, alpha) = {chosen}.")
        elif chosen != HAMP_PAPER_PARAMETERS[dataset]:
            self.logger.print_it(f"Hamp Defender [WARNING]: (gamma, alpha) = {chosen} differ from the paper's {HAMP_PAPER_PARAMETERS[dataset]} for {dataset} (Table III); results are not the paper's configuration.")

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        if self.hamp_configs.mode in ['test_only', 'full']:
            self.logger.print_it(f'Hamp Defender: adding defense layer on outputs using entropy trick...')
            if isinstance(device, str):
                device = self.get_device(dev_str=device)
            # Image inputs are normalized per channel; tabular ones (no image size) are not.
            input_mean, input_std = (None, None) if self.dataset_configs.im_size is None else get_dataset_mean_std(self.dataset_configs.name)
            self.defended_model = HampTestWrapper(model=self.trained_model,
                                                binary_inputs=self.dataset_configs.name in ['purchase', 'texas'],
                                                input_mean=input_mean,
                                                input_std=input_std).to(device)
        elif self.hamp_configs.mode == 'train_only':
            self.logger.print_it("Hamp Defender: train_only mode selected, defend_model does not modify the model at inference time.")
            self.defended_model = self.trained_model
        else:
            raise ValueError(f"Hamp Defender: Unknown mode {self.hamp_configs.mode}!")
        return self.defended_model
    


class HampTrainManager(TrainManager):
    def __init__(self, train_configs, name: str, logger=None, hamp_configs: HampDefenseConfigs = None, num_classes: int = None):
        super().__init__(train_configs=train_configs, name=name, logger=logger)
        self.hamp_configs = hamp_configs
        assert num_classes is not None, "HampTrainManager: num_classes must be provided."
        self.num_classes = num_classes
        # The soft labels depend only on gamma and the number of classes, so they are built once (Algorithm 1, line 2).
        self.soft_label_top1, self.soft_label_other = self.get_soft_label_template()

    def setup_loss(self, loss: Union[str, Callable]):
        self.criterion = HampLoss(gamma=self.hamp_configs.gamma, alpha=self.hamp_configs.alpha)

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Train on the high-entropy soft labels; the rest is the standard step.
        super().train_step(inputs, self.get_soft_labels(original_labels=targets), batch_idx=batch_idx, total_batches=total_batches)

    @torch.no_grad()
    def get_soft_label_template(self):
        # Probabilities of the true class and of each other class (Eq. 5). As in the authors' code, the true class gives
        # up 0.001 at a time until the entropy reaches gamma times the entropy of the uniform label (Eq. 4). The search
        # stops while the true class is still the most likely, which a gamma close to 1 could otherwise step past forever.
        uniform_preds = torch.ones(self.num_classes) / self.num_classes
        logp = torch.log(uniform_preds)
        highest_entropy = torch.sum( -uniform_preds * logp , dim=0 )
        entropy_threshold = highest_entropy*self.hamp_configs.gamma
        reduced_prob = 0.001
        true_target = 0
        preds = torch.zeros(self.num_classes)
        preds[true_target]  = 1.
        while preds[true_target] - reduced_prob > preds[true_target+1] + reduced_prob/(self.num_classes-1):
            preds[true_target] -= reduced_prob
            preds[:true_target] += reduced_prob/(self.num_classes-1)
            preds[true_target+1:] += reduced_prob/(self.num_classes-1)
            if(torch.sum(-preds * torch.log(preds), dim=0) >= entropy_threshold):
                break
        else:
            self.logger.print_it(f'HampTrainManager: no soft label reaches the entropy of gamma={self.hamp_configs.gamma} with the true class most likely, using p={preds[true_target].item():.4f}.')
        return preds[true_target].item(), preds[true_target+1].item()

    @torch.no_grad()
    def get_soft_labels(self, original_labels):
        new_soft_label = torch.full((original_labels.shape[0], self.num_classes), self.soft_label_other, device=original_labels.device)
        return new_soft_label.scatter_(1, original_labels.view(-1, 1), self.soft_label_top1)


class HampLoss(torch.nn.Module):
    # Like nn.CrossEntropyLoss, starts with reduction='none' (per-sample losses, needed by ESAM) and is switched
    # to 'mean' by TrainManager.setup_optimizer for plain optimizers.
    def __init__(self, gamma, alpha, reduction: str = 'none'):
        super(HampLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, targets):
        # Only the train split carries soft labels. Val/test keep integer class ids and get the cross-entropy, as in the
        # authors' code and the benchmark's reports.
        if targets.dim() == 1:
            loss = torch.nn.functional.cross_entropy(outputs, targets.long(), reduction='none')
        else:
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            entropy = -torch.sum(log_probs.exp() * log_probs, dim=1)
            # Eq. 7: the KL divergence to the soft labels minus alpha times the entropy. The KL is averaged over the classes,
            # as in the authors' code (F.kl_div's default 'mean'), not summed as in Eq. 6: their alpha values assume it.
            loss = torch.nn.functional.kl_div(log_probs, targets, reduction='none').mean(dim=1) - self.alpha * entropy
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class HampTestWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, binary_inputs: bool = False, input_mean: Optional[List[float]] = None, input_std: Optional[List[float]] = None):
        super(HampTestWrapper, self).__init__()
        self.model = model
        self.binary_inputs = binary_inputs
        # Per-channel normalization of image inputs. Buffers follow .to(device) and stay out of the state dict.
        self.register_buffer('input_mean', None if input_mean is None else torch.tensor(input_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer('input_std', None if input_std is None else torch.tensor(input_std).view(-1, 1, 1), persistent=False)

    def random_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # Binary features are 0 or 1 with equal probability; other inputs are uniform in [0, 1], normalized if images.
        if self.binary_inputs:
            return torch.randint_like(inputs, 2)
        random_inputs = torch.rand_like(inputs)
        if self.input_mean is not None:
            random_inputs = (random_inputs - self.input_mean) / self.input_std
        return random_inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        random_inputs = self.random_inputs(inputs)
        with torch.no_grad():
            original_outputs = self.model(inputs)
            random_outputs = self.model(random_inputs)
            # Each output takes the sorted scores of a random sample in the order of its own scores, so its ranking of the
            # classes, and with it every top-k label, is unchanged (Algorithm 1, line 13).
            new_outputs = torch.zeros_like(original_outputs).scatter_(1, torch.argsort(original_outputs, dim=1), torch.sort(random_outputs, dim=1).values)
        return new_outputs