"""Noise robustness: a label-only attack on how often the target keeps a sample's label under random noise."""
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.attacks.label_only.helpers.pixels import PixelInputModel, pixel_normalization
from src.mia.attacks.label_only.helpers.threshold_samples import balanced_threshold_samples
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms

# Shadow samples of each kind, members and non-members of the shadow model, on which the noise level and the threshold
# are chosen. The authors' code uses 2,500 of each. For the supervised boundary attack, 500 of each chose as good a
# threshold in a test on CIFAR-10 (docs/attacks/sba.md); here each sample costs n_queries queries per noise level.
# Unlike that attack's threshold, these samples also choose the noise level, which sets the scores: if the AUC is lower
# than expected, try a larger value. With less data outside the shadow dataset, both sides take what it has.
THRESHOLD_SAMPLES_PER_SIDE = 500
# Noisy copies sent to the model in one call, fewer for large inputs so that a call holds at most QUERY_BATCH_VALUES
# input values. Every copy has its own noise, so this sets the speed only.
QUERY_BATCH_SIZE = 500
QUERY_BATCH_VALUES = 2 ** 24


class NoiseRobustnessMIA(BaseMIA):
    """Label-only attack: members keep their label under random input noise more often than non-members.

    Paper: "Label-Only Membership Inference Attacks", https://arxiv.org/abs/2007.14321.
    Documentation: docs/attacks/noise_robustness.md.
    Images are noised as pixels in [0, 1], binary features are flipped, and other tabular features get Gaussian noise.
    """
    def __init__(self,
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Sample the shadow dataset and build the shadow model."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Noise Robustness MIA!")
        assert self.shadow_configs.mode == 'offline', (
            f"Noise Robustness MIA attacker uses an offline shadow model, got mode={self.shadow_configs.mode}.")
        # One shadow model, as in the paper; the configs set it whatever --n_shadows.
        assert self.shadow_configs.n_shadow_datasets == 1, (
            f"Noise Robustness MIA trains one shadow model, set by the configs, "
            f"got {self.shadow_configs.n_shadow_datasets}.")
        # Found on the shadow model before the first samples are scored.
        self.attack_threshold, self.best_sigma = None, None
        # Images are noised as pixels in [0, 1], the model normalising them, as in the authors' code.
        self.normalization = pixel_normalization(self.base_dataset_configs)
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Noise Robustness MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Noise Robustness MIA attacker: definition of noise robustness model...')
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        """Train the shadow model."""
        self.logger.print_it('Noise Robustness MIA attacker: training the shadow model. This will take a while...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f'Noise Robustness MIA attacker: done training the shadow model. '
                             f'It took {h}:{m:02d}:{s:02d}.')

    def find_optimal_sigma_and_threshold(self, device: Union[torch.device, str] = 'cpu'):
        """Choose the noise level and the threshold that best separate the shadow model's members and non-members."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        start = time.time()
        sigmas = self.attack_configs.sigmas
        sigmas = [float(sigmas)] if isinstance(sigmas, (float, int)) else [float(sigma) for sigma in sigmas]
        kind = 'flip probabilities' if self.attack_configs.noise == 'bit_flip' else 'Gaussian noise levels'
        shadow_model = self.shadow_manager.shadow_models.get(index=0).to(device)
        # The same samples for every level, so that the levels are compared on them.
        members, non_members, n_per_side = balanced_threshold_samples(
            self.shadow_manager, THRESHOLD_SAMPLES_PER_SIDE, self.seed)
        self.logger.print_it(f"Noise Robustness MIA attacker: trying {kind} {sigmas} on {n_per_side} members and "
                             f"{n_per_side} non-members of the shadow model...")
        mia_labels = np.r_[np.ones(len(members)), np.zeros(len(non_members))]
        best_accuracy = -np.inf
        for sigma in sigmas:
            scores = np.concatenate([self.dataset_robustness(shadow_model, dataset, sigma, device)
                                     for dataset in (members, non_members)])
            threshold, accuracy = self.best_threshold(scores, mia_labels)
            self.logger.print_it(f"Noise Robustness MIA attacker: level {sigma}: attack accuracy {accuracy:.4f} on "
                                 f"the shadow model at threshold {threshold:.4f}.")
            if accuracy > best_accuracy:
                self.best_sigma, self.attack_threshold, best_accuracy = sigma, threshold, accuracy
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it(f"Noise Robustness MIA attacker: best level {self.best_sigma}, with accuracy "
                             f"{best_accuracy:.4f} at threshold {self.attack_threshold:.4f}. "
                             f"It took {h}:{m:02d}:{s:02d}.")

    @staticmethod
    def best_threshold(scores: np.ndarray, mia_labels: np.ndarray):
        """Return the threshold with the best accuracy for 'member if score > threshold', and that accuracy."""
        _, _, thresholds = roc_curve(mia_labels, scores)
        accuracies = [accuracy_score(mia_labels, scores > threshold) for threshold in thresholds]
        best = int(np.argmax(accuracies))
        return thresholds[best], accuracies[best]

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample by how often the target keeps its label under noise, and threshold it."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        if self.attack_threshold is None:
            self.logger.print_it('Noise Robustness MIA attacker: finding optimal threshold on shadow dataset...')
            self.find_optimal_sigma_and_threshold(device=device)
        self.logger.print_it('Noise Robustness MIA attacker: measuring attack effectiveness...')
        start = time.time()
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f'Noise Robustness MIA attacker: done measuring attack effectiveness. '
                             f'It took {h}:{m:02d}:{s:02d}.')
        self.logger.print_it(f"Noise Robustness MIA attacker: threshold {self.attack_threshold:.4f}, "
                             f"best level {self.best_sigma}.")
        return [MembershipScores(scores, decisions=decisions)]

    @torch.no_grad()
    def dataset_robustness(self, model: torch.nn.Module, dataset, sigma: float,
                           device: torch.device) -> np.ndarray:
        """Return the robustness of every sample of a dataset, at one noise level."""
        model = model.to(device)
        # One sample at a time, for the progress line: its copies are what is batched.
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        scores = []
        start = time.time()
        for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
            h, m, s = convert_to_hms(time.time() - start)
            self.logger.print_it_same_line(f"Noise Robustness MIA attacker: level {sigma}, sample {batch_id + 1}/"
                                           f"{len(dataloader)} [{h:02d}:{m:02d}:{s:02d}]...", console_only=True)
            scores.append(self._features(model=model, inputs=batch_x.to(device), true_labels=batch_y.to(device),
                                         sigma=sigma))
        self.logger.set_logger_newline(console_only=True)
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, true_labels: torch.Tensor,
                  sigma: float) -> torch.Tensor:
        """Return each sample's share of noisy copies the model labels correctly, 0 when it mislabels the sample."""
        model.eval()
        if self.normalization is not None:
            mean, std = (statistic.to(inputs.device) for statistic in self.normalization)
            model, inputs = PixelInputModel(model, mean, std), (inputs * std + mean).clamp(0, 1)
        correct = self.label_pred(model, inputs) == true_labels
        n_queries = self.attack_configs.n_queries
        per_call = max(1, min(QUERY_BATCH_SIZE, QUERY_BATCH_VALUES // inputs[0].numel()))
        robustness = torch.zeros(len(inputs))
        for i in torch.nonzero(correct).flatten().tolist():
            kept = 0
            for first in range(0, n_queries, per_call):
                copies = self.noisy_copies(inputs[i], min(per_call, n_queries - first), sigma)
                kept += int((self.label_pred(model, copies) == true_labels[i]).sum())
            robustness[i] = kept / n_queries
        return robustness

    def noisy_copies(self, sample: torch.Tensor, n_copies: int, sigma: float) -> torch.Tensor:
        """Return noisy copies of one sample: bits flipped with probability sigma, or Gaussian noise of std sigma."""
        copies = sample.unsqueeze(0).expand(n_copies, *sample.shape)
        if self.attack_configs.noise == 'bit_flip':
            return torch.where(torch.rand(copies.shape, device=sample.device) < sigma, 1 - copies, copies)
        noisy = copies + sigma * torch.randn(copies.shape, device=sample.device)
        # Clipped to valid pixels, as in the authors' code. Tabular features are left unbounded.
        return noisy.clamp(0, 1) if self.normalization is not None else noisy

    @torch.no_grad()
    def label_pred(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Return the model's predicted labels."""
        model.eval()
        return torch.argmax(model(inputs), dim=1)

    @torch.no_grad()
    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset,
                      device: Union[torch.device, str] = 'cpu'):
        """Return the target's robustness score of every sample of a dataset, and the attack's decisions."""
        assert self.attack_threshold is not None, "Call find_optimal_sigma_and_threshold before infer_dataset."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        scores = self.dataset_robustness(model, dataset, self.best_sigma, device)
        # Members score strictly above the threshold, as in the paper's code.
        return scores, (scores > self.attack_threshold).astype(np.int64)

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor,
                    device: Union[torch.device, str] = 'cpu'):
        """Return the target's robustness score of every sample of a batch, and the attack's decisions."""
        assert self.attack_threshold is not None, "Call find_optimal_sigma_and_threshold(...) before infer_batch(...)."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        scores = self._features(model=model, inputs=batch_x.to(device), true_labels=batch_y.to(device),
                                sigma=self.best_sigma).numpy()
        return scores, (scores > self.attack_threshold).astype(np.int64)

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                     device: Union[torch.device, str] = 'cpu'):
        """Return the target's robustness score of one sample, and the attack's decision."""
        scores, decisions = self.infer_batch(model, x.unsqueeze(0), y.unsqueeze(0), device)
        return scores.item(), decisions.item()
