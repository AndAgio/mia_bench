from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.utils.variables import BINARY_FEATURE_DATASETS
from src.mia.attacks.label_only.helpers.boundary_search import boundary_distances
from src.mia.attacks.label_only.helpers.pixels import pixel_normalization
from src.mia.attacks.label_only.helpers.qeba import block_basis


class UnsupervisedBoundaryMIA(BaseMIA):
    """Label-only attack on the distance to the decision boundary, with a threshold set on random inputs.

    Paper: "Membership Leakage in Label-Only Exposures" (the boundary attack), https://arxiv.org/abs/2007.15528.
    Documentation: docs/attacks/uba.md.
    The distance is found with HopSkipJump or QEBA. Images are attacked as pixels, so every query is a valid image.
    The attacker needs no data: the threshold comes from random inputs, uniform over the input space, as the paper's
    random images are. The paper covers images only, and on tabular data random records can sit at other distances
    from the boundary than real records, so the decisions can be poor there (docs/attacks/uba.md).
    """
    requires_auxiliary_data = False

    def __init__(self,
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Read the input shape and the pixel normalisation, and build the basis of QEBA's custom mode."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Unsupervised Boundary MIA!")
        self.attack_threshold = None
        # Only the input shape is needed; no candidate values or labels are used.
        self.input_shape = tuple(self.audit_manager.get(labels='original')[0][0].shape)
        # Images are attacked as pixels in [0, 1], the model normalising them, so that every query is a valid image
        # and the distances are in pixels, as in the authors' code. Tabular inputs are attacked as they are.
        self.normalization = pixel_normalization(self.base_dataset_configs)
        self.qeba_basis = None
        if (self.attack_configs.boundary.mode == 'qeba'
                and self.attack_configs.boundary.qeba.reduction_mode == 'custom'):
            self.qeba_basis = block_basis(self.input_shape, self.attack_configs.boundary.qeba.reduction_factor)

    def optimize(self, train_config: TrainConfigs):
        """Set the threshold on random inputs; there is no model to train."""
        self.logger.print_it('Unsupervised Boundary MIA attacker: finding threshold via quantile on random inputs...')
        start = time.time()
        self.find_threshold_via_quantile(device=train_config.device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Unsupervised Boundary MIA attacker: Done finding threshold via quantile. It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def random_calibration_inputs(self):
        """Yield the random inputs the threshold is set on, uniform over the input space, as the datasets hold inputs.

        Images have uniform random pixels, normalised like the dataset's, as in the paper (Sec. 4.4). The paper has no
        tabular data: Purchase and Texas records get random bits, so that each one is a valid record, and other tabular
        data uniform values in [0, 1]. Records as sparse as the real ones would need the data this attacker lacks.
        """
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.attack_configs.n_calibration_samples):
            sample = torch.rand(self.input_shape, generator=generator)
            if self.normalization is not None:
                mean, std = (statistic[0] for statistic in self.normalization)
                yield (sample - mean) / std
            elif self.base_dataset_configs.name in BINARY_FEATURE_DATASETS:
                yield (sample < 0.5).float()
            else:
                yield sample

    def find_threshold_via_quantile(self, device: Union[torch.device, str] = 'cpu'):
        """Set the threshold at the configured quantile of the random inputs' boundary distances.

        A random input belongs to no class, so its distance is the one that changes the label the target predicts
        for it (the paper's Alg. 3).
        """
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        defended_model = self.defender_model.to(device).eval()
        all_feats = []
        start = time.time()
        with torch.no_grad():
            for index, sample in enumerate(self.random_calibration_inputs()):
                h, m, s = convert_to_hms(time.time() - start)
                self.logger.print_it_same_line(
                    f"Unsupervised Boundary MIA attacker: random input {index + 1}/"
                    f"{self.attack_configs.n_calibration_samples} [{h:02d}:{m:02d}:{s:02d}]...", console_only=True)
                batch_x = sample.unsqueeze(0).to(device)
                predicted = self.label_pred(defended_model, batch_x)
                scores = self._features(defended_model, batch_x, predicted)
                all_feats.append(scores.cpu().reshape(-1))
        self.logger.set_logger_newline(console_only=True)
        all_feats = torch.cat(all_feats).numpy()
        if not np.isfinite(all_feats).all():
            raise ValueError('Boundary calibration produced non-finite distances; increase the query budget.')
        # An ascending quantile q is the paper's "top t percentile" with t = 100 * (1 - q); the median is both.
        self.attack_threshold = float(np.quantile(all_feats, self.attack_configs.quantile))
        self.logger.print_it(f'Boundary threshold from random inputs: {self.attack_threshold:.4f}')

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample with the target's distance to its decision boundary, and threshold it."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Unsupervised Boundary MIA attacker: measuring attack effectiveness...')
        start = time.time()
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Unsupervised Boundary MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        return [MembershipScores(scores, decisions=decisions)]

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return each input's boundary distance around the given label.

        Audited samples come with their true label, so a sample the target misclassifies has distance 0, as in the
        paper's Alg. 2 and the authors' code.
        """
        return self.estimate_boundary_distance(model, inputs, targets)

    @torch.no_grad()
    def label_pred(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Return the labels the model predicts for a batch."""
        model.eval()
        logits = model(inputs)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def estimate_boundary_distance(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return each input's distance to the model's decision boundary around its label, with HopSkipJump or QEBA."""
        return boundary_distances(model, inputs, targets, self.attack_configs.boundary, self.normalization,
                                  qeba_basis=self.qeba_basis, logger=self.logger)

    @torch.no_grad()
    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        """Return each sample's boundary distance and whether it is above the threshold (None without one)."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        all_scores = []
        for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
            self.logger.print_it_same_line(f"Boundary Distance MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for inference. This may take a while...", console_only=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            boundary_distance_score = self._features(model=model, inputs=batch_x, targets=batch_y)
            all_scores.append(boundary_distance_score.cpu())
        self.logger.set_logger_newline(console_only=True)
        all_boundary_distance_scores = torch.cat(all_scores, dim=0).numpy()
        if self.attack_threshold is None:
            return all_boundary_distance_scores, None
        # Strictly above: the paper's Alg. 2 calls a sample a non-member when its distance is at most the threshold.
        all_decisions = (all_boundary_distance_scores > self.attack_threshold).astype(np.int64)
        return all_boundary_distance_scores, all_decisions

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        """Return the boundary distances of a batch and whether each is above the threshold (None without one)."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        boundary_distance_score = self._features(model=model, inputs=batch_x, targets=batch_y)
        if self.attack_threshold is None:
            return boundary_distance_score.cpu().numpy(), None
        decisions = (boundary_distance_score > self.attack_threshold).to(torch.int64)
        return boundary_distance_score.cpu().numpy(), decisions.cpu().numpy()

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        """Return one sample's boundary distance and whether it is above the threshold (None without one)."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        boundary_distance_score = self._features(model=model, inputs=x, targets=y)
        if self.attack_threshold is None:
            return boundary_distance_score.item(), None
        decision = (boundary_distance_score > self.attack_threshold).to(torch.int64)
        return boundary_distance_score.item(), decision.item()
