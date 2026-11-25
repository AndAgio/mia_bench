
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from src.data.helpers import MergedDataset, SubsampledDataset
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.mia.attacks.label_only.helpers.boundary_search import boundary_distances
from src.mia.attacks.label_only.helpers.qeba import block_basis, gradient_pca_basis, input_gradients
from src.mia.attacks.label_only.helpers.pixels import PixelInputModel, PixelView, pixel_normalization
from src.mia.attacks.label_only.helpers.threshold_samples import balanced_threshold_samples

# Shadow samples of each kind, members and non-members of the shadow model, on which the threshold is chosen. The
# authors' code tunes it on 2,500 of each; in a test on CIFAR-10, 500 of each chose as good a threshold at a fifth of
# the boundary searches. With less data outside the shadow dataset, both sides take what it has.
THRESHOLD_SAMPLES_PER_SIDE = 500
# Shadow-model gradients from which the sba_qeba-pca mode builds its subspace, on samples outside the shadow dataset:
# the QEBA authors' code fits its CIFAR basis on 8,000 gradients. With less data, the mode takes what there is.
QEBA_PCA_GRADIENTS = 8000


class SupervisedBoundaryMIA(BaseMIA):
    """Label-only attack on the distance to the decision boundary, with a threshold chosen on a shadow model.

    Paper: "Label-Only Membership Inference Attacks", https://arxiv.org/abs/2007.14321.
    Documentation: docs/attacks/sba.md.
    The distance is found with HopSkipJump or QEBA. Images are attacked as pixels, so every query is a valid image.
    """
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Sample the shadow dataset and build the shadow model, and the subspace basis of QEBA's custom mode."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Supervised Boundary MIA!")
        assert self.shadow_configs.mode == 'offline', f"Supervised Boundary MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        # One shadow model, as in the paper (more do not help, its Fig. 7); the configs set it whatever --n_shadows.
        assert self.shadow_configs.n_shadow_datasets == 1, \
            f"Supervised Boundary MIA trains one shadow model, got {self.shadow_configs.n_shadow_datasets}."
        # Images are attacked as pixels in [0, 1], the model normalising them, so that every query is a valid image
        # and the distances are in pixels, as in the paper. Tabular inputs are attacked as they are.
        self.normalization = pixel_normalization(self.base_dataset_configs)
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Supervised Boundary MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Supervised Boundary MIA attacker: definition of boundary model...')
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)
        # QEBA-PCA's basis comes from the shadow model's gradients, once it is trained (optimize).
        self.qeba_basis = None
        reduction_mode = (self.attack_configs.boundary.qeba.reduction_mode
                          if self.attack_configs.boundary.mode == 'qeba' else None)
        if reduction_mode == 'custom':
            input_shape = tuple(self.attacker_data_distribution.get('all')[0][0].shape)
            self.qeba_basis = block_basis(input_shape, self.attack_configs.boundary.qeba.reduction_factor)
        elif reduction_mode == 'pca' and self.normalization is None:
            raise ValueError("QEBA's PCA subspace needs images; on tabular data, use sba or sba_qeba.")

    def optimize(self, train_config: TrainConfigs):
        """Train the shadow model, then choose the threshold on its members and non-members."""
        self.logger.print_it('Supervised Boundary MIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done optimizing shadow models. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        if self.attack_configs.boundary.mode == 'qeba' and self.attack_configs.boundary.qeba.reduction_mode == 'pca':
            self.qeba_basis = self.shadow_gradient_basis(device=train_config.device)

        self.logger.print_it('Supervised Boundary MIA attacker: finding optimal threshold...')
        start = time.time()
        self.find_optimal_threshold(device=train_config.device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done finding optimal threshold. It took {}:{:02d}:{:02d}...'.format(h, m, s))
    
    def shadow_gradient_basis(self, device: Union[torch.device, str] = 'cpu') -> torch.Tensor:
        """Return QEBA-PCA's subspace: the principal components of the shadow model's input gradients.

        The shadow model stands in for the QEBA paper's reference models (Sec. 3.4), on up to QEBA_PCA_GRADIENTS of
        the attacker's samples it was not trained on, as the paper's reference models were not trained on theirs.
        """
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        outside = self.shadow_manager.shadow_data.indices_outside_shadow_dataset(index=0)
        n_gradients = min(QEBA_PCA_GRADIENTS, len(outside))
        chosen = np.random.default_rng(self.seed).choice(outside, n_gradients, replace=False)
        samples = SubsampledDataset(self.attacker_data_distribution.get('all'),
                                    original_indices=sorted(chosen.tolist()), strict=True)
        model = self.shadow_manager.shadow_models.get(index=0)
        mean, std = (statistic.to(device) for statistic in self.normalization)
        model = PixelInputModel(model.to(device), mean, std)
        images = PixelView(samples, self.normalization[0][0], self.normalization[1][0])
        self.logger.print_it(f"Supervised Boundary MIA attacker: building QEBA's PCA subspace from the shadow model's "
                             f"gradients on {n_gradients} samples.")
        gradients = input_gradients(model, images, device)
        return gradient_pca_basis(gradients, self.attack_configs.boundary.qeba.reduction_factor)

    def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
        """Choose the distance threshold that best separates the shadow model's members and non-members."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        # Get the trained shadow model and build a dataset with samples from the shadow dataset labeled as members and samples from outside the shadow dataset labeled as non-members to find the optimal threshold for the noise robustness score at the given sigma on the shadow dataset.
        shadow_model = self.shadow_manager.shadow_models.get(index=0).to(device)
        # As many members as non-members, as the paper's equal-sized sets; each costs a full boundary search.
        shadow_member_dataset, shadow_nonmember_dataset, n_per_side = balanced_threshold_samples(
            self.shadow_manager, THRESHOLD_SAMPLES_PER_SIDE, self.seed)
        self.logger.print_it(f"Supervised Boundary MIA attacker: choosing the threshold on {n_per_side} members and "
                             f"{n_per_side} non-members of the shadow model.")
        datasets_list = [shadow_member_dataset, shadow_nonmember_dataset]
        self.regression_dataset = MergedDataset(*datasets_list)
        mia_labels = torch.tensor([1] * len(shadow_member_dataset) + [0] * len(shadow_nonmember_dataset))
        # Iterate over the shadow dataset and compute noise robustness scores for all samples, keeping track of their membership status according to mia_labels
        dataloader = DataLoader(self.regression_dataset, batch_size=1, shuffle=True)
        all_feats = []
        all_mia_labels = []
        start = time.time()
        with torch.no_grad():
            for batch_id, (batch_x, batch_y, _, sample_ids) in enumerate(dataloader):
                h, m, s = convert_to_hms(time.time()-start)
                self.logger.print_it_same_line(f"Supervised Boundary MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for optimal threshold search [{h:02d}:{m:02d}:{s:02d}]. This may take a while...", console_only=True)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                mia_labels_batch = mia_labels[sample_ids]
                all_mia_labels.append(mia_labels_batch)
                batch_feats = self._features(model=shadow_model, inputs=batch_x, targets=batch_y)
                all_feats.append(batch_feats.cpu())
        self.logger.set_logger_newline(console_only=True)
        # Compute the optimal threshold on the shadow dataset for the given sigma as the one maximizing the attack accuracy when classifying samples as members if their noise robustness score is above the threshold and non-members otherwise.
        all_feats = torch.cat(all_feats, dim=0).numpy()
        all_mia_labels = torch.cat(all_mia_labels).numpy()
        fpr, tpr, thresholds = roc_curve(all_mia_labels, all_feats)
        accuracy_scores = []
        for thresh in thresholds:
            preds = [1 if m > thresh else 0 for m in all_feats]
            accuracy_scores.append(accuracy_score(all_mia_labels, preds))
        optimal_idx = np.argmax(accuracy_scores)
        optimal_threshold = thresholds[optimal_idx]
        h, m, s = convert_to_hms(time.time()-start)
        self.logger.print_it(f"Supervised Boundary MIA attacker: optimal threshold found at {optimal_threshold:.4f} with attack accuracy {accuracy_scores[optimal_idx]:.4f}. Time taken to find optimal threshold: {h:02d}:{m:02d}:{s:02d}.")
        self.attack_threshold = optimal_threshold


    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample with the target's distance to its decision boundary, and threshold it."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Supervised Boundary MIA attacker: measuring attack effectiveness...')
        start = time.time()
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        return [MembershipScores(scores, decisions=decisions)]

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """Return the boundary distance of each input."""
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
        """Return each sample's boundary distance and whether it is above the threshold."""
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_dataset(...) for SBA."
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
        # Strictly above, as when the threshold was chosen (and in the paper, Sec. 3.4, and the authors' code).
        all_decisions = (all_boundary_distance_scores > self.attack_threshold).astype(np.int64)
        return all_boundary_distance_scores, all_decisions

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        """Return the boundary distance of each sample of a batch and whether it is above the threshold."""
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_batch(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        boundary_distance_score = self._features(model=model, inputs=batch_x, targets=batch_y)
        decisions = (boundary_distance_score > self.attack_threshold).to(torch.int64)
        return boundary_distance_score.cpu().numpy(), decisions.cpu().numpy()

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        """Return one sample's boundary distance and whether it is above the threshold."""
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_single(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        boundary_distance_score = self._features(model=model, inputs=x, targets=y)
        decision = (boundary_distance_score > self.attack_threshold).to(torch.int64)
        return boundary_distance_score.item(), decision.item()
