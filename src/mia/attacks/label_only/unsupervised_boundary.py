
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.mia.attacks.base_mia import BaseMIA
from src.data import get_dataset_mean_std
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.mia.attacks.label_only.helpers.hop_skip_jump import HopSkipJump
from src.mia.attacks.label_only.helpers.qeba import Qeba, build_qeba_basis



class UnsupervisedBoundaryMIA(BaseMIA):
    requires_auxiliary_data = False

    # Implementation of unsupervised boundary-based label-only attack using HopSkipJump or QEBA adversarial attack of "Membership Leakage in Label-Only Exposures" (https://dl.acm.org/doi/pdf/10.1145/3460120.3484575)
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Unsupervised Boundary MIA!")
        self.attack_threshold = None
        # Only the input shape is needed; no candidate values or labels are used.
        self.input_shape = tuple(self.audit_manager.get(labels='original')[0][0].shape)
        self.qeba_basis = None
        if (self.attack_configs.boundary.mode == 'qeba'
                and self.attack_configs.boundary.qeba.reduction_mode in ('pca', 'custom')):
            self.qeba_basis = build_qeba_basis(
                samples=list(self.random_calibration_inputs()),
                variant=self.attack_configs.boundary.qeba.reduction_mode,
                reduction_factor=self.attack_configs.boundary.qeba.reduction_factor,
            )

    def optimize(self, train_config: TrainConfigs):        
        self.logger.print_it('Unsupervised Boundary MIA attacker: finding threshold via quantile on random inputs...')
        start = time.time()
        self.find_threshold_via_quantile(device=train_config.device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Unsupervised Boundary MIA attacker: Done finding threshold via quantile. It took {}:{:02d}:{:02d}...'.format(h, m, s))
    
    def random_calibration_inputs(self):
        """Uniform random pixels, normalized like model inputs; no auxiliary data.

        For tabular tasks, use uniform [0, 1] feature-space samples.
        """
        generator = torch.Generator().manual_seed(self.seed)
        mean = std = None
        if len(self.input_shape) == 3:
            mean, std = get_dataset_mean_std(self.base_dataset_configs.name)
            mean = torch.tensor(mean).view(-1, 1, 1)
            std = torch.tensor(std).view(-1, 1, 1)
        for _ in range(self.attack_configs.n_calibration_samples):
            sample = torch.rand(self.input_shape, generator=generator)
            yield (sample - mean) / std if mean is not None else sample

    def find_threshold_via_quantile(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        defended_model = self.defender_model.to(device).eval()
        all_feats = []
        with torch.no_grad():
            for sample in self.random_calibration_inputs():
                batch_x = sample.unsqueeze(0).to(device)
                batch_y = self.label_pred(defended_model, batch_x)
                scores = self._features(defended_model, batch_x, batch_y)
                all_feats.append(scores.cpu().reshape(-1))
        all_feats = torch.cat(all_feats).numpy()
        if not np.isfinite(all_feats).all():
            raise ValueError('Boundary calibration produced non-finite distances; increase the query budget.')
        self.attack_threshold = float(np.quantile(all_feats, self.attack_configs.quantile))
        self.logger.print_it(f'Boundary threshold from random inputs: {self.attack_threshold:.4f}')

    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Unsupervised Boundary MIA attacker: measuring attack effectiveness...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=audit_dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Unsupervised Boundary MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        metrics = self.compute_stats(scores, decisions=decisions)
        return metrics

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.estimate_boundary_distance(model, inputs, self.label_pred(model, inputs))

    @torch.no_grad()
    def label_pred(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns hard labels (argmax) for a batch x.
        """
        model.eval()
        logits = model(inputs)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def estimate_boundary_distance(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute a label-only boundary distance per sample using HSJA.
        Returns:
            dists: shape (N,) distance (L2 or Linf) to the decision boundary.
        Note:
            HSJA is run per-sample for stability. This uses only hard labels internally.
        """
        model.eval()
        N = inputs.shape[0]
        if self.attack_configs.boundary.mode == 'hop_skip_jump':
            # self.logger.print_it(f"Unsupervised Boundary MIA attacker: using HopSkipJump to estimate distance to decision boundary for {N} samples with max {self.attack_configs.n_queries} queries and norm {self.attack_configs.boundary.norm}...")
            boundary_finder = HopSkipJump(norm=self.attack_configs.boundary.norm,
                                        max_queries=self.attack_configs.boundary.n_queries,
                                        logger=self.logger)
        elif self.attack_configs.boundary.mode == 'qeba':
            # self.logger.print_it(f"Unsupervised Boundary MIA attacker: using QEBA to estimate distance to decision boundary for {N} samples with max {self.attack_configs.boundary.n_queries} queries and norm {self.attack_configs.boundary.norm}...")
            boundary_finder = Qeba(norm=self.attack_configs.boundary.norm,
                                    max_queries=self.attack_configs.boundary.n_queries,
                                    logger=self.logger,
                                    variant=self.attack_configs.boundary.qeba.reduction_mode,
                                    reduction_factor=self.attack_configs.boundary.qeba.reduction_factor,
                                    basis=self.qeba_basis)
        else:
            raise ValueError(f"Unsupervised Boundary MIA attacker: bound_mode {self.attack_configs.boundary.mode} not recognized!")

        # start = time.time()
        adversarial_samples, distances, queries = boundary_finder.run(model=model, inputs=inputs, labels=targets, targeted=False, device=inputs.device)
        # stop = time.time()
        # h, m, s = convert_to_hms(stop-start)
        # self.logger.print_it(f"Boundary MIA attacker: adversarial distance to boundary estimation completed in {h}:{m:02d}:{s:02d} for {N} samples.")
        return distances # shape (N, 1)

    @torch.no_grad()
    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_threshold_via_quantile(...) before infer_dataset(...) for UBA."
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
        all_decisions = (all_boundary_distance_scores >= self.attack_threshold).astype(np.int64)
        return all_boundary_distance_scores, all_decisions

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_threshold_via_quantile(...) before infer_batch(...) for UBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        boundary_distance_score = self._features(model=model, inputs=batch_x, targets=batch_y)
        decisions = (boundary_distance_score >= self.attack_threshold).to(torch.int64)
        return boundary_distance_score.cpu().numpy(), decisions.cpu().numpy()

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_threshold_via_quantile(...) before infer_single(...) for UBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        boundary_distance_score = self._features(model=model, inputs=x, targets=y)
        decision = (boundary_distance_score >= self.attack_threshold).to(torch.int64)
        return boundary_distance_score.item(), decision.item()
