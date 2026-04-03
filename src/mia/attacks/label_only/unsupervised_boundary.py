
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.data.helpers import MultiDatasets
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.mia.attacks.label_only.helpers.hop_skip_jump import HopSkipJump
from src.mia.attacks.label_only.helpers.qeba import Qeba



class UnsupervisedBoundaryMIA(BaseMIA):
    # Implementation of unsupervised boundary-based label-only attack using HopSkipJump or QEBA adversarial attack of "Membership Leakage in Label-Only Exposures" (https://dl.acm.org/doi/pdf/10.1145/3460120.3484575)
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Unsupervised Boundary MIA!")
        assert self.shadow_configs.mode == 'offline', f"Unsupervised Boundary MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        if self.shadow_configs.n_shadow_datasets != 1:
            self.logger.print_it(f"Unsupervised Boundary MIA attacker [WARNING]: when using unsupervised boundary MIA, only 1 shadow dataset must be used! Modifying shadow_configs on the fly to set n_shadow_datasets to 1.")
            self.shadow_configs.n_shadow_datasets = 1
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Unsupervised Boundary MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)

    def optimize(self, train_config: TrainConfigs):        
        self.logger.print_it('Unsupervised Boundary MIA attacker: finding threshold via quantile on shadow dataset...')
        start = time.time()
        self.find_threshold_via_quantile(device=train_config.device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Unsupervised Boundary MIA attacker: Done finding threshold via quantile. It took {}:{:02d}:{:02d}...'.format(h, m, s))
    
    def find_threshold_via_quantile(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        # Iterate over the shadow dataset and compute scores for all samples to identify threshold as a quantile of the score distribution on the shadow dataset.
        defended_model = self.defender_model.to(device)
        shadow_member_dataset = self.shadow_manager.get_dataset(index=0, 
                                                                labels='original')
        dataloader = DataLoader(shadow_member_dataset, batch_size=1, shuffle=True)
        all_feats = []
        start = time.time()
        with torch.no_grad():
            for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
                h, m, s = convert_to_hms(time.time()-start)
                self.logger.print_it_same_line(f"Unsupervised Boundary MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for threshold search [{h:02d}:{m:02d}:{s:02d}]. This may take a while...", console_only=True)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_feats = self._features(model=defended_model, inputs=batch_x, targets=batch_y)
                all_feats.append(batch_feats.cpu())
        self.logger.set_logger_newline(console_only=True)
        # Compute the threshold on the shadow dataset as the one at the specified quantile of the score distribution on the shadow dataset.
        all_feats = torch.cat(all_feats, dim=0).numpy()
        self.attack_threshold = np.quantile(all_feats, self.attack_configs.quantile)
        h, m, s = convert_to_hms(time.time()-start)
        self.logger.print_it(f"Unsupervised Boundary MIA attacker: threshold found at {self.attack_threshold:.4f}. Time taken to find threshold: {h:02d}:{m:02d}:{s:02d}.")

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
        
        metrics = self.compute_stats(scores)
        return metrics

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        return self.estimate_boundary_distance(model, inputs, targets)

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
        compiled_model = JITModelWrapper(model)
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
                                    reduction_factor=self.attack_configs.boundary.qeba.reduction_factor)
        else:
            raise ValueError(f"Unsupervised Boundary MIA attacker: bound_mode {self.attack_configs.boundary.mode} not recognized!")

        # start = time.time()
        adversarial_samples, distances, queries = boundary_finder.run(model=compiled_model, inputs=inputs, labels=targets, targeted=False, device=inputs.device)
        # stop = time.time()
        # h, m, s = convert_to_hms(stop-start)
        # self.logger.print_it(f"Boundary MIA attacker: adversarial distance to boundary estimation completed in {h}:{m:02d}:{s:02d} for {N} samples.")
        return distances # shape (N, 1)

    @torch.no_grad()
    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
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
        all_decisions = (all_boundary_distance_scores >= self.attack_threshold).astype(np.int64)
        return all_boundary_distance_scores, all_decisions

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_batch(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        boundary_distance_score = self._features(model=model, inputs=batch_x, targets=batch_y)
        decisions = (boundary_distance_score >= self.attack_threshold).astype(np.int64)
        return boundary_distance_score.cpu().numpy(), decisions

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_single(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        boundary_distance_score = self._features(model=model, inputs=x, targets=y)
        decision = (boundary_distance_score >= self.attack_threshold).astype(np.int64)
        return boundary_distance_score.item(), decision.item()



class JITModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = torch.compile(model)

    def forward(self, x):
        return self.model(x)
