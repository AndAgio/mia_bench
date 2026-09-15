
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from src.data.helpers import MergedDataset
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.mia.attacks.label_only.helpers.hop_skip_jump import HopSkipJump
from src.mia.attacks.label_only.helpers.qeba import Qeba



class SupervisedBoundaryMIA(BaseMIA):
    # Implementation of supervised boundary-based label-only attack using HopSkipJump or QEBA adversarial attack of "Label-Only Membership Inference Attacks" (https://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf)
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Supervised Boundary MIA!")
        assert self.shadow_configs.mode == 'offline', f"Supervised Boundary MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        if self.shadow_configs.n_shadow_datasets != 1:
            self.logger.print_it(f"Supervised Boundary MIA attacker [WARNING]: when using supervised boundary MIA, only 1 shadow dataset must be used! Modifying shadow_configs on the fly to set n_shadow_datasets to 1.")
            self.shadow_configs.n_shadow_datasets = 1
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Supervised Boundary MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Supervised Boundary MIA attacker: definition of boundary model...')
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('Supervised Boundary MIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done optimizing shadow models. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        self.logger.print_it('Supervised Boundary MIA attacker: finding optimal threshold...')
        start = time.time()
        self.find_optimal_threshold(device=train_config.device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done finding optimal threshold. It took {}:{:02d}:{:02d}...'.format(h, m, s))
    
    def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        # Get the trained shadow model and build a dataset with samples from the shadow dataset labeled as members and samples from outside the shadow dataset labeled as non-members to find the optimal threshold for the noise robustness score at the given sigma on the shadow dataset.
        shadow_model = self.shadow_manager.shadow_models.get(index=0).to(device)
        shadow_member_dataset = self.shadow_manager.get_dataset(index=0, 
                                                                labels='original')
        shadow_nonmember_dataset = self.shadow_manager.sample_outside_shadow_dataset(index=0,
                                                                                    num_data=len(shadow_member_dataset), 
                                                                                    labels='original')
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


    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Supervised Boundary MIA attacker: measuring attack effectiveness...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=audit_dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        metrics = self.compute_stats(scores, decisions=decisions)
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
            # self.logger.print_it(f"Supervised Boundary MIA attacker: using HopSkipJump to estimate distance to decision boundary for {N} samples with max {self.attack_configs.boundary.n_queries} queries and norm {self.attack_configs.boundary.norm}...")
            boundary_finder = HopSkipJump(norm=self.attack_configs.boundary.norm,
                                        max_queries=self.attack_configs.boundary.n_queries,
                                        logger=self.logger)
        elif self.attack_configs.boundary.mode == 'qeba':
            # self.logger.print_it(f"Supervised Boundary MIA attacker: using QEBA to estimate distance to decision boundary for {N} samples with max {self.attack_configs.boundary.n_queries} queries and norm {self.attack_configs.boundary.norm}...")
            boundary_finder = Qeba(norm=self.attack_configs.boundary.norm,
                                    max_queries=self.attack_configs.boundary.n_queries,
                                    logger=self.logger,
                                    variant=self.attack_configs.boundary.qeba.reduction_mode,
                                    reduction_factor=self.attack_configs.boundary.qeba.reduction_factor)
        else:
            raise ValueError(f"Supervised Boundary MIA attacker: bound_mode {self.attack_configs.boundary.mode} not recognized!")

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
