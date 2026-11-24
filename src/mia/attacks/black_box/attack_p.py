from typing import Union
import numpy as np
import torch
import time
from src.mia.helpers.reference_thresholds import alpha_star, loss_like
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils import convert_to_hms
from src.utils.configs import AttackerConfigs, TrainConfigs

BATCH_SIZE = 256


class AttackPMIA(BaseMIA):
    """Attack-P: compares the target's score on a sample with its scores on population data.

    Paper: "Enhanced Membership Inference Attacks against Machine Learning Models",
    https://arxiv.org/abs/2111.09679. Documentation: docs/attacks/attack_r_p.md.
    Each sample is scored by -alpha_star, the smallest false-positive rate at which the population's threshold makes it
    a member, computed as for Attack-R, and is a member when alpha_star <= --attacker_pmia_alpha.
    """
    # The target's losses on the population data, computed once for every scored dataset.
    population_losses = None

    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Check the settings, then sample the shadow datasets, of which the first is the population data."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        assert self.shadow_configs.n_shadow_datasets == 1, \
            f"Attack-P uses one shadow dataset, set by the configs, got {self.shadow_configs.n_shadow_datasets}."
        self.shadow_configs.mode = 'offline'
        assert self.attack_configs.p_score_type in ['loss', 'confidence', 'entropy'], f"When using Attack-P MIA, p_score_type must be one of ['loss', 'confidence', 'entropy']!"
        if not 0 <= self.attack_configs.p_alpha <= 1:
            raise ValueError(f"--attacker_pmia_alpha must lie between 0 and 1, got {self.attack_configs.p_alpha}.")
        self.logger.print_it(f"Working with Attack-P and scoring mode {self.attack_configs.p_score_type}!")
        self.name = f"Attack-P {self.attack_configs.p_score_type} attacker"
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it(f"{self.name}: sampling of shadow datasets...")
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)

    def optimize(self, train_config: TrainConfigs):
        """Do nothing: Attack-P trains no model."""
        pass

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample by -alpha_star against the target's scores on population data; members score higher."""
        start = time.time()
        if isinstance(device, str):
            device = AttackPMIA.get_device(dev_str=device)
        self.logger.print_it(f"{self.name}: computing scores. This may take a while...")
        if self.population_losses is None:
            self.population_losses = self.compute_population_losses(device=device)
        # compute target model batch losses
        self.logger.print_it(f"{self.name}: forwarding through auditing and thresholding. This may take a while...")
        start_thresh = time.time()
        audit_scores = []
        audit_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch_index, (data, label, _, _) in enumerate(audit_loader):
            self.logger.print_it_same_line(f"{self.name}: forwarding through auditing and thresholding for batch {batch_index+1}/{len(audit_loader)}...", console_only=True)
            data = data.to(device)
            label = label.to(device)
            audit_scores.append(
                self.compute_batch_scores(model=self.defender_model,
                                        data=data,
                                        label=label)
            )
        self.logger.set_logger_newline(console_only=True)
        audit_scores = np.concatenate(audit_scores, axis=0)
        alpha_stars = alpha_star(self.population_losses, loss_like(audit_scores, self.attack_configs.p_score_type))
        scores = -alpha_stars
        decisions = alpha_stars <= self.attack_configs.p_alpha

        self.logger.print_it(f"{self.name}: auditing forward and thresholding completed in {time.time() - start_thresh:.2f}s.")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f"{self.name}: score computation done! Time taken: {h}:{m:02d}:{s:02d}...")
        return [MembershipScores(scores, decisions=decisions)]

    def compute_population_losses(self, device: torch.device):
        """Return the target's scores on the population data, the shadow dataset, as losses (see loss_like)."""
        shadow_dataset = self.shadow_manager.get_dataset(index=0,
                                                        labels='original')
        shadow_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # Build reference loss matrix
        self.logger.print_it(f"{self.name}: building reference scores matrix. This may take a while...")
        start_ref = time.time()
        # Collect population scores
        pop_scores = []
        for batch_index, (data, label, _, _) in enumerate(shadow_loader):
            self.logger.print_it_same_line(f"{self.name}: computing reference scores for batch {batch_index+1}/{len(shadow_loader)}...", console_only=True)
            data = data.to(device)
            label = label.to(device)
            pop_scores.append(
                self.compute_batch_scores(model=self.defender_model,
                                        data=data,
                                        label=label)
            )
        pop_scores = np.concatenate(pop_scores, axis=0)
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"{self.name}: built reference scores matrix in {time.time() - start_ref:.2f}s.")
        return loss_like(pop_scores, self.attack_configs.p_score_type)

    def compute_batch_scores(self, model: torch.nn.Module, data: torch.Tensor, label: torch.Tensor):
        """Return the model's score of each sample in the batch: its loss, highest confidence or prediction entropy."""
        model.eval()
        with torch.no_grad():
            out = model(data)
            if self.attack_configs.p_score_type == "confidence":
                scores = self.softmax_confidence(out).cpu().numpy()
            elif self.attack_configs.p_score_type == "entropy":
                scores = self.prediction_entropy(out).cpu().numpy()
            elif self.attack_configs.p_score_type == "loss":
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
                scores = criterion(out, label).cpu().numpy()
            else:
                raise ValueError(f"Unsupported scoring type '{self.attack_configs.p_score_type}'.")
        return scores

    @staticmethod
    def softmax_confidence(logits):
        """
        Returns the maximum softmax confidence.
        """
        probs = torch.softmax(logits, dim=-1)
        return torch.max(probs, dim=-1).values

    @staticmethod
    def prediction_entropy(logits):
        """
        Returns entropy of the softmax distribution:
        - sum(p * log(p)) across classes.
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
