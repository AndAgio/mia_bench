from typing import Callable, Union
import numpy as np
import torch
import time
from scipy.stats import norm
from src.data.multi import MultiDatasets
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils import convert_to_hms
from src.utils.configs import AttackerConfigs, TrainConfigs

BATCH_SIZE = 256


class AttackRMIA(BaseMIA):
    # Implementation of Attack-R in Enhanced Membership Inference Attacks against Machine Learning Models (https://arxiv.org/pdf/2111.09679).
    def __init__(self, 
                defender_model: torch.nn.Module,
                defender_dataset: MultiDatasets,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, defender_dataset=defender_dataset, attacker_configs=attacker_configs)
        assert self.shadow_configs.n_shadow_datasets >= 1, f"When using Attack-R MIA, at least 1 shadow dataset must be used!"
        assert self.shadow_configs.mode == 'offline', f"When using Attack-R MIA, only offline shadow datasets are supported!"
        assert self.attack_configs.r_score_type in ['loss', 'confidence', 'entropy'], f"When using Attack-R MIA, r_score_type must be one of ['loss', 'confidence', 'entropy']!"
        self.logger.print_it(f"Working with Attack-R and scoring mode {self.attack_configs.r_score_type}!")
        self.name = f"Attack-R {self.attack_configs.r_score_type} attacker"
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it(f"{self.name}: sampling of shadow datasets...")
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.defender_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it(f"{self.name}: definition of shadow models...")
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it(f"{self.name}: training all shadow models. This will take a while. Sit back and chill...")
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f"{self.name}: done optimizing. It took {h}:{m:02d}:{s:02d}...")

    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        start = time.time()
        if isinstance(device, str):
            device = AttackRMIA.get_device(dev_str=device)
        audit_dataset = self.audit_manager.get(labels='original')
        audit_loader = torch.utils.data.DataLoader(audit_dataset, batch_size=BATCH_SIZE, shuffle=False)
        self.logger.print_it(f"{self.name}: computing scores. This may take a while...")
        # Build reference loss matrix
        self.logger.print_it(f"{self.name}: building reference loss matrix. This may take a while...")
        start_ref = time.time()
        shadow_models = self.shadow_manager.get_all_models()
        ref_matrix = np.zeros((len(shadow_models), len(audit_dataset)))
        for i, shadow_model in shadow_models.items():
            losses = []
            for data, label in audit_loader:
                data = data.to(device)
                label = label.to(device)
                losses.append(
                    self.compute_batch_scores(model=shadow_model, 
                                            data=data, 
                                            label=label)
                )
            ref_matrix[i] = np.concatenate(losses, axis=0)
        self.logger.print_it(f"{self.name}: built reference loss matrix in {time.time() - start_ref:.2f}s.")
        # compute smoothed thresholds for each audit sample 
        self.logger.print_it(f"{self.name}: computing smoothed thresholds. This may take a while...")
        start_smooth = time.time()
        thresholds = AttackRMIA.batched_smoothed_thresholds(ref_matrix, self.attack_configs.r_alpha)
        self.logger.print_it(f"{self.name}: smoothed loss matrix in {time.time() - start_smooth:.2f}s.")
        # compute target model batch losses
        self.logger.print_it(f"{self.name}: forwarding through auditing and thresholding. This may take a while...")
        start_thresh = time.time()
        target_losses = []
        for data, label in audit_loader:
            data = data.to(device)
            label = label.to(device)
            target_losses.append(
                self.compute_batch_scores(model=self.defender_model, 
                                        data=data, 
                                        label=label)
            )
        target_losses = np.concatenate(target_losses, axis=0)
        # decide membership per sample
        scores = target_losses <= thresholds
        self.logger.print_it(f"{self.name}: auditing forward and thresholding completed in {time.time() - start_thresh:.2f}s.")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f"{self.name}: score computation done! Time taken: {h}:{m:02d}:{s:02d}...")
        metrics = self.compute_stats(scores)
        self.logger.print_it(f"{self.name}: Obtained AUC score is: {metrics["auc"]}")
        return metrics

    def compute_batch_scores(self, model: torch.nn.Module, data: torch.Tensor, label: torch.Tensor):
        model.eval()
        with torch.no_grad():
            out = model(data)
            if self.attack_configs.r_score_type == "confidence":
                scores = self.softmax_confidence(out).cpu().numpy()
            elif self.attack_configs.r_score_type == "entropy":
                scores = self.prediction_entropy(out).cpu().numpy()
            elif self.attack_configs.r_score_type == "loss":
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
                scores = criterion(out, label).cpu().numpy()
            else:
                raise ValueError(f"Unsupported scoring type '{self.attack_configs.r_score_type}'.")
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

    @staticmethod
    def batched_smoothed_thresholds(ref_matrix: np.ndarray, alpha: float):
        """
        Given a reference loss matrix (m, n),
        compute alpha-percentile threshold per sample in vectorized form.
        """
        # sort per column
        sorted_ref = np.sort(ref_matrix, axis=0)  # shape (m, n)
        m = sorted_ref.shape[0]

        # linear interpolation thresholds
        u = alpha * (m - 1)
        i = np.floor(u).astype(int)
        j = np.minimum(i + 1, m - 1)
        w = u - i

        # indexed linear interpolation per column
        p_linear = sorted_ref[i] * (1 - w) + sorted_ref[j] * w

        # logit smoothing: clip, transform, fit Gaussian per column
        eps = 1e-8
        clipped = np.clip(sorted_ref, eps, 1 - eps)
        logit_vals = np.log(clipped / (1 - clipped))

        mu = np.mean(logit_vals, axis=0)
        sigma = np.std(logit_vals, axis=0, ddof=1)
        sigma = np.where(sigma < 1e-6, 1e-6, sigma)

        z = norm.ppf(1 - alpha, loc=mu, scale=sigma)
        p_logit = np.exp(z) / (1 + np.exp(z))

        return np.minimum(p_linear, p_logit)
    
