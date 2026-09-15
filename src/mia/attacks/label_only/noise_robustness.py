
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



class NoiseRobustnessMIA(BaseMIA):
    # Implementation of noise robustness based label-only attack of "Label-Only Membership Inference Attacks" (https://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf)
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Noise Robustness MIA!")
        assert self.shadow_configs.mode == 'offline', f"Noise Robustness MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        if self.shadow_configs.n_shadow_datasets != 1:
            self.logger.print_it(f"Noise Robustness MIA attacker [WARNING]: when using noise robustness MIA, only 1 shadow dataset must be used! Modifying shadow_configs on the fly to set n_shadow_datasets to 1.")
            self.shadow_configs.n_shadow_datasets = 1
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Noise Robustness MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Noise Robustness MIA attacker: definition of noise robustness model...')
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('Noise Robustness MIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Noise Robustness MIA attacker: Done optimizing shadow models. It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def find_optimal_sigma_and_threshold(self, sigmas: Union[float, list[float]] = None, device: Union[torch.device, str] = 'cpu'):
        start = time.time()
        sigmas = self.attack_configs.sigmas if sigmas is None else sigmas
        if isinstance(self.attack_configs.sigmas, (float, int)):
            sigmas = [float(self.attack_configs.sigmas)]
        else:
            sigmas = [float(s) for s in self.attack_configs.sigmas]
        self.logger.print_it(f"Noise Robustness MIA attacker: using noise robustness scores with sigmas {sigmas} for noise perturbations...")

        best_sigma = None
        best_sigma_acc = -np.inf
        best_threshold = None
        for sigma in sigmas:
            optimal_threshold, accuracy = self.find_optimal_threshold_at_sigma(sigma=sigma, device=device)
            if accuracy > best_sigma_acc:
                best_sigma = sigma
                best_sigma_acc = accuracy
                best_threshold = optimal_threshold
            self.logger.print_it(f"Noise Robustness MIA attacker: done finding optimal threshold for sigma={sigma} on shadow dataset. Attack accuracy is {accuracy:.4f} at optimal threshold {optimal_threshold:.4f}.")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f"Noise Robustness MIA attacker: done finding optimal sigma and threshold on shadow dataset. Best sigma is {best_sigma} with accuracy {best_sigma_acc:.4f} at optimal threshold {best_threshold:.4f}. It took {h}:{m:02d}:{s:02d}...".format(h, m, s))
        self.best_sigma = best_sigma
        self.attack_threshold = best_threshold

    def find_optimal_threshold_at_sigma(self, sigma: float, device: Union[torch.device, str] = 'cpu'):
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
        with torch.no_grad():
            for batch_id, (batch_x, batch_y, _, sample_ids) in enumerate(dataloader):
                self.logger.print_it_same_line(f"Noise Robustness MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for optimal threshold search with sigma={sigma}...", console_only=True)
                batch_x = batch_x.to(device)
                mia_labels_batch = mia_labels[sample_ids]
                all_mia_labels.append(mia_labels_batch)
                batch_feats = self._features(model=shadow_model, inputs=batch_x, true_labels=batch_y, sigma=sigma)
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
        self.attack_threshold = optimal_threshold
        return optimal_threshold, accuracy_scores[optimal_idx]

    def measure_effectiveness(self, sigmas: Union[float, list[float]] = None, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Noise Robustness MIA attacker: finding optimal threshold on shadow dataset...')
        start = time.time()
        self.find_optimal_sigma_and_threshold(sigmas=sigmas, device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Noise Robustness MIA attacker: Done finding optimal threshold. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        self.logger.print_it('Noise Robustness MIA attacker: measuring attack effectiveness...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=audit_dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Noise Robustness MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        metrics = self.compute_stats(scores, decisions=decisions)

        self.logger.print_it(f"Noise Robustness MIA attacker: attack effectiveness measured! AUC score is {metrics['auc']:.4f} at optimal threshold {self.attack_threshold:.4f} with best sigma {self.best_sigma}.")
        mia_audit_dataset = self.audit_manager.get(labels='mia')
        correct_decisions = (decisions == np.array([label for _, (_, label, _, _) in enumerate(mia_audit_dataset)]))
        attack_accuracy = np.mean(correct_decisions)
        self.logger.print_it(f"Noise Robustness MIA attacker: attack accuracy at optimal threshold is {attack_accuracy:.4f}.")
        return metrics

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, true_labels: torch.Tensor, sigma: float) -> np.ndarray:
        device = inputs.device
        model.eval()
        inputs = inputs.to(device)
        true_labels = true_labels.to(device)
        n_samples_in_batch = inputs.size(0)
        # compute predictions on clean inputs and check which samples are correctly classified
        preds_clean = self.label_pred(model, inputs)
        correct = (preds_clean == true_labels)  # (N,)
        # iterate over samples in batch and compute noise robustness score for each sample generating n_queries noisy versions
        robustness = torch.zeros(n_samples_in_batch, device=device)
        for i in range(n_samples_in_batch):
            if not correct[i]:
                # Keep robustness score at 0 for misclassified samples, as they are not robust to noise by definition (they are already not robust to the clean input).
                continue
            inputs_subset = inputs[i:i+1]
            true_labels_subset = true_labels[i].item()
            # generate T noisy versions
            noise = sigma * torch.randn((self.attack_configs.n_queries, *inputs_subset.shape[1:]), device=device)
            noisy_inputs_subset = inputs_subset + noise # avoid clamping as preprocessing of images is not limiting them to [0,1] range but is for example normalizing them to have mean 0 and std 1, therefore adding noise without clamping can be more realistic of the distribution of noisy samples that the model can see in real life.
            # batched inference
            preds = []
            for j in range(0, self.attack_configs.n_queries, 1):
                preds.append(self.label_pred(model, noisy_inputs_subset[j:j+1]))
            preds = torch.cat(preds, dim=0).cpu().numpy()
            # compute robustness score as the fraction of noisy samples that are still classified the same as the clean input
            robustness[i] = np.mean(preds == true_labels_subset)
        return robustness


    @torch.no_grad()
    def label_pred(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns hard labels (argmax) for a batch x.
        """
        model.eval()
        return torch.argmax(model(inputs), dim=1)
    
    @torch.no_grad()
    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_sigma_and_threshold(...) before infer_dataset(...) for SBA."
        assert self.best_sigma is not None, "Call find_optimal_sigma_and_threshold(...) before infer_dataset(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        all_noise_robustness_score = []
        for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
            self.logger.print_it_same_line(f"Noise Robustness MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for inference...", console_only=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            noise_robustness_score = self._features(model=model, inputs=batch_x, true_labels=batch_y, sigma=self.best_sigma)
            all_noise_robustness_score.append(noise_robustness_score.cpu())
        self.logger.set_logger_newline(console_only=True)
        all_noise_robustness_score = torch.cat(all_noise_robustness_score, dim=0).numpy()
        all_decisions = (all_noise_robustness_score >= self.attack_threshold).astype(np.int64)
        return all_noise_robustness_score, all_decisions

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_sigma_and_threshold(...) before infer_batch(...) for SBA."
        assert self.best_sigma is not None, "Call find_optimal_sigma_and_threshold(...) before infer_batch(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        noise_robustness_score = self._features(model=model, inputs=batch_x, true_labels=batch_y, sigma=self.best_sigma)
        decisions = (noise_robustness_score >= self.attack_threshold).astype(np.int64)
        return noise_robustness_score.cpu().numpy(), decisions

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_sigma_and_threshold(...) before infer_single(...) for SBA."
        assert self.best_sigma is not None, "Call find_optimal_sigma_and_threshold(...) before infer_single(...) for SBA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        noise_robustness_score = self._features(model=model, inputs=x, true_labels=y, sigma=self.best_sigma)
        decision = (noise_robustness_score >= self.attack_threshold).astype(np.int64)
        return noise_robustness_score.item(), decision.item()
