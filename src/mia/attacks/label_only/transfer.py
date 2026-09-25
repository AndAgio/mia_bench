
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from src.data.helpers import MultiDatasets, MergedDataset, TargetOverrideDataset
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms


PROCESSING_BATCH_SIZE = 128


# Apparently the authors in the original implementation of the paper require training the defender model with very few data (only 2000 samples for CIFAR10) and the shadow model over a large set of data (more than 35000 samples on CIFAR10) which should be disjunct from the training dataset of the defender model.
# Given our implementation of the shadow manager, we can only guarantee that the shadow dataset is disjunct from the defender training dataset by sampling only from the testing set. Therefore, we can try the shadow model with fewer data.
# However, our implementation is much more realistic. In practice, it is not possible to consider a setting in which the attacker has access to a dataset which is larger than the victim.

class TransferMIA(BaseMIA):
    # Implementation of transfer attack of "Membership Leakage in Label-Only Exposures" (https://dl.acm.org/doi/pdf/10.1145/3460120.3484575)
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Transfer MIA!")
        assert self.shadow_configs.mode == 'offline', f"Transfer MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        if self.shadow_configs.n_shadow_datasets != 1:
            self.logger.print_it(f"Transfer MIA attacker [WARNING]: when using transfer MIA, only 1 shadow dataset must be used! Modifying shadow_configs on the fly to set n_shadow_datasets to 1.")
            self.shadow_configs.n_shadow_datasets = 1
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Transfer MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Transfer MIA attacker: definition of transfer model...')
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('Transfer MIA attacker: relabelling shadow dataset based on target model...')
        start = time.time()
        distilled_dataset = self.relabel_shadow_dataset(train_config=train_config)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Transfer MIA attacker: done relabelling shadow dataset. It took {}:{:02d}:{:02d}...'.format(h, m, s))

        self.logger.print_it('Transfer MIA attacker: training shadow model with distilled dataset. This will take a while. Sit back and chill...')
        start = time.time()
        wrapped_dataset = MultiDatasets([distilled_dataset], ids=['train'])
        self.shadow_manager.train_single_model_on_given_dataset(id=0,
                                                                train_configs=train_config,
                                                                dataset=wrapped_dataset)
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Transfer MIA attacker: Done optimizing shadow model over the distilled dataset. It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def relabel_shadow_dataset(self, train_config: TrainConfigs):
        # Relabel the shadow dataset according to the predictions of the target model, as done in the transfer attack of "Label-Only Membership Inference Attacks" (https://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf).
        shadow_dataset = self.shadow_manager.get_dataset(index=0, labels='original')
        shadow_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=train_config.batch_size, shuffle=False)
        all_relabels = []
        device = self.get_device(train_config.device)
        self.defender_model.to(device)
        with torch.no_grad():
            for batch_index, (data, original_labels, _, _) in enumerate(shadow_loader):
                # print(f"original_labels shape: {original_labels.shape}")
                self.logger.print_it_same_line(f"Transfer MIA attacker: relabelling batch {batch_index+1}/{len(shadow_loader)}...", console_only=True)
                data = data.to(device)
                preds = torch.argmax(self.defender_model(data), dim=1).cpu()
                all_relabels.append(preds)
        self.logger.set_logger_newline(console_only=True)
        all_relabels = torch.cat(all_relabels, dim=0)
        # print(f"all_relabels shape: {all_relabels.shape}, expected: ({len(shadow_dataset)},)")

        # Relabeling the shadow dataset with the obtained relabels
        return TargetOverrideDataset(shadow_dataset, all_relabels)

    def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        # Calibrate the transfer score on members and non-members of the
        # surrogate model.  _features() normalizes every feature so that larger
        # values always mean stronger membership evidence.
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
        dataloader = DataLoader(self.regression_dataset, batch_size=PROCESSING_BATCH_SIZE, shuffle=True)
        all_feats = []
        all_mia_labels = []
        with torch.no_grad():
            for batch_id, (batch_x, batch_y, _, sample_ids) in enumerate(dataloader):
                self.logger.print_it_same_line(f"Transfer MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for optimal threshold...", console_only=True)
                batch_x = batch_x.to(device)
                mia_labels_batch = mia_labels[sample_ids]
                all_mia_labels.append(mia_labels_batch)
                batch_feats = self._features(model=shadow_model, inputs=batch_x, true_labels=batch_y)
                # print("\n\n")
                # print(f"batch original labels: {batch_y}")
                # print(f"batch predictions: {torch.argmax(shadow_model(batch_x), dim=1)}")
                # print(f"batch features: {batch_feats}")
                # print(f"batch mia labels: {mia_labels_batch}")
                
                all_feats.append(batch_feats.cpu())
        self.logger.set_logger_newline(console_only=True)
        # Pick the threshold that maximizes calibration accuracy.  All feature
        # modes use the same >= rule because their direction was normalized in
        # _features().
        all_feats = torch.cat(all_feats, dim=0).numpy()
        all_mia_labels = torch.cat(all_mia_labels).numpy()
        fpr, tpr, thresholds = roc_curve(all_mia_labels, all_feats)
        accuracy_scores = []
        for thresh in thresholds:
            preds = (all_feats >= thresh).astype(np.int64)
            accuracy_scores.append(accuracy_score(all_mia_labels, preds))
        optimal_idx = np.argmax(accuracy_scores)
        optimal_threshold = thresholds[optimal_idx]
        self.attack_threshold = optimal_threshold
        self.logger.print_it(f"Optimal threshold found on shadow dataset: {optimal_threshold} with accuracy {accuracy_scores[optimal_idx]}")
        return optimal_threshold, accuracy_scores[optimal_idx]

    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Transfer MIA attacker: finding optimal threshold on shadow dataset...')
        start = time.time()
        self.find_optimal_threshold(device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Transfer MIA attacker: Done finding optimal threshold. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        self.logger.print_it('Transfer MIA attacker: measuring attack effectiveness...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        shadow_model = self.shadow_manager.shadow_models.get(index=0).to(device)
        scores, decisions = self.infer_dataset(model=shadow_model,
                                                dataset=audit_dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Transfer MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        metrics = self.compute_stats(scores, decisions=decisions)
        mia_audit_dataset = self.audit_manager.get(labels='mia')
        correct_decisions = (decisions == np.array([label for _, (_, label, _, _) in enumerate(mia_audit_dataset)]))
        attack_accuracy = np.mean(correct_decisions)
        self.logger.print_it(f"Transfer MIA attacker: attack accuracy at optimal threshold is {attack_accuracy:.4f}.")
        return metrics

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Return a score whose larger values always indicate membership.

        Confidence naturally has this direction.  Loss and entropy have the
        opposite direction, so negate them before thresholding or computing
        ROC/AUC metrics.
        """
        device = inputs.device
        model.eval()
        true_labels = true_labels.to(device)
        logits = model(inputs)
        if self.attack_configs.feature_mode == 'loss':
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            losses = criterion(logits, true_labels)
            return -losses.cpu()
        elif self.attack_configs.feature_mode == 'entropy':
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=1)
            return -entropy.cpu()
        elif self.attack_configs.feature_mode == 'max_confidence':
            max_probs = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
            return max_probs.cpu()
        else:
            raise ValueError(f"Unsupported feature_mode {self.attack_configs.feature_mode} for Transfer MIA attacker! Supported modes are: 'loss', 'entropy' and 'max_confidence'.")

    @torch.no_grad()
    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_dataset(...) for Transfer MIA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        dataloader = DataLoader(dataset, batch_size=PROCESSING_BATCH_SIZE, shuffle=False)
        all_scores = []
        for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
            self.logger.print_it_same_line(f"Transfer MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for inference...", console_only=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scores = self._features(model=model, inputs=batch_x, true_labels=batch_y)
            all_scores.append(scores.cpu())
        self.logger.set_logger_newline(console_only=True)
        all_scores = torch.cat(all_scores, dim=0).numpy()
        all_decisions = (all_scores >= self.attack_threshold).astype(np.int64)
        return all_scores, all_decisions
    
    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_batch(...) for Transfer MIA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        scores = self._features(model=model, inputs=batch_x, true_labels=batch_y)
        decisions = (scores >= self.attack_threshold).to(torch.int64)
        return scores.cpu().numpy(), decisions.cpu().numpy()

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_single(...) for Transfer MIA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        score = self._features(model=model, inputs=x, true_labels=y)
        decision = (score >= self.attack_threshold).to(torch.int64)
        return score.item(), decision.item()
