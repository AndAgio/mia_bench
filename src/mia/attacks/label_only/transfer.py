
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from src.data.helpers import MultiDatasets, MergedDataset, TargetOverrideDataset
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.mia.attacks.label_only.helpers.threshold_samples import balanced_threshold_samples


PROCESSING_BATCH_SIZE = 128


class TransferMIA(BaseMIA):
    """Label-only transfer attack: a shadow model trained on the target's labels scores the samples.

    Paper: "Membership Leakage in Label-Only Exposures" (the transfer attack), https://arxiv.org/abs/2007.15528.
    Documentation: docs/attacks/transfer.md.
    The target labels the shadow dataset, the shadow model learns those labels, and each sample is scored on the
    shadow model alone, by its loss at its true label, its confidence or its entropy, so scoring sends no query to the
    target.
    """
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Sample the shadow dataset and build the shadow model."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.attack_threshold = None
        self.logger.print_it(f"Working with Transfer MIA!")
        assert self.shadow_configs.mode == 'offline', f"Transfer MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        assert self.shadow_configs.n_shadow_datasets == 1, \
            f"Transfer MIA trains one shadow model, set by the configs, got {self.shadow_configs.n_shadow_datasets}."
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Transfer MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Transfer MIA attacker: definition of transfer model...')
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        """Relabel the shadow dataset with the target's predicted labels and train the shadow model on them."""
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
        """Return the shadow dataset labelled with the target's predicted labels, the only output the attack reads."""
        shadow_dataset = self.shadow_manager.get_dataset(index=0, labels='original')
        shadow_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=train_config.batch_size, shuffle=False)
        all_relabels = []
        device = self.get_device(train_config.device)
        self.defender_model.to(device).eval()
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
        """Choose the score threshold that best separates the shadow model's members and non-members by accuracy.

        The paper suggests a threshold from shadow samples held out as non-members alone, which is what the calibrated
        decision rule does; this one also uses the shadow model's members (see docs/attacks/transfer.md).
        """
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        # _features() orients every score so that larger values mean stronger membership evidence.
        shadow_model = self.shadow_manager.shadow_models.get(index=0).to(device)
        # As many members as non-members, every one there is: each costs one pass through the shadow model only.
        shadow_member_dataset, shadow_nonmember_dataset, n_per_side = balanced_threshold_samples(
            self.shadow_manager, None, self.seed)
        self.logger.print_it(f"Transfer MIA attacker: choosing the threshold on {n_per_side} members and "
                             f"{n_per_side} non-members of the shadow model.")
        datasets_list = [shadow_member_dataset, shadow_nonmember_dataset]
        self.regression_dataset = MergedDataset(*datasets_list)
        mia_labels = torch.tensor([1] * len(shadow_member_dataset) + [0] * len(shadow_nonmember_dataset))
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
                all_feats.append(batch_feats.cpu())
        self.logger.set_logger_newline(console_only=True)
        # The threshold that maximizes accuracy, with >=: a sample is a member when its loss is at most the
        # threshold (the paper's Alg. 1), and every score is oriented by _features().
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

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample with the transfer model's score on it, and threshold it."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        if self.attack_threshold is None:
            self.logger.print_it('Transfer MIA attacker: finding optimal threshold on shadow dataset...')
            start = time.time()
            self.find_optimal_threshold(device=device)
            stop = time.time()
            h, m, s = convert_to_hms(stop-start)
            self.logger.print_it('Transfer MIA attacker: Done finding optimal threshold. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        self.logger.print_it('Transfer MIA attacker: measuring attack effectiveness...')
        start = time.time()
        shadow_model = self.shadow_manager.shadow_models.get(index=0).to(device)
        scores, decisions = self.infer_dataset(model=shadow_model,
                                                dataset=dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Transfer MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        return [MembershipScores(scores, decisions=decisions)]

    @torch.no_grad()
    def _features(self, model: torch.nn.Module, inputs: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Return each input's score on the shadow model, larger for likelier members.

        The loss is at the true label (the paper's Eq. 1). Confidence already has this direction; loss and entropy are
        negated. The paper's entropy is divided by log K, which ranks the samples the same.
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
        """Return each sample's score on the shadow model and whether it reaches the threshold."""
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
        """Return the score of each sample of a batch on the shadow model and whether it reaches the threshold."""
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
        """Return one sample's score on the shadow model and whether it reaches the threshold."""
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
