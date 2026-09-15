from typing import Union
import pathlib
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data import get_defender_datas, get_attacker_datas
from src.mia.helpers.auditing_data_manager import AuditingDatasetManager
from src.mia.helpers.results_manager import ResultManager
from src.utils.configs import AttackerConfigs
from src.utils.log import Loggable, get_logger_from_configs


class BaseMIA(Loggable):
    def __init__(self, defender_model: torch.nn.Module, attacker_configs: AttackerConfigs):
        logger = get_logger_from_configs(attacker_configs.log)
        super().__init__(logger=logger)
        self.logger.print_it('Setting up and MIA attacker. First thing to do is sampling the auditing dataset...')
        self.defender_model = defender_model

        self.base_dataset_configs = attacker_configs.dataset.base
        self.audit_configs = attacker_configs.dataset.auditing
        self.shadow_configs = attacker_configs.dataset.shadow
        self.model_configs = attacker_configs.model
        self.attack_configs = attacker_configs.attack
        self.attacker_hash = attacker_configs.hash
        self.seed = self.base_dataset_configs.seed

        self.defender_datasets = get_defender_datas(dataset=self.base_dataset_configs.name,
                                datasets_folder=self.base_dataset_configs.data_folder,
                                def_split=self.base_dataset_configs.def_split,
                                att_split=self.base_dataset_configs.att_split,
                                seed=self.seed,
                                logger=self.logger)
        self.attacker_data_distribution = get_attacker_datas(dataset=self.base_dataset_configs.name,
                                                        datasets_folder=self.base_dataset_configs.data_folder,
                                                        def_split=self.base_dataset_configs.def_split,
                                                        att_split=self.base_dataset_configs.att_split,
                                                        seed=self.seed,
                                                        logger=self.logger)
        self.audit_manager = AuditingDatasetManager(defender_datasets=self.defender_datasets,
                                                    configs=self.audit_configs,
                                                    logger=logger)
        self.results_manager = ResultManager()

        # self.defender_dataset = defender_dataset
        # self.seed = attacker_configs.audit.seed
        # self.audit_manager = AuditingDatasetManager(original_datasets=defender_dataset,
        #                                             configs=attacker_configs.audit,
        #                                             logger=logger)
        # self.results_manager = ResultManager()
        # self.audit_configs=attacker_configs.audit
        # self.shadow_configs=attacker_configs.shadow

    def optimize(self):
        raise NotImplementedError('MIA should implement the method to optimize it!')
    
    def measure_effectiveness(self):
        raise NotImplementedError('MIA should implement the method to measure its effectiveness!')
    
    def compute_stats(self, scores: np.array, decisions: np.array = None, params: dict = None):
        """Compute aggregate and per-sample MIA results.

        ``scores`` must be ordered like the auditing dataset and larger scores must
        indicate stronger membership evidence.  Attacks with their own calibrated
        threshold should pass ``decisions``.  For score-only attacks, the decision
        threshold is selected by maximizing Youden's J statistic on the ROC curve.
        """
        params = {} if params is None else params
        audit_data = self.audit_manager.get(labels='mia')
        self.reset_logger()
        audit_labels = np.asarray([int(label) for _, label, _, _ in audit_data], dtype=np.int64)
        audit_ids = [int(sample_id) for _, _, sample_id, _ in audit_data]
        scores = np.asarray(scores).reshape(-1)
        assert scores.shape == audit_labels.shape, f"Unexpected shape for scores: {scores.shape}, expected {audit_labels.shape}"
        assert len(audit_ids) == len(scores)

        if not hasattr(self, 'audit_class_labels') or not hasattr(self, 'audit_model_correctness'):
            self.prepare_audit_analysis()
        class_labels = np.asarray(self.audit_class_labels, dtype=np.int64)
        model_correctness = np.asarray(self.audit_model_correctness, dtype=bool)
        assert class_labels.shape == audit_labels.shape
        assert model_correctness.shape == audit_labels.shape

        fpr, tpr, thresholds = roc_curve(audit_labels, scores)
        auc_score = auc(fpr, tpr)
        if decisions is None:
            if scores.dtype == np.bool_ or np.all(np.isin(scores, [0, 1])):
                decision_threshold = 0.5
            else:
                finite = np.isfinite(thresholds)
                candidates = np.flatnonzero(finite)
                best_index = candidates[np.argmax((tpr - fpr)[finite])]
                decision_threshold = float(thresholds[best_index])
            decisions = (scores >= decision_threshold).astype(np.int64)
            decision_source = 'roc_youden'
        else:
            decisions = np.asarray(decisions).reshape(-1).astype(np.int64)
            assert decisions.shape == audit_labels.shape, f"Unexpected shape for decisions: {decisions.shape}, expected {audit_labels.shape}"
            if not np.all(np.isin(decisions, [0, 1])):
                raise ValueError('Membership decisions must contain only 0 (non-member) and 1 (member).')
            decision_threshold = None
            decision_source = 'attack'

        correct = decisions == audit_labels
        tp_mask = (decisions == 1) & (audit_labels == 1)
        tn_mask = (decisions == 0) & (audit_labels == 0)
        fp_mask = (decisions == 1) & (audit_labels == 0)
        fn_mask = (decisions == 0) & (audit_labels == 1)
        tp, tn, fp, fn = (int(mask.sum()) for mask in (tp_mask, tn_mask, fp_mask, fn_mask))

        def safe_div(numerator, denominator):
            return float(numerator / denominator) if denominator else 0.0

        attack_accuracy = float(correct.mean()) if len(correct) else 0.0
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_div(2 * precision * recall, precision + recall)
        tpr_at_fpr = {
            str(target): float(np.max(tpr[fpr <= target])) if np.any(fpr <= target) else 0.0
            for target in (0.001, 0.01, 0.1)
        }
        per_sample = [
            {
                'dataset_id': sample_id,
                'membership_label': int(label),
                'membership': 'member' if label else 'non_member',
                'score': float(score),
                'predicted_membership': int(decision),
                'correctly_identified': bool(is_correct),
                'class_label': int(class_label),
                'model_prediction_correct': bool(model_correct),
            }
            for sample_id, label, score, decision, is_correct, class_label, model_correct in
            zip(audit_ids, audit_labels, scores, decisions, correct, class_labels, model_correctness)
        ]

        def group_stats(mask):
            group_labels = audit_labels[mask]
            group_scores = scores[mask]
            group_decisions = decisions[mask]
            group_correct = group_decisions == group_labels
            group_tp = int(((group_decisions == 1) & (group_labels == 1)).sum())
            group_tn = int(((group_decisions == 0) & (group_labels == 0)).sum())
            group_fp = int(((group_decisions == 1) & (group_labels == 0)).sum())
            group_fn = int(((group_decisions == 0) & (group_labels == 1)).sum())
            if len(group_labels) == 0:
                return {
                    'n_samples': 0, 'n_members': 0, 'n_non_members': 0,
                    'attack_accuracy': None, 'balanced_accuracy': None,
                    'precision': None, 'recall': None, 'f1': None, 'specificity': None,
                    'true_positives': 0, 'true_negatives': 0,
                    'false_positives': 0, 'false_negatives': 0,
                    'auc': None, 'tpr_at_fpr_0.001': None,
                    'tpr_at_fpr_0.01': None, 'tpr_at_fpr_0.1': None,
                }
            group_recall = safe_div(group_tp, group_tp + group_fn)
            group_specificity = safe_div(group_tn, group_tn + group_fp)
            group_precision = safe_div(group_tp, group_tp + group_fp)
            output = {
                'n_samples': int(mask.sum()),
                'n_members': int(group_labels.sum()),
                'n_non_members': int((group_labels == 0).sum()),
                'attack_accuracy': float(group_correct.mean()) if len(group_correct) else None,
                'balanced_accuracy': float((group_recall + group_specificity) / 2),
                'precision': group_precision,
                'recall': group_recall,
                'f1': safe_div(2 * group_precision * group_recall,
                               group_precision + group_recall),
                'specificity': group_specificity,
                'true_positives': group_tp,
                'true_negatives': group_tn,
                'false_positives': group_fp,
                'false_negatives': group_fn,
            }
            if len(np.unique(group_labels)) == 2:
                group_fpr, group_tpr, _ = roc_curve(group_labels, group_scores)
                output['auc'] = float(auc(group_fpr, group_tpr))
                for target in (0.001, 0.01, 0.1):
                    output[f'tpr_at_fpr_{target}'] = (
                        float(np.max(group_tpr[group_fpr <= target]))
                        if np.any(group_fpr <= target) else 0.0
                    )
            else:
                output.update({'auc': None, 'tpr_at_fpr_0.001': None,
                               'tpr_at_fpr_0.01': None, 'tpr_at_fpr_0.1': None})
            return output

        per_class = {
            str(class_label): group_stats(class_labels == class_label)
            for class_label in np.unique(class_labels)
        }
        valid_class_auc = [item['auc'] for item in per_class.values() if item['auc'] is not None]
        valid_class_tpr = {
            target: [item[f'tpr_at_fpr_{target}'] for item in per_class.values()
                     if item[f'tpr_at_fpr_{target}'] is not None]
            for target in (0.001, 0.01, 0.1)
        }
        by_model_correctness = {
            'correct': group_stats(model_correctness),
            'incorrect': group_stats(~model_correctness),
        }
        valid_auc_classes = [label for label, item in per_class.items() if item['auc'] is not None]
        worst_auc_class = (max(valid_auc_classes, key=lambda label: per_class[label]['auc'])
                           if valid_auc_classes else None)
        results = {'auc': auc_score,
                'attack_accuracy': attack_accuracy,
                'balanced_accuracy': float((recall + specificity) / 2),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'membership_advantage': float(np.max(tpr - fpr)),
                'tpr_at_fpr_0.001': tpr_at_fpr['0.001'],
                'tpr_at_fpr_0.01': tpr_at_fpr['0.01'],
                'tpr_at_fpr_0.1': tpr_at_fpr['0.1'],
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'n_audit_samples': int(len(audit_labels)),
                'n_members': int(audit_labels.sum()),
                'n_non_members': int((audit_labels == 0).sum()),
                'recovered_member_count': tp,
                'decision_source': decision_source,
                'decision_threshold': decision_threshold,
                'correctly_identified_ids': [audit_ids[i] for i in np.flatnonzero(correct)],
                'recovered_member_ids': [audit_ids[i] for i in np.flatnonzero(tp_mask)],
                'correctly_rejected_non_member_ids': [audit_ids[i] for i in np.flatnonzero(tn_mask)],
                'false_positive_ids': [audit_ids[i] for i in np.flatnonzero(fp_mask)],
                'missed_member_ids': [audit_ids[i] for i in np.flatnonzero(fn_mask)],
                'per_sample': per_sample,
                'per_class': per_class,
                'macro_class_auc': float(np.mean(valid_class_auc)) if valid_class_auc else None,
                'worst_class_auc': float(np.max(valid_class_auc)) if valid_class_auc else None,
                'best_class_auc': float(np.min(valid_class_auc)) if valid_class_auc else None,
                'worst_class_auc_label': int(worst_auc_class) if worst_auc_class is not None else None,
                'worst_class_tpr_at_fpr_0.001': max(valid_class_tpr[0.001], default=None),
                'worst_class_tpr_at_fpr_0.01': max(valid_class_tpr[0.01], default=None),
                'worst_class_tpr_at_fpr_0.1': max(valid_class_tpr[0.1], default=None),
                'by_model_correctness': by_model_correctness,
                'tpr': tpr.tolist(),
                'fpr': fpr.tolist(),
                'roc': thresholds.tolist()}
        self.results_manager.add_results(params=params,
                                        results=results)
        return results

    def prepare_audit_analysis(self, device: Union[torch.device, str] = 'cpu'):
        """Cache original class labels and final-model correctness for each audit sample."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        dataset = self.audit_manager.get(labels='original')
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        self.defender_model = self.defender_model.to(device)
        self.defender_model.eval()
        class_labels = []
        correctness = []
        with torch.no_grad():
            for inputs, labels, _, _ in loader:
                outputs = self.defender_model(inputs.to(device))
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions = torch.argmax(outputs, dim=1).cpu()
                class_labels.extend(int(label) for label in labels)
                correctness.extend(bool(value) for value in predictions == labels)
        self.audit_class_labels = class_labels
        self.audit_model_correctness = correctness

    def add_run_metrics(self, **metrics):
        """Attach run-level metrics (for example model utility) to every result."""
        for params in self.results_manager.list_configs():
            for name, value in metrics.items():
                self.results_manager.set_result(params, name, value)
            result = self.results_manager.get_result(params)
            query_count = result.get('target_model_queries_total', 0)
            recovered_count = result.get('recovered_member_count', 0)
            recovered_per_1000 = (1000.0 * recovered_count / query_count
                                  if query_count else None)
            self.results_manager.set_result(
                params, 'recovered_members_per_1000_target_queries', recovered_per_1000)

    def get_best_result(self, metric_name: str = 'auc', mode: str = 'max', aggregate: str = 'mean'):
        assert metric_name in ['auc', 'tpr', 'fpr', 'roc']
        assert mode in ['min', 'max']
        best_params, best_result = self.results_manager.get_best(metric_name=metric_name,
                                                            mode=mode,
                                                            aggregate=aggregate)
        return best_params, best_result
    
    def summarize_results(self):
        return self.results_manager.to_rows()
    
    def save_results_to_json(self, file: Union[str, pathlib.Path]):
        file = pathlib.Path(file)
        self.results_manager.save_json(str(file))
        self.results_manager.save_mia_artifacts(file.parent)

    def get_stats(self, scores: np.array, plot: bool = False):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        fpr, tpr, roc = roc_curve(audit_labels, scores)
        auc_score = auc(fpr, tpr)
        if plot:
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f"ROC_MIA.png")
        return auc_score, tpr, fpr, roc
    
    def compute_auc(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        fpr, tpr, _ = roc_curve(audit_labels, scores)
        auc_score = auc(fpr, tpr)
        return auc_score
    
    def compute_tpr(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        _, tpr, _ = roc_curve(audit_labels, scores)
        return tpr.tolist()
    
    def compute_fpr(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        fpr, _, _ = roc_curve(audit_labels, scores)
        return fpr.tolist()
    
    def compute_roc(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        _, _, roc = roc_curve(audit_labels, scores)
        return roc.tolist()
    
    @staticmethod
    def get_device(dev_str: str = 'cpu'):
        # Set appropriate devices
        if torch.cuda.is_available() and dev_str != 'cpu':
            dev_str = 'cuda:{}'.format(dev_str)
            device = torch.device(dev_str)
        elif torch.backends.mps.is_available() and dev_str != 'cpu':
            dev_str = 'mps'
            device = torch.device(dev_str)
        else:
            device = torch.device('cpu')
        return device
