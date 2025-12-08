from typing import Callable, Union
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from src.data.multi import MultiDatasets
from .auditing_data_manager import AuditingDatasetManager
from src.utils.configs import AuditingDataConfigs
from src.utils.log import Loggable, SmartLogger, DumbLogger


class BaseMIA(Loggable):
    def __init__(self, victim_model: torch.nn.Module, victim_dataset: MultiDatasets, audit_configs: AuditingDataConfigs, logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        self.logger.print_it('Setting up and MIA attacker. First thing to do is sampling the auditing dataset...')
        self.victim_model = victim_model
        self.victim_dataset = victim_dataset
        self.seed = audit_configs.seed
        self.audit_manager = AuditingDatasetManager(original_datasets=victim_dataset,
                                                    configs=audit_configs,
                                                    logger=logger)

    def optimize(self):
        raise NotImplementedError('MIA should implement the method to optimize it!')
    
    def measure_effectiveness(self):
        raise NotImplementedError('MIA should implement the method to measure its effectiveness!')
    
    def compute_stats(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        tpr, fpr, roc = roc_curve(audit_labels, scores)
        auc_score = auc(fpr, tpr)
        return {'auc': auc_score,
                'tpr': tpr,
                'fpr': fpr,
                'roc': roc}

    def get_stats(self, scores: np.array, plot: bool = False):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        tpr, fpr, roc = roc_curve(audit_labels, scores)
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
            plt.savefig(f'ROC_MIA.png')
        return auc_score, tpr, fpr, roc
    
    def compute_auc(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        tpr, fpr, _ = roc_curve(audit_labels, scores)
        auc_score = auc(fpr, tpr)
        return auc_score
    
    def compute_tpr(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        tpr, _, _ = roc_curve(audit_labels, scores)
        return tpr
    
    def compute_fpr(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        _, fpr, _ = roc_curve(audit_labels, scores)
        return fpr
    
    def compute_roc(self, scores: np.array):
        audit_data = self.audit_manager.get(labels='mia')
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        _, _, roc = roc_curve(audit_labels, scores)
        return roc
