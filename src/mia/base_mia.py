from typing import Callable, Union
import pathlib
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from src.data.multi import MultiDatasets
from .auditing_data_manager import AuditingDatasetManager
from .results_manager import ResultManager
from src.utils.configs import AttackerConfigs
from src.utils.log import Loggable, get_logger_from_configs


class BaseMIA(Loggable):
    def __init__(self, victim_model: torch.nn.Module, victim_dataset: MultiDatasets, attacker_configs: AttackerConfigs):
        logger = get_logger_from_configs(attacker_configs.log)
        super().__init__(logger=logger)
        self.logger.print_it('Setting up and MIA attacker. First thing to do is sampling the auditing dataset...')
        self.victim_model = victim_model
        self.victim_dataset = victim_dataset
        self.seed = attacker_configs.audit.seed
        self.audit_manager = AuditingDatasetManager(original_datasets=victim_dataset,
                                                    configs=attacker_configs.audit,
                                                    logger=logger)
        self.results_manager = ResultManager()
        self.audit_configs=attacker_configs.audit
        self.shadow_configs=attacker_configs.shadow
        self.model_configs=attacker_configs.model
        self.attack_configs=attacker_configs.attack
        self.attacker_hash = attacker_configs.hash

    def optimize(self):
        raise NotImplementedError('MIA should implement the method to optimize it!')
    
    def measure_effectiveness(self):
        raise NotImplementedError('MIA should implement the method to measure its effectiveness!')
    
    def compute_stats(self, scores: np.array, params: dict = {}):
        audit_data = self.audit_manager.get(labels='mia')
        self.reset_logger()
        audit_labels = [label for _, (_, label) in enumerate(audit_data)]
        fpr, tpr, roc = roc_curve(audit_labels, scores)
        auc_score = auc(fpr, tpr)
        results = {'auc': auc_score,
                'tpr': tpr.tolist(),
                'fpr': fpr.tolist(),
                'roc': roc.tolist()}
        self.results_manager.add_results(params=params,
                                        results=results)
        return results
    
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
        self.results_manager.save_json(file)

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
            plt.savefig(f'ROC_MIA.png')
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

