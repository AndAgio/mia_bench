import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import norm
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import TrainConfigs, AttackerConfigs
from src.utils import convert_to_hms
from typing import Union
import time


class LiRA(BaseMIA):
    """LiRA: a likelihood-ratio test on the sample's score, using shadow models.

    Paper: "Membership inference attacks from first principles",
    https://ieeexplore.ieee.org/abstract/document/9833649.
    Scores are log likelihood ratios; larger values indicate membership, and
    zero corresponds to equal in/out likelihoods.
    """
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with LiRA!")
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('LiRA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('LiRA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('LiRA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('LiRA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        self.logger.print_it('Computing LiRA scores. This may take a while...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        audit_dataset_indices = self.audit_manager.get_all_ids()

        all_models_indices = self.shadow_manager.get_all_model_indeces()
        trained_in_shadow_models_for_sample = []
        trained_out_shadow_models_for_sample = []
        for index in audit_dataset_indices:
            in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index)
            out_models_indices = [index for index in all_models_indices if index not in in_models_indices]
            trained_in_shadow_models_for_sample.append(in_models_indices)
            trained_out_shadow_models_for_sample.append(out_models_indices)

        self.logger.print_it(f"Computing phis with in models...")
        phis_in = self.compute_phis(audit_dataset=audit_dataset,
                                        models_for_sample=trained_in_shadow_models_for_sample,
                                        device=device)

        self.logger.print_it(f"Computing phis with out models...")
        phis_out = self.compute_phis(audit_dataset=audit_dataset,
                                        models_for_sample=trained_out_shadow_models_for_sample,
                                        device=device)

        self.logger.print_it(f"Computing phis with defender model...")
        phis_defender = self.compute_phis(audit_dataset=audit_dataset,
                                        models_for_sample=[[self.defender_model] for _ in range(len(audit_dataset))],
                                        device=device)
        scores = lira_log_likelihood_ratio(phis_in, phis_out, phis_defender)
        # self.logger.print_it(f"LiRA attacker: scores = {scores}")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('LiRA attacker: score computation done! Time taken to compute LR: {}:{:02d}:{:02d}...'.format(h, m, s))

        metrics = self.compute_stats(scores)
        return metrics

        
    def compute_phis(self, audit_dataset: Dataset, models_for_sample: list[list[Union[int, torch.nn.Module]]], device: Union[torch.device, str] = 'cpu'):
        tot_samples = len(audit_dataset)
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        phis = np.zeros((len(audit_dataset), len(models_for_sample[0])))
        s = time.time()
        self.logger.print_it(f"Computing phi for all {tot_samples} samples. This may take a while...")
        for sample_index, (sample, label, _, _) in enumerate(audit_loader):
            self.logger.print_it_same_line(f"Computing phi for sample {sample_index+1}/{tot_samples}...", console_only=True)
            models = models_for_sample[sample_index]
            for model_index, model in enumerate(models):
                if isinstance(model, torch.nn.Module):
                    pass
                elif isinstance(model, int):
                    model = self.shadow_manager.get_model(model)
                else:
                    raise ValueError('Model should be either a torch Module or an integer referring to the id of the shadow model!')
                phi = LiRA.compute_phi(model=model,
                                        data=sample,
                                        target=label,
                                        device=device)
                phis[sample_index, model_index] = phi.item()
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Computed phis in {time.time() - s} seconds.")
        return phis

    @staticmethod
    def compute_phi(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = LiRA.get_device(dev_str=device)
    
        model.eval()
        with torch.no_grad():
            output = model(data.to(device))

        # log(p_true / (1 - p_true)) = true_logit - logsumexp(other_logits).
        # Work directly in logit space so confident models do not underflow all
        # non-target probabilities to zero. CPU float64 also supports MPS models.
        logits = output.detach().cpu().to(torch.float64)
        if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
            raise ValueError("LiRA requires finite logits with at least two classes")
        labels = target.detach().cpu().long().reshape(-1, 1)
        true_logits = logits.gather(1, labels).reshape(-1)
        other_logits = logits.scatter(1, labels, float('-inf'))
        return true_logits - torch.logsumexp(other_logits, dim=1)


def lira_log_likelihood_ratio(phis_in, phis_out, phis_target):
    """Fit regularized in/out Gaussians and return finite log likelihood ratios."""
    targets = np.asarray(phis_target, dtype=np.float64).reshape(-1)
    references = [np.asarray(values, dtype=np.float64) for values in (phis_in, phis_out)]
    if not len(targets) or not np.isfinite(targets).all():
        raise ValueError("LiRA target scores must be nonempty and finite")
    for values in references:
        if (values.ndim != 2 or values.shape[0] != len(targets) or values.shape[1] == 0
                or not np.isfinite(values).all()):
            raise ValueError("LiRA needs finite, nonempty in/out references for every target")

    # The reference implementation also adds 1e-30 to standard deviations:
    # https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py
    # A single in/out reference (two shadows overall) has exactly zero variance.
    log_densities = [norm.logpdf(targets, loc=values.mean(axis=1),
                                scale=values.std(axis=1) + 1e-30)
                     for values in references]
    scores = log_densities[0] - log_densities[1]
    if not np.isfinite(scores).all():
        raise ValueError("LiRA log likelihood ratios are non-finite; check reference scores and scales")
    return scores
