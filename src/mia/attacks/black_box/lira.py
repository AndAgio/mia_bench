import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import norm
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import TrainConfigs, AttackerConfigs
from src.utils import convert_to_hms
from typing import Union
import time

# The fewest shadow models of each mode, so that every Gaussian fitted to a sample has at least two values, hence a
# variance: online, two models trained with the sample and two without it (each sample is in half of the shadow models,
# rounded down); offline, two models without it.
MIN_SHADOW_MODELS = {'online': 4, 'offline': 2}


class LiRA(BaseMIA):
    """LiRA: a likelihood-ratio test on the target's logit-scaled confidence in each sample, against shadow models.

    Paper: "Membership Inference Attacks From First Principles", https://arxiv.org/abs/2112.03570.
    Documentation: docs/attacks/lira.md.
    Online, the score is the log likelihood ratio between Gaussians fitted to the shadow models trained with and
    without the sample; offline, the paper's one-sided test against the Gaussian of the shadow models trained without
    it.
    """
    # The paper reports ROC curves only, so the attack decision rule calibrates the threshold.
    has_own_decisions = False

    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Check the mode and the number of shadow models, then sample the shadow datasets and build the models."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.mode = self.attack_configs.mode
        if self.shadow_configs.mode != self.mode:
            raise ValueError(f"LiRA's shadow datasets must be sampled in its own mode, {self.mode}, "
                             f"not {self.shadow_configs.mode}.")
        n_shadows, needed = self.shadow_configs.n_shadow_datasets, MIN_SHADOW_MODELS[self.mode]
        if n_shadows < needed:
            reason = ("each sample is in half of them, and the Gaussians of the models trained with and without it "
                      "each need two values for a variance" if self.mode == 'online' else
                      "the Gaussian of the models trained without each sample needs two values for a variance")
            hint = "; with fewer, offline_lira needs only 2" if self.mode == 'online' else ""
            raise ValueError(f"{self.mode.capitalize()} LiRA needs at least {needed} shadow models (--n_shadows), got "
                             f"{n_shadows}: {reason}{hint}.")
        self.logger.print_it(f"Working with LiRA in {self.mode.upper()} mode!")
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('LiRA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('LiRA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        """Train every shadow model on its shadow dataset."""
        self.logger.print_it('LiRA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('LiRA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def score(self, dataset: Dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample: online, the log likelihood ratio of the target's phi under its IN and OUT shadow models;
        offline, the one-sided test against its OUT shadow models."""
        self.logger.print_it('Computing LiRA scores. This may take a while...')
        start = time.time()
        audit_dataset = dataset
        audit_dataset_indices = dataset.get_indices(mode='original')

        all_models_indices = self.shadow_manager.get_all_model_indeces()
        trained_in_shadow_models_for_sample = []
        trained_out_shadow_models_for_sample = []
        for index in audit_dataset_indices:
            in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index)
            out_models_indices = [index for index in all_models_indices if index not in in_models_indices]
            trained_in_shadow_models_for_sample.append(in_models_indices)
            trained_out_shadow_models_for_sample.append(out_models_indices)

        if self.mode == 'online':
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
        if self.mode == 'online':
            scores = lira_log_likelihood_ratio(phis_in, phis_out, phis_defender)
        else:
            scores = lira_offline_scores(phis_out, phis_defender)
        # self.logger.print_it(f"LiRA attacker: scores = {scores}")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('LiRA attacker: score computation done! Time taken to compute LR: {}:{:02d}:{:02d}...'.format(h, m, s))
        return [MembershipScores(scores)]

        
    def compute_phis(self, audit_dataset: Dataset, models_for_sample: list[list[Union[int, torch.nn.Module]]], device: Union[torch.device, str] = 'cpu'):
        """Return phi for each sample (row) under each of its models (column): shadow model ids or modules."""
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
        """Return phi = log(p / (1 - p)) of the model's probability p of each sample's true class, computed stably."""
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


def checked_phis(phis_target, *phis_references):
    """Return the target's phis as a vector and each set of reference phis as a matrix, one row per target, after
    checking that they are finite and that every target has references."""
    targets = np.asarray(phis_target, dtype=np.float64).reshape(-1)
    references = [np.asarray(values, dtype=np.float64) for values in phis_references]
    if not len(targets) or not np.isfinite(targets).all():
        raise ValueError("LiRA target scores must be nonempty and finite")
    for values in references:
        if (values.ndim != 2 or values.shape[0] != len(targets) or values.shape[1] == 0
                or not np.isfinite(values).all()):
            raise ValueError("LiRA needs finite, nonempty in/out references for every target")
    return targets, references


def lira_offline_scores(phis_out, phis_target):
    """The paper's offline one-sided test (Eq. 4), Pr[Z <= phi] with Z ~ N(mu_out, sigma_out^2) fitted per sample, as
    the score -log Pr[Z > phi]: the same ranking, without losing resolution for the largest phis."""
    targets, (references,) = checked_phis(phis_target, phis_out)
    scores = -norm.logsf(targets, loc=references.mean(axis=1), scale=references.std(axis=1) + 1e-30)
    if not np.isfinite(scores).all():
        raise ValueError("LiRA offline scores are non-finite; check reference scores and scales")
    return scores


def lira_log_likelihood_ratio(phis_in, phis_out, phis_target):
    """Fit regularized in/out Gaussians and return finite log likelihood ratios."""
    targets, references = checked_phis(phis_target, phis_in, phis_out)

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
