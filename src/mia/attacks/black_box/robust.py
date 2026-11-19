import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from collections import Counter
from typing import Union
import time

# The default a of each dataset. The paper's values where it reports one (App. B.2.2); the others were found with the
# paper's procedure by tests/tune_robust_a.py, run once offline, not by the attack.
# TODO: tune svhn, fmnist, texas, news, gtsrb and tinyimagenet with tests/tune_robust_a.py, once the training defaults
# Issue URL: https://github.com/AndAgio/mia_bench/issues/53
# of the image datasets are checked: a tuned a depends on how the shadow models are trained.
DEFAULT_A = {'cifar10': 0.3, 'cinic10': 0.3, 'cifar100': 0.6, 'imagenet': 1.0, 'purchase': 0.2}


def offline_p_x(p_x_out: np.ndarray, a: float) -> np.ndarray:
    """The paper's equation 10: Pr(x) with the IN models approximated as Pr(x)_IN = a Pr(x)_OUT + 1 - a."""
    return 0.5 * ((1 + a) * p_x_out + (1 - a))


def rmia_scores(ratio_x: np.ndarray, ratio_z: np.ndarray, gamma: float) -> np.ndarray:
    """The share of population samples z that each x gamma-dominates: Pr(x|theta)/Pr(x) > gamma Pr(z|theta)/Pr(z)."""
    return np.mean(ratio_x[:, np.newaxis] / ratio_z[np.newaxis, :] > gamma, axis=1)


class RobustMIA(BaseMIA):
    """RMIA: a likelihood ratio between the sample and population samples, computed with reference models.

    Paper: "Low-Cost High-Power Membership Inference Attacks",
    https://arxiv.org/pdf/2312.03262.
    """
    # The paper reports ROC curves only, so the attack decision rule calibrates the threshold.
    has_own_decisions = False

    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        assert self.shadow_configs.mode == self.attack_configs.mode, f"Whenever working with RobustMIA the mode for shadow datasets and attack should be the same!"
        self.mode = self.attack_configs.mode
        if self.mode == 'online' and self.shadow_configs.n_shadow_datasets < 2:
            raise ValueError(f"Online RobustMIA needs at least 2 shadow models (--n_shadows), got "
                             f"{self.shadow_configs.n_shadow_datasets}: it averages the models trained with each "
                             f"sample (IN) and those without it (OUT), and with 1 model no model is IN.")
        if self.mode == 'offline':
            a_values = self.a_values()
            if not all(0 <= a <= 1 for a in a_values):
                raise ValueError(f"RobustMIA's a (--attacker_robust_a) must lie between 0 and 1, got {a_values}.")
        # The population Z and its probabilities under each shadow model and the target, drawn once so that every value
        # of a and every scored dataset (auditing and calibration samples) is compared with the same population.
        self.population_ids = None
        self.population_probs = None
        self.population_target_probs = None
        # Which shadow models were trained on each population sample, one column per model.
        self.population_trained = None
        self.logger.print_it(f"Working with RobustMIA in {self.mode.upper()} mode!")
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('RobustMIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('RobustMIA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('RobustMIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('RobustMIA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def a_values(self) -> list[float]:
        """The values of the paper's a to score with in offline mode, one result each: the given ones, or else the
        dataset's default (DEFAULT_A)."""
        a = self.attack_configs.a
        if a is not None:
            return [float(value) for value in (a if isinstance(a, list) else [a])]
        dataset = self.base_dataset_configs.name
        if dataset not in DEFAULT_A:
            raise ValueError(f"RobustMIA has no default a for {dataset}: pass --attacker_robust_a, or find one with "
                             f"tests/tune_robust_a.py and add it to DEFAULT_A in src/mia/attacks/black_box/robust.py.")
        return [DEFAULT_A[dataset]]

    def score(self, dataset: Dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample with the share of population samples it dominates, once for each value of a offline."""
        gamma = self.attack_configs.gamma
        if self.mode == 'online':
            # The online test averages the IN and OUT models: it has no a, so its result carries no a either.
            return [MembershipScores(self.compute_with_a(dataset=dataset, a=None, gamma=gamma, device=device))]
        return [MembershipScores(self.compute_with_a(dataset=dataset,
                                                     a=a,
                                                     gamma=gamma,
                                                     device=device),
                                 params={'a': a}) for a in self.a_values()]

    def population_exclusions(self) -> set:
        """Pool samples whose Pr(z) cannot be computed like Pr(x): offline, those every shadow model was trained on
        (no OUT model); online, also those no shadow model was trained on (no IN model)."""
        shadow_data = self.shadow_manager.shadow_data
        model_ids = self.shadow_manager.get_all_model_indeces()
        counts = Counter(sample for model in model_ids for sample in shadow_data.get_map(model))
        excluded = {sample for sample, count in counts.items() if count == len(model_ids)}
        if self.mode == 'online':
            pool = shadow_data.attacker_data_distribution.get('all').get_indices(mode='original')
            excluded |= {sample for sample in pool if sample not in counts}
        return excluded

    def ensure_population(self, device: Union[torch.device, str] = 'cpu'):
        """Draw the population Z once and compute its probabilities under each shadow model and the target."""
        if self.population_probs is not None:
            return
        self.population_ids = self.shadow_manager.sample_random_population_indices(
            num_data=self.attack_configs.random_pop_size, exclude=self.population_exclusions())
        population = self.shadow_manager.get_random_population(indices=self.population_ids, labels='original')
        self.reset_logger()
        model_ids = self.shadow_manager.get_all_model_indeces()
        trained_sets = [set(self.shadow_manager.shadow_data.get_map(model)) for model in model_ids]
        self.population_trained = np.array([[z in trained for trained in trained_sets] for z in self.population_ids],
                                           dtype=bool).reshape(len(self.population_ids), len(model_ids))
        self.logger.print_it(f"Computing p(z)...")
        self.population_probs = self.compute_p_x_theta(audit_dataset=population,
                                                       models_for_sample=[model_ids for _ in self.population_ids],
                                                       device=device)
        self.logger.print_it(f"Computing p(z|theta) with defender model...")
        self.population_target_probs = self.compute_p_x_theta(
            audit_dataset=population, models_for_sample=[[self.defender_model] for _ in self.population_ids],
            device=device)[:, 0]

    def population_ratios(self, a: float = None) -> np.ndarray:
        """Pr(z|theta) / Pr(z) for each population sample, with Pr(z) computed like Pr(x), as both of the authors'
        codebases do: offline from the shadow models not trained on z, through a; online as the mean of those trained
        on z and those not."""
        trained = self.population_trained
        p_z_out = (self.population_probs * ~trained).sum(axis=1) / (~trained).sum(axis=1)
        if self.mode == 'offline':
            p_z = offline_p_x(p_z_out, a)
        else:
            p_z_in = (self.population_probs * trained).sum(axis=1) / trained.sum(axis=1)
            p_z = 0.5 * p_z_in + 0.5 * p_z_out
        return self.population_target_probs / (p_z + 1e-15)

    def compute_with_a(self, dataset: Dataset, a: float = None, gamma: float = 2,
                       device: Union[torch.device, str] = 'cpu'):
        """Score the dataset for one value of a (offline; None online): the share of population samples it dominates."""
        # Compute P(x|theta) for all samples in the auditing dataset
        if self.mode == 'offline':
            self.logger.print_it('Computing RobustMIA scores in OFFLINE mode with a={:.3f}'.format(a))
        elif self.mode == 'online':
            self.logger.print_it('Computing RobustMIA scores in ONLINE mode')
        start = time.time()
        audit_dataset = dataset
        audit_dataset_indices = dataset.get_indices(mode='original')
        self.reset_logger()

        left_lr = self.compute_p_x_theta_over_p_x(dataset=audit_dataset, dataset_indices=audit_dataset_indices, a=a,
                                                  device=device)

        self.ensure_population(device=device)
        right_lr = self.population_ratios(a)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('RobustMIA attacker: LR computation done! Time taken to compute LR: {}:{:02d}:{:02d}...'.format(h, m, s))
        self.logger.print_it(f'RobustMIA attacker: scoring with gamma={gamma}')
        return rmia_scores(left_lr, right_lr, gamma)


    def compute_p_x_theta_over_p_x(self, dataset: Dataset, dataset_indices: list[int], a: float = None,
                                   device: Union[torch.device, str] = 'cpu'):
        """Ratio Pr(x|theta) / Pr(x) for each sample, with Pr(x) from the shadow models (offline, through a)."""
        if self.mode == 'offline':
            # Load indices for all shadow models and set them as offline models
            all_models_indices = self.shadow_manager.get_all_model_indeces()
            trained_out_shadow_models_for_sample = [all_models_indices for _ in range(len(dataset))]
            # Indices for online shadow models are empty
            trained_in_shadow_models_for_sample = [[] for _ in range(len(dataset))]
        if self.mode == 'online':
            all_models_indices = self.shadow_manager.get_all_model_indeces()
            trained_in_shadow_models_for_sample = []
            trained_out_shadow_models_for_sample = []
            for index in dataset_indices:
                in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index)
                out_models_indices = [index for index in all_models_indices if index not in in_models_indices]
                trained_in_shadow_models_for_sample.append(in_models_indices)
                trained_out_shadow_models_for_sample.append(out_models_indices)
                # self.logger.print_it(f"For sample with index {index}, I found {len(in_models_indices)} online shadow models and {len(out_models_indices)} offline shadow models...")

        if self.mode == 'online':
            self.logger.print_it(f"Computing p(x) with in models...")
            p_x_thetas_in = self.compute_p_x_theta(audit_dataset=dataset,
                                                    models_for_sample=trained_in_shadow_models_for_sample,
                                                    device=device)
        self.logger.print_it(f"Computing p(x) with out models...")
        p_x_thetas_out = self.compute_p_x_theta(audit_dataset=dataset,
                                                    models_for_sample=trained_out_shadow_models_for_sample,
                                                    device=device)
        # p_x_theta_out = np.mean(p_x_thetas_out, axis=1)
        p_x_out = np.mean(p_x_thetas_out, axis=1)
        if self.mode == 'offline':
            p_x = offline_p_x(p_x_out, a)
        elif self.mode == 'online':
            # p_x_theta_in = np.mean(p_x_thetas_in, axis=1)
            p_x_in = np.mean(p_x_thetas_in, axis=1)
            p_x = 0.5 * p_x_in +  0.5 * p_x_out

        self.logger.print_it(f"Computing p(x|theta) with the defender model...")
        p_x_thetas_defender = self.compute_p_x_theta(audit_dataset=dataset,
                                                models_for_sample=[[self.defender_model] for _ in range(len(dataset))],
                                                device=device)
        p_x_thetas_defender = p_x_thetas_defender.squeeze()

        ratio = p_x_thetas_defender/(p_x + 1e-15)
        return ratio


    def compute_p_x_theta(self, audit_dataset: Dataset, models_for_sample: list[list[Union[int, torch.nn.Module]]], device: Union[torch.device, str] = 'cpu'):
        tot_samples = len(audit_dataset)
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        p_x_thetas = np.zeros((len(audit_dataset), len(models_for_sample[0])))
        s = time.time()
        self.logger.print_it(f"Computing p(x|theta) for all {tot_samples} samples. This may take a while...")
        for sample_index, (sample, label, _, _) in enumerate(audit_loader):
            self.logger.print_it_same_line(f'Computing p(x|theta) for sample {sample_index+1}/{tot_samples}...', console_only=True)
            models = models_for_sample[sample_index]
            for model_index, model in enumerate(models):
                if isinstance(model, torch.nn.Module):
                    pass
                elif isinstance(model, int):
                    model = self.shadow_manager.get_model(model)
                else:
                    raise ValueError('Model should be either a torch Module or an integer referring to the id of the shadow model!')
                p_x_theta = RobustMIA.get_prob(model=model,
                                        data=sample,
                                        target=label,
                                        device=device)
                p_x_thetas[sample_index, model_index] = p_x_theta
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Computed all p(x|theta) in {time.time() - s} seconds.")
        return p_x_thetas

    @staticmethod
    def get_prob(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = RobustMIA.get_device(dev_str=device)
        model.eval()
        lab=target[0].item()
        with torch.no_grad():
            output = model(data.to(device))
            softmax_output = torch.nn.Softmax(dim=1)(output)
            prob = softmax_output[0][lab].detach().cpu().item()
        return prob
