import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.data.multi import MultiDatasets
from src.mia.base_mia import BaseMIA
from src.mia.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from typing import Union
import time


class RMIA(BaseMIA):
    # Implementation of Low-Cost High-Power Membership Inference Attacks (https://arxiv.org/pdf/2312.03262)
    def __init__(self, 
                victim_model: torch.nn.Module,
                victim_dataset: MultiDatasets,
                attacker_configs: AttackerConfigs, 
                exp_hash: str):
        super().__init__(victim_model=victim_model, victim_dataset=victim_dataset, attacker_configs=attacker_configs, exp_hash=exp_hash)
        assert self.shadow_configs.mode == self.attack_configs.mode, f'Whenever working with RMIA the mode for shadow datasets and attack should be the same!'
        self.mode = self.attack_configs.mode
        self.logger.print_it(f'Working with RMIA in {self.mode.upper()} mode!')
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('RMIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.victim_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    exp_hash=self.exp_hash)
        self.logger.print_it('RMIA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('RMIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('RMIA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def measure_effectiveness(self, random_pop_size: int = None, alpha: Union[float, list[float]] = None, gamma: float = None, device: Union[torch.device, str] = 'cpu'):
        random_pop_size = self.attack_configs.random_pop_size if random_pop_size is None else random_pop_size
        alpha = self.attack_configs.alpha if alpha is None else alpha
        gamma = self.attack_configs.gamma if gamma is None else gamma
        if self.mode == 'online':
            self.logger.print_it('Found RMIA in ONLINE mode, setting alpha to 0 for ease')
            alphas = [0]
        elif alpha is None and self.mode == 'offline':
            self.logger.print_it('Found alpha to be None for OFFLINE RMIA, which is not possible! Testing over all alphas...')
            alphas = np.arange(0, 1.05, 0.1)
        elif isinstance(alpha, list):
            assert all([0 < al <= 1 for al in alpha]), f'All given alphas should be between 0 and 1!'
            alphas = alpha
        elif isinstance(alpha, float):
            assert 0 < alpha <= 1, f'The given alpha should be between 0 and 1! Found alpha={alpha} instead!'
            alphas = [alpha]
        scores = {alpha: self.compute_with_alpha(random_pop_size=random_pop_size,
                                                alpha=alpha,
                                                gamma=gamma,
                                                device=device) for alpha in alphas}
        metrics = {alpha: self.compute_stats(scores[alpha]) for alpha in alphas}
        # self.logger.print_it('RMIA attacker: Obtained scores are: {}'.format(metrics))
        return metrics


    def compute_with_alpha(self, random_pop_size: int, alpha: float, gamma: float = 1, device: Union[torch.device, str] = 'cpu'):
        # assert self.mode == 'offline', f'compute_with_given_alpha should be called only for OFFLINE mode!'
        # Compute P(x|theta) for all samples in the auditing dataset
        if self.mode == 'offline':
            assert 0 < alpha <= 1, f'The given alpha should be between 0 and 1! Found alpha={alpha} instead!'
            self.logger.print_it('Computing RMIA scores in OFFLINE mode with alpha={:.3f}'.format(alpha))
        elif self.mode == 'online':
            self.logger.print_it('Computing RMIA scores in ONLINE mode')
            assert alpha == 0
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        audit_dataset_indices = self.audit_manager.get_all_ids()
        self.reset_logger()

        left_lr = self.compute_p_x_theta_over_p_x(dataset=audit_dataset, dataset_indices=audit_dataset_indices, alpha=alpha, device=device)

        random_data_indices = self.shadow_manager.sample_random_population_indices(num_data=random_pop_size)
        random_data = self.shadow_manager.get_random_population(indices=random_data_indices,
                                                                labels='original')
        self.reset_logger()

        right_lr = self.compute_p_z_theta_over_p_z(dataset=random_data, device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('RMIA attacker: LR computation done! Time taken to compute LR: {}:{:02d}:{:02d}...'.format(h, m, s))

        lr_ratios = np.divide(left_lr[:, np.newaxis], right_lr)
        gamma=1
        positives_indices = lr_ratios > gamma
        positives = np.sum(positives_indices, axis=1)
        total = right_lr.shape[0]
        scores = positives/total
        return scores        


    def compute_p_x_theta_over_p_x(self, dataset: Dataset, dataset_indices:list[int], alpha: float = 0.5, device: Union[torch.device, str] = 'cpu'):
        if self.mode == 'offline':
            assert 0 <= alpha <= 1, f'RMIA for OFFLINE mode should have a valid alpha! alpha={alpha} was given!'
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
                in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index,
                                                                                                    split='all')
                out_models_indices = [index for index in all_models_indices if index not in in_models_indices]
                trained_in_shadow_models_for_sample.append(in_models_indices)
                trained_out_shadow_models_for_sample.append(out_models_indices)
                # self.logger.print_it(f'For sample with index {index}, I found {len(in_models_indices)} online shadow models and {len(out_models_indices)} offline shadow models...')

        if self.mode == 'online':
            self.logger.print_it(f'Computing p(x) with in models...')
            p_x_thetas_in = self.compute_p_x_theta(audit_dataset=dataset,
                                                    models_for_sample=trained_in_shadow_models_for_sample,
                                                    device=device)
        self.logger.print_it(f'Computing p(x) with out models...')
        p_x_thetas_out = self.compute_p_x_theta(audit_dataset=dataset,
                                                    models_for_sample=trained_out_shadow_models_for_sample,
                                                    device=device)
        # p_x_theta_out = np.mean(p_x_thetas_out, axis=1)
        p_x_out = np.mean(p_x_thetas_out, axis=1)
        if self.mode == 'offline':
            p_x = 0.5*((1+alpha)*p_x_out + (1-alpha))
        elif self.mode == 'online':
            # p_x_theta_in = np.mean(p_x_thetas_in, axis=1)
            p_x_in = np.mean(p_x_thetas_in, axis=1)
            p_x = 0.5 * p_x_in +  0.5 * p_x_out

        self.logger.print_it(f'Computing p(x|theta) with the victim model...')
        p_x_thetas_victim = self.compute_p_x_theta(audit_dataset=dataset,
                                                models_for_sample=[[self.victim_model] for _ in range(len(dataset))],
                                                device=device)
        p_x_thetas_victim = p_x_thetas_victim.squeeze()

        ratio = p_x_thetas_victim/(p_x + 1e-15)
        return ratio
    

    def compute_p_z_theta_over_p_z(self, dataset: Dataset, device: Union[torch.device, str] = 'cpu'):
        all_models_indices = self.shadow_manager.get_all_model_indeces()
        trained_shadow_models_for_sample = [all_models_indices for _ in range(len(dataset))]
        self.logger.print_it(f'Computing p(z)...')
        p_z_thetas = self.compute_p_x_theta(audit_dataset=dataset,
                                            models_for_sample=trained_shadow_models_for_sample,
                                            device=device)
        p_z = np.mean(p_z_thetas, axis=1)

        self.logger.print_it(f'Computing p(z|theta) with victim model...')
        p_z_thetas_victim = self.compute_p_x_theta(audit_dataset=dataset,
                                                models_for_sample=[[self.victim_model] for _ in range(len(dataset))],
                                                device=device)
        p_z_thetas_victim = p_z_thetas_victim.squeeze()

        ratio = p_z_thetas_victim/(p_z + 1e-15)
        return ratio
        
        
    def compute_p_x_theta(self, audit_dataset: Dataset, models_for_sample: list[list[Union[int, torch.nn.Module]]], device: Union[torch.device, str] = 'cpu'):
        tot_samples = len(audit_dataset)
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        p_x_thetas = np.zeros((len(audit_dataset), len(models_for_sample[0])))
        s = time.time()
        self.logger.print_it(f'Computing p(x|theta) for all {tot_samples} samples. This may take a while...')
        for sample_index, (sample, label) in enumerate(audit_loader):
            models = models_for_sample[sample_index]
            for model_index, model in enumerate(models):
                if isinstance(model, torch.nn.Module):
                    pass
                elif isinstance(model, int):
                    model = self.shadow_manager.get_model(model)
                else:
                    raise ValueError('Model should be either a torch Module or an integer referring to the id of the shadow model!')
                p_x_theta = RMIA.get_prob(model=model,
                                        data=sample,
                                        target=label,
                                        device=device)
                p_x_thetas[sample_index, model_index] = p_x_theta
        self.logger.print_it(f'Computed all p(x|theta) in {time.time() - s} seconds.')
        return p_x_thetas

    @staticmethod
    def get_prob(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = RMIA.get_device(dev_str=device)
        model.eval()
        lab=target[0].item()
        with torch.no_grad():
            output = model(data.to(device))
            softmax_output = torch.nn.Softmax(dim=1)(output)
            prob = softmax_output[0][lab].detach().cpu().item()
        return prob
    