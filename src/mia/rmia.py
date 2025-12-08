import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.data.multi import MultiDatasets
from src.mia.base_mia import BaseMIA
from src.mia.shadow_manager import ShadowManager
from src.utils.configs import TrainConfigs, AuditingDataConfigs, ShadowDataConfigs, ModelConfigs
from src.utils.log import SmartLogger, DumbLogger
from src.utils import convert_to_hms
from typing import Union
import time


class RMIA(BaseMIA):
    def __init__(self, 
                victim_model: torch.nn.Module,
                victim_dataset: MultiDatasets,
                audit_configs: AuditingDataConfigs,
                shadow_configs: ShadowDataConfigs, 
                model_configs: ModelConfigs,
                logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(victim_model=victim_model, victim_dataset=victim_dataset, audit_configs=audit_configs, logger=logger)
        self.mode = shadow_configs.mode
        self.logger.print_it(f'Working with RMIA in {self.mode.upper()} mode!')
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('RMIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.victim_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=shadow_configs)
        self.logger.print_it('RMIA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=shadow_configs.n_shadow_datasets,
                                                model_configs=model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('RMIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('RMIA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def measure_effectiveness(self, random_pop_size: int = 1000, alpha: Union[float, list[float]] = None, device: Union[torch.device, str] = 'cpu'):
        if alpha is None and self.mode == 'offline':
            self.logger.print_it('Found alpha to be None for OFFLINE RMIA, which is not possible! Testing over all alphas...')
            alphas = np.arange(0, 1.05, 0.1)
        elif alpha is None and self.mode == 'online':
            self.logger.print_it('Found alpha to be None for ONLINE RMIA, setting it to 0 for ease')
            alphas = [0]
        elif isinstance(alpha, list):
            assert all([0 < al <= 1 for al in alpha]), f'All given alphas should be between 0 and 1!'
            alphas = alpha
        elif isinstance(alpha, float):
            assert 0 < alpha <= 1, f'The given alpha should be between 0 and 1! Found alpha={alpha} instead!'
            alphas = [alpha]
        scores = {alpha: self.compute_with_alpha(alpha=alpha,
                                                random_pop_size=random_pop_size,
                                                device=device) for alpha in alphas}
        metrics = {alpha: self.compute_stats(scores[alpha]) for alpha in alphas}
        self.logger.print_it('RMIA attacker: Obtained scores are: {}'.format(metrics))
        return metrics
        

    def compute_with_alpha(self, random_pop_size: int, alpha: float, device: Union[torch.device, str] = 'cpu'):
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

        left_lr = self.compute_p_x_theta_over_p_x(dataset=audit_dataset, dataset_indices=audit_dataset_indices, alpha=alpha, device=device)
        print('left_lr.shape: {}'.format(left_lr.shape))

        random_data_indices = self.shadow_manager.sample_random_population_indices(num_data=random_pop_size)
        random_data = self.shadow_manager.get_random_population(indices=random_data_indices,
                                                                labels='original')

        right_lr = self.compute_p_x_theta_over_p_x(dataset=random_data, dataset_indices=random_data_indices['all_ids'], alpha=alpha, device=device)
        print('right_lr.shape: {}'.format(left_lr.shape))
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('RMIA attacker: LR computation done! Time taken to compute LR: {}:{:02d}:{:02d}...'.format(h, m, s))

        lr_ratios = np.divide(left_lr[:, np.newaxis], right_lr)
        print('lr_ratios: {}'.format(lr_ratios))
        print('lr_ratios.shape: {}'.format(lr_ratios.shape))
        gamma=1
        positives_indices = lr_ratios > gamma
        print(positives_indices)
        positives = np.sum(positives_indices, axis=1)
        total = right_lr.shape[0]
        scores = positives/total
        print(scores.shape)
        print(scores)
        return scores        


    def compute_p_x_theta_over_p_x(self, dataset: Dataset, dataset_indices:list[int], alpha: float = 0.5, device: Union[torch.device, str] = 'cpu'):
        if self.mode == 'offline':
            assert 0 <= alpha <= 1, f'RMIA for OFFLINE mode should have a valid alpha! alpha={alpha} was given!'
            # Load indices for all shadow models and set them as offline models
            all_models_indices = self.shadow_manager.get_all_model_indeces()
            trained_offline_shadow_models_for_sample = [all_models_indices for _ in range(len(dataset))]
            # Indices for online shadow models are empty
            trained_online_shadow_models_for_sample = [[] for _ in range(len(dataset))]
        if self.mode == 'online':
            all_models_indices = self.shadow_manager.get_all_model_indeces()
            trained_online_shadow_models_for_sample = []
            trained_offline_shadow_models_for_sample = []
            for index in dataset_indices:
                online_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index,
                                                                                                    split='all')
                offline_models_indices = [index for index in all_models_indices if index not in online_models_indices]
                trained_online_shadow_models_for_sample.append(online_models_indices)
                trained_offline_shadow_models_for_sample.append(offline_models_indices)

        p_x_thetas_online = self.compute_p_x_theta(audit_dataset=dataset,
                                                models_for_sample=trained_online_shadow_models_for_sample,
                                                device=device)
        print('p_x_thetas_online.shape: {}'.format(p_x_thetas_online.shape))
        p_x_thetas_offline = self.compute_p_x_theta(audit_dataset=dataset,
                                                    models_for_sample=trained_offline_shadow_models_for_sample,
                                                    device=device)
        print('p_x_thetas_offline.shape: {}'.format(p_x_thetas_offline.shape))
        # p_x_theta_offline = np.mean(p_x_thetas_offline, axis=1)
        p_x_offline = np.mean(p_x_thetas_offline, axis=1)
        if self.mode == 'offline':
            p_x = 0.5*((1+alpha)*p_x_offline + (1-alpha))
        elif self.mode == 'online':
            # p_x_theta_online = np.mean(p_x_thetas_online, axis=1)
            p_x_online = np.mean(p_x_thetas_online, axis=1)
            p_x = 0.5 * p_x_online +  0.5 * p_x_offline
        print('p_x.shape: {}'.format(p_x.shape))

        p_x_thetas_victim = self.compute_p_x_theta(audit_dataset=dataset,
                                                models_for_sample=[[self.victim_model] for _ in range(len(dataset))],
                                                device=device)
        p_x_thetas_victim = p_x_thetas_victim.squeeze()
        print('p_x_thetas_victim.shape: {}'.format(p_x_thetas_victim.shape))

        ratio = p_x_thetas_victim/(p_x + 1e-15)
        print('ratio.shape: {}'.format(ratio.shape))
        return ratio


    def compute_p_x_theta(self, audit_dataset: Dataset, models_for_sample: list[list[Union[int, torch.nn.Module]]], device: Union[torch.device, str] = 'cpu'):
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        p_x_thetas = np.zeros((len(audit_dataset), len(models_for_sample[0])))
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


    # def measure_effectiveness(self):
    #     audit_dataset = self.audit_manager.get(labels='mia')
    #     scores=[]
    #     for sample in audit_dataset:
    #         target_prob=prob_target[i]
    #         target_prob_given_target_model=get_prob(target_model,target_data[i:i+1],target_labels[i:i+1],device)
    #         lr_target=target_prob_given_target_model/(target_prob+1e-15)
    #         C=0
    #         for j in range(random_data.size(0)):
    #             rand_prob=rand_probs_per_target[i][j]
    #             rand_prob_given_target_model=get_prob(target_model,random_data[j:j+1],random_labs[j:j+1],device)
    #             lr_rand=rand_prob_given_target_model/(rand_prob+1e-15)
    #             C+=1 if lr_target/lr_rand>gamma else 0

    #         scores.append(C/random_data.size(0))

    #     return np.array(scores)