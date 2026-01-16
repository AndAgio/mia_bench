import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import norm
from src.data.multi import MultiDatasets
from src.mia.base_mia import BaseMIA
from src.mia.shadow_manager import ShadowManager
from src.utils.configs import TrainConfigs, AttackerConfigs
from src.utils import convert_to_hms
from typing import Union
import time


class LiRA(BaseMIA):
    # Implementation of Membership inference attacks from first principles (https://ieeexplore.ieee.org/abstract/document/9833649).
    def __init__(self, 
                victim_model: torch.nn.Module,
                victim_dataset: MultiDatasets,
                attacker_configs: AttackerConfigs, 
                exp_hash: str):
        super().__init__(victim_model=victim_model, victim_dataset=victim_dataset, attacker_configs=attacker_configs, exp_hash=exp_hash)
        self.logger.print_it(f'Working with LiRA!')
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('LiRA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.victim_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    exp_hash=self.exp_hash)
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
            in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index,
                                                                                                split='all')
            out_models_indices = [index for index in all_models_indices if index not in in_models_indices]
            trained_in_shadow_models_for_sample.append(in_models_indices)
            trained_out_shadow_models_for_sample.append(out_models_indices)

        self.logger.print_it(f'Computing phis with in models...')
        phis_in = self.compute_phis(audit_dataset=audit_dataset,
                                        models_for_sample=trained_in_shadow_models_for_sample,
                                        device=device)
        shadow_in_means = np.mean(phis_in, axis=1)
        shadow_in_stds = np.std(phis_in, axis=1)

        self.logger.print_it(f'Computing phis with out models...')
        phis_out = self.compute_phis(audit_dataset=audit_dataset,
                                        models_for_sample=trained_out_shadow_models_for_sample,
                                        device=device)
        shadow_out_means = np.mean(phis_out, axis=1)
        shadow_out_stds = np.std(phis_out, axis=1)

        self.logger.print_it(f'Computing phis with victim model...')
        phis_victim = self.compute_phis(audit_dataset=audit_dataset,
                                        models_for_sample=[[self.victim_model] for _ in range(len(audit_dataset))],
                                        device=device)
        phis_victim = phis_victim.squeeze()

        p_in = norm.pdf(phis_victim, loc=shadow_in_means, scale=shadow_in_stds)
        p_out = norm.pdf(phis_victim, loc=shadow_out_means, scale=shadow_out_stds)

        scores = p_in/(p_out+1e-15)

        self.logger.print_it(f'LiRA attacker: scores = {scores}')

        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('LiRA attacker: score computation done! Time taken to compute LR: {}:{:02d}:{:02d}...'.format(h, m, s))

        metrics = self.compute_stats(scores)
        # self.logger.print_it('RMIA attacker: Obtained scores are: {}'.format(metrics))
        return metrics

        
    def compute_phis(self, audit_dataset: Dataset, models_for_sample: list[list[Union[int, torch.nn.Module]]], device: Union[torch.device, str] = 'cpu'):
        tot_samples = len(audit_dataset)
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        phis = np.zeros((len(audit_dataset), len(models_for_sample[0])))
        s = time.time()
        self.logger.print_it(f'Computing phi for all {tot_samples} samples. This may take a while...')
        for sample_index, (sample, label) in enumerate(audit_loader):
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
                phis[sample_index, model_index] = phi
        self.logger.print_it(f'Computed phis in {time.time() - s} seconds.')
        return phis

    @staticmethod
    def compute_phi(model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = LiRA.get_device(dev_str=device)
    
        model.eval()
        with torch.no_grad():
            output = model(data.to(device))

        num_classes = output.size(1)
        batch_size = output.size(0)
        
        # Compute CrossEntropyLoss for the entire batch
        loss = torch.nn.CrossEntropyLoss(reduction='none')(output, target.to(device))
        loss = torch.clamp(loss, min=1e-15, max=10.0)
        
        # Create anti-targets for each sample in the batch
        anti_targets = []
        for i in range(batch_size):
            sample_anti_targets = [j for j in range(num_classes) if j != target[i].item()]
            anti_targets.append(torch.tensor(sample_anti_targets).to(device))
        
        # Compute exp terms
        exp_term = torch.exp(-loss)
        
        # Compute anti_exp_term for each sample
        anti_exp_term = []
        for i in range(batch_size):
            sample_anti_exp = torch.exp(-torch.nn.CrossEntropyLoss(reduction='none')(
                output[i].unsqueeze(0).repeat(num_classes-1, 1),
                anti_targets[i]
            ))
            anti_exp_term.append(sample_anti_exp)
        
        anti_exp_term = torch.stack(anti_exp_term)
        
        # Compute phi for each sample
        positive_term = torch.log(exp_term).detach().cpu()
        negative_term = torch.log(torch.sum(anti_exp_term, dim=1)).detach().cpu()
        phi = positive_term - negative_term
        
        return phi
    