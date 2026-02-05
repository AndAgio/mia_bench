import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.data.helpers import MultiDatasets
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils.log import get_logger
from src.utils import convert_to_hms
from typing import Union
import time
from src.trainer.train_manager import TrainManager


class QuantileMIA(BaseMIA):
    # Implementation of Scalable membership inference attacks via quantile regression (https://proceedings.neurips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html).
    def __init__(self, 
                defender_model: torch.nn.Module,
                defender_dataset: MultiDatasets,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, defender_dataset=defender_dataset, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Quantile MIA!")
        
        # TODO: add silent check for n_shadow_datasets == 1 and avoid raising an error, but rather modify configurations on the fly.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/19
        # assignees: AndAgio.

        assert self.shadow_configs.n_shadow_datasets == 1, f"When using quantile MIA, only 1 shadow dataset must be used!"
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Quantile MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.defender_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Quantile MIA attacker: definition of quantile model...')
        self.model_configs.num_classes = 2 if self.attack_configs.use_gaussian else self.attack_configs.n_quantile
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('Quantile MIA attacker: setting quantiles and loss function...')
        # Set up quantiles
        device = self.get_device(dev_str=train_config.device)
        if self.attack_configs.use_logscale:
            log_low = np.log10(self.attack_configs.low_quantile)
            log_high = np.log10(self.attack_configs.high_quantile)
            self.quantile = torch.sort(
                                torch.logspace(log_low, log_high, self.attack_configs.n_quantile)
                            )[0].reshape([1, -1]).to(device)
        else:
            self.quantile = torch.linspace(self.attack_configs.low_quantile,
                                    self.attack_configs.high_quantile,
                                    self.attack_configs.n_quantile).reshape([1, -1]).to(device)
        self.quantile_loss_fn = GaussianLoss().to(device) if self.attack_configs.use_gaussian else PinballLoss(quantile=self.quantile).to(device)
        train_config.loss = self.quantile_loss_fn

        self.logger.print_it('Quantile MIA attacker: constructing quantile dataset...')
        self.defender_model.eval()
        features = []
        target_scores = []
        shadow_dataset = self.shadow_manager.get_dataset(index=0,
                                                        labels='original')
        shadow_loader = DataLoader(shadow_dataset, batch_size=1, shuffle=False)
        start_data = time.time()
        with torch.no_grad():
            for data, target in shadow_loader:
                self.logger.print_it_same_line(f'Quantile MIA attacker: processing sample {len(features)+1}/{len(shadow_dataset)}...', console_only=True)
                features.append(data)
                target_score, _ = self.defender_scoring_fn(data, target, device=train_config.device)
                target_scores.append(target_score)
            self.logger.set_logger_newline(console_only=True)
        features = torch.cat(features)
        target_scores = torch.cat(target_scores)
        quantile_dataset = TensorDataset(features, target_scores)

        # TODO: wrap the quantile_dataset in MultiDataset to ease TrainManager handling and maybe in IndexedDataset to preserve original indices.
        # assignees: AndAgio.

        self.logger.print_it(f"Quantile MIA attacker: constructed quantile dataset in {time.time() - start_data:.2f}s.")

        self.logger.print_it('Quantile MIA attacker: training quantile model. This will take a while. Sit back and chill...')
        start = time.time()
        logger = get_logger(name='{} quantile shadow'.format(self.logger.name),
                            log_folder=self.logger.get_folder(),
                            mode=self.logger.get_mode())
        train_manager = TrainManager(train_configs=train_config,
                                    name='quantile shadow',
                                    logger=logger)
        train_manager.initialize_train(dataset=quantile_dataset,
                                        model=self.shadow_manager.get_model(index=0),
                                        configs=train_config)
        quantile_model = train_manager.train(extra_configs={'quantiles': self.quantile},
                                            return_last_model=True,
                                            return_best_model=False,
                                            return_stats=False)
        self.shadow_manager.update_model(index=0,
                                        model=quantile_model)
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Quantile MIA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def defender_scoring_fn(self, data: torch.Tensor, target: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = QuantileMIA.get_device(dev_str=device)
        self.defender_model.eval()
        with torch.no_grad():
            logits = self.defender_model(data.to(device)).detach().cpu()
            onehot_label = torch.nn.functional.one_hot(target, num_classes=logits.shape[-1]).bool()
            score = logits[onehot_label]
            # Mask out the true label before taking max over incorrect labels
            logits_masked = logits.masked_fill(onehot_label, float('-inf'))
            score -= torch.max(logits_masked, dim=1)[0]
        return score, logits

    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        self.logger.print_it('Computing Quantile MIA scores. This may take a while...')
        start = time.time()
        if isinstance(device, str):
            device = QuantileMIA.get_device(dev_str=device)
        quantile_model = self.shadow_manager.get_model(index=0).to(device)
        quantile_model.eval()
        self.defender_model.eval()
        audit_dataset = self.audit_manager.get(labels='original')
        self.reset_logger()
        tot_samples = len(audit_dataset)
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        scores = np.zeros((len(audit_dataset), ))
        self.logger.print_it(f"Computing scores for all {tot_samples} samples. This may take a while...")
        for sample_index, (sample, label) in enumerate(audit_loader):
            with torch.no_grad():
                target_score, _ = self.defender_scoring_fn(sample, label, device=device)
                predicted_scores = quantile_model(sample.to(device))
                
                if self.attack_configs.use_gaussian:
                    mu = predicted_scores[:, 0]
                    log_std = predicted_scores[:, 1]
                    predicted_scores = mu.reshape([-1, 1]) + torch.exp(log_std).reshape([-1, 1]) * torch.erfinv(2 * self.quantile.to(predicted_scores.device) - 1).reshape([1, -1]) * math.sqrt(2)
                quantile_value = 1 - self.attack_configs.quantile_alpha
                quantile_index = torch.argmin(torch.abs(self.quantile - quantile_value))
                score = target_score.detach().cpu().item() - predicted_scores[0, quantile_index].detach().cpu().item()
                scores[sample_index] = score
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Quantile MIA attacker: score computation done! Time taken to compute: {}:{:02d}:{:02d}...'.format(h, m, s))

        metrics = self.compute_stats(scores)
        self.logger.print_it('Quantile MIA attacker: Obtained AUC score is: {}'.format(metrics['auc']))
        return metrics
    

# Custom loss functions
class PinballLoss(torch.nn.Module):
    def __init__(self, quantile: torch.Tensor):
        super(PinballLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = target.reshape([-1, 1])
        delta_score = target - input
        loss = torch.nn.functional.relu(delta_score) * self.quantile + torch.nn.functional.relu(-delta_score) * (1.0 - self.quantile)
        return loss


class GaussianLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        mu = input[:, 0]
        log_std = input[:, 1]
        loss = log_std + 0.5 * torch.exp(-2 * log_std) * (target - mu) ** 2
        return loss