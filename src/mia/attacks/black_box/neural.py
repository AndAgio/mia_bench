from typing import Union
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time
from src.data.helpers import IndexedDataset
from src.data.helpers import MultiDatasets
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils import convert_to_hms
from src.utils.configs import AttackerConfigs, TrainConfigs


PROCESSING_BATCH_SIZE = 256

class NeuralMIA(BaseMIA):
    # Implementation of Neural Network-based MIA (https://ieeexplore.ieee.org/document/7958568)
    def __init__(self, 
                defender_model: torch.nn.Module,
                defender_dataset: MultiDatasets,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, defender_dataset=defender_dataset, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Neural MIA!")
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Neural MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.defender_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Neural MIA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('Neural MIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Neural MIA attacker: done training shadow models, it took {}:{:02d}:{:02d}...'.format(h, m, s))
        self.logger.print_it('Neural MIA attacker: constructing the dataset of shadow predictions to train the attacking model...')
        all_shadow_data = self.shadow_manager.get_all_samples_in_all_shadow_datasets()
        all_shadow_models_indices = self.shadow_manager.get_all_model_indeces()

        tot_n_samples = len(all_shadow_data) * len(all_shadow_models_indices)
        sample_label_map = {}
        self.logger.print_it('Neural MIA attacker: building sample label map...')
        start_map = time.time()
        for enumerate_index, (_, _, _, sample_id) in enumerate(all_shadow_data):
            # self.logger.print_it_same_line(f"Processing sample {enumerate_index}/{len(all_indexed_shadow_data)} for attacking model dataset construction...")
            in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=sample_id,
                                                                                                split='all')
            out_models_indices = [idx for idx in all_shadow_models_indices if idx not in in_models_indices]
            for in_ids in in_models_indices:
                sample_label_map[(sample_id, in_ids)] = 1 # member
            for out_ids in out_models_indices:
                sample_label_map[(sample_id, out_ids)] = 0 # non-member
        # self.logger.set_logger_newline()
        self.logger.print_it(f"Neural MIA attacker: sample label map created in {time.time() - start_map:.2f} seconds!")
            
        # Get feature shape by passing a dummy sample through one model
        dummy_data, _, _, _ = all_shadow_data[0]
        dummy_data = dummy_data.unsqueeze(0)  # add batch dimension
        dummy_model_index = all_shadow_models_indices[0]
        dummy_model = self.shadow_manager.get_model(index=dummy_model_index)
        dummy_feature = NeuralMIA.get_model_out(model=dummy_model,
                                                data=dummy_data,
                                                device=train_config.device,
                                                mode=self.attack_configs.neural_input_mode)
        feature_shape = dummy_feature.shape[1]
        self.logger.print_it(f"Neural MIA attacker: feature shape determined as {feature_shape}.")

        # Build dataset for training the attacking model
        self.logger.print_it('Neural MIA attacker: building training dataset for the attacking model...')
        start_build = time.time()
        features = torch.zeros((tot_n_samples, feature_shape))
        targets = torch.zeros((tot_n_samples, 1))
        current_index = 0
        for model_index in all_shadow_models_indices:
            shadow_model = self.shadow_manager.get_model(index=model_index)
            dataloader = DataLoader(all_shadow_data, batch_size=PROCESSING_BATCH_SIZE, shuffle=False)
            for enumerate_index, (batch_data, _, _, batch_sample_ids) in enumerate(dataloader):
                self.logger.print_it_same_line(f"Processing batch {enumerate_index+1}/{len(dataloader)} of shadow model {model_index+1}/{len(all_shadow_models_indices)} for attacking model dataset construction...", console_only=True)
                batch_size = batch_data.size(0)
                batch_features = NeuralMIA.get_model_out(model=shadow_model,
                                                        data=batch_data,
                                                        device=train_config.device,
                                                        mode=self.attack_configs.neural_input_mode)
                features[current_index:current_index+batch_size, :] = batch_features
                for i in range(batch_size):
                    targets[current_index + i, 0] = sample_label_map[(batch_sample_ids[i].item(), model_index)]
                current_index += batch_size
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Neural MIA attacker: training dataset built in {time.time() - start_build:.2f} seconds!")
        # Build and train the attacking model
        dataset = IndexedDataset(TensorDataset(features, targets))
        train_loader = DataLoader(dataset, batch_size=PROCESSING_BATCH_SIZE, shuffle=True)
        layers = []
        prev_dim = features.shape[1]
        for hidden_dim in self.attack_configs.model_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # Final output layer (1 logit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        attack_model = nn.Sequential(*layers)
        start_training = time.time()
        attack_model = self.train_attack_model(model=attack_model,
                                                train_loader=train_loader,
                                                device=train_config.device)
        self.attack_model = attack_model
        stop = time.time()
        h, m, s = convert_to_hms(stop-start_training)
        self.logger.print_it(f"Neural MIA attacker: training of the attacking model completed in {h}:{m:02d}:{s:02d}.")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f"Neural MIA attacker: optimization of the attack took {h}:{m:02d}:{s:02d}.")

    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        # Compute scores on the auditing dataset
        self.logger.print_it('Neural MIA attacker: computing scores on the auditing dataset...')
        audit_dataset = self.audit_manager.get(labels='mia')
        start = time.time()
        if isinstance(device, str):
            device = NeuralMIA.get_device(dev_str=device)
        self.attack_model.to(device)
        self.defender_model.to(device)
        self.attack_model.eval()
        self.defender_model.eval()
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        scores = np.zeros((len(audit_dataset), ))
        for sample_index, (sample, _, _, _) in enumerate(audit_loader):
            feature = NeuralMIA.get_model_out(model=self.defender_model,
                                                data=sample,
                                                device=device,
                                                mode=self.attack_configs.neural_input_mode)
            with torch.no_grad():
                output = self.attack_model(feature.to(device)).detach().cpu().item()
                scores[sample_index] = output
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Neural MIA attacker: score computation done! Time taken to compute: {}:{:02d}:{:02d}...'.format(h, m, s))

        metrics = self.compute_stats(scores)
        self.logger.print_it('Neural MIA attacker: Obtained AUC score is: {}'.format(metrics['auc']))
        return metrics

    def train_attack_model(self, model: nn.Module, train_loader: DataLoader, device: Union[torch.device, str] = 'cpu'):
        self.logger.print_it('Neural MIA attacker: training the attacking model. This may take a while...')
        if isinstance(device, str):
            device = NeuralMIA.get_device(dev_str=device)
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.attack_configs.model_lr)
        model.train()
        for epoch in range(self.attack_configs.model_epochs):
            run_loss = 0.0
            for features, labels, _, _ in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                run_loss += loss.item()
            self.logger.print_it_same_line(f"Epoch {epoch+1}/{self.attack_configs.model_epochs}, Attack Loss: {run_loss/len(train_loader):.4f}", console_only=True)
        self.logger.set_logger_newline(console_only=True)
        return model

    @staticmethod
    def get_model_out(model: torch.nn.Module, data: torch.Tensor, device: Union[torch.device, str], mode: str = 'logit'):
        assert mode in ['prob', 'logit', 'feat'], f"Mode {mode} not recognized. Available modes are prob, logit and feature!"

        if isinstance(device, str):
            device = NeuralMIA.get_device(dev_str=device)
    
        model.eval()
        with torch.no_grad():
            if mode == 'prob':
                output = torch.nn.functional.softmax(model(data.to(device)), dim=1).cpu()
            elif mode == 'logit':
                output = model(data.to(device)).cpu()
            elif mode == 'feat':
                assert hasattr(model, 'feature'), 'Model does not have feature extraction method which is required for running NeuralMIA with feature mode!'
                output = model.feature(data.to(device)).cpu()
        return output
