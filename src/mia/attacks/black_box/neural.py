from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from src.data.helpers import IndexedDataset, SubsampledDataset
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils import convert_to_hms
from src.utils.configs import AttackerConfigs, TrainConfigs


PROCESSING_BATCH_SIZE = 256
# With a single shadow model every training row would be a member of it: the non-members of a shadow model are the
# samples of the other shadow datasets.
MIN_SHADOW_MODELS = 2


class NeuralMIA(BaseMIA):
    """Shadow-model attack: one neural network per class, trained on shadow models' outputs, predicts membership.

    Paper: "Membership Inference Attacks Against Machine Learning Models", https://arxiv.org/abs/1610.05820.
    Documentation: docs/attacks/neural.md.
    Offline (the paper), the shadow datasets hold none of the audited samples; online (as in "Membership Inference
    Attacks From First Principles", Sec. V-C), each audited sample is in half of them.
    """
    @staticmethod
    def reads_features(attacker_configs: AttackerConfigs) -> bool:
        """Return whether the attack reads the target model's internal features: only in feat mode."""
        return attacker_configs.attack.neural_input_mode == 'feat'

    def __init__(self,
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Check the mode and the number of shadow models, then sample the shadow datasets and build the models."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        # offline: the shadow datasets hold none of the audited samples (the paper); online: each is in half of them.
        self.mode = self.attack_configs.mode
        if self.shadow_configs.mode != self.mode:
            raise ValueError(f"Neural MIA in {self.mode} mode needs shadow datasets of its own mode, "
                             f"not {self.shadow_configs.mode}.")
        n_shadows = self.shadow_configs.n_shadow_datasets
        if n_shadows < MIN_SHADOW_MODELS:
            raise ValueError(f"Neural MIA needs at least {MIN_SHADOW_MODELS} shadow models (--n_shadows), got "
                             f"{n_shadows}: the non-members of each shadow model are the samples of the others.")
        self.logger.print_it(f"Working with Neural MIA in {self.mode.upper()} mode!")
        self.attacker_models = {}
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Neural MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Neural MIA attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        """Train the shadow models, then one attack model per class on their outputs and memberships."""
        self.logger.print_it('Neural MIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        # Keeping every trained shadow ResNet on CUDA can exhaust or heavily
        # fragment GPU memory before the feature-building GEMMs begin.
        for shadow_model in self.shadow_manager.get_all_models().values():
            shadow_model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Neural MIA attacker: done training shadow models, it took {}:{:02d}:{:02d}...'.format(h, m, s))
        all_shadow_data = self.shadow_manager.get_all_samples_in_all_shadow_datasets()
        model_indices = self.shadow_manager.get_all_model_indeces()
        # Shadow membership is by original index; sets make each lookup constant-time.
        members = {model_index: set(self.shadow_manager.shadow_data.get_map(model_index))
                   for model_index in model_indices}
        sample_labels = NeuralMIA.labels_of(all_shadow_data)
        start_training = time.time()
        # One class at a time, so that only that class's rows are ever in memory.
        for label in np.unique(sample_labels):
            label = int(label)
            positions = np.flatnonzero(sample_labels == label).tolist()
            class_data = SubsampledDataset.from_positions(all_shadow_data, positions)
            features, targets = self.build_class_rows(class_data=class_data, members=members,
                                                      model_indices=model_indices, device=train_config.device)
            n_members = int(targets.sum().item())
            n_non_members = len(targets) - n_members
            self.logger.print_it(f"Neural MIA attacker: class {label}, {len(targets)} rows ({n_members} members, "
                                 f"{n_non_members} non-members) from {len(model_indices)} shadow models.")
            # The paper's shadow test sets are as large as their training sets; weighting the members' loss by the
            # ratio gives both groups the same weight, whatever the share of the attacker's data in each shadow model.
            if n_members and n_non_members:
                pos_weight = torch.tensor([n_non_members / n_members])
            else:
                pos_weight = None
                self.logger.print_it(f"Neural MIA attacker: WARNING, class {label} has only "
                                     f"{'members' if n_members else 'non-members'}; its attack model cannot tell "
                                     f"them apart.")
            train_loader = DataLoader(IndexedDataset(TensorDataset(features, targets)),
                                      batch_size=PROCESSING_BATCH_SIZE, shuffle=True)
            self.attacker_models[label] = self.train_attacker_model(model=self.build_attacker_model(features.shape[1]),
                                                                    train_loader=train_loader,
                                                                    device=train_config.device,
                                                                    pos_weight=pos_weight)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start_training)
        self.logger.print_it(f"Neural MIA attacker: training of the {len(self.attacker_models)} attacking models "
                             f"completed in {h}:{m:02d}:{s:02d}.")
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it(f"Neural MIA attacker: optimization of the attack took {h}:{m:02d}:{s:02d}.")

    def build_class_rows(self, class_data: Dataset, members: dict, model_indices: list,
                         device: Union[torch.device, str]):
        """Return every shadow model's outputs on one class's samples, and whether each sample was in that model."""
        features, targets = [], []
        for model_index in model_indices:
            shadow_model = self.shadow_manager.get_model(index=model_index)
            for batch_data, _, batch_sample_ids, _ in DataLoader(class_data, batch_size=PROCESSING_BATCH_SIZE,
                                                                 shuffle=False):
                features.append(NeuralMIA.get_model_out(model=shadow_model, data=batch_data, device=device,
                                                        mode=self.attack_configs.neural_input_mode))
                targets.append(torch.tensor([[float(int(sample_id) in members[model_index])]
                                             for sample_id in batch_sample_ids]))
            shadow_model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(features), torch.cat(targets)

    def build_attacker_model(self, n_inputs: int) -> nn.Module:
        """Return an untrained attack model: the hidden layers of --attacker_neural_model_layers, then one logit."""
        layers = []
        prev_dim = n_inputs
        for hidden_dim in self.attack_configs.model_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # Final output layer (1 logit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample with the logit of its class's attack model on the target's output."""
        self.logger.print_it('Neural MIA attacker: computing scores...')
        start = time.time()
        if isinstance(device, str):
            device = NeuralMIA.get_device(dev_str=device)
        self.defender_model.to(device)
        self.defender_model.eval()
        outputs, labels = [], []
        for samples, sample_labels, _, _ in DataLoader(dataset, batch_size=PROCESSING_BATCH_SIZE, shuffle=False):
            outputs.append(NeuralMIA.get_model_out(model=self.defender_model, data=samples, device=device,
                                                   mode=self.attack_configs.neural_input_mode))
            labels.append(sample_labels.reshape(-1))
        outputs, labels = torch.cat(outputs), torch.cat(labels).numpy()
        scores = np.zeros((len(dataset), ))
        for label in np.unique(labels):
            label = int(label)
            if label not in self.attacker_models:
                raise ValueError(f"Neural MIA has no attack model for class {label}: the shadow datasets hold no "
                                 f"sample of it.")
            attacker_model = self.attacker_models[label].to(device)
            attacker_model.eval()
            rows = labels == label
            with torch.no_grad():
                scores[rows] = attacker_model(outputs[rows].to(device))[:, 0].cpu().numpy()
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Neural MIA attacker: score computation done! Time taken to compute: {}:{:02d}:{:02d}...'.format(h, m, s))
        # The attack model's own rule: a member when it gives membership a probability of at least 0.5.
        return [MembershipScores(scores, decisions=scores >= 0)]

    def train_attacker_model(self, model: nn.Module, train_loader: DataLoader,
                             device: Union[torch.device, str] = 'cpu', pos_weight: Optional[torch.Tensor] = None):
        """Train an attack model with Adam on the binary cross-entropy, the members' loss weighted by pos_weight."""
        self.logger.print_it('Neural MIA attacker: training the attacking model. This may take a while...')
        if isinstance(device, str):
            device = NeuralMIA.get_device(dev_str=device)
        model.to(device)
        # The final layer emits unrestricted logits (there is no Sigmoid).
        criterion = nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else pos_weight.to(device))
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
    def labels_of(dataset: Dataset) -> np.ndarray:
        """Return the class label of every sample of a dataset, in order."""
        labels = [batch_labels.reshape(-1) for _, batch_labels, _, _ in
                  DataLoader(dataset, batch_size=PROCESSING_BATCH_SIZE, shuffle=False)]
        return torch.cat(labels).numpy()

    @staticmethod
    def get_model_out(model: torch.nn.Module, data: torch.Tensor, device: Union[torch.device, str], mode: str = 'logit'):
        """Return a model's probabilities, logits or internal features on a batch, on the CPU."""
        assert mode in ['prob', 'logit', 'feat'], f"Mode {mode} not recognized. Available modes are prob, logit and feature!"

        if isinstance(device, str):
            device = NeuralMIA.get_device(dev_str=device)

        model.to(device)
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
