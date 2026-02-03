from typing import Union, Callable
import copy
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from src.data.helpers import MultiDatasets
from src.utils.configs import DefenderConfigs, SelenaDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender
from src.mia.helpers.shadow_models_manager import ShadowModelsManager
from src.trainer.train_manager import TrainManager


class SelenaDefender(BaseDefender):
    # Implementation of training time optimization component of Selena defense from "Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture" (https://www.usenix.org/system/files/sec22-tang.pdf).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, SelenaDefenseConfigs), f"Selena Defender can only be used with SelenaDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'selena_defender'
        self.selena_configs = defender_configs.defense
        # TODO: Implement the Selena defense.
        # assignees: AndAgio

        self.split_data_manager = SplitDataManager(dataset=self.dataset,
                                                    selena_configs=self.selena_configs,
                                                    logger=self.logger)
        self.split_model_manager = ShadowModelsManager(n_models=self.selena_configs.K,
                                                    model_configs=self.model_configs,
                                                    logger=self.logger)

    def train_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it(f"Selena Defender: training defender model with K={self.selena_configs.K} split models and L={self.selena_configs.L} exclusions per sample...")
        self.train_all_split_models(train_configs=train_configs)
        # self.train_distilled_model(train_configs=train_configs)
        return self.train_distilled_model(train_configs=train_configs,
                                        return_stats=return_stats)
    
    def train_all_split_models(self, train_configs: TrainConfigs):
        data_splits = self.split_data_manager.get_all_datasets()
        models = self.split_model_manager.get_all()
        assert len(models) == self.selena_configs.K, f"Selena Defender: number of split models {len(models)} does not match K={self.selena_configs.K}!"
        assert len(models) == data_splits.n_splits(), f"Selena Defender: number of data splits {data_splits.n_splits()} does not match K={self.selena_configs.K}!"
        for k in range(self.selena_configs.K):
            self.logger.print_it(f"Selena Defender: training split model {k+1}/{self.selena_configs.K}...")
            train_manager = TrainManager(train_configs=train_configs,
                                        name='selena_split_{}'.format(k),
                                        logger=self.logger)
            train_manager.initialize_train(dataset=data_splits.get(id=k),
                                            model=models[k],
                                            configs=train_configs)
            model = train_manager.train(return_best_model=True,
                                        return_last_model=False,
                                        return_stats=False)
            self.split_model_manager.update(index=k,
                                            model=model)
            self.logger.print_it(f"Selena Defender: finished training split model {k+1}/{self.selena_configs.K}!")

    def train_distilled_model(self, train_configs: TrainConfigs, return_stats: bool = False):
        self.logger.print_it("Selena Defender: training distilled model on outputs of split models...")
        dataset = self.gather_dataset_for_distillation(train_configs=train_configs)
        train_manager = DistillTrainManager(train_configs=train_configs,
                                            name='selena_distilled',
                                            logger=self.logger)
        train_manager.initialize_train(dataset=dataset,
                                        model=self.untrained_model,
                                        configs=train_configs)
        if return_stats:
            self.trained_model, train_stats = train_manager.train(return_best_model=True,
                                                        return_last_model=False,
                                                        return_stats=return_stats)
        else:
            self.trained_model = train_manager.train(return_best_model=True,
                                            return_last_model=False,
                                            return_stats=return_stats)
        self.logger.print_it("Selena Defender: finished training distilled model!")
        if return_stats:
            return self.trained_model, train_stats
        else:
            return self.trained_model

    def gather_dataset_for_distillation(self, train_configs: TrainConfigs) -> TensorDataset:
        start = time.time()
        device = self.get_device(dev_str=train_configs.device)
        dataset_to_return = MultiDatasets()
        train_dataset = self.dataset.get('train')
        predictions_matrix = torch.zeros(self.selena_configs.K, len(train_dataset), self.model_configs.num_classes)
        self.logger.print_it("Selena Defender: gathering distillation outputs from split models. This may take a while...")
        start_pred = time.time()
        for k in range(self.selena_configs.K):
            self.logger.print_it_same_line(f"Selena Defender: gathering distillation outputs from split model {k+1}/{self.selena_configs.K}...", console_only=True)
            model = self.split_model_manager.get(index=k)
            model = model.to(device)
            model.eval()
            dataloader = DataLoader(train_dataset, batch_size=train_configs.batch_size, shuffle=False)
            all_outputs = []
            with torch.no_grad():
                for batch_idx, (samples, _) in enumerate(dataloader):
                    samples = samples.to(device)
                    outputs = model(samples)
                    all_outputs.append(torch.nn.functional.softmax(outputs, dim=1).cpu())
            all_outputs_tensor = torch.cat(all_outputs, dim=0)
            predictions_matrix[k, :, :] = all_outputs_tensor.to(predictions_matrix.device)
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Selena Defender: gathered distillation outputs in {time.time()-start_pred:.2f} seconds.")

        self.logger.print_it(f"Selena Defender: Gathering masking matrix for distillation...")
        start_mask = time.time()
        mask_matrix = torch.zeros(self.selena_configs.K, len(train_dataset))
        for i in range(len(train_dataset)):
            self.logger.print_it_same_line(f"Selena Defender: gathering masking info for sample {i+1}/{len(train_dataset)}...", console_only=True)
            included_models = self.split_data_manager.get_models_for_sample(sample_index=i)
            for k in included_models:
                mask_matrix[k, i] = 1.0
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Selena Defender: gathered masking matrix in {time.time()-start_mask:.2f} seconds.")

        self.logger.print_it(f"Selena Defender: applying masking and averaging outputs for distillation dataset...")
        mask_matrix = mask_matrix.bool().to(predictions_matrix.device)
        mask_exp = mask_matrix.unsqueeze(-1)
        masked_sum = (predictions_matrix * mask_exp.to(predictions_matrix.dtype)).sum(dim=0)
        counts = mask_matrix.sum(dim=0).unsqueeze(-1)
        mean_outputs = torch.where(counts > 0,
                        masked_sum / counts.to(predictions_matrix.dtype),
                        torch.tensor(float('nan'), device=predictions_matrix.device))
        assert torch.isfinite(mean_outputs).all(), "Found NaN or Inf in `mean_outputs`"
        distilled_dataset = copy.deepcopy(train_dataset)
        distilled_dataset.targets = mean_outputs
        dataset_to_return.add(distilled_dataset, id='train')
        test_dataset = self.dataset.get('test')
        dataset_to_return.add(test_dataset, id='test')
        self.logger.print_it(f"Selena Defender: gathered distilled dataset in {time.time()-start:.2f} seconds.")
        return dataset_to_return

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        self.logger.print_it("Selena Defender: defend_model does not modify the model at inference time.")
        self.defended_model = self.trained_model
        return self.defended_model
    

class SplitDataManager:
    def __init__(self, dataset: MultiDatasets, selena_configs: SelenaDefenseConfigs, logger=None):
        self.dataset = dataset
        self.selena_configs = selena_configs
        self.logger = logger
        self.split_data()

    def split_data(self):
        all_train_data = self.dataset.get('train').data
        all_train_labels = self.dataset.get('train').targets
        all_train_indices = np.arange(len(self.dataset.get('train')))
        exclusion_matrix = np.zeros((all_train_indices.shape[0], self.selena_configs.L))
        for i in range(len(exclusion_matrix)):
            tmp = np.arange(self.selena_configs.K)
            np.random.shuffle(tmp)
            exclusion_matrix[i, :] = tmp[:self.selena_configs.L]
        exclusion_matrix = exclusion_matrix.astype(np.int32)
        self.exclusion_matrix = exclusion_matrix
        inclusion_matrix = np.zeros((all_train_indices.shape[0], self.selena_configs.K - self.selena_configs.L))
        for i in range(len(inclusion_matrix)):
            inc = []
            for k in range(self.selena_configs.K):
                if k not in exclusion_matrix[i, :]:
                    inc.append(k)
            inclusion_matrix[i, :] = np.array(inc)
        inclusion_matrix = inclusion_matrix.astype(np.int32)
        self.inclusion_matrix = inclusion_matrix
        self.logger.print_it("Selena Defender: data split into K={} models with L={} exclusions per sample.".format(self.selena_configs.K, self.selena_configs.L))

    def get_dataset_for_model(self, model_index: int) -> TensorDataset:
        dataset_to_return = MultiDatasets()
        indices = []
        for i in range(self.exclusion_matrix.shape[0]):
            if model_index not in self.exclusion_matrix[i, :]:
                indices.append(i)
        indices = np.array(indices)
        train_dataset = Subset(self.dataset.get('train'), indices=indices)
        dataset_to_return.add(train_dataset, id='train')
        test_dataset = self.dataset.get('test')
        dataset_to_return.add(test_dataset, id='test')
        return dataset_to_return
    
    def get_models_for_sample(self, sample_index: int) -> list[int]:
        return self.inclusion_matrix[sample_index, :].tolist()
    
    def get_excluded_models_for_sample(self, sample_index: int) -> list[int]:
        return self.exclusion_matrix[sample_index, :].tolist()

    def get_all_datasets(self) -> MultiDatasets:
        datasets = MultiDatasets()
        for k in range(self.selena_configs.K):
            datasets.add(self.get_dataset_for_model(model_index=k), id=k)
        return datasets
    

class DistillTrainManager(TrainManager):
    def __init__(self, train_configs: TrainConfigs, name: str, logger=None):
        super().__init__(train_configs=train_configs,
                        name=name,
                        logger=logger)
        
    def setup_loss(self, loss: Union[str, Callable]):
        self.logger.print_it('Setting up distillation loss...')
        self.criterion = DistillLoss()


class DistillLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (-torch.sum(targets*torch.log(torch.nn.functional.softmax(outputs,dim=1))))/outputs.shape[0]
        return loss