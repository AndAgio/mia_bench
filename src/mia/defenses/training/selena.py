from typing import Union, Callable
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from src.data.helpers import MultiDatasets, TargetOverrideDataset
from src.utils.configs import DefenderConfigs, SelenaDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender
from src.mia.helpers.shadow_models_manager import ShadowModelsManager
from src.trainer.train_manager import TrainManager
from src.trainer.stats_tracker import StageSummary
from src.trainer.distributed import is_rank0


class SelenaDefender(BaseDefender):
    # Implementation of training time optimization component of Selena defense from "Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture" (https://www.usenix.org/system/files/sec22-tang.pdf).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, SelenaDefenseConfigs), f"Selena Defender can only be used with SelenaDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'selena_defender'
        self.selena_configs = defender_configs.defense
        self.split_data_manager = SplitDataManager(dataset=self.dataset,
                                                    selena_configs=self.selena_configs,
                                                    seed=defender_configs.train.seed,
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
            train_manager = SelenaSplitTrainManager(train_configs=train_configs,
                                                    name='selena_split_{}'.format(k),
                                                    logger=self.logger)
            train_manager.initialize_train(dataset=data_splits.get(id=k),
                                            model=models[k],
                                            configs=train_configs)
            model = train_manager.train(return_best_model=True,
                                        return_last_model=False,
                                        return_stats=False)
            # Keep finished split models off the accelerator; each is moved back only while it produces soft labels.
            self.split_model_manager.update(index=k,
                                            model=model.cpu())
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
        # exclusion_mask[i, k] is True when model k never saw sample i, i.e. k is one of its L teachers.
        exclusion_mask = self.split_data_manager.get_exclusion_mask()
        # Running sum of the teachers' softmax outputs, so only one model's predictions are held at a time.
        masked_sum = torch.zeros(len(train_dataset), self.model_configs.num_classes)
        self.logger.print_it("Selena Defender: gathering distillation outputs from split models. This may take a while...")
        start_pred = time.time()
        for k in range(self.selena_configs.K):
            self.logger.print_it_same_line(f"Selena Defender: gathering distillation outputs from split model {k+1}/{self.selena_configs.K}...", console_only=True)
            # Model k is only a teacher for the samples it never saw.
            excluded_indices = np.nonzero(exclusion_mask[:, k])[0]
            model = self.split_model_manager.get(index=k)
            model = model.to(device)
            model.eval()
            dataloader = DataLoader(Subset(train_dataset, indices=excluded_indices), batch_size=train_configs.batch_size, shuffle=False)
            all_outputs = []
            with torch.no_grad():
                for batch_idx, (samples, _, _, _) in enumerate(dataloader):
                    samples = samples.to(device)
                    outputs = model(samples)
                    all_outputs.append(torch.nn.functional.softmax(outputs, dim=1).cpu())
            if all_outputs:
                masked_sum[torch.from_numpy(excluded_indices)] += torch.cat(all_outputs, dim=0)
            model.cpu()
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it(f"Selena Defender: gathered distillation outputs in {time.time()-start_pred:.2f} seconds.")

        counts = torch.from_numpy(exclusion_mask.sum(axis=1)).unsqueeze(-1).to(masked_sum.dtype)
        mean_outputs = masked_sum / counts
        assert torch.isfinite(mean_outputs).all(), "Found NaN or Inf in `mean_outputs`"

        distilled_dataset = TargetOverrideDataset(train_dataset, mean_outputs)

        dataset_to_return.add(distilled_dataset, id='train')
        try:
            val_dataset = self.dataset.get('val')
            dataset_to_return.add(val_dataset, id='val')
        except KeyError:
            pass
        test_dataset = self.dataset.get('test')
        dataset_to_return.add(test_dataset, id='test')
        self.logger.print_it(f"Selena Defender: gathered distilled dataset in {time.time()-start:.2f} seconds.")
        return dataset_to_return

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        self.logger.print_it("Selena Defender: defend_model does not modify the model at inference time.")
        self.defended_model = self.trained_model
        return self.defended_model
    

class SplitDataManager:
    def __init__(self, dataset: MultiDatasets, selena_configs: SelenaDefenseConfigs, seed: int = 12345, logger=None):
        self.dataset = dataset
        self.selena_configs = selena_configs
        self.seed = seed
        self.logger = logger
        self.split_data()

    def split_data(self):
        n_samples = len(self.dataset.get('train'))
        K, L = self.selena_configs.K, self.selena_configs.L
        rng = np.random.default_rng(self.seed)
        # One random permutation of the K model ids per sample: the first L are the models that never see it.
        permutations = rng.permuted(np.tile(np.arange(K), (n_samples, 1)), axis=1)
        self.exclusion_matrix = np.sort(permutations[:, :L], axis=1).astype(np.int32)
        self.inclusion_matrix = np.sort(permutations[:, L:], axis=1).astype(np.int32)
        self.exclusion_mask = np.zeros((n_samples, K), dtype=bool)
        np.put_along_axis(self.exclusion_mask, self.exclusion_matrix, True, axis=1)
        self.logger.print_it("Selena Defender: data split into K={} models with L={} exclusions per sample.".format(self.selena_configs.K, self.selena_configs.L))

    def get_dataset_for_model(self, model_index: int) -> TensorDataset:
        dataset_to_return = MultiDatasets()
        indices = np.nonzero(~self.exclusion_mask[:, model_index])[0]

        #TODO: Refactor selena as well to avoid creating Subset datasets and instead use the custom dataset classes defined in data helpers.
        #Issue URL: https://github.com/AndAgio/mia_bench/issues/46
        # assignees: AndAgio

        train_dataset = Subset(self.dataset.get('train'), indices=indices)
        dataset_to_return.add(train_dataset, id='train')
        try:
            val_dataset = self.dataset.get('val')
            dataset_to_return.add(val_dataset, id='val')
        except KeyError:
            pass
        try:
            test_dataset = self.dataset.get('test')
            dataset_to_return.add(test_dataset, id='test')
        except KeyError:
            pass
        return dataset_to_return
    
    def get_models_for_sample(self, sample_index: int) -> list[int]:
        return self.inclusion_matrix[sample_index, :].tolist()
    
    def get_excluded_models_for_sample(self, sample_index: int) -> list[int]:
        return self.exclusion_matrix[sample_index, :].tolist()

    def get_exclusion_mask(self) -> np.ndarray:
        # (N, K) boolean: True where model k is excluded from (never trained on) sample i.
        return self.exclusion_mask

    def get_all_datasets(self) -> MultiDatasets:
        datasets = MultiDatasets()
        for k in range(self.selena_configs.K):
            datasets.add(self.get_dataset_for_model(model_index=k), id=k)
        return datasets
    

class SelenaSplitTrainManager(TrainManager):
    def __init__(self, train_configs: TrainConfigs, name: str, logger=None):
        super().__init__(train_configs=train_configs,
                        name=name,
                        logger=logger)
        
    def build_message_for_stage_end(self, stage_summary: StageSummary) -> str:
        message = f"{self.device.type.upper()}:{self.local_rank} | "
        message += f"SPLIT ID: {self.name.split('_')[-1]} | "
        message += f"EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        message += ' {}: '.format(stage_summary.stage.upper())
        if is_rank0():
            metrics = stage_summary.metrics
        message = self.append_metrics(message, metrics)
        message = self.append_lr(message)
        message = self.append_times(message)
        return message
    
    def build_message_for_batch_end(self, index_batch, total_batches) -> str:
        message = f"{self.device.type.upper()}:{self.local_rank} | "
        message += f"SPLIT ID: {self.name.split('_')[-1]} | "
        message += f"EPOCH: {self.epoch}/{self.train_configs.scheduler_config.epochs} |"
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(self.epoch_stats_tracker.get_stage().upper())
        if is_rank0():
            metrics = self.epoch_stats_tracker._require_active_stage().current_avgs()
        message = self.append_metrics(message, metrics)
        message = self.append_lr(message)
        message = self.append_times(message)
        return message
    

class DistillTrainManager(TrainManager):
    def __init__(self, train_configs: TrainConfigs, name: str, logger=None):
        super().__init__(train_configs=train_configs,
                        name=name,
                        logger=logger)
        
    def setup_loss(self, loss: Union[str, Callable]):
        self.logger.print_it('Setting up distillation loss...')
        self.criterion = DistillLoss()


class DistillLoss(torch.nn.Module):
    # Like nn.CrossEntropyLoss, starts with reduction='none' (per-sample losses, needed by ESAM) and is switched
    # to 'mean' by TrainManager.setup_optimizer for plain optimizers.
    def __init__(self, reduction: str = 'none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Only the train split carries soft labels; val/test keep integer class ids.
        if targets.dim() == 1:
            loss = torch.nn.functional.cross_entropy(outputs, targets.long(), reduction='none')
        else:
            loss = -torch.sum(targets*torch.nn.functional.log_softmax(outputs,dim=1), dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
