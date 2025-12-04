import math
import time
import numpy as np
import copy
from torch.utils.data import Subset, ConcatDataset, Dataset
from src.data.multi import MultiDatasets
from src.mia.auditing_data_manager import AuditingDatasetManager, FixedLabelDataset
from src.utils.configs import ShadowDataConfigs
from src.utils.log import Loggable, SmartLogger, DumbLogger
from typing import Union

class ShadowDatasetsManager(Loggable):
    def __init__(self,
                original_datasets: MultiDatasets, 
                auditing_dataset: AuditingDatasetManager,
                shadow_configs: ShadowDataConfigs,
                logger: Union[SmartLogger, DumbLogger] = None):
        super().__init__(logger=logger)
        assert original_datasets is not None
        assert original_datasets.n_splits() >= 2
        self.original_datasets = original_datasets
        assert 'train' in original_datasets.get_ids() and 'test' in original_datasets.get_ids()

        assert shadow_configs.mode in ['online', 'offline']
        self.mode = shadow_configs.mode
        assert auditing_dataset is not None
        assert auditing_dataset.get_all_ids() is not None
        assert auditing_dataset.get_all_ids() != []
        self.auditing_indices = auditing_dataset.get_all_ids()

        assert 0 < shadow_configs.n_shadow_datasets < 101
        self.n_shadow_datasets = shadow_configs.n_shadow_datasets
        assert 0 < shadow_configs.n_samples_per_dataset < 100000
        self.n_samples_per_dataset = shadow_configs.n_samples_per_dataset
        assert 0 < shadow_configs.test_perc < 1
        self.test_perc = shadow_configs.test_perc

        self.seed = shadow_configs.seed
        self._rng = np.random.default_rng(seed=self.seed)

        self.basic_dictionary = {'train_ids': None, 
                            'test_ids': None,
                            'all_ids': None}
        self.shadow_datasets_map = None
        self.logger.print_it(f'Sampling {self.n_shadow_datasets} shadow datasets. This may take a while...')
        start = time.time()
        self.sample()
        stop = time.time()
        self.logger.print_it('Sampling of {} shadow datasets completed in {:.3f} seconds'.format(self.n_shadow_datasets, stop-start))

    def sample(self):
        train_data = self.original_datasets.get('train')
        test_data = self.original_datasets.get('test')
        shadow_datasets_indices = self.sample_indices_for_offline_shadow_datasets()
        if self.mode == 'online':
            in_indices_to_add = copy.deepcopy(self.auditing_indices)
            for index_to_add in in_indices_to_add:
                n_datasets_to_randomly_sample = math.floor(self.n_shadow_datasets / 2)
                datasets_to_modify = self._rng.choice(np.arange(self.n_shadow_datasets), n_datasets_to_randomly_sample, replace=False).tolist()
                for dataset_to_modify in datasets_to_modify:
                    if index_to_add < len(train_data):
                        indices_to_replace_from = [i for i in shadow_datasets_indices[dataset_to_modify]['tr_ids'] if i not in in_indices_to_add]
                    else:
                        indices_to_replace_from = [i for i in shadow_datasets_indices[dataset_to_modify]['te_ids'] if i not in in_indices_to_add]
                    index_to_substitute = self._rng.choice(indices_to_replace_from, 1, replace=False)
                    indices_to_replace_from.remove(index_to_substitute)
                    indices_to_replace_from.append(index_to_add)
                    if index_to_add < len(train_data):
                        shadow_datasets_indices[dataset_to_modify]['tr_ids'] = indices_to_replace_from
                    else:
                        shadow_datasets_indices[dataset_to_modify]['te_ids'] = indices_to_replace_from
            self.shadow_datasets_map = {i: copy.deepcopy(self.basic_dictionary) for i in range(self.n_shadow_datasets)}
            for index in range(self.n_shadow_datasets):
                # shadow_dataset = ConcatDataset([Subset(train_data, 
                #                                         shadow_datasets_indices[index]['tr_ids']),
                #                                 Subset(test_data, 
                #                                         [id-len(train_data) for id in shadow_datasets_indices[index]['te_ids']])])
                # shadow_datasets[index]['data'] = shadow_dataset
                self.shadow_datasets_map[index]['train_ids'] = shadow_datasets_indices[index]['tr_ids']
                self.shadow_datasets_map[index]['test_ids'] = shadow_datasets_indices[index]['te_ids']
                self.shadow_datasets_map[index]['all_ids'] = shadow_datasets_indices[index]['tr_ids'] + shadow_datasets_indices[index]['te_ids']
        else:
            self.shadow_datasets_map = {i: copy.deepcopy(self.basic_dictionary) for i in range(self.n_shadow_datasets)}
            for index in range(self.n_shadow_datasets):
                self.shadow_datasets_map[index]['train_ids'] = shadow_datasets_indices[index]['tr_ids']
                self.shadow_datasets_map[index]['test_ids'] = shadow_datasets_indices[index]['te_ids']
                self.shadow_datasets_map[index]['all_ids'] = shadow_datasets_indices[index]['ids']
            # shadow_datasets = copy.deepcopy(shadow_datasets_indices)
            # for index in range(n_shadow_datasets):
            #     shadow_dataset = ConcatDataset([Subset(train_data, 
            #                                             shadow_datasets_indices[index]['tr_ids']),
            #                                     Subset(test_data, 
            #                                             [id-len(train_data) for id in shadow_datasets_indices[index]['te_ids']])])
            #     shadow_datasets[index]['data'] = shadow_dataset
            # return shadow_datasets

    def sample_indices_for_offline_shadow_datasets(self):
        train_data = self.original_datasets.get('train')
        test_data = self.original_datasets.get('test')
        indices_to_avoid = copy.deepcopy(self.auditing_indices)
        available_indices_train = [i for i in range(len(train_data)) if i not in indices_to_avoid]
        n_samples_from_victim_train = math.floor(self.n_samples_per_dataset * (1 - self.test_perc))
        n_samples_from_victim_test = self.n_samples_per_dataset - n_samples_from_victim_train
        shadow_datasets_indices = {i: {} for i in range(self.n_shadow_datasets)}
        for i in range(self.n_shadow_datasets):
            self.logger.print_it_same_line(f'Sampling shadow dataset {i+1}/{self.n_shadow_datasets}. This may take a while...')
            train_indexes = self._rng.choice(available_indices_train,
                                            n_samples_from_victim_train,
                                            replace=False).tolist()
            test_indexes = self._rng.choice(np.arange(len(train_data),len(test_data)+len(train_data)),
                                            n_samples_from_victim_test,
                                            replace=False).tolist()
            all_indexes = train_indexes + test_indexes
            shadow_datasets_indices[i] = {'tr_ids': train_indexes,
                                        'te_ids': test_indexes,
                                        'ids': all_indexes}
        self.logger.set_logger_newline()
        return shadow_datasets_indices
    
    def get(self, index: int, labels: str = 'mia'):
        assert self.check_id(index), f"Invalid ID for shadow dataset you are trying to get with id {index}"
        original_train_data = self.original_datasets.get('train')
        original_test_data = self.original_datasets.get('test')
        if labels == 'mia':
            train_data = Subset(original_train_data, self.shadow_datasets_map[index]['train_ids'])
            test_data = Subset(original_test_data, [id-len(original_train_data) for id in self.shadow_datasets_map[index]['test_ids']])
            return ConcatDataset([FixedLabelDataset(train_data,
                                                    fixed_label=1),
                                    FixedLabelDataset(test_data,
                                                    fixed_label=0),])
        elif labels == 'original':
            train_data = Subset(original_train_data, self.shadow_datasets_map[index]['train_ids'])
            test_data = Subset(original_test_data, [id-len(original_train_data) for id in self.shadow_datasets_map[index]['test_ids']])
            return ConcatDataset([train_data,test_data])
        else:
            raise ValueError('Labels mode should be either mia or original!')

    def get_by_indices(self, indices: list[int], labels: str = 'mia'):
        datasets = MultiDatasets()
        for index in indices:
            datasets.add(self.get(index, labels=labels), index)
        return datasets

    def get_all(self, labels: str = 'mia'):
        datasets = MultiDatasets()
        for index in range(self.n_shadow_datasets):
            datasets.add(self.get(index, labels=labels), index)
        return datasets
    
    def find_shadow_datasets_containing_sample_id(self, id: int, split: str = 'train'):
        indices_found = []
        assert split in ['train', 'test', 'all']
        field = 'train_ids' if split == 'train' else 'test_ids' if split == 'test' else 'all_ids'
        for index in range(self.n_shadow_datasets):
            if id in self.shadow_datasets_map[index][field]:
                indices_found.append(index)
        return indices_found
    
    def get_shadow_datasets_containing_sample_id(self, id: int, split: str = 'train', labels: str = 'mia'):
        return self.get_by_indices(self.find_shadow_datasets_containing_sample_id(id=id, split=split),
                                    labels=labels)
    
    def get_num_dataset(self):
        return len(self.shadow_datasets_map.keys())
    
    def check_id(self, index: int):
        return index in self.shadow_datasets_map.keys()
    
    def get_all_ids(self):
        return list(self.shadow_datasets_map.keys())
    