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
                exp_hash: str,
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

        if self.mode == 'online':
            if not shadow_configs.n_samples_per_dataset > len(self.auditing_indices):
                self.logger.print_it(f'Number of samples per shadow dataset should be larger than number of auditing samples in online mode! Resetting n_samples_per_dataset to {len(self.auditing_indices)*2}...')
                shadow_configs.n_samples_per_dataset = len(self.auditing_indices) * 2
                self.n_samples_per_dataset = shadow_configs.n_samples_per_dataset

        assert 0 < shadow_configs.n_shadow_datasets < 101
        self.n_shadow_datasets = shadow_configs.n_shadow_datasets
        assert 0 < shadow_configs.n_samples_per_dataset < 100000
        self.n_samples_per_dataset = shadow_configs.n_samples_per_dataset
        assert 0 < shadow_configs.test_perc < 1
        self.test_perc = shadow_configs.test_perc

        self.seed = shadow_configs.seed
        self._rng = np.random.default_rng(seed=self.seed)

        self.exp_hash = exp_hash

        self.basic_dictionary = {'train_ids': None, 
                            'test_ids': None,
                            'all_ids': None}
        self.shadow_datasets_map = None
        start = time.time()
        self.logger.print_it(f'Sampling, refining and checking {self.n_shadow_datasets} shadow datasets. This may take a while...')
        self.sample()
        stop = time.time()
        self.logger.print_it('Sampling of {} shadow datasets completed in {:.3f} seconds'.format(self.n_shadow_datasets, stop-start))

    def sample(self):
        # TODO: add method to store and reload shadow datasets maps.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/13
        # assignees: AndAgio

        # use self.exp_hash to store/retrieve shadow datasets maps.
        
        train_data = self.original_datasets.get('train')
        test_data = self.original_datasets.get('test')
        shadow_datasets_indices = self.sample_indices_for_offline_shadow_datasets()
        if self.mode == 'online':
            # Use a set for fast membership checks and avoid rebuilding 'ids' on every small change.
            assert self.n_samples_per_dataset > len(self.auditing_indices), f'Number of samples per shadow dataset should be larger than number of auditing samples in online mode!'
            in_indices_to_add = copy.deepcopy(self.auditing_indices)
            in_indices_set = set(in_indices_to_add)
            train_len = len(train_data)
            s = time.time()
            self.logger.print_it(f'Refining online shadow datasets for all {len(in_indices_to_add)} samples. This may take a while...')
            for _, index_to_add in enumerate(in_indices_to_add):
                n_datasets_to_randomly_sample = math.floor(self.n_shadow_datasets / 2)
                datasets_to_modify = self._rng.choice(np.arange(self.n_shadow_datasets), n_datasets_to_randomly_sample, replace=False).tolist()
                for dataset_to_modify in datasets_to_modify:
                    # Select replacement candidates excluding auditing indices using set membership (O(1)).
                    if index_to_add < train_len:
                        candidates = shadow_datasets_indices[dataset_to_modify]['tr_ids']
                    else:
                        candidates = shadow_datasets_indices[dataset_to_modify]['te_ids']
                    indices_to_replace_from = [i for i in candidates if i not in in_indices_set]
                    if not indices_to_replace_from:
                        continue
                    # Ensure scalar Python int from numpy choice
                    index_to_substitute = int(self._rng.choice(indices_to_replace_from, 1, replace=False)[0])
                    if index_to_add < train_len:
                        shadow_datasets_indices[dataset_to_modify]['tr_ids'].remove(index_to_substitute)
                        shadow_datasets_indices[dataset_to_modify]['tr_ids'].append(index_to_add)
                    else:
                        shadow_datasets_indices[dataset_to_modify]['te_ids'].remove(index_to_substitute)
                        shadow_datasets_indices[dataset_to_modify]['te_ids'].append(index_to_add)
            self.logger.print_it(f'Refinement executed in {time.time() - s} seconds.')
            # Rebuild combined ids once per dataset instead of on every modification
            for dataset_idx in range(self.n_shadow_datasets):
                shadow_datasets_indices[dataset_idx]['ids'] = shadow_datasets_indices[dataset_idx]['tr_ids'] + shadow_datasets_indices[dataset_idx]['te_ids']
        elif self.mode == 'offline':
            pass
        else:
            raise ValueError(f'Mode should be either online or offline! Found "{self.mode}" instead!')
        # Copying correct indices to final map
        self.shadow_datasets_map = {i: copy.deepcopy(self.basic_dictionary) for i in range(self.n_shadow_datasets)}
        for index in range(self.n_shadow_datasets):
            self.shadow_datasets_map[index]['train_ids'] = copy.deepcopy(shadow_datasets_indices[index]['tr_ids'])
            self.shadow_datasets_map[index]['test_ids'] = copy.deepcopy(shadow_datasets_indices[index]['te_ids'])
            self.shadow_datasets_map[index]['all_ids'] = copy.deepcopy(shadow_datasets_indices[index]['ids'])
        # Checking for correctness
        assert self.check_in_out_correctness(), f'Something went wrong with shadow dataset sampling!'
        self.logger.print_it(f'Shadow datasets are OK for {self.mode} mode!')

    def sample_indices_for_offline_shadow_datasets(self):
        train_data = self.original_datasets.get('train')
        test_data = self.original_datasets.get('test')
        indices_to_avoid = copy.deepcopy(self.auditing_indices)
        available_indices_train = [i for i in range(len(train_data)) if i not in indices_to_avoid]
        available_indices_test = [i for i in range(len(train_data),len(test_data)+len(train_data)) if i not in indices_to_avoid]
        n_samples_from_victim_train = math.floor(self.n_samples_per_dataset * (1 - self.test_perc))
        n_samples_from_victim_test = self.n_samples_per_dataset - n_samples_from_victim_train
        shadow_datasets_indices = {i: {} for i in range(self.n_shadow_datasets)}
        s = time.time()
        self.logger.print_it(f'Sampling all {self.n_shadow_datasets} shadow datasets...')
        for i in range(self.n_shadow_datasets):
            train_indexes = self._rng.choice(available_indices_train,
                                            n_samples_from_victim_train,
                                            replace=False).tolist()
            test_indexes = self._rng.choice(available_indices_test,
                                            n_samples_from_victim_test,
                                            replace=False).tolist()
            all_indexes = train_indexes + test_indexes
            shadow_datasets_indices[i] = {'tr_ids': train_indexes,
                                        'te_ids': test_indexes,
                                        'ids': all_indexes}
        self.logger.print_it(f'Sampling executed in {time.time() - s} seconds.')
        return shadow_datasets_indices
    
    def check_in_out_correctness(self):
        if self.mode == 'online':
            expected_num_ins = math.floor(self.n_shadow_datasets / 2)
            expected_num_outs = self.n_shadow_datasets - expected_num_ins
        elif self.mode == 'offline':
            expected_num_ins = 0
            expected_num_outs = self.n_shadow_datasets
        else:
            raise ValueError(f'Mode should be either online or offline! Found "{self.mode}" instead!')
        train_data = self.original_datasets.get('train')
        found_outcomes = []
        s = time.time()
        self.logger.print_it(f'Checking correctness of shadow datasets in {self.mode} mode for all {len(self.auditing_indices)} samples. This may take a while...')
        for k, index in enumerate(self.auditing_indices):
            n_ins_found = 0
            n_outs_found = 0
            n_ins_found_all = 0
            n_outs_found_all = 0
            for shadow_index in range(self.n_shadow_datasets):
                if index < len(train_data):
                    if index in self.shadow_datasets_map[shadow_index]['train_ids']:
                        n_ins_found += 1
                    else:
                        n_outs_found += 1
                else:
                    if index in self.shadow_datasets_map[shadow_index]['test_ids']:
                        n_ins_found += 1
                    else:
                        n_outs_found += 1
                if index in self.shadow_datasets_map[shadow_index]['all_ids']:
                    n_ins_found_all += 1
                else:
                    n_outs_found_all += 1
            outcome = [n_ins_found == expected_num_ins,
                        n_outs_found == expected_num_outs,
                        n_ins_found_all == expected_num_ins,
                        n_outs_found_all == expected_num_outs]
            found_outcomes += outcome
        self.logger.print_it(f'Checking executed in {time.time() - s} seconds with {"positive" if all(found_outcomes) else "negative"} outcome.')
        return all(found_outcomes)

    def sample_random_indices(self, num_data: int = 1000):
        train_data = self.original_datasets.get('train')
        test_data = self.original_datasets.get('test')
        indices_to_avoid = copy.deepcopy(self.auditing_indices)
        available_indices_train = [i for i in range(len(train_data)) if i not in indices_to_avoid]
        n_samples_from_victim_train = math.floor(num_data * (1 - self.test_perc))
        n_samples_from_victim_test = num_data - n_samples_from_victim_train
        self.logger.print_it(f'Sampling random sample dataset...')
        train_indexes = self._rng.choice(available_indices_train,
                                        n_samples_from_victim_train,
                                        replace=False).tolist()
        test_indexes = self._rng.choice(np.arange(len(train_data),len(test_data)+len(train_data)),
                                        n_samples_from_victim_test,
                                        replace=False).tolist()
        all_indexes = train_indexes + test_indexes
        indices = {'train_ids': train_indexes,
                    'test_ids': test_indexes,
                    'all_ids': all_indexes}
        return indices
    
    def get_random_population(self, indices: dict = None, num_data: int = None, labels: str = 'original'):
        original_train_data = self.original_datasets.get('train')
        original_test_data = self.original_datasets.get('test')
        if indices is None:
            assert 0 < num_data <= 1000, f'Number of data to sample random population should be between 1 and 1000, received {num_data} instead!'
            indices = self.sample_random_indices(num_data=num_data)
        if labels == 'mia':
            train_data = Subset(original_train_data, indices['train_ids'])
            test_data = Subset(original_test_data, [id-len(original_train_data) for id in indices['test_ids']])
            return ConcatDataset([FixedLabelDataset(train_data,
                                                    fixed_label=1),
                                    FixedLabelDataset(test_data,
                                                    fixed_label=0),])
        elif labels == 'original':
            train_data = Subset(original_train_data, indices['train_ids'])
            test_data = Subset(original_test_data, [id-len(original_train_data) for id in indices['test_ids']])
            return ConcatDataset([train_data,test_data])
        else:
            raise ValueError('Labels mode should be either mia or original!')

    
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
    