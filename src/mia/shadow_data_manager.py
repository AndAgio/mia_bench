import math
import numpy as np
import copy
from torch.utils.data import Subset, ConcatDataset, Dataset
from src.data.multi import MultiDatasets
from src.mia.auditing_data_manager import AuditingDatasetManager, FixedLabelDataset


class ShadowDatasetsManager():
    def __init__(self, original_datasets: MultiDatasets, auditing_dataset: AuditingDatasetManager, n_shadow_datasets: int, n_samples_per_dataset: int, mode: str, test_perc: float = 0.5, seed: int = 12345):
        assert original_datasets is not None
        assert original_datasets.n_splits() >= 2
        self.original_datasets = original_datasets
        assert 'train' in original_datasets.get_ids() and 'test' in original_datasets.get_ids()

        assert mode in ['online', 'offline']
        self.mode = mode
        assert auditing_dataset is not None
        assert auditing_dataset.get_all_ids() is not None
        assert auditing_dataset.get_all_ids() != []
        self.auditing_indices = auditing_dataset.get_all_ids()

        assert 0 < n_shadow_datasets < 20
        self.n_shadow_datasets = n_shadow_datasets
        assert 0 < n_samples_per_dataset < 100000
        self.n_samples_per_dataset = n_samples_per_dataset
        assert 0 < test_perc < 1
        self.test_perc = test_perc

        self.basic_dictionary = {'train_ids': None, 
                            'test_ids': None,
                            'all_ids': None}
        self.shadow_datasets_map = None
        self.sample()

    def sample(self):
        shadow_datasets_indices = self.sample_indices_for_offline_shadow_datasets()
        if self.mode == 'online':
            in_indices_to_add = copy.deepcopy(self.auditing_indices)
            for index_to_add in in_indices_to_add:
                n_datasets_to_randomly_sample = math.floor(self.n_shadow_datasets / 2)
                datasets_to_modify = self._rng.choice(np.arange(self.n_shadow_datasets), n_datasets_to_randomly_sample, replace=False).tolist()
                for dataset_to_modify in datasets_to_modify:
                    if in_indices_to_add < len(self.train_dataset):
                        indices_to_replace_from = [i for i in shadow_datasets_indices[dataset_to_modify]['tr_ids'] if i not in in_indices_to_add]
                    else:
                        indices_to_replace_from = [i for i in shadow_datasets_indices[dataset_to_modify]['te_ids'] if i not in in_indices_to_add]
                    index_to_substitute = self._rng.choice(indices_to_replace_from, 1, replace=False)
                    indices_to_replace_from.remove(index_to_substitute)
                    indices_to_replace_from.append(index_to_add)
                    if in_indices_to_add < len(self.train_dataset):
                        shadow_datasets_indices[dataset_to_modify]['tr_ids'] = indices_to_replace_from
                    else:
                        shadow_datasets_indices[dataset_to_modify]['te_ids'] = indices_to_replace_from
            self.shadow_datasets_map = {i: copy.deepcopy(self.basic_dictionary) for i in range(self.n_shadow_datasets)}
            for index in range(self.n_shadow_datasets):
                # shadow_dataset = ConcatDataset([Subset(self.train_dataset, 
                #                                         shadow_datasets_indices[index]['tr_ids']),
                #                                 Subset(self.test_dataset, 
                #                                         [id-len(self.train_dataset) for id in shadow_datasets_indices[index]['te_ids']])])
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
            #     shadow_dataset = ConcatDataset([Subset(self.train_dataset, 
            #                                             shadow_datasets_indices[index]['tr_ids']),
            #                                     Subset(self.test_dataset, 
            #                                             [id-len(self.train_dataset) for id in shadow_datasets_indices[index]['te_ids']])])
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
            train_indexes = self._rng.choice(available_indices_train,
                                            n_samples_from_victim_train,
                                            replace=False).tolist()
            test_indexes = self._rng.choice(np.arange(len(test_data)+len(train_data)),
                                            n_samples_from_victim_test,
                                            replace=False).tolist()
            all_indexes = train_indexes + test_indexes
            shadow_datasets_indices[i] = {'tr_ids': train_indexes,
                                        'te_ids': test_indexes,
                                        'ids': all_indexes}
        return shadow_datasets_indices
    
    def get(self, index: int, labels: str = 'mia'):
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
    

