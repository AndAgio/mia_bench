import math
import numpy as np
import copy
from torch.utils.data import Dataset, Subset, ConcatDataset


class DatasetManager():
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, seed: int = 12345):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.seed = seed
        self._rng = np.random.default_rng(seed=self.seed)

    def get_victim_train(self):
        return self.train_dataset
    
    def get_victim_test(self):
        return self.test_dataset
    
    def sample_auditing_dataset(self, n_auditing_samples: int, in_perc: float = 0.5):
        assert 0 < in_perc < 1
        n_samples_to_pick_from_train = math.floor(n_auditing_samples * in_perc)
        n_samples_to_pick_from_test = n_auditing_samples - n_samples_to_pick_from_train
        assert n_samples_to_pick_from_train < len(self.train_dataset)
        assert n_samples_to_pick_from_test < len(self.test_dataset)
        member_indexes = self._rng.choice(np.arange(len(self.train_dataset)), 
                                        n_samples_to_pick_from_train,
                                        replace=False).tolist()
        non_member_indexes = self._rng.choice(np.arange(len(self.test_dataset)+len(self.train_dataset)),
                                            n_samples_to_pick_from_test,
                                            replace=False).tolist()
        all_indexes = member_indexes + non_member_indexes
        victim_dataset = {'in': Subset(self.train_dataset,
                                    member_indexes),
                        'out': Subset(self.test_dataset,
                                    [id-len(self.train_dataset) for id in non_member_indexes]),
                        'in_ids': member_indexes,
                        'out_ids': non_member_indexes,
                        'all_ids': all_indexes}
        return victim_dataset
    
    def sample_shadow_datasets(self, auditing_dataset: dict, n_shadow_datasets: int, n_samples_per_dataset: int, mode: str, test_perc: float = 0.5):
        assert mode in ['online', 'offline']
        if mode == 'online':
            assert auditing_dataset is not None
            assert auditing_dataset['all_ids'] is not None
            assert auditing_dataset['all_ids'] != []
        shadow_datasets_indices = self.sample_indices_for_offline_shadow_datasets(auditing_dataset=auditing_dataset,
                                                                    n_shadow_datasets=n_shadow_datasets,
                                                                    n_samples_per_dataset=n_samples_per_dataset,
                                                                    test_perc=test_perc)
        if mode == 'online':
            in_indices_to_add = auditing_dataset['all_ids']
            for index_to_add in in_indices_to_add:
                n_datasets_to_randomly_sample = math.floor(n_shadow_datasets / 2)
                datasets_to_modify = self._rng.choice(np.arange(n_shadow_datasets), n_datasets_to_randomly_sample, replace=False).tolist()
                for dataset_to_modify in datasets_to_modify:
                    if in_indices_to_add < len(self.train_dataset):
                        indices_to_replace_from = [i for i in shadow_datasets[dataset_to_modify]['tr_ids'] if i not in in_indices_to_add]
                    else:
                        indices_to_replace_from = [i for i in shadow_datasets[dataset_to_modify]['te_ids'] if i not in in_indices_to_add]
                    index_to_substitute = self._rng.choice(indices_to_replace_from, 1, replace=False)
                    indices_to_replace_from.remove(index_to_substitute)
                    indices_to_replace_from.append(index_to_add)
                    if in_indices_to_add < len(self.train_dataset):
                        shadow_datasets_indices[dataset_to_modify]['tr_ids'] = indices_to_replace_from
                    else:
                        shadow_datasets_indices[dataset_to_modify]['te_ids'] = indices_to_replace_from
            shadow_datasets = {i: {} for i in range(n_shadow_datasets)}
            for index in range(n_shadow_datasets):
                shadow_dataset = ConcatDataset([Subset(self.train_dataset, 
                                                        shadow_datasets_indices[index]['tr_ids']),
                                                Subset(self.test_dataset, 
                                                        [id-len(self.train_dataset) for id in shadow_datasets_indices[index]['te_ids']])])
                shadow_datasets[index]['data'] = shadow_dataset
                shadow_datasets[index]['tr_ids'] = shadow_datasets_indices[index]['tr_ids']
                shadow_datasets[index]['te_ids'] = shadow_datasets_indices[index]['te_ids']
                shadow_datasets[index]['ids'] = shadow_datasets_indices[index]['tr_ids'] + shadow_datasets_indices[index]['te_ids']
        else:
            shadow_datasets = copy.deepcopy(shadow_datasets_indices)
            for index in range(n_shadow_datasets):
                shadow_dataset = ConcatDataset([Subset(self.train_dataset, 
                                                        shadow_datasets_indices[index]['tr_ids']),
                                                Subset(self.test_dataset, 
                                                        [id-len(self.train_dataset) for id in shadow_datasets_indices[index]['te_ids']])])
                shadow_datasets[index]['data'] = shadow_dataset
            return shadow_datasets
        
    def sample_indices_for_offline_shadow_datasets(self, auditing_dataset: dict, n_shadow_datasets: int, n_samples_per_dataset: int, test_perc: float = 0.5):
        indices_to_avoid = auditing_dataset['all_ids']
        available_indices_train = [i for i in range(len(self.train_dataset)) if i not in indices_to_avoid]
        n_samples_from_victim_train = math.floor(n_samples_per_dataset * (1 - test_perc))
        n_samples_from_victim_test = n_samples_per_dataset - n_samples_from_victim_train
        shadow_datasets_indices = {i: {} for i in range(n_shadow_datasets)}
        for i in range(n_shadow_datasets):
            train_indexes = self._rng.choice(available_indices_train,
                                            n_samples_from_victim_train,
                                            replace=False).tolist()
            test_indexes = self._rng.choice(np.arange(len(self.test_dataset)+len(self.train_dataset)),
                                            n_samples_from_victim_test,
                                            replace=False).tolist()
            all_indexes = train_indexes + test_indexes
            shadow_datasets_indices[i] = {'tr_ids': train_indexes,
                                        'te_ids': test_indexes,
                                        'ids': all_indexes}
        return shadow_datasets_indices
