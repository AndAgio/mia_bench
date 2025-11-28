import math
import numpy as np
from torch.utils.data import Subset, ConcatDataset, Dataset
from src.data.multi import MultiDatasets


class AuditingDatasetManager():
    def __init__(self, original_datasets: MultiDatasets, n_auditing_samples: int, in_perc: float = 0.5, seed: int = 12345):
        assert original_datasets is not None
        assert original_datasets.n_splits() >= 2
        self.original_datasets = original_datasets
        assert 'train' in original_datasets.get_ids() and 'test' in original_datasets.get_ids()
        assert n_auditing_samples > 0
        assert 0 < in_perc < 1
        n_samples_to_pick_from_train = math.floor(n_auditing_samples * in_perc)
        n_samples_to_pick_from_test = n_auditing_samples - n_samples_to_pick_from_train
        assert n_samples_to_pick_from_train < len(self.original_datasets.get('train'))
        assert n_samples_to_pick_from_test < len(self.original_datasets.get('test'))
        self.n_auditing_samples = n_auditing_samples
        self._rng = np.random.default_rng(seed=seed)
        self.in_auditing_dataset = None
        self.in_ids = None
        self.out_auditing_dataset = None
        self.out_ids = None
        self.full_auditing_dataset = None
        self.all_ids = None
        self.audit_dataset = None
        self._sample(n_samples_to_pick_from_train, n_samples_to_pick_from_test)

    def _sample(self, n_samples_to_pick_from_train: int, n_samples_to_pick_from_test: int):
        member_indexes = self._rng.choice(np.arange(len(self.original_datasets.get('train'))), 
                                        n_samples_to_pick_from_train,
                                        replace=False).tolist()
        non_member_indexes = self._rng.choice(np.arange(len(self.original_datasets.get('test'))+len(self.original_datasets.get('train'))),
                                            n_samples_to_pick_from_test,
                                            replace=False).tolist()
        all_indexes = member_indexes + non_member_indexes
        self.in_auditing_dataset = Subset(self.original_datasets.get('train'),
                                        member_indexes)
        self.in_ids = member_indexes
        self.out_auditing_dataset = Subset(self.original_datasets.get('test'),
                                        [id-len(self.original_datasets.get('train')) for id in non_member_indexes])
        self.out_ids = [id-len(self.original_datasets.get('train')) for id in non_member_indexes]
        self.full_auditing_dataset = ConcatDataset([self.in_auditing_dataset, self.out_auditing_dataset])
        self.all_ids = all_indexes
        self.audit_dataset = ConcatDataset([FixedLabelDataset(self.in_auditing_dataset,
                                                            fixed_label=1),
                                            FixedLabelDataset(self.out_auditing_dataset,
                                                            fixed_label=0),])
    
    def get_audit(self):
        assert self.audit_dataset is not None
        return self.audit_dataset
    
    def get_members_only(self):
        assert self.in_auditing_dataset is not None
        return self.in_auditing_dataset
    
    def get_non_members_only(self):
        assert self.out_auditing_dataset is not None
        return self.out_auditing_dataset
    
    def get_members_ids(self):
        assert self.in_ids is not None
        return self.in_ids
    
    def get_non_members_ids(self):
        assert self.out_ids is not None
        return self.out_ids
    
    def get_all_ids(self):
        assert self.all_ids is not None
        return self.all_ids
    
    def get(self, labels: str = 'mia'):
        if labels == 'mia':
            return ConcatDataset([FixedLabelDataset(self.in_auditing_dataset,
                                                    fixed_label=1),
                                    FixedLabelDataset(self.out_auditing_dataset,
                                                    fixed_label=0),])
        elif labels == 'original':
            return ConcatDataset([self.in_auditing_dataset, self.out_auditing_dataset])
        else:
            raise ValueError('Labels mode should be either mia or original!')



class FixedLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, fixed_label: int = 0):
        super().__init__()
        self.data = dataset
        self.fixed_label = fixed_label

    def __getitem__(self, index):
        return self.data[index][0], self.fixed_label

    def __len__(self):
        return len(self.data)