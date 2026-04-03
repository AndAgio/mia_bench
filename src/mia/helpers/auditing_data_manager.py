import math
import numpy as np
from torch.utils.data import Subset, ConcatDataset
from src.data.helpers import MultiDatasets, FixedLabelDataset, MyOriginalIndexSubset, MyConcatDataset
from src.utils.configs import AuditingDataConfigs
from src.utils.log import Loggable, MyLogger


class AuditingDatasetManager(Loggable):
    def __init__(self, defender_datasets: MultiDatasets, configs: AuditingDataConfigs, logger: MyLogger = None): 
        super().__init__(logger=logger)
        assert defender_datasets is not None
        assert defender_datasets.n_splits() >= 2
        self.defender_datasets = defender_datasets
        assert 'train' in defender_datasets.get_ids() and 'test' in defender_datasets.get_ids()
        assert configs.n_auditing_samples > 0
        assert 0 < configs.in_perc < 1
        n_samples_to_pick_from_train = math.floor(configs.n_auditing_samples * configs.in_perc)
        n_samples_to_pick_from_test = configs.n_auditing_samples - n_samples_to_pick_from_train
        assert n_samples_to_pick_from_train < len(self.defender_datasets.get('train')), f"Trying to pick {n_samples_to_pick_from_train} samples from train but it only contains {len(self.defender_datasets.get('train'))} samples!"
        assert n_samples_to_pick_from_test < len(self.defender_datasets.get('test')), f"Trying to pick {n_samples_to_pick_from_test} samples from test but it only contains {len(self.defender_datasets.get('test'))} samples!"
        
        # TODO: add option to sample from the attacker dataset instead of the defender test dataset for the non-members, to be used when the number of non-members requested is large.
        # assignees: AndAgio.

        self.n_auditing_samples = configs.n_auditing_samples
        self.seed = configs.seed
        self._rng = np.random.default_rng(seed=self.seed)
        self.in_auditing_dataset = None
        self.in_ids = None
        self.out_auditing_dataset = None
        self.out_ids = None
        self.full_auditing_dataset = None
        self.all_ids = None
        self.audit_dataset = None
        self._sample(n_samples_to_pick_from_train, n_samples_to_pick_from_test)

    def _sample(self, n_samples_to_pick_from_train: int, n_samples_to_pick_from_test: int):
        self.logger.print_it(f"Sampling auditing dataset with {n_samples_to_pick_from_train} samples picked from train and {n_samples_to_pick_from_test} samples picked from test...")
        defender_train_original_sample_indexes = self.defender_datasets.get('train').get_all_original_indices() 
        defender_test_original_sample_indexes = self.defender_datasets.get('test').get_all_original_indices()
        member_original_sample_indexes = self._rng.choice(defender_train_original_sample_indexes,
                                                        n_samples_to_pick_from_train,
                                                        replace=False).tolist()
        non_member_original_sample_indexes = self._rng.choice(defender_test_original_sample_indexes,
                                                            n_samples_to_pick_from_test,
                                                            replace=False).tolist()
        self.in_ids = member_original_sample_indexes
        self.in_auditing_dataset = MyOriginalIndexSubset(self.defender_datasets.get('train'), 
                                                        member_original_sample_indexes,
                                                        strict=True,
                                                        return_indexed_tuple=True)
        self.out_ids = non_member_original_sample_indexes
        self.out_auditing_dataset = MyOriginalIndexSubset(self.defender_datasets.get('test'),
                                                        non_member_original_sample_indexes, 
                                                        strict=True, 
                                                        return_indexed_tuple=True)
        self.all_ids = member_original_sample_indexes + non_member_original_sample_indexes
        self.full_auditing_dataset = MyConcatDataset([self.in_auditing_dataset, self.out_auditing_dataset])

        # member_indexes = self._rng.choice(np.arange(len(self.original_datasets.get('train'))), 
        #                                 n_samples_to_pick_from_train,
        #                                 replace=False).tolist()
        # non_member_indexes = self._rng.choice(np.arange(len(self.original_datasets.get('train')), len(self.original_datasets.get('test'))+len(self.original_datasets.get('train'))),
        #                                     n_samples_to_pick_from_test,
        #                                     replace=False).tolist()
        # all_indexes = member_indexes + non_member_indexes
        # self.in_auditing_dataset = Subset(self.original_datasets.get('train'),
        #                                 member_indexes)
        # self.in_ids = member_indexes
        # self.out_auditing_dataset = Subset(self.original_datasets.get('test'),
        #                                 [id-len(self.original_datasets.get('train')) for id in non_member_indexes])
        # self.out_ids = [id-len(self.original_datasets.get('train')) for id in non_member_indexes]
        # self.full_auditing_dataset = ConcatDataset([self.in_auditing_dataset, self.out_auditing_dataset])
        # self.all_ids = all_indexes
        # self.audit_dataset = ConcatDataset([FixedLabelDataset(self.in_auditing_dataset,
        #                                                     fixed_label=1),
        #                                     FixedLabelDataset(self.out_auditing_dataset,
        #                                                     fixed_label=0),])
    
    def get_original_dataset(self):
        return self.original_datasets

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
        self.logger.print_it(f"Getting auditing dataset with {labels} labels...")
        if labels == 'mia':
            return MyConcatDataset([FixedLabelDataset(self.in_auditing_dataset,
                                                    fixed_label=1),
                                    FixedLabelDataset(self.out_auditing_dataset,
                                                    fixed_label=0),])
        elif labels == 'original':
            return MyConcatDataset([self.in_auditing_dataset, self.out_auditing_dataset])
        else:
            raise ValueError('Labels mode should be either mia or original!')

