from typing import Union
import torch
from torch.utils.data import Dataset, Subset


class IndexedDataset:
    def __init__(self, ds):
        if isinstance(ds, IndexedDataset):
            self.ds = ds.ds
            self.orig_indices = ds.orig_indices.copy()
            self.base_dataset = ds.base_dataset
            return

        self.ds = ds
        self.base_dataset = self._resolve_base(ds)
        self.orig_indices = self._resolve_orig_indices(ds)

    def _resolve_base(self, ds):
        # unwrap Subset chains to ultimate base dataset
        base = ds
        while isinstance(base, Subset):
            base = base.dataset
        return base

    def _resolve_orig_indices(self, ds):
        # start with positions in ds
        indices = list(range(len(ds)))
        base = ds
        while True:
            if isinstance(base, Subset):
                indices = [base.indices[i] for i in indices]
                base = base.dataset
                continue
            if isinstance(base, IndexedDataset):
                indices = [base.orig_indices[i] for i in indices]
            break
        return indices

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        orig_idx = self.orig_indices[idx]
        new_idx = idx
        return x, y, orig_idx, new_idx

    def get_all_targets(self, to_torch: bool = False):
        if isinstance(self.ds, Subset):
            targets = torch.tensor(self.ds.dataset.targets)[self.ds.indices]
            return targets if to_torch else targets.tolist()
        elif isinstance(self.ds, Dataset):
            return self.ds.targets if not to_torch else torch.tensor(self.ds.targets)
        else:
            raise ValueError(f"Found dataset of type {type(self.ds)} inside IndexedDataset. It is not supported for getting all targets!")
        
    def get_original_indices(self, to_torch: bool = False):
        return self.orig_indices if not to_torch else torch.tensor(self.orig_indices)
    
    def get_indices(self, to_torch: bool = False):
        indices = list(range(len(self.ds)))
        if to_torch:
            indices = torch.tensor(indices)
        return indices
    
    def get_indices_mapping(self):
        return {i: self.orig_indices[i] for i in range(len(self.ds))}
    
    def set_targets(self, new_targets: Union[torch.Tensor, list]):
        if isinstance(self.ds, Subset):
            if isinstance(new_targets, list):
                assert len(new_targets) == len(self.ds), f"Length of new_targets list ({len(new_targets)}) does not match length of Subset dataset ({len(self.ds)})!"
                assert isinstance(new_targets[0], (torch.Tensor, list, int)), f"Unsupported type for new_targets elements: {type(new_targets[0])}!"
                single_target_shape = list(new_targets[0].shape) if isinstance(new_targets[0], torch.Tensor) else len(new_targets[0]) if isinstance(new_targets[0], list) else 1
                new_targets = torch.tensor(new_targets)
            if isinstance(new_targets, torch.Tensor):
                assert new_targets.shape[0] == len(self.ds), f"Length of new_targets tensor ({new_targets.shape[0]}) does not match length of Subset dataset ({len(self.ds)})!"
                single_target_shape = list(new_targets.shape[1:]) if len(new_targets.shape) > 1 else 1
            new_targets_to_set_shape = [len(self.ds.dataset)] + single_target_shape if isinstance(single_target_shape, list) else (len(self.ds.dataset), single_target_shape)
            new_targets_to_set = - torch.ones(new_targets_to_set_shape)
            for i in range(len(self.ds.dataset)):
                if i in self.ds.indices:
                    j = self.ds.indices.index(i)
                    new_targets_to_set[i, :] = new_targets[j, :]
            self.ds.dataset.targets = new_targets_to_set
        elif isinstance(self.ds, Dataset):
            self.ds.targets = new_targets if isinstance(new_targets, torch.Tensor) else torch.tensor(new_targets)
        else:
            raise ValueError(f"Found dataset of type {type(self.ds)} inside IndexedDataset. It is not supported for getting all targets!")


class MultiDatasets():
    def __init__(self, datasets: list[Dataset] = None, ids: list[str] = None, info: dict = None):
        if datasets is not None and ids is not None:
            assert len(datasets) == len(ids)
            self.datasets = {ids[i]: datasets[i] for i in range(len(datasets))}
        elif datasets is None and ids is None:
            self.datasets = {}
        if info is not None:
            self.info = info
        
    def add(self, dataset: Dataset, id: str):
        self.datasets[id] = dataset
    
    def remove(self, id: str):
        self.datasets.pop(id)
    
    def get(self, id: str):
        if id in self.datasets.keys():
            return self.datasets[id]
        else:
            raise KeyError(f"ID '{id}' not in datasets managed by {self}!")
        
    def n_splits(self):
        return len(self.datasets.keys())
    
    def get_ids(self):
        return list(self.datasets.keys())
    
    def add_info(self, info: dict):
        self.info = info

    def get_info(self):
        return self.info
    
    def wrap(self, wrapper_class, **kwargs):
        for id in self.datasets.keys():
            self.datasets[id] = wrapper_class(self.datasets[id], **kwargs)


class FixedLabelDataset(Dataset):
    """Wrap a dataset and replace all labels with a single fixed label.

    This wrapper *always* returns a tuple (x, fixed_label, orig_idx, new_idx) where:
    - x: the sample input
    - fixed_label: the provided fixed label
    - orig_idx: original index in the base dataset (resolved through Subset/IndexedDataset chains)
    - new_idx: the position inside this wrapped dataset (i.e. the `index` argument)

    The wrapper is robust to underlying datasets that return (x,y), (x,y,orig_idx,...),
    or just x. Original index resolution is delegated to `IndexedDataset` for consistency
    across nested dataset wrappers.
    """
    def __init__(self, dataset: Dataset, fixed_label: int = 0):
        super().__init__()
        self.data = dataset
        self.fixed_label = fixed_label
        # leverage IndexedDataset to resolve base-dataset indices once
        self._indexed_helper = IndexedDataset(dataset)
        self._orig_indices = list(self._indexed_helper.orig_indices)

    def __getitem__(self, index):
        item = self.data[index]
        # Extract input x
        if isinstance(item, (tuple, list)):
            x = item[0]
        else:
            x = item

        # Resolve original index via IndexedDataset bookkeeping
        orig_idx = self._orig_indices[index] if index < len(self._orig_indices) else index

        # new_idx is simply the position inside this dataset
        new_idx = index

        return x, self.fixed_label, orig_idx, new_idx

    def __len__(self):
        return len(self.data)

    def get_all_original_indices(self):
        """Public accessor for the full list of originals (as resolved by IndexedDataset)."""
        return list(self._orig_indices)
    
    @property
    def transform(self):
        if isinstance(self.data, Subset):
            return self.data.dataset.transform
        elif isinstance(self.data, Dataset):
            return self.data.transform
        else:
            raise ValueError(f"Found dataset of type {type(self.data)} inside FixedLabelDataset. It is not supported for getting transform!")



class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        batch_x, batch_y, original_id, _ = self.datasets[dataset_idx][sample_idx]
        return batch_x, batch_y, original_id, idx

    @property
    def transform(self):
        transforms = []
        for dataset in self.datasets:
            if isinstance(dataset, Subset):
                transforms.append(dataset.dataset.transform)
            elif isinstance(dataset, Dataset):
                transforms.append(dataset.transform)
            else:
                try:
                    transforms.append(dataset.transform)
                except AttributeError:
                    raise ValueError(f"Found dataset of type {type(dataset)} inside MyConcatDataset. It is not supported for getting transform!")
        if not all(are_transforms_equal(transforms[0], t) for t in transforms):
            print(f"WARNING: not all datasets in MyConcatDataset have the same transform! Found transforms: {transforms}. Returning the first one by default.")
        return transforms[0]


def are_transforms_equal(t1: transforms, t2: transforms):
    # Check if they have the same number of steps
    if len(t1.transforms) != len(t2.transforms):
        return False
    
    # Iterate through and compare each step
    for step1, step2 in zip(t1.transforms, t2.transforms):
        # Check if they are the exact same type of transform
        if type(step1) != type(step2):
            return False
        # Check if their internal parameters match
        if step1.__dict__ != step2.__dict__:
            return False
            
    return True