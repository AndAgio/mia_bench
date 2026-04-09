import warnings
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from typing import List, Sequence, Any
import numpy as np
from collections import defaultdict


def _unpack_item(item):
    """Safely unpacks standard (x, y), unsupervised (x,), or already tracked tuples."""
    if isinstance(item, (tuple, list)):
        if len(item) == 1:
            return item[0], None
        elif len(item) >= 2:
            return item[0], item[1]
    return item, None

class IndexTrackingMixin:
    def get_indices(self, mode: str = 'original') -> List[int]:
        if mode == 'current':
            return self.get_current_current_indices()
        elif mode == 'original':
            return self.get_current_original_indices()
        else:
            raise ValueError("Invalid mode. Use 'current' or 'original'.")

    def get_current_original_indices(self) -> List[int]:
        """
        1. Returns the original/global indices of the samples currently 
        present in this specific dataset layer.
        """
        return list(self.original_indices)
    
    def get_current_current_indices(self) -> List[int]:
        """
        1. Returns the current local indices of the samples in this dataset layer.
        This is just a range from 0 to len(self) - 1, but it's useful for clarity and consistency.
        """
        return list(range(len(self)))

    def get_root_original_indices(self) -> List[int]:
        """
        2. Traverses down the wrapper chain to the very first tracked dataset 
        (e.g., the MergedDataset) and returns the full list of all originally available indices.
        """
        current = self
        # Keep digging into the wrapped datasets as long as they also track indices
        while hasattr(current, 'dataset') and hasattr(current.dataset, 'original_indices'):
            current = current.dataset
            
        return list(current.original_indices)
    

# Notice we inherit from Dataset AND IndexTrackingMixin
class IndexedDataset(Dataset, IndexTrackingMixin):
    def __init__(self, dataset: Dataset, original_indices: List[int] = None):
        self.dataset = dataset
        if original_indices is not None:
            self.original_indices = original_indices
        elif hasattr(dataset, 'original_indices'):
            self.original_indices = dataset.original_indices
        else:
            self.original_indices = list(range(len(dataset)))
        self._orig_to_local = {orig: local for local, orig in enumerate(self.original_indices)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            x, y, orig_idx = item[0], item[1], item[2]
        else:
            x, y = _unpack_item(item)
            orig_idx = self.original_indices[idx]
        return x, y, orig_idx, idx


class MergedDataset(Dataset, IndexTrackingMixin):
    def __init__(self, *datasets: Dataset):
        if not datasets:
            raise ValueError("At least one dataset must be provided to MergedDataset.")

        # 1. Check for uniformity: all or nothing
        has_indices = [hasattr(ds, 'original_indices') for ds in datasets]
        
        if any(has_indices) and not all(has_indices):
            raise ValueError(
                "Mixed datasets detected! You must provide either ALL datasets with "
                "'original_indices' tracked, or ALL raw datasets. Mixing is not allowed."
            )

        tracked_datasets = []

        # 2. Scenario A: NONE of the datasets have tracking (Raw Datasets)
        if not any(has_indices):
            current_offset = 0
            for ds in datasets:
                new_indices = list(range(current_offset, current_offset + len(ds)))
                tracked_ds = IndexedDataset(ds, original_indices=new_indices)
                tracked_datasets.append(tracked_ds)
                current_offset += len(ds)

        # 3. Scenario B: ALL of the datasets have tracking
        else:
            tracked_datasets = list(datasets)

        # Build the underlying concatenation (contains duplicates)
        self.concat_dataset = ConcatDataset(tracked_datasets)
        
        # 4. Deduplicate Original Indices and map valid ConcatDataset targets
        self.original_indices = []
        self.kept_concat_indices = []
        seen_indices = set()
        duplicates_found = 0
        
        current_concat_idx = 0
        for ds in tracked_datasets:
            for orig_idx in ds.original_indices:
                if orig_idx not in seen_indices:
                    # First time seeing this sample! Keep it.
                    seen_indices.add(orig_idx)
                    self.original_indices.append(orig_idx)
                    self.kept_concat_indices.append(current_concat_idx)
                else:
                    # It's a duplicate! We skip appending it to our kept lists.
                    duplicates_found += 1
                
                # We must advance the concat index regardless, so it stays in sync 
                # with the underlying ConcatDataset
                current_concat_idx += 1
                
        # 5. Issue Warning if necessary
        if duplicates_found > 0:
            warnings.warn(
                f"\nMergedDataset Warning: {duplicates_found} overlapping sample(s) "
                f"detected and deduplicated.\nOnly the first occurrence of each "
                f"overlapping original index was kept."
            )

        # 6. Create the lookup dictionary mapping global -> local merged index
        self._orig_to_local = {orig: local for local, orig in enumerate(self.original_indices)}

    def __len__(self) -> int:
        # The true length is now the number of unique samples
        return len(self.kept_concat_indices)

    def __getitem__(self, idx: int):
        # Route the local idx through our filter array to get the correct item
        # from the underlying ConcatDataset
        concat_idx = self.kept_concat_indices[idx]
        item = self.concat_dataset[concat_idx]
        
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            x, y, orig_idx = item[0], item[1], item[2]
        else:
            x, y = _unpack_item(item)
            orig_idx = self.original_indices[idx]
            
        return x, y, orig_idx, idx


class SubsampledDataset(Dataset, IndexTrackingMixin):
    def __init__(self, dataset: Dataset, original_indices: Sequence[int], strict: bool = True):
        if not hasattr(dataset, '_orig_to_local'):
            dataset = IndexedDataset(dataset)
        self.dataset = dataset
        self.original_indices = [int(i) for i in original_indices]
        self.local_to_parent_local = []
        
        missing = []
        for orig_idx in self.original_indices:
            if orig_idx in self.dataset._orig_to_local:
                self.local_to_parent_local.append(self.dataset._orig_to_local[orig_idx])
            else:
                missing.append(orig_idx)
                
        if strict and missing:
            raise ValueError(f"Missing {len(missing)} original indices. First 10: {missing[:10]}")
        if not strict:
            self.original_indices = [i for i in self.original_indices if i not in missing]

        self._orig_to_local = {orig: local for local, orig in enumerate(self.original_indices)}

    def __len__(self) -> int:
        return len(self.local_to_parent_local)

    def __getitem__(self, idx: int):
        # We only take the first 3 items from the parent, dropping the parent's `idx`.
        # Then we append our own `idx`.
        x, y, orig_idx, _ = self.dataset[self.local_to_parent_local[idx]]
        return x, y, orig_idx, idx


class ConstantLabelDataset(Dataset, IndexTrackingMixin):
    def __init__(self, dataset: Dataset, constant_label: Any):
        if not hasattr(dataset, 'original_indices'):
            dataset = IndexedDataset(dataset)
        self.dataset = dataset
        self.constant_label = constant_label
        self.original_indices = self.dataset.original_indices
        self._orig_to_local = self.dataset._orig_to_local

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            x, _, orig_idx = item[0], item[1], item[2]
        else:
            x, _ = _unpack_item(item)
            orig_idx = self.original_indices[idx]
        return x, self.constant_label, orig_idx, idx
    

class AugmentWrappedDataset(Dataset, IndexTrackingMixin):
    """Wraps a dataset to apply additional transformations."""
    def __init__(self, base_dataset: Dataset, extra_transform: transforms.Compose = None):
        self.base_dataset = base_dataset
        self.extra_transform = extra_transform
        assert hasattr(base_dataset, 'original_indices'), "Base dataset must have 'original_indices' for tracking original indices in this project!"
        self.original_indices = self.base_dataset.original_indices
        assert hasattr(base_dataset, '_orig_to_local'), "Base dataset must have '_orig_to_local' mapping for tracking original indices in this project!"
        self._orig_to_local = self.base_dataset._orig_to_local

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            x, y, orig_idx = item[0], item[1], item[2]
        else:
            x, y = _unpack_item(item)
            orig_idx = self.original_indices[idx]        
        # 2. Apply the delayed transformations
        if self.extra_transform:
            x = self.extra_transform(x)
            
        return x, y, orig_idx, idx


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

    def update(self, dataset: Dataset, id: str):
        if id in self.datasets.keys():
            self.datasets[id] = dataset
        else:
            raise KeyError(f"ID '{id}' not in datasets managed by {self}! Use add() to add new datasets.")
    
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


class DatasetSplitter:
    @staticmethod
    def _proportions_to_counts(total_len: int, proportions: Sequence[float]) -> List[int]:
        """Converts proportions (e.g., [0.2, 0.8]) to absolute counts."""
        proportions = np.array(proportions) / np.sum(proportions)
        counts = np.floor(proportions * total_len).astype(int)
        counts[-1] = total_len - counts[:-1].sum() # Add remainder to last chunk
        return counts.tolist()

    @staticmethod
    def split_stratified(dataset, proportions: Sequence[float], seed: int = 42):
        """
        Splits into chunks while maintaining class balance.
        Automatically extracts labels from the dataset's (x, y, orig_idx, curr_idx) tuples.
        """
        np.random.seed(seed)
        
        # 1. Automatically extract labels from the dataset
        label_to_indices = defaultdict(list)
        for local_idx in range(len(dataset)):
            # Unpack the 4-entry tuple
            _, y, _, _ = dataset[local_idx]
            
            # If the label is a tensor (e.g., tensor(1)), convert to a python int/float
            if hasattr(y, 'item'):
                y = y.item()
                
            label_to_indices[y].append(local_idx)
            
        chunk_local_indices = [[] for _ in proportions]
        
        # 2. Distribute indices of each class across the chunks
        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)
            counts = DatasetSplitter._proportions_to_counts(len(indices), proportions)
            
            start = 0
            for i, count in enumerate(counts):
                chunk_local_indices[i].extend(indices[start : start + count])
                start += count
                
        # 3. Shuffle within the chunks and map to original indices
        chunks = []
        for local_idxs in chunk_local_indices:
            np.random.shuffle(local_idxs)
            orig_idxs = [dataset.original_indices[idx] for idx in local_idxs]
            chunks.append(SubsampledDataset(dataset, orig_idxs))
            
        return chunks

    @staticmethod
    def split_by_metric(dataset, metrics: Sequence[float], proportions: Sequence[float], strategy: str = 'contiguous', seed: int = 42):
        """
        Splits dataset based on a metric (e.g., memorization scores).
        strategy="contiguous": Chunk 1 gets top scores, chunk 2 gets next, etc.
        strategy="balanced": All chunks get a similar distribution/histogram of scores.
        """
        np.random.seed(seed)
        metrics = np.array(metrics)
        sorted_local_indices = np.argsort(metrics)[::-1] # Descending order (High mem first)
        
        chunk_local_indices = [[] for _ in proportions]
        counts = DatasetSplitter._proportions_to_counts(len(dataset), proportions)
        
        if strategy == 'contiguous':
            # E.g., Chunk 1: Highest mem, Chunk 2: Medium mem, Chunk 3: Lowest mem
            start = 0
            for i, count in enumerate(counts):
                chunk_local_indices[i] = sorted_local_indices[start : start + count].tolist()
                start += count
                
        elif strategy == 'balanced':
            # Group into small buckets based on score, and distribute evenly among chunks
            # so every chunk gets a slice of high, medium, and low memorization scores.
            start = 0
            # We bucket the sorted indices. Number of buckets = total items / sum of smallest common denominator, 
            # but a simpler way is to split the sorted array into many tiny stratified segments.
            n_chunks = len(proportions)
            for start_idx in range(0, len(sorted_local_indices), n_chunks):
                bucket = sorted_local_indices[start_idx : start_idx + n_chunks]
                np.random.shuffle(bucket) # Shuffle within the bucket to avoid strict deterministic patterns
                for i, idx in enumerate(bucket):
                    # Assign to chunk i if it hasn't exceeded its count
                    # Fallback assignment logic to respect proportions
                    target_chunk = i % n_chunks
                    while len(chunk_local_indices[target_chunk]) >= counts[target_chunk]:
                        target_chunk = (target_chunk + 1) % n_chunks
                    chunk_local_indices[target_chunk].append(idx)
                    
        else:
            raise ValueError("strategy must be 'contiguous' or 'balanced'")

        # Map to original indices and create datasets
        chunks = []
        for local_idxs in chunk_local_indices:
            np.random.shuffle(local_idxs) # Shuffle so subsequent training isn't monotonically sorted
            orig_idxs = [dataset.original_indices[idx] for idx in local_idxs]
            chunks.append(SubsampledDataset(dataset, orig_idxs))
            
        return chunks