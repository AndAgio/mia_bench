from typing import Union, Optional, Sequence
import bisect
import torch
import numpy as np
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision.transforms as transforms

from typing import TypeVar, List
_T_co = TypeVar("_T_co", covariant=True)

class MySubset(Dataset[_T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        self.base_dataset = self._resolve_base(dataset)
        self.orig_indices = self._resolve_orig_indices(dataset)
        self.dataset = dataset
        self.indices = indices

    def _resolve_base(self, ds):
        # unwrap Subset chains to ultimate base dataset
        base = ds
        while isinstance(base, Subset) or isinstance(base, MySubset):
            base = base.dataset
        return base

    def _resolve_orig_indices(self, ds):
        # start with positions in ds
        indices = list(range(len(ds)))
        base = ds
        while True:
            # print(f"Resolving indices for dataset of type {type(base)} with {len(indices)} samples. indices = {indices}")
            if isinstance(base, Subset) or isinstance(base, MySubset):
                # print(f"Reached Subset base with orig_indices = {base.indices}")
                indices = [base.indices[i] for i in indices]
                base = base.dataset
                continue
            if isinstance(base, IndexedDataset):
                # print(f"Reached IndexedDataset base with orig_indices = {base.orig_indices}")
                indices = [base.orig_indices[i] for i in indices]
            break
        return indices
    
    def get_all_original_indices(self, to_torch: bool = False):
        return self.orig_indices if not to_torch else torch.tensor(self.orig_indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]) -> List[_T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)


class MyOriginalIndexSubset(Dataset[_T_co]):
    r"""Subset a dataset using *original* indices (global IDs), not local positions.

    Example:
    - Start from dataset with 10000 samples (original IDs 0..9999)
    - Subsample 5000 samples using `MySubset`
    - Subsample again by original IDs (e.g. 2000 IDs from 0..9999)

    This class maps requested original IDs to local positions in the provided
    (possibly already-subsampled) dataset.
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(
        self,
        dataset: Dataset[_T_co],
        original_indices: Sequence[int],
        strict: bool = True,
        return_indexed_tuple: bool = False,
    ) -> None:
        self.dataset = dataset
        self.return_indexed_tuple = return_indexed_tuple
        self.base_dataset = self._resolve_base(dataset)
        # original IDs currently available in `dataset` order
        self.orig_indices = self._resolve_orig_indices(dataset)

        # map original_id -> local position(s) inside provided dataset
        inv_map = {}
        for local_idx, orig_idx in enumerate(self.orig_indices):
            inv_map.setdefault(int(orig_idx), []).append(local_idx)

        local_indices = []
        missing = []
        for orig_idx in original_indices:
            orig_idx = int(orig_idx)
            if orig_idx not in inv_map:
                missing.append(orig_idx)
                continue
            # datasets should normally have unique original IDs; if repeated,
            # keep first occurrence deterministically.
            local_indices.append(inv_map[orig_idx][0])

        if strict and len(missing) > 0:
            missing_preview = missing[:20]
            raise ValueError(
                f"Requested {len(original_indices)} original indices, but {len(missing)} are missing "
                f"in the provided dataset. Missing sample (first up to 20): {missing_preview}"
            )

        self.requested_original_indices = [int(i) for i in original_indices]
        self.missing_original_indices = missing
        self.indices = local_indices
        self.selected_original_indices = [self.orig_indices[i] for i in self.indices]

    def _resolve_base(self, ds):
        base = ds
        while isinstance(base, Subset) or isinstance(base, MySubset) or isinstance(base, MyOriginalIndexSubset):
            base = base.dataset
        return base

    def _resolve_orig_indices(self, ds):
        indices = list(range(len(ds)))
        base = ds
        while True:
            if isinstance(base, Subset) or isinstance(base, MySubset) or isinstance(base, MyOriginalIndexSubset):
                indices = [base.indices[i] for i in indices]
                base = base.dataset
                continue
            if isinstance(base, IndexedDataset):
                indices = [base.orig_indices[i] for i in indices]
            break
        return indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        item = self.dataset[self.indices[idx]]
        if not self.return_indexed_tuple:
            return item

        if isinstance(item, (tuple, list)):
            if len(item) >= 2:
                x, y = item[0], item[1]
            elif len(item) == 1:
                x, y = item[0], None
            else:
                raise ValueError("Found empty tuple/list item in MyOriginalIndexSubset!")
        else:
            x, y = item, None

        orig_idx = self.selected_original_indices[idx]
        new_idx = idx
        return x, y, orig_idx, new_idx

    def __getitems__(self, indices: List[int]) -> List[_T_co]:
        return [self.__getitem__(idx) for idx in indices]

    def __len__(self):
        return len(self.indices)

    def get_all_original_indices(self, to_torch: bool = False):
        selected = self.selected_original_indices
        return selected if not to_torch else torch.tensor(selected)

    def get_missing_original_indices(self):
        return list(self.missing_original_indices)

    @property
    def transform(self):
        if isinstance(self.dataset, Subset) or isinstance(self.dataset, MySubset) or isinstance(self.dataset, MyOriginalIndexSubset):
            return self.dataset.dataset.transform
        elif isinstance(self.dataset, Dataset):
            return self.dataset.transform
        else:
            raise ValueError(f"Found dataset of type {type(self.dataset)} inside MyOriginalIndexSubset. It is not supported for getting transform!")

    @classmethod
    def sample_n_available_from_candidates(
        cls,
        dataset: Dataset[_T_co],
        candidate_original_indices: Sequence[int],
        n_samples: int,
        seed: int = 12345,
        replace: bool = False,
        strict_if_not_enough: bool = True,
        return_indexed_tuple: bool = False,
    ):
        """Sample exactly N available original IDs from a candidate list.

        Returns:
            subset: `MyOriginalIndexSubset` built from sampled original IDs
            sampled_original_ids: list[int]
            unavailable_candidate_ids: list[int]
        """
        if n_samples <= 0:
            raise ValueError(f"`n_samples` must be > 0, found {n_samples}")

        # Build a temporary view to resolve original IDs available in `dataset`.
        tmp = cls(dataset=dataset,
            original_indices=[],
            strict=False,
            return_indexed_tuple=return_indexed_tuple)
        available_set = set(int(i) for i in tmp.orig_indices)

        candidates = [int(i) for i in candidate_original_indices]
        available_candidates = [i for i in candidates if i in available_set]
        unavailable_candidates = [i for i in candidates if i not in available_set]

        if len(available_candidates) == 0:
            raise ValueError("No candidate original indices are available in the provided dataset.")

        if not replace and n_samples > len(available_candidates):
            if strict_if_not_enough:
                raise ValueError(
                    f"Cannot sample exactly n_samples={n_samples} without replacement: only "
                    f"{len(available_candidates)} candidates are available in the provided dataset."
                )
            n_to_sample = len(available_candidates)
        else:
            n_to_sample = n_samples

        rng = np.random.default_rng(seed=seed)
        sampled_original_ids = rng.choice(
            np.array(available_candidates, dtype=np.int64),
            size=n_to_sample,
            replace=replace,
        ).tolist()

        subset = cls(
            dataset=dataset,
            original_indices=sampled_original_ids,
            strict=True,
            return_indexed_tuple=return_indexed_tuple,
        )
        return subset, sampled_original_ids, unavailable_candidates
    

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
        while isinstance(base, Subset) or isinstance(base, MySubset) or isinstance(base, MyOriginalIndexSubset):
            base = base.dataset
        return base

    def _resolve_orig_indices(self, ds):
        # start with positions in ds
        indices = list(range(len(ds)))
        base = ds
        while True:
            # print(f"Resolving indices for dataset of type {type(base)} with {len(indices)} samples. indices = {indices}")
            if isinstance(base, Subset) or isinstance(base, MySubset) or isinstance(base, MyOriginalIndexSubset):
                # print(f"Reached Subset base with orig_indices = {base.indices}")
                indices = [base.indices[i] for i in indices]
                base = base.dataset
                continue
            if isinstance(base, IndexedDataset):
                # print(f"Reached IndexedDataset base with orig_indices = {base.orig_indices}")
                indices = [base.orig_indices[i] for i in indices]
            break
        return indices

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # print(f"IndexedDataset: fetching item with idx={idx} from dataset of type {type(self.ds)}")
        # print(f"self.ds[idx]: {self.ds[idx]}")
        item = self.ds[idx]
        x = item[0]
        y = item[1]
        # x, y = self.ds[idx]
        orig_idx = self.orig_indices[idx]
        new_idx = idx
        # if len(item) >= 3:
        #     print(f"orig_idx: {orig_idx}, item[2]: {item[2]}")
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
    
    def get_all_original_indices(self, to_torch: bool = False):
        return self.orig_indices if not to_torch else torch.tensor(self.orig_indices)
    
    def get_indices(self, to_torch: bool = False):
        indices = list(range(len(self.ds)))
        if to_torch:
            indices = torch.tensor(indices)
        return indices
    
    def get_indices_mapping(self):
        return {i: self.orig_indices[i] for i in range(len(self.ds))}
    
    def set_targets(self, new_targets: Union[torch.Tensor, list]):
        if isinstance(self.ds, Subset) or isinstance(self.ds, MySubset) or isinstance(self.ds, MyOriginalIndexSubset):
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
        
    @property
    def transform(self):
        if isinstance(self.ds, Subset) or isinstance(self.ds, MySubset) or isinstance(self.ds, MyOriginalIndexSubset):
            return self.ds.dataset.transform
        elif isinstance(self.ds, Dataset):
            return self.ds.transform
        else:
            raise ValueError(f"Found dataset of type {type(self.ds)} inside IndexedDataset. It is not supported for getting transform!")

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
        if isinstance(self.data, Subset) or isinstance(self.data, MySubset) or isinstance(self.data, MyOriginalIndexSubset):
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
            if isinstance(dataset, Subset) or isinstance(dataset, MySubset) or isinstance(dataset, MyOriginalIndexSubset):
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
    
    def get_all_original_indices(self, to_torch: bool = False):
        all_orig_indices = []
        for dataset in self.datasets:
            original_indices_i = [self.__getitem__(idx)[2] for idx in range(len(dataset))]
            all_orig_indices.extend(original_indices_i)
        return all_orig_indices if not to_torch else torch.tensor(all_orig_indices)


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


class MergedIndexedDataset(Dataset):
    """Merge multiple datasets and expose contiguous merged indices.

    Samples are returned as ``(x, y, merged_idx)`` where ``merged_idx`` is in
    ``[0, len(self)-1]`` and is stable across all chunks obtained from this
    dataset via ``torch.utils.data.Subset``.
    """
    def __init__(self,
                datasets: Sequence[Dataset],
                dataset_names: Optional[Sequence[str]] = None):
        super().__init__()
        if datasets is None or len(datasets) == 0:
            raise ValueError("`datasets` must contain at least one dataset.")
        self.datasets = list(datasets)
        if dataset_names is None:
            self.dataset_names = [f"set_{i}" for i in range(len(self.datasets))]
        else:
            if len(dataset_names) != len(self.datasets):
                raise ValueError("`dataset_names` length must match `datasets` length.")
            self.dataset_names = list(dataset_names)
        self.dataset = ConcatDataset(self.datasets)
        self._lengths = [len(ds) for ds in self.datasets]
        self._offsets = np.cumsum([0] + self._lengths).tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)):
            if len(item) >= 2:
                x, y = item[0], item[1]
            elif len(item) == 1:
                x, y = item[0], None
            else:
                raise ValueError("Found empty tuple/list item in merged dataset!")
        else:
            x, y = item, None
        return x, y, idx

    def get_split_and_local_index(self, merged_idx: int):
        if not 0 <= merged_idx < len(self):
            raise IndexError(f"merged_idx={merged_idx} out of bounds for length {len(self)}")
        set_idx = bisect.bisect_right(self._offsets, merged_idx) - 1
        local_idx = merged_idx - self._offsets[set_idx]
        return self.dataset_names[set_idx], local_idx

    def get_merged_index(self, dataset_name: str, local_idx: int):
        if dataset_name not in self.dataset_names:
            raise KeyError(f"Unknown dataset_name={dataset_name}. Available: {self.dataset_names}")
        set_idx = self.dataset_names.index(dataset_name)
        if not 0 <= local_idx < self._lengths[set_idx]:
            raise IndexError(
                f"local_idx={local_idx} out of bounds for dataset '{dataset_name}' with length {self._lengths[set_idx]}"
            )
        return self._offsets[set_idx] + local_idx

    def get_merged_to_source_mapping(self):
        mapping = {}
        for i in range(len(self)):
            mapping[i] = self.get_split_and_local_index(i)
        return mapping


class SampleMetricTracker:
    """External tracker for per-merged-index metrics.

    This class is intentionally decoupled from dataset classes.
    """
    def __init__(self, n_samples: int):
        if n_samples <= 0:
            raise ValueError("`n_samples` must be > 0")
        self.n_samples = int(n_samples)
        self._metrics = {}

    def set_metrics(self, metric_values: Union[dict, list, np.ndarray, torch.Tensor]):
        if isinstance(metric_values, dict):
            missing = [i for i in range(self.n_samples) if i not in metric_values]
            if missing:
                raise ValueError(f"Metric dict is missing {len(missing)} indices.")
            self._metrics = {int(k): float(v) for k, v in metric_values.items()}
            return

        if isinstance(metric_values, torch.Tensor):
            metric_values = metric_values.detach().cpu().numpy()
        values = np.asarray(metric_values)
        if values.shape[0] != self.n_samples:
            raise ValueError(f"Expected {self.n_samples} metric values, found {values.shape[0]}.")
        self._metrics = {i: float(values[i]) for i in range(self.n_samples)}

    def set_metric(self, merged_idx: int, value: float):
        if not 0 <= merged_idx < self.n_samples:
            raise IndexError(f"merged_idx={merged_idx} out of bounds for length {self.n_samples}")
        self._metrics[int(merged_idx)] = float(value)

    def get_metric(self, merged_idx: int):
        if merged_idx not in self._metrics:
            raise KeyError(f"No metric found for merged_idx={merged_idx}")
        return self._metrics[int(merged_idx)]

    def get_metric_from_item(self, item, index_position: int = 2):
        """Get metric value directly from a dataset item.

        Expected item formats are typically:
        - `(x, y, merged_idx)` from `MergedIndexedDataset`
        - `(x, y, merged_idx, new_idx)` from `IndexedDataset` / `MyConcatDataset`

        By default, `index_position=2` assumes the third element stores the
        stable merged index.
        """
        merged_idx = get_stable_index_from_item(item=item, index_position=index_position)
        return self.get_metric(merged_idx)

    def get_metric_map(self):
        return self._metrics.copy()

    def as_array(self, fill_value: float = 0.0):
        arr = np.full(self.n_samples, fill_value, dtype=np.float64)
        for idx, val in self._metrics.items():
            arr[int(idx)] = float(val)
        return arr

    def to_sampling_weights(self,
                            mode: str = 'linear',
                            epsilon: float = 1e-12,
                            temperature: float = 1.0,
                            fill_value: float = 0.0):
        """Convert tracked metric values to non-negative sampling weights.

        Supported modes:
        - `linear`: shift to be >= 0
        - `softmax`: $w_i = \exp(m_i / T)$
        - `rank`: larger metric -> larger rank weight
        """
        values = self.as_array(fill_value=fill_value)

        if mode == 'linear':
            weights = values - np.min(values)
            weights = weights + epsilon
        elif mode == 'softmax':
            if temperature <= 0:
                raise ValueError("`temperature` must be > 0 for softmax mode.")
            z = values / float(temperature)
            z = z - np.max(z)
            weights = np.exp(z) + epsilon
        elif mode == 'rank':
            order = np.argsort(values)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(values) + 1)
            weights = ranks.astype(np.float64) + epsilon
        else:
            raise ValueError("Unsupported mode. Use one of: ['linear', 'softmax', 'rank']")

        if np.all(weights <= 0):
            raise ValueError("Computed non-positive weights for all samples.")
        return weights


def _sizes_from_spec(n_samples: int,
                    chunk_sizes: Optional[Sequence[int]] = None,
                    chunk_fractions: Optional[Sequence[float]] = None,
                    chunk_percentages: Optional[Sequence[float]] = None,
                    keep_remainder_in_last: bool = True):
    specs = [chunk_sizes is not None, chunk_fractions is not None, chunk_percentages is not None]
    if sum(specs) != 1:
        raise ValueError("Provide exactly one among chunk_sizes, chunk_fractions or chunk_percentages.")

    if chunk_sizes is not None:
        sizes = [int(s) for s in chunk_sizes]
    elif chunk_fractions is not None:
        if any((f <= 0 or f > 1) for f in chunk_fractions):
            raise ValueError("All chunk fractions must be in (0, 1].")
        sizes = [int(np.floor(float(f) * n_samples)) for f in chunk_fractions]
    else:
        if any((p <= 0 or p > 100) for p in chunk_percentages):
            raise ValueError("All chunk percentages must be in (0, 100].")
        sizes = [int(np.floor((float(p) / 100.0) * n_samples)) for p in chunk_percentages]

    if any(s <= 0 for s in sizes):
        raise ValueError(f"All chunk sizes must be > 0. Found sizes={sizes}")

    total_requested = sum(sizes)
    if total_requested > n_samples:
        raise ValueError(f"Requested {total_requested} samples but dataset has only {n_samples} samples.")

    if total_requested < n_samples and keep_remainder_in_last:
        sizes[-1] += (n_samples - total_requested)

    return sizes


def split_merged_dataset_into_chunks(merged_dataset: Dataset,
                                    chunk_sizes: Optional[Sequence[int]] = None,
                                    chunk_fractions: Optional[Sequence[float]] = None,
                                    chunk_percentages: Optional[Sequence[float]] = None,
                                    sampling_weights: Optional[Union[dict, list, np.ndarray, torch.Tensor]] = None,
                                    replace: bool = False,
                                    seed: int = 12345,
                                    shuffle_without_weights: bool = True,
                                    keep_remainder_in_last: bool = True,
                                    return_chunk_indices: bool = True):
    """Split a merged dataset into chunks with variable sizes.

    Chunk size can be specified by absolute sizes, fractions, or percentages.
    Optionally, weighted sampling can be used so chunks are sampled with
    probability depending on a per-merged-index score/metric.

    Returns:
    - chunks: list[Subset]
    - indices_per_chunk (optional): list[list[int]] with merged indices
    """
    n_samples = len(merged_dataset)
    sizes = _sizes_from_spec(n_samples=n_samples,
                            chunk_sizes=chunk_sizes,
                            chunk_fractions=chunk_fractions,
                            chunk_percentages=chunk_percentages,
                            keep_remainder_in_last=keep_remainder_in_last)

    rng = np.random.default_rng(seed=seed)

    if sampling_weights is None:
        base_indices = np.arange(n_samples)
        if shuffle_without_weights:
            rng.shuffle(base_indices)
        ordered_indices = base_indices.tolist()
    else:
        if isinstance(sampling_weights, dict):
            weights = np.array([sampling_weights.get(i, 0.0) for i in range(n_samples)], dtype=np.float64)
        elif isinstance(sampling_weights, torch.Tensor):
            weights = sampling_weights.detach().cpu().numpy().astype(np.float64)
        else:
            weights = np.asarray(sampling_weights, dtype=np.float64)

        if weights.shape[0] != n_samples:
            raise ValueError(f"Expected {n_samples} weights, found {weights.shape[0]} instead!")
        if np.any(weights < 0):
            raise ValueError("Sampling weights must be >= 0.")
        if np.all(weights == 0):
            raise ValueError("All sampling weights are 0. Cannot sample.")

        if replace:
            probs = weights / weights.sum()
            ordered_indices = rng.choice(np.arange(n_samples), size=sum(sizes), replace=True, p=probs).tolist()
        else:
            remaining_indices = np.arange(n_samples)
            remaining_weights = weights.copy()
            ordered_indices = []
            for _ in range(sum(sizes)):
                probs = remaining_weights / remaining_weights.sum()
                pos = int(rng.choice(np.arange(len(remaining_indices)), size=1, replace=False, p=probs)[0])
                ordered_indices.append(int(remaining_indices[pos]))
                remaining_indices = np.delete(remaining_indices, pos)
                remaining_weights = np.delete(remaining_weights, pos)

    chunks = []
    indices_per_chunk = []
    cursor = 0
    for s in sizes:
        ids = ordered_indices[cursor:cursor+s]
        cursor += s
        chunks.append(MySubset(merged_dataset, ids))
        indices_per_chunk.append(ids)

    if return_chunk_indices:
        return chunks, indices_per_chunk
    return chunks


def build_merged_dataset(datasets: Union[Sequence[Dataset], Dataset],
                        dataset_names: Optional[Sequence[str]] = None,
                        test_dataset: Optional[Dataset] = None):
    """Helper to build a merged dataset from multiple datasets.

    Example:
    ``build_merged_dataset([train_ds, val_ds, test_ds], ['train', 'val', 'test'])``
    """
    # Backward compatibility:
    # build_merged_dataset(train_dataset, test_dataset=<...>)
    # build_merged_dataset(train_dataset, dataset_names=['train','test'], test_dataset=<...>)
    if isinstance(datasets, Dataset):
        if test_dataset is None:
            raise ValueError("When passing a single Dataset, `test_dataset` must be provided for backward compatibility.")
        datasets = [datasets, test_dataset]
        if dataset_names is None:
            dataset_names = ['train', 'test']
    return MergedIndexedDataset(datasets=datasets, dataset_names=dataset_names)


def get_stable_index_from_item(item, index_position: int = 2):
    """Extract the stable merged index from a returned dataset sample.

    This is useful after additional `Subset` sampling and/or concatenation when
    you still want to query metrics via merged indices.
    """
    if not isinstance(item, (tuple, list)):
        raise ValueError("Cannot extract stable index from a non tuple/list sample.")
    if len(item) <= index_position:
        raise ValueError(
            f"Sample has length {len(item)}; cannot read stable index at position {index_position}."
        )
    return int(item[index_position])


def get_metric_for_sample(dataset: Dataset,
                        sample_idx: int,
                        metric_tracker: SampleMetricTracker,
                        index_position: int = 2):
    """Retrieve metric value for a specific sample in a possibly wrapped dataset."""
    item = dataset[sample_idx]
    merged_idx = get_stable_index_from_item(item=item, index_position=index_position)
    return metric_tracker.get_metric(merged_idx)


def subset_by_original_indices(dataset: Dataset,
                            original_indices: Sequence[int],
                            strict: bool = True,
                            return_indexed_tuple: bool = False):
    """Convenience wrapper to subset a dataset using original/global indices."""
    return MyOriginalIndexSubset(dataset=dataset,
                                original_indices=original_indices,
                                strict=strict,
                                return_indexed_tuple=return_indexed_tuple)


def sample_subset_by_available_original_indices(
    dataset: Dataset,
    candidate_original_indices: Sequence[int],
    n_samples: int,
    seed: int = 12345,
    replace: bool = False,
    strict_if_not_enough: bool = True,
    return_indexed_tuple: bool = False,
):
    """Convenience wrapper around `MyOriginalIndexSubset.sample_n_available_from_candidates`."""
    return MyOriginalIndexSubset.sample_n_available_from_candidates(
        dataset=dataset,
        candidate_original_indices=candidate_original_indices,
        n_samples=n_samples,
        seed=seed,
        replace=replace,
        strict_if_not_enough=strict_if_not_enough,
        return_indexed_tuple=return_indexed_tuple,
    )