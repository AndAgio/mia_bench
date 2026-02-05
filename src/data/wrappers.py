from torch.utils.data import Dataset, Subset


class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, sample_indices, sample_weights):
        self.wrapped_dataset = dataset
        self.sample_indices = sample_indices
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, idx):
        image, label = self.wrapped_dataset.__getitem__(idx)
        indices = self.sample_indices[idx]
        weights = self.sample_weights[idx]
        return image, label, indices, weights


class IndexedDataset:
    def __init__(self, ds):
        self.ds = ds
        self.orig_indices = self._resolve_orig_indices(ds)

    def _resolve_orig_indices(self, ds):
        if not isinstance(ds, Subset):
            return list(range(len(ds)))
        indices = ds.indices.copy()
        base = ds.dataset
        while isinstance(base, Subset):
            indices = [base.indices[i] for i in indices]
            base = base.dataset
        return indices

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        orig_idx = self.orig_indices[idx]
        new_idx = idx
        return x, y, orig_idx, new_idx

