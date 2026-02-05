from torch.utils.data import Dataset


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
            raise ValueError(f"ID '{id}' not in datasets managed by {self}!")
        
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
    def __init__(self, dataset: Dataset, fixed_label: int = 0):
        super().__init__()
        self.data = dataset
        self.fixed_label = fixed_label

    def __getitem__(self, index):
        return self.data[index][0], self.fixed_label

    def __len__(self):
        return len(self.data)