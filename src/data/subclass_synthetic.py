from typing import Union
import math
import numpy as np
import torch
from torch.utils.data import Dataset


class SubclassSynthetic(Dataset):
    def __init__(self, n_samples: int=400, sample_dimension: int=50, n_classes: int=2, signal_noise: float=0.5, 
                 n_subclasses: int=2, subclasses_p: Union[float, tuple[float]]=0.1, 
                 subclasses_noises: Union[float, tuple[float]]=0.5, subclasses_distances: Union[float, tuple[float]]=0.2):
        self.n_samples = n_samples
        self.sample_dimension = sample_dimension
        self.n_classes = n_classes
        if n_classes%2 == 0:
            self.classes = [item for item in range(-int(n_classes/2), int(n_classes/2)+1, 1) if item != 0]
        else:
            self.classes = [item for item in range(-math.floor(n_classes/2), math.floor(n_classes/2)+1, 1)]
        # print('Defined classes: {}'.format(self.classes))
        if isinstance(subclasses_p, float):
            subclasses_p = [subclasses_p for _ in range(n_subclasses)]
        if isinstance(subclasses_noises, float):
            subclasses_noises = [subclasses_noises for _ in range(n_subclasses)]
        if isinstance(subclasses_distances, float):
            subclasses_distances = [subclasses_distances for _ in range(n_subclasses)]
        assert n_subclasses == len(subclasses_p) == len(subclasses_noises) == len(subclasses_distances)
        assert all(subclass_p < 1 for subclass_p in subclasses_p)
        assert sum(subclasses_p) < 1
        assert 1-sum(subclasses_p) > 0
        self.signal_noise = signal_noise
        self.n_subclasses = n_subclasses
        self.all_subclasses_p = subclasses_p + [1-sum(subclasses_p)]
        assert sum(self.all_subclasses_p) == 1
        self.all_subclasses_noises = subclasses_noises + [signal_noise]
        self.all_subclasses_distances = subclasses_distances + [0.]
        if n_subclasses%2 == 0:
            self.all_subclasses_distance_multipliers = [item for item in range(-int(n_subclasses/2), int(n_subclasses/2)+1, 1) if item != 0] + [0.]
        else:
            self.all_subclasses_distance_multipliers = [item for item in range(-math.ceil(n_subclasses/2), math.floor(n_subclasses/2)+1, 1) if item != 0] + [0.]
        self.generate_data()

    def generate_data(self):
        self.data = []
        self.labels = []

        for _ in range(self.n_samples):
            y = np.random.choice(self.classes, 1)[0]
            # print('self.n_subclasses: {}'.format(self.n_subclasses))
            elements = [i for i in range(self.n_subclasses + 1)]
            # print('elements: {}'.format(elements))
            selected_subclass = np.random.choice(elements, p=self.all_subclasses_p)
            # print('selected_subclass: {}'.format(selected_subclass))
            # print('self.all_subclasses_distance_multipliers: {}'.format(self.all_subclasses_distance_multipliers))
            # print('self.all_subclasses_distances: {}'.format(self.all_subclasses_distances))
            selected_average = y + self.all_subclasses_distance_multipliers[selected_subclass]*self.all_subclasses_distances[selected_subclass]
            selected_noise = self.all_subclasses_noises[selected_subclass]
            x = torch.tensor(np.random.normal(selected_average, selected_noise, size=(self.sample_dimension)))
            self.data.append(x)
            self.labels.append(y)

        self.data = torch.stack(self.data).float()
        self.labels = torch.tensor(self.labels).unsqueeze(1).float()
        # print('labels shape: {}'.format(self.labels.shape))
        # self.labels[self.labels == -1] = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    @property
    def input_dimension(self):
        return self.sample_dimension

