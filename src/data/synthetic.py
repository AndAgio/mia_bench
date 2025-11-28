import numpy as np
import torch
from torch.utils.data import Dataset


class Synthetic(Dataset):
    def __init__(self, n_samples: int=400, sample_dimension: int=100, n_patches: int=2, noise: float=1., signal_mu: float=1., label_flip_p: float=0.1):
        self.n_samples = n_samples
        assert n_patches <= sample_dimension
        assert sample_dimension % n_patches == 0
        self.sample_dimension = sample_dimension
        self.n_patches = n_patches
        self.patch_dimension = int(self.sample_dimension / self.n_patches)

        self.noise = noise
        self.signal_mu = signal_mu
        self.label_flip_p = label_flip_p

        self.n_clusters = 1
        self.features = torch.zeros(self.n_clusters, self.patch_dimension)
        pos = 0
        for i in range(self.n_clusters):
            self.features[i,pos] = self.signal_mu
            pos+=1

        self.generate_data()

    def generate_data(self):
        self.data = []
        self.labels = []

        for _ in range(self.n_samples):
            original_y = np.random.choice([-1,1], 1)[0]
            k = torch.randint(0, self.n_clusters, (1,))
            signal_features = self.features[k][0] * original_y
            xis = []
            for _ in range(self.n_patches - 1):
                xis.append(torch.tensor(np.random.normal(0, self.noise, size=(self.patch_dimension))))     
            x = torch.vstack([signal_features.unsqueeze(0), torch.stack(xis)])
            idx = torch.randperm(len(x))
            x = x[idx].flatten()
            if np.random.binomial(1,self.label_flip_p,1)[0] == 0:
                y = original_y
            else:
                y = -1*original_y
            self.data.append(x)
            self.labels.append(y)

        self.data = torch.stack(self.data).unsqueeze(1).float()
        self.labels = torch.tensor(self.labels)
        self.labels[self.labels == -1] = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    @property
    def input_dimension(self):
        return self.sample_dimension
    
    @property
    def num_patches(self):
        return self.n_patches

