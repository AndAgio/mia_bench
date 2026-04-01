import os
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

class News(Dataset):
    def __init__(self, root, train=True, transform=None, max_features=134410, download=True):
        """
        Args:
            root (str): Directory where the dataset will be stored.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable, optional): A function/transform to apply to the features.
            max_features (int): The exact number of TF-IDF dimensions to match the paper.
            download (bool): If True, downloads the dataset if it is not found locally.
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.max_features = max_features
        self.download = download

        # Create the root directory if it doesn't exist
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        # Load the dataset (either from cache or by processing it)
        self._load_data()

    def _load_data(self):
        """Loads data from a cached .npz file, or generates it via TF-IDF if missing."""
        
        subset = 'train' if self.train else 'test'
        
        # Define a unique filename for the cached features
        cache_file = os.path.join(self.root, f"news20_{subset}_{self.max_features}.npz")
        
        # ==========================================
        # FAST PATH: Load from cached .npz file
        # ==========================================
        if os.path.exists(cache_file):
            print(f"Loading cached '{subset}' split from {cache_file}...")
            loader = np.load(cache_file)
            
            # Reconstruct the scipy sparse CSR matrix
            self.samples_sparse = sp.csr_matrix(
                (loader['data'], loader['indices'], loader['indptr']), 
                shape=loader['shape']
            )
            
            # Load the targets
            self.targets = torch.tensor(loader['targets'], dtype=torch.long)
            self.targets_list = self.targets.tolist()
            print(f"Data loaded from cache! Shape: {self.samples_sparse.shape}")
            return

        # ==========================================
        # SLOW PATH: Download, fit TF-IDF, and cache
        # ==========================================
        print(f"Cache not found. Generating '{subset}' split using TF-IDF...")
        
        try:
            # 1. Fetch the raw text data for the requested split
            data = fetch_20newsgroups(
                data_home=self.root,
                subset=subset, 
                download_if_missing=self.download
            )
            
            # 2. Build the TF-IDF vocabulary space using the full dataset
            print("Building the TF-IDF vocabulary space (this may take a moment)...")
            full_data = fetch_20newsgroups(
                data_home=self.root,
                subset='all', 
                download_if_missing=self.download
            )
        except IOError:
            raise FileNotFoundError(
                f"The 20 Newsgroups dataset was not found in {self.root}. "
                "Please set `download=True` to download it."
            )
        
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.vectorizer.fit(full_data.data)
        
        # 3. Transform the specific subset
        print("Applying TF-IDF transformation...")
        self.samples_sparse = self.vectorizer.transform(data.data)
        
        # 4. Save the sparse matrix components and targets to an .npz file for next time
        print(f"Caching features to {cache_file}...")
        np.savez(
            cache_file,
            data=self.samples_sparse.data,
            indices=self.samples_sparse.indices,
            indptr=self.samples_sparse.indptr,
            shape=self.samples_sparse.shape,
            targets=data.target
        )
        
        # 5. Extract labels for the current run
        self.targets = torch.tensor(data.target, dtype=torch.long)
        self.targets_list = self.targets.tolist()
        
        print(f"Data generated and cached! Shape: {self.samples_sparse.shape}")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.samples_sparse.shape[0]

    def __getitem__(self, idx):
        """
        Returns a sample and its target.
        Converts the sparse row to a dense PyTorch tensor on the fly to save RAM.
        """
        # Extract the single row, convert to dense numpy array, and squeeze to 1D
        sample_np = self.samples_sparse[idx].toarray().squeeze()
        
        # Convert to PyTorch tensor
        sample = torch.tensor(sample_np, dtype=torch.float32)
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def get_dataloader(self, batch_size=32, shuffle=True):
        """Returns a DataLoader for the dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    try:
        # Run 1: Will download the text, apply TF-IDF, and save the .npz files
        print("--- FIRST RUN (Generates Cache) ---")
        news_train = News(root='./data/news', train=True, download=True)
        
        # Run 2: Will instantly load from the .npz files it just created
        print("\n--- SECOND RUN (Loads from Cache) ---")
        news_train_cached = News(root='./data/news', train=True, download=False)

    except FileNotFoundError as e:
        print(e)