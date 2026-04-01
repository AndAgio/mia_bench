import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

class News20(Dataset):
    def __init__(self, train=True, transform=None, max_features=134410, download=True):
        """
        Args:
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable, optional): A function/transform to apply to the features.
            max_features (int): The exact number of TF-IDF dimensions to match the paper.
            download (bool): If True, downloads the dataset if it is not found locally.
        """
        self.train = train
        self.transform = transform
        self.max_features = max_features
        self.download = download

        # Load the dataset and apply TF-IDF
        self._load_data()

    def _load_data(self):
        """Fetches the 20 Newsgroups text data and applies TF-IDF vectorization."""
        
        subset = 'train' if self.train else 'test'
        print(f"Loading 20 Newsgroups '{subset}' split...")
        
        try:
            # 1. Fetch the raw text data for the requested split
            data = fetch_20newsgroups(
                subset=subset, 
                download_if_missing=self.download
            )
            
            # 2. Build the TF-IDF vocabulary space using the full dataset.
            # We fit on 'all' to ensure train and test features align perfectly.
            print("Building the TF-IDF vocabulary space...")
            full_data = fetch_20newsgroups(
                subset='all', 
                download_if_missing=self.download
            )
        except IOError:
            raise FileNotFoundError(
                "The 20 Newsgroups dataset was not found on disk. "
                "Please set `download=True` to download it."
            )
        
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.vectorizer.fit(full_data.data)
        
        # 3. Transform the specific subset (train or test) into a sparse TF-IDF matrix
        print("Applying TF-IDF transformation...")
        self.samples_sparse = self.vectorizer.transform(data.data)
        
        # 4. Extract labels
        self.targets = torch.tensor(data.target, dtype=torch.long)
        self.targets_list = self.targets.tolist()
        
        print(f"Data loaded! Shape: {self.samples_sparse.shape}")

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
        # Initialize the train and test datasets with the optional download parameter
        news_train = News20(train=True, download=True)
        news_test = News20(train=False, download=True)

        # Output dataset sizes
        print(f"Training samples: {len(news_train)}")
        print(f"Testing samples: {len(news_test)}")

        # Fetch DataLoaders
        train_loader = news_train.get_dataloader(batch_size=64, shuffle=True)
        test_loader = news_test.get_dataloader(batch_size=64, shuffle=False)
        
        # Grab one batch to check shapes
        features, labels = next(iter(train_loader))
        print(f"Batch features shape: {features.shape}")
        print(f"Batch labels shape: {labels.shape}")
        
    except FileNotFoundError as e:
        print(e)