import os
import torch
import numpy as np
import urllib.request
from torch.utils.data import Dataset, DataLoader


class Purchase(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        """
        Args:
            root (str): Directory where the dataset will be stored.
            train (bool): If True, loads the training set (80%); otherwise, loads the test set (20%).
            transform (callable, optional): A function/transform to apply to the features.
            download (bool): If True, downloads the dataset if it is not found.
        """
        self.root_dir = root
        self.train = train
        self.transform = transform
        self.download = download
        
        self.file_path = os.path.join(self.root_dir, 'purchase100.npz')

        # Download the dataset if necessary
        if self.download:
            self._download_dataset()

        # Load the dataset arrays and split
        self._load_data()

    def _download_dataset(self):
        """Downloads the Purchase100 dataset if not already present."""
        if not os.path.exists(self.file_path):
            print("Downloading Purchase100 dataset...")
            url = "https://github.com/xehartnort/Purchase100-Texas100-datasets/releases/download/v3.0/purchase100.npz"
            
            # Create directory if it doesn't exist
            os.makedirs(self.root_dir, exist_ok=True)
            
            # Download the file
            print("Pulling from dataset url...")
            urllib.request.urlretrieve(url, self.file_path)
            print("Download complete.")

    def _load_data(self):
        """Loads the .npz file, extracts features/labels, and performs a train/test split."""
        data = np.load(self.file_path)
        
        # Datasets packaged in .npz format usually use standard keys. 
        # We handle 'features'/'labels', 'X'/'y', or fallback to default numpy names.
        if 'features' in data and 'labels' in data:
            features = data['features']
            labels = data['labels']
        elif 'X' in data and 'y' in data:
            features = data['X']
            labels = data['y']
        else:
            features = data['arr_0']
            labels = data['arr_1']

        # Convert numpy arrays to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        
        # Depending on how the labels were saved, they might be 2D. We ensure they are 1D.
        labels = torch.tensor(labels, dtype=torch.long).squeeze()

        # The raw .npz file is generally unsplit. We will do a deterministic 80/20 split here.
        # Note: If your experiment requires a specific fixed random seed for splitting, you 
        # would add a random permutation here. For simplicity, we split sequentially.
        num_samples = len(features)
        split_idx = int(num_samples * 0.8)

        if self.train:
            self.samples = features[:split_idx]
            self.targets = labels[:split_idx]
        else:
            self.samples = features[split_idx:]
            self.targets = labels[split_idx:]

        # Create a Python list of targets for easier handling in some cases (e.g., when using certain metrics or libraries that expect lists).
        self.targets_list = self.targets.tolist()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns a sample and its target from the dataset at the given index."""
        sample = self.samples[idx]
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
    # Initialize the train and test datasets
    purchase_train = Purchase(root='./data/purchase100', train=True, download=True)
    purchase_test = Purchase(root='./data/purchase100', train=False, download=True)

    # Output dataset sizes to verify the 80/20 split
    print(f"Training samples: {len(purchase_train)}")
    print(f"Testing samples: {len(purchase_test)}")

    # Fetch DataLoaders
    train_loader = purchase_train.get_dataloader(batch_size=64, shuffle=True)
    test_loader = purchase_test.get_dataloader(batch_size=64, shuffle=False)
    
    # Grab one batch to check shapes
    features, labels = next(iter(train_loader))
    print(f"Batch features shape: {features.shape}")
    print(f"Batch labels shape: {labels.shape}")