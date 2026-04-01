import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class GTSRB(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        """
        Args:
            root (str): Directory where the dataset will be stored.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable, optional): A function/transform to apply to the images.
            download (bool): If True, downloads the dataset if it is not found.
        """
        self.root_dir = root
        self.train = train
        self.download = download
        
        # GTSRB images vary in size. If no transform is provided, 
        # we default to resizing them to 32x32 and converting to tensors.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # torchvision uses 'split' instead of a train boolean
        split_name = 'train' if self.train else 'test'

        # Load the dataset using torchvision's built-in GTSRB class
        self.dataset = datasets.GTSRB(
            root=self.root_dir,
            split=split_name,
            transform=self.transform,
            download=self.download
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns a sample (image) and its target (class label) at the given index."""
        return self.dataset[idx]

    def get_dataloader(self, batch_size=32, shuffle=True):
        """Returns a DataLoader for the dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    # Initialize the train and test datasets
    gtsrb_train = GTSRB(root='./data/gtsrb', train=True, download=True)
    gtsrb_test = GTSRB(root='./data/gtsrb', train=False, download=True)

    # Output dataset sizes
    print(f"Training samples: {len(gtsrb_train)}")
    print(f"Testing samples: {len(gtsrb_test)}")

    # Fetch DataLoaders
    train_loader = gtsrb_train.get_dataloader(batch_size=64, shuffle=True)
    test_loader = gtsrb_test.get_dataloader(batch_size=64, shuffle=False)
    
    # Grab one batch to check shapes
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")