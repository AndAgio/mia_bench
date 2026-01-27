import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import urllib.request
import tarfile


class Cinic10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        """
        Args:
            root_dir (str): Directory where the dataset will be stored.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable, optional): A function/transform to apply to the images.
            download (bool): If True, downloads the dataset if it is not found.
        """
        self.root_dir = root
        self.train = train
        self.transform = transform or self._default_transform()
        self.download = download

        # Define dataset paths
        self.train_dir = os.path.join(root, 'train')
        self.test_dir = os.path.join(root, 'test')

        # Download the dataset if necessary
        if self.download:
            self._download_dataset()

        # Load the dataset
        self.dataset = datasets.ImageFolder(
            root=self.train_dir if self.train else self.test_dir,
            transform=self.transform
        )

    def _default_transform(self):
        """Default transform for the CINIC-10 dataset."""
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835)
            )
        ])

    def _download_dataset(self):
        """Downloads and extracts the CINIC-10 dataset if not already present."""
        if not os.path.exists(self.train_dir) or not os.path.exists(self.test_dir):
            print("Downloading CINIC-10 dataset...")
            url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
            filename = os.path.join(self.root_dir, "CINIC-10.tar.gz")

            # Create directory if it doesn't exist
            os.makedirs(self.root_dir, exist_ok=True)

            # Download the dataset
            print("Pulling from dataset url...")
            urllib.request.urlretrieve(url, filename)

            # Extract the dataset
            print("Extracting images from tar file...")
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=self.root_dir)

            # Remove the tar file
            os.remove(filename)
            print("Download and extraction complete.")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns a sample from the dataset at the given index."""
        return self.dataset[idx]

    def get_dataloader(self, batch_size=32, shuffle=True):
        """Returns a DataLoader for the dataset."""
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage:
# cinic10_train = CINIC10Dataset(root_dir='./data/cinic10', train=True, download=True)
# cinic10_test = CINIC10Dataset(root_dir='./data/cinic10', train=False, download=True)
# train_loader = cinic10_train.get_dataloader(batch_size=64, shuffle=True)
# test_loader = cinic10_test.get_dataloader(batch_size=64, shuffle=False)
