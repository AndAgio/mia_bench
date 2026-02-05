import os
from pathlib import Path
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
        data_path = self.train_dir if self.train else self.test_dir

        # Download the dataset if necessary
        if self.download:
            self._download_dataset()

        # Load the dataset
        self.dataset = datasets.ImageFolder(
            root=data_path,
            transform=self.transform
        )

        self.classes, self.class_to_idx = self._find_classes(data_path)
        self.samples = self.make_dataset(data_path, self.class_to_idx)
        self.targets = [s[1] for s in self.samples]

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

    def _find_classes(self, dir):
        """Creates classes from the folder structure.

        Args:
            dir: (string) Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
            and class_to_idx is a dictionary.
        """

        classes = []

        dir = Path(dir)

        for d in dir.rglob("*"):
            if d.is_dir():
                # Split path into strings 
                parts = d.parts # type: tuple, strings
                item = f"{parts[-2]}_{parts[-1]}"
                classes.append(item)

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def _get_target(self, file_path):
        """Returns a target_class from the parent and grandparent folders.

        Args:
            file_path: (string) path to the file.

        Returns:
            target_class: (string) target class for that file.
        """

        parts = file_path.parts
        target_class = f"{parts[-3]}_{parts[-2]}"
        return target_class

    def make_dataset(self, dir, class_to_idx):
        """Returns a list of image path, and target index

        Args:
            dir: (string) The path of each image sample
            class_to_idx: (dict: string, int) Sorted classes, mapped to int

        Returns:
            images: (list of tuples) Path and mapped class for each sample
        """

        images = []

        dir = Path(dir)

        for d in dir.rglob("*.png"):
            if not d.is_dir():
                target = self._get_target(d)
                item = (d, class_to_idx[target])
                images.append(item)

        return images

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

# cinic10_train = Cinic10(root='./datas/cinic10', train=True, download=True)
# print(cinic10_train.targets)