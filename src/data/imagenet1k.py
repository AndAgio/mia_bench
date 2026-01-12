
import os
import pathlib
import shutil
import tempfile
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg


# ---- Official ILSVRC2012 archives and their MD5 checksums (ImageNet-1K) ----
# Sources for filenames / MD5: TorchVision docs + widely cited scripts.  # see notes below
ARCHIVE_META = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val":   ("ILSVRC2012_img_val.tar",   "29b22e2961454d5413ddabcf34fc5622"),
    "devkit":("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}

META_FILE = "meta.bin"


class ImageNet1K(ImageFolder):
    """
    ImageNet-1K (ILSVRC 2012) Classification Dataset.

    Works with:
      1) Official ILSVRC archives (train/val/devkit tarballs).
      2) Kaggle/CLS-LOC style directory layout (train/<wnid>/..., val initially flat).

    Args:
        root (str | Path): Root directory for the dataset (where archives or folders live).
        split (str): 'train' or 'val'.
        layout (str): 'ilsvrc' for official tar archives, 'kaggle' for Kaggle/CLS-LOC folders.
        download (bool): If True and layout='ilsvrc', attempt to download the official tar files.
                         (Requires that you already have access / authentication.)
        transform, target_transform, loader: Same as torchvision.datasets.ImageFolder.

    Attributes (after __init__):
        classes: list of class name tuples (human readable).
        class_to_idx: dict mapping each class name to index [0..999].
        wnids: list of WordNet IDs (synsets) for the 1,000 classes.
        wnid_to_idx: dict mapping synset -> index.
        imgs / samples: list of (path, class_index) tuples.
        targets: list[int] class_index per image.
    """
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        layout: str = "ilsvrc",
        download: bool = False,
        **kwargs: Any,
    ) -> None:
        root = self.root = os.path.expanduser(str(root))
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.layout = verify_str_arg(layout, "layout", ("ilsvrc", "kaggle"))

        if self.layout == "ilsvrc":
            # Optionally download official archives (requires access; may 403 without auth).
            if download and not self._check_archives_present():
                self._download_official_archives()  # see notes below

            # Parse devkit and split archives as needed.
            self._parse_archives_ilsvrc()

            # Build label metadata and initialize ImageFolder.
            wnid_to_classes = load_meta_file(self.root)[0]
            super().__init__(self.split_folder, **kwargs)

            # Convert wnids/classes to human-readable classes like TorchVision does.
            self.wnids = self.classes
            self.wnid_to_idx = self.class_to_idx
            self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
            self.class_to_idx = {
                cls: idx
                for idx, clss in enumerate(self.classes)
                for cls in clss
            }

        else:  # layout == "kaggle"
            # Expect a directory layout like: root/train/<wnid>/*.JPEG and root/val/*.JPEG (flat).
            # If val is flat, call arrange_kaggle_val(...) once to sort into <wnid>/ folders.
            # Then we can use ImageFolder directly.
            # Load devkit (or pre-existing meta.bin) to get wnid->classes mapping for human-readable names.
            if not check_integrity(os.path.join(self.root, META_FILE)):
                parse_devkit_archive(self.root)  # requires ILSVRC devkit to be present

            wnid_to_classes = load_meta_file(self.root)[0]
            split_dir = self.split_folder  # train or val
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"Kaggle layout expected a '{self.split}' folder at {split_dir}, "
                    "but it does not exist."
                )

            # If val is flat (files directly under root/val), arrange into wnid folders once.
            if self.split == "val":
                maybe_flat = any(
                    os.path.isfile(os.path.join(split_dir, f))
                    for f in os.listdir(split_dir)
                )
                if maybe_flat:
                    print("Detected flat Kaggle val layout—arranging into class folders...")
                    arrange_kaggle_val(self.root, val_folder="val")  # uses meta.bin ground-truth

            super().__init__(split_dir, **kwargs)

            # Remap to human-readable classes.
            self.wnids = self.classes
            self.wnid_to_idx = self.class_to_idx
            self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
            self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    # --------------------------- helpers & properties ---------------------------

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def _check_archives_present(self) -> bool:
        ok = True
        for _, (fname, _) in ARCHIVE_META.items():
            path = os.path.join(self.root, fname)
            if not os.path.exists(path):
                print(f"[ImageNet1K] Missing archive: {path}")
                ok = False
        return ok

    def _download_official_archives(self) -> None:
        """
        Attempt to download the official ILSVRC2012 archives into `root`.
        Requires that you have a valid session / access to image-net.org.
        """
        this_file_dir = pathlib.Path(__file__).parent.resolve()
        data_dir = os.path.join(this_file_dir, "..", "..", self.root)
        os.makedirs(data_dir, exist_ok=True)
        print(f"[ImageNet1K] Downloading into: {data_dir}")

        # NOTE: Official downloads often require authentication and may return 403 without cookies.
        # If you have access, these direct paths correspond to the known filenames.
        commands = [
            f"wget https://image-net.org/data/ILSVRC/2012/{fname}"
            for fname, _ in ARCHIVE_META.values()
        ]
        procs = [subprocess.Popen(cmd, shell=True, cwd=data_dir) for cmd in commands]
        for p in procs:
            p.wait()

    def _parse_archives_ilsvrc(self) -> None:
        # Devkit -> meta.bin with wnid mapping + val wnids
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        # Train / Val folders
        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                parse_train_archive(self.root)
            else:
                parse_val_archive(self.root)

    def extra_repr(self) -> str:
        return f"Split: {self.split}, Layout: {self.layout}"


# ----------------------------- shared utilities ------------------------------

def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, Tuple[str, ...]], List[str]]:
    """Load meta.bin, which contains (wnid->classes dict, val_wnids list)."""
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)
    if check_integrity(file):
        return torch.load(file)
    msg = ("The meta file {} is not present in the root directory or is corrupted. "
           "It is automatically created by parse_devkit_archive.")
    raise RuntimeError(msg.format(file))


def _verify_archive(root: str, file: str, md5: str) -> None:
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root: str, file: Optional[str] = None) -> None:
    """
    Parse the ILSVRC2012 devkit and save meta information in meta.bin:
    - wnid_to_classes: Dict[wnid, tuple(human-readable class names)]
    - val_wnids: List[wnid] for each validation image (in original order)

    Requires: scipy to read MATLAB .mat
    """
    import scipy.io as sio

    @contextmanager
    def tmpdir() -> Any:
        d = tempfile.mkdtemp()
        try:
            yield d
        finally:
            shutil.rmtree(d)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    _verify_archive(root, file, md5)

    with tmpdir() as d:
        extract_archive(os.path.join(root, file), d)
        devkit_root = os.path.join(d, "ILSVRC2012_devkit_t12")

        # meta.mat: synsets + classes; validation_ground_truth.txt: idx per val image
        meta = sio.loadmat(os.path.join(devkit_root, "data", "meta.mat"), squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        leaves = [meta[idx] for idx, n in enumerate(nums_children) if n == 0]
        idcs, wnids, classes = list(zip(*leaves))[:3]
        classes = [tuple(cls.split(', ')) for cls in classes]

        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: cls for wnid, cls in zip(wnids, classes)}

        with open(os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt"), "r") as f:
            val_idcs = [int(line.strip()) for line in f.readlines()]
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(root: str, file: Optional[str] = None, folder: str = "train") -> None:
    """Extract train tar and nested class tars into root/train/<wnid>/..."""
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    _verify_archive(root, file, md5)

    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)
    # Each class has its own nested .tar that we must extract:
    archives = [os.path.join(train_root, a) for a in os.listdir(train_root)]
    for a in archives:
        extract_archive(a, os.path.splitext(a)[0], remove_finished=True)


def parse_val_archive(root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val") -> None:
    """Extract val tar and move images into root/val/<wnid>/... according to ground-truth."""
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]
    _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), val_root)
    images = sorted([os.path.join(val_root, f) for f in os.listdir(val_root)])

    for wnid in set(wnids):
        os.makedirs(os.path.join(val_root, wnid), exist_ok=True)
    for wnid, img in zip(wnids, images):
        shutil.move(img, os.path.join(val_root, wnid, os.path.basename(img)))


def arrange_kaggle_val(root: str, val_folder: str = "val") -> None:
    """
    For Kaggle/CLS-LOC layout where val images are flat under root/val,
    create subdirs root/val/<wnid>/ and move images accordingly,
    using the meta.bin (generated by parse_devkit_archive).

    Expects filenames like ILSVRC2012_val_00000001.JPEG
    """
    wnids = load_meta_file(root)[1]  # list[str] of wnids per val image
    val_root = os.path.join(root, val_folder)
    files = sorted([f for f in os.listdir(val_root) if f.lower().endswith(".jpeg")])

    if len(files) != len(wnids):
        raise RuntimeError(
            f"Mismatch: {len(files)} val images vs {len(wnids)} entries in ground-truth."
        )
    for wnid in set(wnids):
        os.makedirs(os.path.join(val_root, wnid), exist_ok=True)
    for wnid, fname in zip(wnids, files):
        src = os.path.join(val_root, fname)
        dst = os.path.join(val_root, wnid, fname)
        shutil.move(src, dst)
