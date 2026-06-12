import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(IMG_EXTENSIONS)
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No image files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def make_div2k_kodak_loaders(
    root: str = "./data_set",    # placement of data directory
    crop_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = False,
):
    train_dir = os.path.join(root, "train", "DIV2K_train_HR")
    val_dir   = os.path.join(root, "val", "DIV2K_valid_HR")
    test_dir  = os.path.join(root, "test", "kodak")

    tf_train = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    tf_val = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    tf_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    train_set = ImageFolderDataset(train_dir, transform=tf_train)
    val_set   = ImageFolderDataset(val_dir, transform=tf_val)
    test_set  = ImageFolderDataset(test_dir, transform=tf_test)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader