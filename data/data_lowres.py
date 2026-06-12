import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_cifar10_loaders(
    root: str = "./data",    #placement of the data directory
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_size: int = 5000,
    seed: int = 42,
):
    tf_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_set = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=tf_train,
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=tf_test,
    )

    train_size = len(full_train_set) - val_size
    if train_size <= 0:
        raise ValueError(f"val_size={val_size} is too large for CIFAR-10 train split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=generator,
    )

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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader