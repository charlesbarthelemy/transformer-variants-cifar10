import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_dataloaders(batch_size, mean=None, std=None):
    """
    Prepare CIFAR-10 dataloaders with normalization stats.

    Args:
        batch_size (int): Batch size for training/testing.
        mean (tuple, optional): Mean for normalization. Auto-computed if None.
        std (tuple, optional): Std for normalization. Auto-computed if None.

    Returns:
        trainset, testset (torchvision.datasets.CIFAR10): Preprocessed datasets.
    """
    # Load CIFAR-10 once to compute mean/std if not provided
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    data = torch.tensor(cifar10.data, dtype=torch.float32) / 255.0
    data = data.permute(0, 3, 1, 2)  # (N, C, H, W)

    # Compute mean and std if missing
    if mean is None or std is None:
        mean = data.mean(dim=(0, 2, 3))
        std = data.std(dim=(0, 2, 3))
        mean, std = tuple(mean.numpy()), tuple(std.numpy())

    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load train and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset