import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_dataloaders():
    """
    Prepare CIFAR-10 dataloaders with normalization stats.
    Returns:
        trainset, testset (torchvision.datasets.CIFAR10): Preprocessed datasets.
    """
    # Load CIFAR-10 data to compute mean and standard deviation
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for img, _ in cifar10:
        mean += img.mean([1, 2])
        std += img.std([1, 2])

    mean /= len(cifar10)
    std /= len(cifar10)

    mean_tuple = tuple(mean.numpy())
    std_tuple = tuple(std.numpy())

    # Transformations for training data with advanced augmentations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_tuple, std_tuple)
    ])

    # Transformations for test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_tuple, std_tuple)
    ])

    # Load training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset, mean_tuple, std_tuple