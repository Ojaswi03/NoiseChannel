# File: data/cifar10.py

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

def load_cifar10(test_size=0.2, random_state=42, normalize=True, binary=False, flatten=True):
    """
    Load CIFAR-10 dataset and return train/test splits as NumPy arrays.

    Args:
        test_size (float): Fraction for test split.
        random_state (int): Seed for reproducibility.
        normalize (bool): Normalize pixels to [0, 1].
        binary (bool): If True, cat/dog vs. others. Else full 10-way.
        flatten (bool): If True, output shape is (N, 3072). Else (N, 3, 32, 32).

    Returns:
        X_train, X_test, y_train, y_test: NumPy arrays
    """
    print("[INFO] Downloading CIFAR-10 dataset...")
    transform = transforms.ToTensor()
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    X = np.concatenate([trainset.data, testset.data], axis=0)  # (60000, 32, 32, 3)
    y = np.concatenate([trainset.targets, testset.targets], axis=0)

    if binary:
        print("[INFO] Converting to binary classification: cat/dog = 1, others = 0...")
        y = np.array([1 if label in [3, 5] else 0 for label in y])  # 3 = cat, 5 = dog
    else:
        y = np.array(y)

    if normalize:
        print("[INFO] Normalizing pixel values to [0, 1]...")
        X = X.astype(np.float32) / 255.0

    if flatten:
        print("[INFO] Flattening images to shape (N, 3072)...")
        X = X.reshape(X.shape[0], -1)

    print("[INFO] Splitting dataset into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
