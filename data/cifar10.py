import os
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

def load_cifar10(test_size=0.2, random_state=42, normalize=True, binary=True, flatten=True):
    data_root = "./data"

    print("[INFO] Downloading CIFAR-10 (if not already)...")
    transform = transforms.ToTensor()
    train = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test = CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # Clean up tar.gz after download
    archive_path = os.path.join(data_root, "cifar-10-python.tar.gz")
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print("[CLEANUP] Removed raw CIFAR archive:", archive_path)

    # Concatenate and process data
    X = np.concatenate([train.data, test.data])
    y = np.array(train.targets + test.targets)

    if binary:
        print("[INFO] Converting labels to binary (cat/dog = 1, others = 0)")
        y = np.array([1 if label in [3, 5] else 0 for label in y])  # 3=cat, 5=dog

    if normalize:
        X = X.astype(np.float32) / 255.0

    if flatten:
        X = X.reshape((X.shape[0], -1))  # (N, 3072)

    print("[INFO] Splitting into train/test sets...")
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
