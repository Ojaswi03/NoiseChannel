
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(test_size=0.2, random_state=42, normalize=True, binary=False):
    """
    Load MNIST dataset from OpenML, return train/test splits.
    
    Args:
        test_size (float): Fraction of data to use as test set.
        random_state (int): Seed for reproducibility.
        normalize (bool): Whether to normalize pixel values to [0, 1].
        binary (bool): If True, convert to binary classification (even vs odd).
        
    Returns:
        X_train, X_test, y_train, y_test: numpy arrays
    """
    print("[INFO] Fetching MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(np.int32)
    
    if binary:
        print("[INFO] Converting labels to binary (even=0, odd=1)...")
        y = (y % 2).astype(np.int32)  # Even = 0, Odd = 1
    
    if normalize:
        print("[INFO] Normalizing pixel values...")
        X = X / 255.0
    
    print("[INFO] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
