# File: centralized.py

import numpy as np
from data.mnist import load_mnist
from sklearn.metrics import accuracy_score
import wandb
# Initialize Weights & Biases
run = wandb.init(
    project="Nosiy Federated Learning",
    entity="ojaswisinha2001-ohio-university"
)
def hinge_loss(w, X, y, lambd):
    """Compute hinge loss + L2 regularization"""
    margins = 1 - y * (X @ w)
    # loss = np.mean(np.maximum(0, margins)) + lambd * np.sum(w ** 2)
    loss = np.mean(np.log1p(np.maximum(0, margins))) + lambd * np.sum(w ** 2)

    return loss

def compute_gradient(w, X, y, lambd):
    """Compute gradient of hinge loss with L2 regularization"""
    margins = 1 - y * (X @ w)
    indicator = margins > 0
    grad = -np.mean((indicator * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return grad

def centralized_training(num_epochs=20, lr=0.01, lambd=0.01, binary=True):
    """
    Centralized training of a linear SVM using hinge loss from scratch.

    Args:
        num_epochs (int): Training epochs.
        lr (float): Learning rate.
        lambd (float): L2 regularization strength.
        binary (bool): If True, even=0, odd=1 conversion.

    Returns:
        w: Final model weights.
        train_accs, test_accs, losses: Metrics per epoch.
    """
    print("[INFO] Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist(binary=binary)

    print("[INFO] Initializing model...")
    n_features = X_train.shape[1]
    # w = np.zeros(n_features)
    w = np.random.normal(loc=0.0, scale=0.01, size=n_features)


    # Convert labels to {-1, +1}
    y_train_bin = 2 * y_train - 1
    y_test_bin = 2 * y_test - 1

    train_accs, test_accs, losses = [], [], []

    print("[INFO] Starting training loop...")
    for epoch in range(num_epochs):
        grad = compute_gradient(w, X_train, y_train_bin, lambd)
        w -= lr * grad

        # Evaluate
        train_preds = np.sign(X_train @ w)
        test_preds = np.sign(X_test @ w)
        train_acc = accuracy_score(y_train_bin, train_preds)
        test_acc = accuracy_score(y_test_bin, test_preds)
        loss = hinge_loss(w, X_train, y_train_bin, lambd)
        run.log({
            "epoch": epoch + 1,
            "Centralized/train_accuracy": train_acc,
            "centralized/loss": loss
        })

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        losses.append(loss)

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}")

    return w, train_accs, test_accs, losses
