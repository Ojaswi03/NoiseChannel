import numpy as np
from sklearn.metrics import accuracy_score
from data.cifar10 import load_cifar10
import wandb

def hinge_loss(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    return np.mean(np.maximum(0, margins)) + lambd * np.sum(w ** 2)


def compute_gradient(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    active = margins > 0
    grad = -np.mean((active * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return grad


def local_wcm_update(w_t, X, y, lambd, sigma, rho, gamma, prev_g):
    # Compute current gradient
    grad = compute_gradient(w_t, X, y, lambd)

    # Surrogate linearized gradient (SCA step)
    surrogate_grad = rho * grad + (1 - rho) * prev_g
    w_intermediate = w_t - gamma * surrogate_grad

    # Worst-case bounded adversarial noise (unit norm scaled to σ)
    noise = np.random.randn(*w_t.shape)
    noise /= max(np.linalg.norm(noise), 1e-12)
    noise *= sigma

    # Apply noise
    w_local = w_intermediate + noise
    new_grad = compute_gradient(w_local, X, y, lambd)

    return w_local, new_grad


def aggregate_weights(weights, sizes):
    total = np.sum(sizes)
    return sum(w * (sz / total) for w, sz in zip(weights, sizes))


def wcm_training(num_clients=10,
                 num_rounds=50,
                 lr=0.05,
                 lambd=0.01,
                 sigma=0.1,
                 binary=True,
                 alpha=0.7,
                 beta=0.6):
    """
    Worst-Case Model training for robust federated learning on CIFAR-10.

    Returns:
        w_global: final model weights
        accs: list of test accuracies per round
        losses: list of hinge loss values per round
    """
    print("[INFO] Loading CIFAR-10 dataset...")
    X_train, X_test, y_train, y_test = load_cifar10(binary=binary)

    # Convert labels to {-1, +1} already done in load_cifar10
    # y_train = 2 * y_train - 1
    # y_test = 2 * y_test - 1

    n_samples, n_features = X_train.shape
    w_global = np.random.normal(0, 0.01, size=n_features)

    print("[INFO] Partitioning data across clients...")
    indices = np.random.permutation(n_samples)
    X_splits = np.array_split(X_train[indices], num_clients)
    y_splits = np.array_split(y_train[indices], num_clients)
    sizes = [len(X) for X in X_splits]
    grads = [np.zeros(n_features) for _ in range(num_clients)]

    accs, losses = [], []

    print("[INFO] Starting Worst-Case Model training...")
    for t in range(1, num_rounds + 1):
        gamma_t = lr / (t ** alpha)
        rho_t = 1.0 / (t ** beta)

        local_models = []
        new_grads = []

        for i in range(num_clients):
            X_local, y_local = X_splits[i], y_splits[i]
            w_new, g_new = local_wcm_update(
                w_t=w_global,
                X=X_local,
                y=y_local,
                lambd=lambd,
                sigma=sigma,
                rho=rho_t,
                gamma=gamma_t,
                prev_g=grads[i]
            )
            local_models.append(w_new)
            new_grads.append(g_new)

        w_global = aggregate_weights(local_models, sizes)
        grads = new_grads

        preds = np.sign(X_test @ w_global)
        acc = accuracy_score(y_test, preds)
        loss = hinge_loss(w_global, X_train, y_train, lambd)
        wandb.log({
            "round": t,
            "WCM/accuracy": acc,
            "WCM/loss": loss
        })

        accs.append(acc)
        losses.append(loss)

        print(f"[Round {t:2d}] Test Accuracy: {acc:.4f} | Train Loss: {loss:.4f}")

    return w_global, accs, losses
