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
    margins = 1 - y * (X @ w)
    return np.mean(np.maximum(0, margins)) + lambd * np.sum(w ** 2)

def compute_gradient(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    active = margins > 0
    grad = -np.mean((active * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return grad

def local_wcm_update(w_t, X, y, lambd, sigma, rho, gamma, prev_g, loss_type="hinge"):
    # Estimate gradient
    grad = compute_gradient(w_t, X, y, lambd)

    # Inner surrogate loss (linearized)
    surrogate = rho * grad + (1 - rho) * prev_g + 2 * lambd * (w_t - w_t)
    w_local = w_t - gamma * surrogate

    # Worst-case noise injection (bounded norm <= sigma)
    delta = sigma * np.random.randn(*w_local.shape)
    delta = delta / max(np.linalg.norm(delta), 1e-12) * sigma
    w_local_noisy = w_local + delta

    # Update local model
    new_g = compute_gradient(w_local_noisy, X, y, lambd)
    return w_local_noisy, new_g

def aggregate_weights(weights, sizes):
    total = np.sum(sizes)
    return sum(w * (sz / total) for w, sz in zip(weights, sizes))

def wcm_training(num_clients=5, num_rounds=35, lr=0.05, lambd=0.01, sigma=0.01,
                 binary=True, alpha=0.7, beta=0.6):
    print("[INFO] Loading data...")
    X_train, X_test, y_train, y_test = load_mnist(binary=binary)
    y_train_bin = 2 * y_train - 1
    y_test_bin = 2 * y_test - 1

    n_samples, n_features = X_train.shape
    w_global = np.random.normal(0, 0.01, size=n_features)

    indices = np.random.permutation(n_samples)
    splits_X = np.array_split(X_train[indices], num_clients)
    splits_y = np.array_split(y_train_bin[indices], num_clients)
    sizes = [len(x) for x in splits_X]

    grads = [np.zeros(n_features) for _ in range(num_clients)]

    accs, losses = [], []

    print("[INFO] Starting WCM training...")
    for t in range(1, num_rounds + 1):
        gamma_t = 1.0 / (t ** alpha)
        rho_t = 1.0 / (t ** beta)

        local_models = []
        new_grads = []

        for i in range(num_clients):
            X_local, y_local = splits_X[i], splits_y[i]
            w_new, g_new = local_wcm_update(
                w_global, X_local, y_local, lambd, sigma,
                rho=rho_t, gamma=gamma_t, prev_g=grads[i]
            )
            local_models.append(w_new)
            new_grads.append(g_new)

        w_global = aggregate_weights(local_models, sizes)
        grads = new_grads

        # Evaluation
        preds = np.sign(X_test @ w_global)
        acc = accuracy_score(y_test_bin, preds)
        loss = hinge_loss(w_global, X_train, y_train_bin, lambd)
        
        run.log({
            "round": t,
            "WCM/accuracy": acc,
            "WCM/loss": loss
        })

        accs.append(acc)
        losses.append(loss)

        print(f"[Round {t}] Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    return w_global, accs, losses
