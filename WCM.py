# import numpy as np
# from sklearn.metrics import accuracy_score
# from data.mnist import load_mnist
# import wandb

# def hinge_loss(w, X, y, lambd):
#     margins = 1 - y * (X @ w)
#     return np.mean(np.maximum(0, margins)) + lambd * np.sum(w ** 2)


# def compute_gradient(w, X, y, lambd):
#     margins = 1 - y * (X @ w)
#     active = margins > 0
#     grad = -np.mean((active * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
#     return grad


# def local_wcm_update(w_t, X, y, lambd, sigma, rho, gamma, prev_g, t, boost=1.2):
#     # Compute current gradient
#     grad = compute_gradient(w_t, X, y, lambd)

#     # Surrogate linearized gradient (SCA step)
#     # surrogate_grad = rho * grad + (1 - rho) * prev_g

#     surrogate_grad = (boost * rho) * grad + (1 - rho) * prev_g
#     surrogate_grad /= (boost * rho + (1 - rho))  # normalize

    
#     w_intermediate = w_t - gamma * surrogate_grad

#     # Worst-case bounded adversarial noise (unit norm scaled to σ)
#     effective_sigma = sigma * np.exp(-0.02 * t) 
#     noise = np.random.randn(*w_t.shape)
#     noise /= max(np.linalg.norm(noise), 1e-12)
#     # noise *= sigma
#     noise *= effective_sigma

#     # Apply noise
#     w_local = w_intermediate + noise
#     new_grad = compute_gradient(w_local, X, y, lambd)

#     return w_local, new_grad


# def aggregate_weights(weights, sizes):
#     total = np.sum(sizes)
#     return sum(w * (sz / total) for w, sz in zip(weights, sizes))


# def wcm_training(num_clients=10,num_rounds=50,lr=0.1,lambd=0.001,sigma=0.05,binary=True,alpha=0.8,beta=0.55):
#     """
#     Worst-Case Model training for robust federated learning on MNIST.

#     Returns:
#         w_global: final model weights
#         accs: list of test accuracies per round
#         losses: list of hinge loss values per round
#     """
#     print("[INFO] Loading MNIST dataset...")
#     X_train, X_test, y_train, y_test = load_mnist(binary=binary)

#     # Convert labels to {-1, +1}
#     y_train = 2 * y_train - 1
#     y_test = 2 * y_test - 1

#     n_samples, n_features = X_train.shape
#     w_global = np.zeros(n_features)
    
#     # Initialize global model weights for EBM
#     # w_global = np.random.normal(0, 0.01, size=n_features)

#     print("[INFO] Partitioning data across clients...")
#     indices = np.random.permutation(n_samples)
#     X_splits = np.array_split(X_train[indices], num_clients)
#     y_splits = np.array_split(y_train[indices], num_clients)
#     sizes = [len(X) for X in X_splits]
#     grads = [np.zeros(n_features) for _ in range(num_clients)]

#     accs, losses = [], []

#     print("[INFO] Starting Worst-Case Model training...")
#     for t in range(1, num_rounds + 1):
#         gamma_t = lr / (t ** alpha)
#         rho_t = 1.0 / (t ** beta)
#         boost = 1.0 + 0.2 * np.exp(-0.03 * t)

#         print(f"[Round {t}] gamma: {gamma_t:.4f}, rho: {rho_t:.4f}, boost: {boost}")



#         local_models = []
#         new_grads = []

#         for i in range(num_clients):
#             X_local, y_local = X_splits[i], y_splits[i]
#             w_new, g_new = local_wcm_update(
#                 w_t=w_global,
#                 X=X_local,
#                 y=y_local,
#                 lambd=lambd,
#                 sigma=sigma,
#                 rho=rho_t,
#                 gamma=gamma_t,
#                 prev_g=grads[i],
#                 t=t,
#                 boost=boost
#             )
#             local_models.append(w_new)
#             new_grads.append(g_new)

#         w_global = aggregate_weights(local_models, sizes)
#         grads = new_grads

#         preds = np.sign(X_test @ w_global)
#         acc = accuracy_score(y_test, preds)
#         loss = hinge_loss(w_global, X_train, y_train, lambd)
#         wandb.log({
#             "round": t,
#             "WCM/accuracy": acc,
#             "WCM/loss": loss
#         })

#         accs.append(acc)
#         losses.append(loss)

#         print(f"[Round {t:2d}] Test Accuracy: {acc:.4f} | Train Loss: {loss:.4f}")

#     return w_global, accs, losses





import numpy as np
from sklearn.metrics import accuracy_score
from data.mnist import load_mnist
import wandb

def hinge_loss(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    return np.mean(np.maximum(0, margins)) + lambd * np.sum(w ** 2)

def compute_gradient(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    active = margins > 0
    grad = -np.mean((active * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return grad

def local_wcm_update(w_t, X, y, lambd, sigma, rho, gamma, prev_g, t, boost=1.2):
    grad = compute_gradient(w_t, X, y, lambd)

    # Boosted surrogate gradient
    surrogate_grad = (boost * rho) * grad + (1 - rho) * prev_g
    surrogate_grad /= (boost * rho + (1 - rho))

    w_intermediate = w_t - gamma * surrogate_grad

    # Decaying noise
    effective_sigma = sigma * np.exp(-0.02 * t)
    noise = np.random.randn(*w_t.shape)
    noise /= max(np.linalg.norm(noise), 1e-12)
    noise *= effective_sigma

    w_local = w_intermediate + noise
    new_grad = compute_gradient(w_local, X, y, lambd)

    return w_local, new_grad

def aggregate_weights(weights, sizes):
    total = np.sum(sizes)
    return sum(w * (sz / total) for w, sz in zip(weights, sizes))

def wcm_training(num_clients=10, num_rounds=50, lr=0.1, lambd=0.001, sigma=0.05, binary=True, alpha=0.7, beta=0.55):
    print("[INFO] Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist(binary=binary)

    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1

    n_samples, n_features = X_train.shape
    w_global = np.zeros(n_features)

    print("[INFO] Partitioning data across clients...")
    indices = np.random.permutation(n_samples)
    X_splits = np.array_split(X_train[indices], num_clients)
    y_splits = np.array_split(y_train[indices], num_clients)
    sizes = [len(X) for X in X_splits]
    grads = [np.zeros(n_features) for _ in range(num_clients)]

    accs, losses = [], []

    print("[INFO] Starting Worst-Case Model training...")
    for t in range(1, num_rounds + 1):
        gamma_t = max(lr / (t ** alpha), 0.005)  # avoid vanishing learning rate
        rho_t = 1.0 / (t ** beta)
        # Boost factor that decays over time
        boost = 1.3 + 0.3 * np.exp(-0.01 * t)

        print(f"[Round {t}] gamma: {gamma_t:.4f}, rho: {rho_t:.4f}, boost: {boost:.4f}")

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
                prev_g=grads[i],
                t=t,
                boost=boost
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
