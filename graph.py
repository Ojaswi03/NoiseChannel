# File: graph.py

import matplotlib.pyplot as plt
from centralized import centralized_training
from FedAvg import federated_training
from EBM import ebm_training

def compare_all_models():
    rounds = 35
    lr = 0.01
    lambd = 0.001
    sigma = 0.1
    num_clients = 5
    binary = True

    print("\n=== Centralized Training ===")
    _, centralized_accs, _, _ = centralized_training(
        num_epochs=rounds,
        lr=lr,
        lambd=lambd,
        binary=binary
    )

    print("\n=== FedAvg (Conventional FL) ===")
    _, fedavg_accs, _ = federated_training(
        num_clients=num_clients,
        num_rounds=rounds,
        lr=lr,
        lambd=lambd,
        sigma=sigma,
        binary=binary
    )

    print("\n=== EBM (Proposed FL) ===")
    _, ebm_accs, _ = ebm_training(
        num_clients=num_clients,
        num_rounds=rounds,
        lr=lr,
        lambd=lambd,
        sigma=sigma,
        binary=binary
    )

    # Plotting accuracy vs iteration
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, rounds + 1), centralized_accs, label="Centralized Learning")
    plt.plot(range(1, rounds + 1), fedavg_accs, label="Conventional Federated Learning")
    plt.plot(range(1, rounds + 1), ebm_accs, label="Proposed Federated Learning (EBM)")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Iteration (MNIST, Binary Classification)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_all_models()
