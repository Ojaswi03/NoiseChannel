import os
import matplotlib.pyplot as plt
from centralized import centralized_training
from FedAvg import federated_training
from EBM import ebm_training
from WCM import wcm_training

import wandb
run = wandb.init(
    project="Nosiy Federated Learning",
    entity="ojaswisinha2001-ohio-university"
)

def run_mnist(rounds, lr, lambd, sigma, num_clients, binary=True):
    print("\n[MNIST] Running Centralized Training")
    _, c_accs, _, c_losses = centralized_training(rounds, lr, lambd, binary, dataset="mnist")

    print("\n[MNIST] Running FedAvg")
    _, f_accs, f_losses = federated_training(num_clients, rounds, lr, lambd, sigma, binary, dataset="mnist")

    print("\n[MNIST] Running EBM")
    _, e_accs, e_losses = ebm_training(num_clients, rounds, lr, lambd, sigma, binary)

    plot_results(c_accs, f_accs, e_accs, c_losses, f_losses, e_losses, "mnist", "diagram")

def run_cifar(rounds, lr, lambd, sigma, num_clients, binary=True):
    print("\n[CIFAR-10] Running Centralized Training")
    _, c_accs, _, c_losses = centralized_training(rounds, lr, lambd, binary, dataset="cifar10")

    print("\n[CIFAR-10] Running FedAvg")
    _, f_accs, f_losses = federated_training(num_clients, rounds, lr, lambd, sigma, binary, dataset="cifar10")

    print("\n[CIFAR-10] Running WCM")
    _, w_accs, w_losses = wcm_training(num_clients, rounds, lr, lambd, sigma, binary)

    plot_results(c_accs, f_accs, w_accs, c_losses, f_losses, w_losses, "cifar10", "diagram-cifar")

def plot_results(c_acc, f_acc, alt_acc, c_loss, f_loss, alt_loss, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x = list(range(1, len(c_acc) + 1))

    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, c_acc, label="Centralized")
    plt.plot(x, f_acc, label="FedAvg")
    plt.plot(x, alt_acc, label="EBM" if dataset == "mnist" else "WCM")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset.upper()} | Accuracy vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = f"{out_dir}/{dataset.lower()}_accuracy_plot.png"
    plt.savefig(acc_path)
    plt.show()
    print(f"[Saved] Accuracy plot → {acc_path}")

    # Loss Plot (log scale)
    plt.figure(figsize=(8, 6))
    plt.plot(x, c_loss, label="Centralized")
    plt.plot(x, f_loss, label="FedAvg")
    plt.plot(x, alt_loss, label="EBM" if dataset == "mnist" else "WCM")
    plt.xlabel("Round")
    plt.ylabel("Loss (log scale)")
    plt.yscale("log")
    plt.title(f"{dataset.upper()} | Loss vs Iteration")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    loss_path = f"{out_dir}/{dataset.lower()}_loss_plot.png"
    plt.savefig(loss_path)
    plt.show()
    print(f"[Saved] Loss plot → {loss_path}")

def main():
    print("=" * 50)
    print(" Federated Learning Experiment (MNIST / CIFAR-10)")
    print("=" * 50)

    dataset = input("Select dataset (mnist / cifar10): ").strip().lower()
    if dataset not in ["mnist", "cifar10"]:
        print("❌ Invalid dataset choice. Exiting.")
        return

    rounds = int(input("Enter number of rounds (e.g., 35): "))
    lr = float(input("Enter learning rate (e.g., 0.05): "))
    lambd = float(input("Enter lambda (e.g., 0.01): "))
    sigma = float(input("Enter noise std. dev sigma (e.g., 0.01): "))
    num_clients = int(input("Enter number of clients (e.g., 10): "))
    binary = input("Binary classification (cat/dog or even/odd)? [y/n]: ").strip().lower() == 'y'

    if dataset == "mnist":
        run_mnist(rounds, lr, lambd, sigma, num_clients, binary, )
    elif dataset == "cifar10":
        run_cifar(rounds, lr, lambd, sigma, num_clients, binary)
    else:
        raise ValueError("Unsupported dataset: " + dataset)
    run.finish()

if __name__ == "__main__":
    main()
