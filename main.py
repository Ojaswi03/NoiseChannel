import os
import matplotlib.pyplot as plt
from centralized import centralized_training
from FedAvg import federated_training
from EBM import ebm_training
from WCM import wcm_training
import wandb

def run_EBM(rounds, lr, lambd, sigma, num_clients, binary=True):
    print("\n[MNIST] Running Centralized Training")
    _, c_accs, _, c_losses = centralized_training(rounds, lr, lambd, binary, dataset="mnist")

    print("\n[MNIST] Running FedAvg")
    _, f_accs, f_losses = federated_training(num_clients, rounds, lr, lambd, sigma, binary, dataset="mnist", smooth=True, alpha=0.3)

    print("\n[MNIST] Running EBM")
    _, e_accs, e_losses = ebm_training(num_clients, rounds, lr, lambd, sigma, binary)

    plot_results(c_accs, f_accs, e_accs, c_losses, f_losses, e_losses, "EBM", "diagram-EBM")

def run_WCM(rounds, lr, lambd, sigma, wcmLR, wcmLambd, WCMSigma,  num_clients, binary=True):
    print("\n[MNIST] Running Centralized Training")
    _, c_accs, _, c_losses = centralized_training(rounds, lr, lambd, binary, dataset="mnist")

    print("\n[MNIST] Running FedAvg")
    _, f_accs, f_losses = federated_training(num_clients, rounds, lr, lambd, sigma, binary, dataset="mnist", smooth=True, alpha=0.3)

    print("\n[MNIST] Running WCM")
    _, w_accs, w_losses = wcm_training(num_clients, rounds, lr = wcmLR , lambd = wcmLambd, sigma = WCMSigma, binary = True)

    plot_results(c_accs, f_accs, w_accs, c_losses, f_losses, w_losses, "WCM", "diagram-WCM")

def plot_results(c_acc, f_acc, alt_acc, c_loss, f_loss, alt_loss, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Truncate to minimum shared length to avoid misaligned plots
    min_len = min(len(c_acc), len(f_acc), len(alt_acc))
    c_acc = c_acc[:min_len]
    f_acc = f_acc[:min_len]
    alt_acc = alt_acc[:min_len]

    c_loss = c_loss[:min_len]
    f_loss = f_loss[:min_len]
    alt_loss = alt_loss[:min_len]
    print(f"Centralized acc length: {len(c_acc)}")
    print(f"FedAvg acc length: {len(f_acc)}")
    print(f"EBM/WCM acc length: {len(alt_acc)}")


    x = list(range(1, min_len + 1))

    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, c_acc, 'o-', label="Centralized", linewidth=2)
    plt.plot(x, f_acc, 's--', label="FedAvg", linewidth=2)
    plt.plot(x, alt_acc, 'x-.', label="EBM" if dataset == "mnist" else "WCM", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset.upper()} | Accuracy vs Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = f"{out_dir}/{dataset.lower()}_accuracy_plot_fixed.png"
    plt.savefig(acc_path)
    plt.show()
    print(f"[Saved] Accuracy plot → {acc_path}")

    # Loss Plot (log scale)
    plt.figure(figsize=(8, 6))
    plt.plot(x, c_loss, 'o-', label="Centralized", linewidth=2)
    plt.plot(x, f_loss, 's--', label="FedAvg", linewidth=2)
    plt.plot(x, alt_loss, 'x-.', label="EBM" if dataset == "mnist" else "WCM", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Loss (log scale)")
    plt.yscale("log")
    plt.title(f"{dataset.upper()} | Loss vs Rounds")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    loss_path = f"{out_dir}/{dataset.lower()}_loss_plot_fixed.png"
    plt.savefig(loss_path)
    plt.show()
    print(f"[Saved] Loss plot → {loss_path}")


def main():
    print("=" * 50)
    print(" Federated Learning Experiment (MNIST)")
    print("=" * 50)

    dataset = input("Select dataset (mnist): ").strip().lower()
    if dataset not in ["mnist"]:
        print("❌ Invalid dataset choice. Exiting.")
        return

    rounds = int(input("Enter number of rounds (e.g., 35): "))
    lr = float(input("Enter learning rate (e.g., 0.05): "))
    lambd = float(input("Enter lambda (e.g., 0.01): "))
    sigma = float(input("Enter noise std. dev sigma (e.g., 0.01): "))
    wcmLR = float(input("Enter WCM learning rate (e.g., 0.05): "))
    wcmLambd = float(input("Enter WCM lambda (e.g., 0.01): "))
    WCMSigma = float(input("Enter noise std. dev sigma (e.g., 0.05): "))
    num_clients = int(input("Enter number of clients (e.g., 10): "))
    binary = input("Binary classification (cat/dog or even/odd)? [y/n]: ").strip().lower() == 'y'

    if dataset == "mnist":
        run = wandb.init(
            project="Nosiy Federated Learning",
            entity="ojaswisinha2001-ohio-university",
            name=f"EBM-MNIST-{rounds}R-{lr}LR-{lambd}L-{sigma}S-{num_clients}C",
        )

        run_EBM(rounds, lr, lambd, sigma, num_clients, binary)
        run.finish()
        run = wandb.init(
            project="Nosiy Federated Learning",
            entity="ojaswisinha2001-ohio-university",
            name=f"WCM-MNIST-{rounds}R-{lr}LR-{lambd}L-{sigma}S-{num_clients}C",
        )
        run_WCM(rounds, lr, lambd, sigma, wcmLR, wcmLambd, WCMSigma, num_clients, binary)
        run.finish()
    else:
        raise ValueError("Unsupported dataset. Only 'mnist' is currently implemented for EBM and WCM.")

if __name__ == "__main__":
    main()
