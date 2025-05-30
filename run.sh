#!/bin/bash

# Core parameters that main.py expects
dataset="mnist"  # mnist
num_epochs=50 # 35 for mnist
lr=0.05 # 0.01 for mnist
lambda=0.001 # 0.001 for mnist
sigma=0.1 # 0.1 for mnist
wcmLR=0.25 # 0.01 for mnist
wcmLambda=0.0015 # 0.001 for mnist
wcmSigma=0.0075 # 0.1 for mnist
num_clients=10 # 10 for mnist
binary="y" # "y" for binary mode, "n" for non-binary

echo "=========== FEDERATED LEARNING CONFIG ==========="
echo "Dataset           : $dataset"
echo "Epochs            : $num_epochs"
echo "Learning Rate     : $lr"
echo "Lambda            : $lambda"
echo "Sigma (noise)     : $sigma"
echo "WCM Learning Rate : $wcmLR"
echo "WCM Lambda        : $wcmLambda"
echo "WCM Sigma (noise) : $wcmSigma"
echo "Num Clients       : $num_clients"
echo "Binary Mode       : $binary"
echo "================================================="

# Feed parameters to main.py non-interactively
 python3 main.py <<EOF
$dataset
$num_epochs
$lr
$lambda
$sigma
$wcmLR
$wcmLambda
$wcmSigma
$num_clients
$binary
EOF

echo "================================================="
echo "✅ Experiment complete!"
echo "📁 Plots saved to: diagram/${dataset}_accuracy_plot.png and _loss_plot.png"
echo "================================================="





# echo "🚀 Running Federated Learning (FedAvg vs WCM) on CIFAR-10 for Figure 5..."

# # Core parameters that main.py expects
# dataset="cifar10"  # mnist, cifar10
# num_epochs=50 # 35 for mnist
# lr=0.05 # 0.01 for mnist
# lambda=0.01 # 0.001 for mnist
# sigma=0.1 # 0.1 for mnist
# num_clients=10 # 10 for mnist
# binary="y" # "y" for binary mode, "n" for non-binary

# echo "=========== FEDERATED LEARNING CONFIG ==========="
# echo "Dataset       : $dataset"
# echo "Epochs        : $num_epochs"
# echo "Learning Rate : $lr"
# echo "Lambda        : $lambda"
# echo "Sigma (noise) : $sigma"
# echo "Num Clients   : $num_clients"
# echo "Binary Mode   : $binary"
# echo "================================================="


# python3 main.py <<EOF
# $dataset
# $num_epochs
# $lr
# $lambda
# $sigma
# $num_clients
# $binary
# EOF

# echo "================================================="
# echo "✅ Experiment complete!"
# echo "📁 Plots saved to: diagram/${dataset}_accuracy_plot.png and _loss_plot.png"
# echo "================================================="

