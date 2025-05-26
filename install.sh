#!/bin/bash

echo "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "🧪 Creating Python virtual environment..."
python3 -m venv venv-noise

echo "✅ Virtual environment created."
echo "   To activate it, run: source venv-noise/bin/activate"

echo "📦 Installing required Python packages..."
source venv-noise/bin/activate

pip install --upgrade pip
pip install torch torchvision matplotlib numpy wandb scikit-learn pandas

echo ""
read -p "🔐 Enter your Weights & Biases API key (or press Enter to skip): " 2965b3902211b5ec68d7e973e43800ba4fb1791b
if [[ ! -z "$wandb_key" ]]; then
    wandb login "$wandb_key"
else
    echo "⚠️  Skipping wandb login. You can run 'wandb login' manually later."
fi

echo ""
echo "✅ Setup complete."
echo "   To begin working, run: source venv-ebm/bin/activate"
