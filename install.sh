#!/bin/bash


echo "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "🧪 Creating Python virtual environment..."
python3 -m venv venv-noise

if [ ! -f "venv-noise/bin/activate" ]; then
    echo "❌ Failed to create virtual environment. Make sure 'python3-full' is installed."
    exit 1
fi

echo "✅ Virtual environment created."
echo "   To activate it, run: source venv-noise/bin/activate"

echo "📦 Installing required Python packages..."
source venv-noise/bin/activate

# Use --break-system-packages if needed due to Debian PEP 668
pip install --upgrade pip --break-system-packages
pip install torch torchvision matplotlib numpy wandb scikit-learn pandas --break-system-packages

echo ""
read -p "🔐 Enter your Weights & Biases API key (or press Enter to skip): " wandb_key
if [[ ! -z "$wandb_key" ]]; then
    wandb login "$wandb_key"
else
    echo "⚠️  Skipping wandb login. You can run 'wandb login' manually later."
fi

echo ""
echo "✅ Setup complete."
echo "   To begin working, run: source venv-noise/bin/activate"
