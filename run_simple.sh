#!/bin/bash

echo "Starting Vision Transformer training for plant identification with CUDA..."

# Set Python path - adjust if needed
export PYTHONPATH=$(pwd)

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python not found! Please make sure Python is installed and in your PATH."
    read -p "Press Enter to continue..."
    exit 1
fi

# Check for CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Check if required directories exist
if [ ! -d "data/plantnet_300K" ]; then
    echo "Data directory not found! Please make sure the dataset is available."
    read -p "Press Enter to continue..."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output/transformer

# Set CUDA environment variables to force GPU usage
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Execute training with default parameters and GPU settings
echo "Training with GPU acceleration..."
python train_transformer.py --data-dir data/plantnet_300K --output-dir output/transformer --batch-size 8 --lr 5e-5 --epochs 30

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Training failed! Please check the error message above."
else
    echo "Training completed successfully!"
    echo "Model saved to output/transformer/"
fi

read -p "Press Enter to continue..."