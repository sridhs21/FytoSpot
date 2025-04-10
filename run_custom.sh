#!/bin/bash

echo "Advanced Vision Transformer Training with CUDA Optimization"

# Set default parameters
DATA_DIR="data/plantnet_300K"
OUTPUT_DIR="output/transformer"
BATCH_SIZE=8
LEARNING_RATE=5e-5
EPOCHS=30
MODEL_TYPE="vit_base_patch16_224"
RESUME=""
CUDA_DEVICE=0
MAX_MEMORY_MB=512

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --max-memory)
            MAX_MEMORY_MB="$2"
            shift 2
            ;;
        --help)
            show_help=true
            break
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help=true
            break
            ;;
    esac
done

# Help function
show_help() {
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --data-dir DIR       Dataset directory (default: data/plantnet_300K)"
    echo "  --output-dir DIR     Output directory (default: output/transformer)"
    echo "  --batch-size N       Batch size (default: 8)"
    echo "  --lr RATE            Learning rate (default: 5e-5)"
    echo "  --epochs N           Number of training epochs (default: 30)"
    echo "  --model-type TYPE    Model type (default: vit_base_patch16_224)"
    echo "                       Available: vit_base_patch16_224, vit_small_patch16_224, custom_vit"
    echo "  --resume PATH        Resume training from checkpoint"
    echo "  --cuda-device N      CUDA device index (default: 0)"
    echo "  --max-memory MB      Maximum CUDA memory split size in MB (default: 512)"
    echo "  --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --batch-size 4 --model-type vit_small_patch16_224 --max-memory 1024"
    echo ""
    read -p "Press Enter to continue..."
    exit 0
}

# Show help if requested
if [ "$show_help" = true ]; then
    show_help
fi

# Set Python path
export PYTHONPATH=$(pwd)

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python not found! Please make sure Python is installed and in your PATH."
    read -p "Press Enter to continue..."
    exit 1
fi

# Check for CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; available = torch.cuda.is_available(); print('CUDA Available:', available); print('CUDA Device Count:', torch.cuda.device_count() if available else 0); print('CUDA Device Name:', torch.cuda.get_device_name(0) if available else 'None'); exit(1 if not available else 0)"

if [ $? -ne 0 ]; then
    echo "WARNING: CUDA is not available! Training will use CPU and be very slow."
    echo "Please ensure your GPU drivers are correctly installed."
    read -p "Do you want to continue anyway? (Y/N) " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if required directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "Please make sure the dataset is available."
    read -p "Press Enter to continue..."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Clear CUDA cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$MAX_MEMORY_MB

# Display training parameters
echo ""
echo "Training with the following parameters:"
echo "  Data directory:     $DATA_DIR"
echo "  Output directory:   $OUTPUT_DIR"
echo "  Batch size:         $BATCH_SIZE"
echo "  Learning rate:      $LEARNING_RATE"
echo "  Epochs:             $EPOCHS"
echo "  Model type:         $MODEL_TYPE"
echo "  CUDA device:        $CUDA_DEVICE"
echo "  Max memory split:   $MAX_MEMORY_MB MB"
if [ -n "$RESUME" ]; then
    echo "  Resuming from:      ${RESUME#--resume }"
fi
echo ""

# Run garbage collection before starting (helps with memory)
python -c "import gc; gc.collect()"

echo "Starting GPU-accelerated training..."
echo ""

# Execute training
python train_transformer.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --batch-size "$BATCH_SIZE" --lr "$LEARNING_RATE" --epochs "$EPOCHS" --model-type "$MODEL_TYPE" $RESUME

# Check if training was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "Training failed! Please check the error message above."
else
    echo ""
    echo "Training completed successfully!"
    echo "Model saved to $OUTPUT_DIR"
fi

# Clear CUDA cache after finishing
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

read -p "Press Enter to continue..."