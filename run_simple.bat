@echo off
echo Starting Vision Transformer training for plant identification with CUDA...

REM Set Python path - adjust if needed
set PYTHONPATH=%~dp0

REM Check if Python is in PATH
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH! Please make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

REM Check for CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

REM Check if required directories exist
if not exist "data\plantnet_300K" (
    echo Data directory not found! Please make sure the dataset is available.
    pause
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "output\transformer" mkdir output\transformer

REM Set CUDA environment variables to force GPU usage
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Execute training with default parameters and GPU settings
echo Training with GPU acceleration...
python train_transformer.py --data-dir data/plantnet_300K --output-dir output/transformer --batch-size 8 --lr 5e-5 --epochs 30

REM Check if training was successful
if %ERRORLEVEL% NEQ 0 (
    echo Training failed! Please check the error message above.
) else (
    echo Training completed successfully!
    echo Model saved to output/transformer/
)

pause