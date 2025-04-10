@echo off
setlocal enabledelayedexpansion

echo Advanced Vision Transformer Training with CUDA Optimization

REM Set default parameters
set DATA_DIR=data/plantnet_300K
set OUTPUT_DIR=output/transformer
set BATCH_SIZE=8
set LEARNING_RATE=5e-5
set EPOCHS=30
set MODEL_TYPE=vit_base_patch16_224
set RESUME=
set CUDA_DEVICE=0
set MAX_MEMORY_MB=512

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--data-dir" (
    set DATA_DIR=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--lr" (
    set LEARNING_RATE=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--model-type" (
    set MODEL_TYPE=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--resume" (
    set RESUME=--resume %~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--cuda-device" (
    set CUDA_DEVICE=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--max-memory" (
    set MAX_MEMORY_MB=%~2
    shift & shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :show_help
)
echo Unknown parameter: %~1
goto :show_help
:end_parse

REM Set Python path
set PYTHONPATH=%~dp0

REM Check if Python is in PATH
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH! Please make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

REM Check for CUDA availability
echo Checking CUDA availability...
python -c "import torch; available = torch.cuda.is_available(); print('CUDA Available:', available); print('CUDA Device Count:', torch.cuda.device_count() if available else 0); print('CUDA Device Name:', torch.cuda.get_device_name(0) if available else 'None'); exit(1 if not available else 0)"

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: CUDA is not available! Training will use CPU and be very slow.
    echo Please ensure your GPU drivers are correctly installed.
    echo Do you want to continue anyway? (Y/N)
    set /p CONTINUE=
    if /i "!CONTINUE!" NEQ "Y" exit /b 1
)

REM Check if required directories exist
if not exist "%DATA_DIR%" (
    echo Data directory not found: %DATA_DIR%
    echo Please make sure the dataset is available.
    pause
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Clear CUDA cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

REM Set CUDA environment variables
set CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:%MAX_MEMORY_MB%

REM Display training parameters
echo.
echo Training with the following parameters:
echo   Data directory:     %DATA_DIR%
echo   Output directory:   %OUTPUT_DIR%
echo   Batch size:         %BATCH_SIZE%
echo   Learning rate:      %LEARNING_RATE%
echo   Epochs:             %EPOCHS%
echo   Model type:         %MODEL_TYPE%
echo   CUDA device:        %CUDA_DEVICE%
echo   Max memory split:   %MAX_MEMORY_MB% MB
if not "%RESUME%"=="" echo   Resuming from:      %RESUME:--resume =%
echo.

REM Run garbage collection before starting (helps with memory)
python -c "import gc; gc.collect()"

echo Starting GPU-accelerated training...
echo.

REM Execute training
python train_transformer.py --data-dir %DATA_DIR% --output-dir %OUTPUT_DIR% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --epochs %EPOCHS% --model-type %MODEL_TYPE% %RESUME%

REM Check if training was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed! Please check the error message above.
) else (
    echo.
    echo Training completed successfully!
    echo Model saved to %OUTPUT_DIR%
)

REM Clear CUDA cache after finishing
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

pause
exit /b 0

:show_help
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --data-dir DIR       Dataset directory (default: data/plantnet_300K)
echo   --output-dir DIR     Output directory (default: output/transformer)
echo   --batch-size N       Batch size (default: 8)
echo   --lr RATE            Learning rate (default: 5e-5)
echo   --epochs N           Number of training epochs (default: 30)
echo   --model-type TYPE    Model type (default: vit_base_patch16_224)
echo                        Available: vit_base_patch16_224, vit_small_patch16_224, custom_vit
echo   --resume PATH        Resume training from checkpoint
echo   --cuda-device N      CUDA device index (default: 0)
echo   --max-memory MB      Maximum CUDA memory split size in MB (default: 512)
echo   --help               Show this help message
echo.
echo Example:
echo   %~nx0 --batch-size 4 --model-type vit_small_patch16_224 --max-memory 1024
echo.
pause
exit /b 0