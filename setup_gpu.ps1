# NeuroLens GPU Setup Script
# Replaces CPU-only PyTorch with CUDA version
# Usage: .\setup_gpu.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuroLens GPU Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU
Write-Host "[1/3] Checking GPU..." -ForegroundColor Yellow
$gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: No NVIDIA GPU found or driver not installed" -ForegroundColor Red
    exit 1
}
Write-Host "  GPU: $gpuInfo" -ForegroundColor Green

# Uninstall CPU PyTorch
Write-Host "[2/3] Uninstalling CPU-only PyTorch..." -ForegroundColor Yellow
pip uninstall torch torchvision torchaudio -y 2>&1 | Out-Null
Write-Host "  Uninstalled" -ForegroundColor Green
Write-Host ""

# Install CUDA PyTorch
Write-Host "[3/3] Installing CUDA PyTorch (cu124)..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes depending on network speed..." -ForegroundColor Gray
Write-Host ""

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Verify
$verify = python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('PyTorch:', torch.__version__); print('CUDA ver:', torch.version.cuda)" 2>&1
Write-Host $verify -ForegroundColor Green

if ($verify -match "CUDA: True") {
    Write-Host ""
    Write-Host "  GPU setup SUCCESS!" -ForegroundColor Green
    Write-Host "  Test with:" -ForegroundColor Gray
    Write-Host '    python scripts/quick_test.py --model-path models/Qwen2.5-1.5B-Instruct --level quick' -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "  WARNING: GPU setup may have issues, check output above" -ForegroundColor Red
}

Write-Host "========================================" -ForegroundColor Cyan
