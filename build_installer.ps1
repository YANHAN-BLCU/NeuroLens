# NeuroLens Build Script
# Usage: .\build_installer.ps1 [-SkipPyInstaller] [-SkipInnoSetup] [-Clean] [-Debug]

param(
    [switch]$SkipPyInstaller,
    [switch]$SkipInnoSetup,
    [switch]$Clean,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "visualization\backend"
$RootExe = Join-Path $ProjectRoot "NeuroLens.exe"
$RootInternal = Join-Path $ProjectRoot "_internal"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuroLens Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 0: Check environment
Write-Host "[0/4] Checking environment..." -ForegroundColor Yellow

# Use full path to avoid PATH issues
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}
Write-Host "  Python: $pythonExe" -ForegroundColor Gray

$cudaAvail = & $pythonExe -c "import torch; print(torch.cuda.is_available())" 2>&1
$gpuName = & $pythonExe -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
Write-Host "  CUDA available: $cudaAvail" -ForegroundColor Gray
Write-Host "  GPU: $gpuName" -ForegroundColor Gray

if ($cudaAvail -ne "True") {
    Write-Host "  WARNING: PyTorch has NO CUDA support! Packaged exe will not use GPU." -ForegroundColor Red
    Write-Host "  Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Yellow
    $continue = Read-Host "  Continue packaging anyway? (y/N)"
    if ($continue -ne "y") { exit 0 }
}

$version = (Get-Content (Join-Path $ProjectRoot "VERSION") -Raw).Trim()
Write-Host "  Version: $version" -ForegroundColor Gray

# Generate _AppVersion.iss from VERSION file
$versionParts = $version -split '\.'
while ($versionParts.Count -lt 4) { $versionParts += '0' }
$versionInfo = ($versionParts[0..3] -join '.')
$appVersionIss = @"
; 由 build_installer.ps1 根据仓库根目录 VERSION 生成（不要手动维护）
#define MyAppVersion "$version"
#define MyVersionInfo "$versionInfo"
"@
Set-Content (Join-Path $ProjectRoot "installer\_AppVersion.iss") $appVersionIss -Encoding UTF8
Write-Host "  Generated installer\_AppVersion.iss" -ForegroundColor Gray
Write-Host ""

# Step 1: Clean old build
if ($Clean) {
    Write-Host "[1/4] Cleaning old build files..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force (Join-Path $BackendDir "dist"), (Join-Path $BackendDir "build") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $ProjectRoot "build") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $RootExe, $RootInternal -ErrorAction SilentlyContinue
    Write-Host "  Cleaned" -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping clean (use -Clean to remove old files)" -ForegroundColor Gray
}
Write-Host ""

# Step 2: PyInstaller
if (-not $SkipPyInstaller) {
    Write-Host "[2/4] PyInstaller packaging..." -ForegroundColor Yellow
    Push-Location $BackendDir

    $pyinstallerArgs = @("NeuroLens.spec", "--noconfirm")
    if ($Debug) {
        $pyinstallerArgs += "--log-level=DEBUG"
    }

    & $pythonExe -m PyInstaller @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  PyInstaller FAILED!" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    Write-Host "  PyInstaller done" -ForegroundColor Green
} else {
    Write-Host "[2/4] Skipping PyInstaller" -ForegroundColor Gray
}
Write-Host ""

# Step 3: Verify build output at project root
Write-Host "[3/4] Verifying build output..." -ForegroundColor Yellow

if (-not (Test-Path $RootExe)) {
    Write-Host "  ERROR: Build output not found: $RootExe" -ForegroundColor Red
    Write-Host "  PyInstaller should write NeuroLens.exe to the project root (see NeuroLens.spec DISTPATH)." -ForegroundColor Gray
    exit 1
}
if (-not (Test-Path $RootInternal)) {
    Write-Host "  ERROR: Build output not found: $RootInternal" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: $RootExe" -ForegroundColor Green
Write-Host "  OK: $RootInternal" -ForegroundColor Green

$nerModelNames = @('chinese-ner-per-addr-rbt3', 'chinese-ner-per-addr')
$nerWeight = $null
foreach ($nerName in $nerModelNames) {
    $nerWeight = Get-ChildItem -Path (Join-Path $RootInternal "Desensization-dashboard\models\$nerName") -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -in @('model.safetensors', 'pytorch_model.bin') } |
      Select-Object -First 1
    if ($nerWeight) { break }
}
if (-not $nerWeight) {
    Write-Host "  ERROR: NER model weights missing under _internal\Desensization-dashboard\models\chinese-ner-per-addr-rbt3" -ForegroundColor Red
    Write-Host "  Expected model.safetensors or pytorch_model.bin from NeuroLens.spec." -ForegroundColor Gray
    exit 1
}
Write-Host "  OK: NER model ($($nerWeight.Name), $([math]::Round($nerWeight.Length / 1MB, 1)) MB)" -ForegroundColor Green

$rootModels = Join-Path $ProjectRoot "models"
if (Test-Path $rootModels) {
    $rootModelCount = (Get-ChildItem $rootModels -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "  OK: root models\ ($rootModelCount files, will be included in installer)" -ForegroundColor Green
} else {
    Write-Host "  WARN: root models\ not found; installer will skip it (skipifsourcedoesntexist)" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Inno Setup installer
if (-not $SkipInnoSetup) {
    Write-Host "[4/4] Building installer with Inno Setup..." -ForegroundColor Yellow

    $iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) {
        Write-Host "  WARNING: Inno Setup 6 not found, skipping installer" -ForegroundColor Yellow
        Write-Host "  Download: https://jrsoftware.org/isinfo.php" -ForegroundColor Gray
    } else {
        & $iscc (Join-Path $ProjectRoot "installer\NeuroLens_Setup.iss")
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Inno Setup FAILED!" -ForegroundColor Red
            exit 1
        }
        Write-Host "  Installer created: installer\Output\" -ForegroundColor Green
    }
} else {
    Write-Host "[4/4] Skipping Inno Setup" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BUILD COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output:" -ForegroundColor Yellow
Write-Host "  Main exe: $ProjectRoot\NeuroLens.exe" -ForegroundColor Gray
Write-Host "  Installer: $ProjectRoot\installer\Output\" -ForegroundColor Gray
