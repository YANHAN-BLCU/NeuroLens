# Sync NeuroBreak-Reproduction files to Docker container (PowerShell)
# Usage: .\scripts\sync_to_container.ps1 [container_name]

param(
    [string]$ContainerName = "neurobreak-container"
)

Write-Host "Syncing files to container: $ContainerName" -ForegroundColor Green

# Check if container is running
$containerRunning = docker ps --format "{{.Names}}" | Select-String -Pattern "^$ContainerName$"
if (-not $containerRunning) {
    Write-Host "Error: Container $ContainerName is not running" -ForegroundColor Red
    exit 1
}

# Key files to sync
$files = @(
    "engine/assessment/evaluate.py",
    "engine/assessment/report.py",
    "scripts/download_salad.py",
    "scripts/eval_utility.py",
    "scripts/verify_salad.py",
    "scripts/download_models.py",
    "configs/runtime/salad.yaml"
)

# Sync individual files
Write-Host "Syncing files..." -ForegroundColor Yellow
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  Copying $file" -ForegroundColor Cyan
        docker cp $file "${ContainerName}:/workspace/$file"
    } else {
        Write-Host "  Warning: $file not found" -ForegroundColor Yellow
    }
}

Write-Host "Sync complete!" -ForegroundColor Green
Write-Host "Verifying files in container..." -ForegroundColor Yellow

# Verify key files
docker exec $ContainerName bash -c @"
    echo 'Checking files...'
    test -f /workspace/engine/assessment/evaluate.py && echo '  ✓ evaluate.py' || echo '  ✗ evaluate.py missing'
    test -f /workspace/engine/assessment/report.py && echo '  ✓ report.py' || echo '  ✗ report.py missing'
    test -f /workspace/scripts/download_salad.py && echo '  ✓ download_salad.py' || echo '  ✗ download_salad.py missing'
    test -f /workspace/scripts/eval_utility.py && echo '  ✓ eval_utility.py' || echo '  ✗ eval_utility.py missing'
    test -f /workspace/configs/runtime/salad.yaml && echo '  ✓ salad.yaml' || echo '  ✗ salad.yaml missing'
"@

