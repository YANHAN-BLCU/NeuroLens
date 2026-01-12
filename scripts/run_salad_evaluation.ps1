# PowerShell 脚本：在 Docker 容器内运行 SALAD-Bench 评估

param(
    [string]$ContainerName = "neurobreak-container",
    [string]$DataDir = "/workspace/data/salad/raw",
    [string]$OutputDir = "/workspace/logs",
    [string]$Config = "base_set",
    [int]$MaxSamples = 100,
    [int]$StartFrom = 0
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SALAD-Bench Evaluation Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Container: $ContainerName" -ForegroundColor Yellow
Write-Host "Data directory: $DataDir" -ForegroundColor Yellow
Write-Host "Output directory: $OutputDir" -ForegroundColor Yellow
Write-Host "Config: $Config" -ForegroundColor Yellow
Write-Host "Max samples: $MaxSamples" -ForegroundColor Yellow
Write-Host "Start from: $StartFrom" -ForegroundColor Yellow
Write-Host ""

# 检查容器是否存在
$containerExists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerExists) {
    Write-Host "Error: Container '$ContainerName' does not exist!" -ForegroundColor Red
    exit 1
}

# 检查容器是否运行
$containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "Warning: Container '$ContainerName' is not running, starting..." -ForegroundColor Yellow
    docker start $ContainerName
    Start-Sleep -Seconds 2
}

# 生成输出文件名
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = "$OutputDir/salad_evaluation_${Config}_${timestamp}.jsonl"

Write-Host "Running evaluation in container..." -ForegroundColor Green
Write-Host "Output file: $outputFile" -ForegroundColor Gray
Write-Host ""

# 在容器内运行评估脚本
docker exec $ContainerName bash -c "
    cd /workspace && \
    python scripts/evaluate_salad_pipeline.py \
        --data_dir $DataDir \
        --output $outputFile \
        --config $Config \
        --max_samples $MaxSamples \
        --start_from $StartFrom
"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Evaluation complete!" -ForegroundColor Green
    Write-Host "Results saved to: $outputFile" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Error: Evaluation failed!" -ForegroundColor Red
    exit 1
}

