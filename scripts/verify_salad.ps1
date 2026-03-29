# PowerShell 脚本：验证 SALAD-Bench 数据集
# 使用方法: .\scripts\verify_salad.ps1 [数据目录路径]

param(
    [string]$DataDir = "data/salad/raw"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "验证 SALAD-Bench 数据集" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "数据目录: $DataDir" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $DataDir)) {
    Write-Host "[ERROR] 目录不存在: $DataDir" -ForegroundColor Red
    exit 1
}

$files = Get-ChildItem -Path $DataDir -Filter "*.jsonl" -ErrorAction SilentlyContinue

if ($files.Count -eq 0) {
    Write-Host "[WARNING] 未找到 JSONL 文件" -ForegroundColor Yellow
    exit 1
}

Write-Host "找到 $($files.Count) 个文件:" -ForegroundColor Green
Write-Host ""

$totalSamples = 0
$totalSize = 0

foreach ($file in $files) {
    $sizeMB = [math]::Round($file.Length / 1MB, 2)
    $totalSize += $file.Length
    
    # 计算行数（样本数）
    $lineCount = (Get-Content $file.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    
    Write-Host "文件: $($file.Name)" -ForegroundColor White
    Write-Host "  大小: $sizeMB MB" -ForegroundColor Gray
    Write-Host "  样本数: $lineCount" -ForegroundColor Gray
    Write-Host "  修改时间: $($file.LastWriteTime)" -ForegroundColor Gray
    Write-Host ""
    
    $totalSamples += $lineCount
}

$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "总计:" -ForegroundColor Cyan
Write-Host "  文件数: $($files.Count)" -ForegroundColor White
Write-Host "  总样本数: $totalSamples" -ForegroundColor White
Write-Host "  总大小: $totalSizeMB MB" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

# 验证容器中的数据（如果容器存在）
$containerName = "neurobreak-container"
$containerExists = docker ps -a --filter "name=$containerName" --format "{{.Names}}" 2>$null

if ($containerExists) {
    Write-Host ""
    Write-Host "检查容器中的数据..." -ForegroundColor Cyan
    docker exec $containerName ls -lh /workspace/data/salad/raw/ 2>$null | Select-String "jsonl"
}

