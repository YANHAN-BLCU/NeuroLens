# PowerShell 脚本：同步文件到 Docker 容器
# 使用方法: .\scripts\sync_to_docker.ps1

param(
    [string]$ContainerName = "neurobreak-container",
    [string]$WorkspacePath = "/workspace"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "同步文件到 Docker 容器" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "容器名称: $ContainerName" -ForegroundColor Yellow
Write-Host "工作目录: $WorkspacePath" -ForegroundColor Yellow
Write-Host ""

# 检查容器是否存在
$containerExists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerExists) {
    Write-Host "错误: 容器 '$ContainerName' 不存在！" -ForegroundColor Red
    Write-Host "请先启动容器或检查容器名称。" -ForegroundColor Red
    exit 1
}

# 检查容器是否运行
$containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "警告: 容器 '$ContainerName' 未运行，正在启动..." -ForegroundColor Yellow
    docker start $ContainerName
    Start-Sleep -Seconds 2
}

Write-Host "正在同步脚本文件..." -ForegroundColor Green

# 同步脚本文件
$scripts = @(
    "scripts/run_io_tests.py",
    "scripts/download_models.py",
    "scripts/check_models.py"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Write-Host "  复制 $script ..." -ForegroundColor Gray
        docker cp $script "${ContainerName}:${WorkspacePath}/$script"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $script 已同步" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $script 同步失败" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⚠ $script 不存在，跳过" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "验证文件..." -ForegroundColor Green

# 验证文件
docker exec $ContainerName ls -lh "${WorkspacePath}/scripts/" | Select-String -Pattern "run_io_tests|download_models|check_models"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "同步完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "现在可以在容器内运行:" -ForegroundColor Yellow
Write-Host "  docker exec -it $ContainerName /bin/bash" -ForegroundColor White
Write-Host "  python scripts/run_io_tests.py" -ForegroundColor White

