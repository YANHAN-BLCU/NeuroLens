# PowerShell 脚本：在 Docker 容器内下载 Alpaca 数据集
# 使用方法: .\scripts\download_alpaca_docker.ps1

param(
    [string]$ContainerName = "neurolens",
    [string]$OutputPath = "/workspace/data/alpaca",
    [switch]$KeepJson = $false,
    [switch]$NoConvert = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "在 Docker 容器内下载 Alpaca 数据集" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "容器名称: $ContainerName" -ForegroundColor Yellow
Write-Host "输出路径: $OutputPath" -ForegroundColor Yellow
Write-Host ""

# 检查容器是否存在
$containerExists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerExists) {
    Write-Host "错误: 容器 '$ContainerName' 不存在！" -ForegroundColor Red
    Write-Host "请先启动容器，或检查容器名称是否正确。" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "启动容器示例:" -ForegroundColor Cyan
    Write-Host "  .\scripts\run_container.ps1" -ForegroundColor White
    exit 1
}

# 检查容器是否运行
$containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "容器未运行，正在启动..." -ForegroundColor Yellow
    docker start $ContainerName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 无法启动容器 '$ContainerName'" -ForegroundColor Red
        exit 1
    }
    Start-Sleep -Seconds 2
}

Write-Host "检查脚本是否存在..." -ForegroundColor Green
$scriptExists = docker exec $ContainerName test -f /workspace/scripts/download_alpaca.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 下载脚本不存在于容器内" -ForegroundColor Yellow
    Write-Host "正在同步脚本到容器..." -ForegroundColor Yellow
    
    # 检查本地脚本是否存在
    $localScript = Join-Path $PSScriptRoot "download_alpaca.py"
    if (-not (Test-Path $localScript)) {
        Write-Host "错误: 本地脚本不存在: $localScript" -ForegroundColor Red
        exit 1
    }
    
    docker cp $localScript "${ContainerName}:/workspace/scripts/download_alpaca.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 无法同步脚本到容器" -ForegroundColor Red
        exit 1
    }
    Write-Host "脚本同步完成" -ForegroundColor Green
}

# 构建命令参数
$args = @("--yes", "--output", $OutputPath)
if ($KeepJson) {
    $args += "--keep-json"
}
if ($NoConvert) {
    $args += "--no-convert"
}

Write-Host ""
Write-Host "开始下载数据集..." -ForegroundColor Green
Write-Host "执行命令: python /workspace/scripts/download_alpaca.py $($args -join ' ')" -ForegroundColor Cyan
Write-Host ""

# 执行下载命令
docker exec $ContainerName python /workspace/scripts/download_alpaca.py $args

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "下载完成！" -ForegroundColor Green
    Write-Host ""
    Write-Host "数据集位置: ${ContainerName}:$OutputPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使用示例:" -ForegroundColor Yellow
    Write-Host "  docker exec $ContainerName python /workspace/scripts/run_snip_scorer.py \`" -ForegroundColor White
    Write-Host "      --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \`" -ForegroundColor White
    Write-Host "      --dataset-path $OutputPath/alpaca_data.jsonl \`" -ForegroundColor White
    Write-Host "      --output-path /workspace/outputs/utility_neurons \`" -ForegroundColor White
    Write-Host "      --mode utility \`" -ForegroundColor White
    Write-Host "      --batch-size 8 \`" -ForegroundColor White
    Write-Host "      --num-samples 0" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "下载失败！" -ForegroundColor Red
    Write-Host "请检查:" -ForegroundColor Yellow
    Write-Host "  1. 容器是否有网络访问权限" -ForegroundColor White
    Write-Host "  2. 容器内是否有足够的磁盘空间" -ForegroundColor White
    Write-Host "  3. 查看容器日志: docker logs $ContainerName" -ForegroundColor White
    exit 1
}
