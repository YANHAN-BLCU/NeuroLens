# PowerShell 脚本：在 Docker 容器内删除模型文件
# 使用方法: .\scripts\remove_models_from_container.ps1 [--container <container_name>]

param(
    [string]$ContainerName = "neurobreak-container",
    [switch]$Confirm = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "在 Docker 容器内删除模型文件" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "容器名称: $ContainerName" -ForegroundColor Yellow
Write-Host ""

# 检查容器是否存在
$containerExists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerExists) {
    Write-Host "错误: 容器 '$ContainerName' 不存在！" -ForegroundColor Red
    Write-Host "请检查容器名称或先启动容器。" -ForegroundColor Yellow
    exit 1
}

# 检查容器是否运行
$containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "警告: 容器 '$ContainerName' 未运行。" -ForegroundColor Yellow
    $start = Read-Host "是否启动容器? (y/n)"
    if ($start -eq "y" -or $start -eq "Y") {
        Write-Host "启动容器..." -ForegroundColor Yellow
        docker start $ContainerName
        Start-Sleep -Seconds 2
    } else {
        Write-Host "无法在未运行的容器中执行命令。" -ForegroundColor Red
        exit 1
    }
}

# 检查脚本是否存在
$scriptExists = docker exec $ContainerName test -f /workspace/scripts/remove_models.sh
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: 容器内未找到 remove_models.sh 脚本" -ForegroundColor Yellow
    Write-Host "尝试直接执行删除命令..." -ForegroundColor Yellow
    
    # 直接执行删除命令
    $models = @(
        "meta-llama_Llama-3.2-3B-Instruct",
        "meta-llama_Llama-Guard-3-1B",
        "Meta-Llama-3-8B-Instruct",
        "Llama-Guard-3-8B"
    )
    
    $paths = @(
        "/cache",
        "/workspace/hf_models",
        "/workspace/models"
    )
    
    Write-Host ""
    Write-Host "查找模型文件..." -ForegroundColor Yellow
    
    foreach ($model in $models) {
        foreach ($path in $paths) {
            $fullPath = "$path/$model"
            $exists = docker exec $ContainerName test -d $fullPath 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "找到: $fullPath" -ForegroundColor Green
                
                if (-not $Confirm) {
                    $response = Read-Host "删除 $fullPath? (y/n)"
                    if ($response -ne "y" -and $response -ne "Y") {
                        continue
                    }
                }
                
                Write-Host "删除: $fullPath" -ForegroundColor Yellow
                docker exec $ContainerName rm -rf $fullPath
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  ✓ 已删除: $model" -ForegroundColor Green
                } else {
                    Write-Host "  ✗ 删除失败: $model" -ForegroundColor Red
                }
            }
        }
    }
} else {
    # 使用脚本删除
    Write-Host "在容器内执行删除脚本..." -ForegroundColor Yellow
    
    $confirmFlag = ""
    if ($Confirm) {
        $confirmFlag = "--confirm"
    }
    
    docker exec -it $ContainerName bash /workspace/scripts/remove_models.sh $confirmFlag
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "完成" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

