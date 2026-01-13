# PowerShell 脚本：使用 neurolens 镜像启动 Docker 容器
# 使用方法: .\scripts\run_container.ps1

param(
    [string]$ImageName = "neurolens:v1.0",
    [string]$ContainerName = "neurobreak-container",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 4173,
    [string]$ModelPath = "F:/models",
    [switch]$Detached = $false
)

# 获取当前脚本所在目录的父目录（项目根目录）
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "启动 Docker 容器" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "镜像名称: $ImageName" -ForegroundColor Yellow
Write-Host "容器名称: $ContainerName" -ForegroundColor Yellow
Write-Host "项目目录: $ProjectRoot" -ForegroundColor Yellow
Write-Host "后端端口: $BackendPort" -ForegroundColor Yellow
Write-Host "前端端口: $FrontendPort" -ForegroundColor Yellow
Write-Host ""

# 检查镜像是否存在
Write-Host "检查 Docker 镜像..." -ForegroundColor Green
$imageExists = docker images --format "{{.Repository}}" | Select-String -Pattern "^$ImageName$"
if (-not $imageExists) {
    Write-Host "警告: 镜像 '$ImageName' 不存在！" -ForegroundColor Yellow
    Write-Host "请先拉取或构建镜像，或检查镜像名称是否正确。" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "是否继续尝试启动容器? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

# 检查容器是否已存在
$containerExists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
if ($containerExists) {
    Write-Host "警告: 容器 '$ContainerName' 已存在！" -ForegroundColor Yellow
    Write-Host ""
    $action = Read-Host "选择操作: [s]top and remove / [r]estart / [c]ancel (s/r/c)"
    
    switch ($action.ToLower()) {
        "s" {
            Write-Host "停止并删除现有容器..." -ForegroundColor Yellow
            docker stop $ContainerName 2>$null
            docker rm $ContainerName 2>$null
            Write-Host "容器已删除" -ForegroundColor Green
        }
        "r" {
            Write-Host "重启现有容器..." -ForegroundColor Yellow
            docker restart $ContainerName
            Write-Host "容器已重启" -ForegroundColor Green
            Write-Host ""
            Write-Host "进入容器:" -ForegroundColor Cyan
            Write-Host "  docker exec -it $ContainerName /bin/bash" -ForegroundColor White
            exit 0
        }
        default {
            Write-Host "已取消操作" -ForegroundColor Yellow
            exit 0
        }
    }
}

# 构建 Docker 运行命令
Write-Host "启动容器..." -ForegroundColor Green

# Windows 路径转换为 Docker 挂载格式
# 使用绝对路径并转换为Unix风格
$workspaceMount = "${ProjectRoot}:/workspace"
$workspaceMount = $workspaceMount -replace '\\', '/'
$cacheMount = "/workspace/.cache"

# 构建命令
$dockerArgs = @(
    "run"
)

# 根据是否后台运行选择参数
if ($Detached) {
    $dockerArgs += "-d"
} else {
    $dockerArgs += "-it"
}

$dockerArgs += @(
    "--gpus", "all",
    "--name", $ContainerName,
    "-p", "${BackendPort}:8000",
    "-p", "${FrontendPort}:4173",
    "-v", $workspaceMount,
    "-v", "${cacheMount}:/workspace/.cache"
)

# 如果提供了模型路径，添加模型挂载
if ($ModelPath -and (Test-Path $ModelPath)) {
    Write-Host "检测到模型路径: $ModelPath" -ForegroundColor Green
    $modelMount = $ModelPath -replace '\\', '/'
    $dockerArgs += "-v"
    $dockerArgs += "${modelMount}:/workspace/models"
}

# 添加镜像名称和启动命令
$dockerArgs += $ImageName

if ($Detached) {
    # 后台运行时使用sleep保持容器运行
    $dockerArgs += "sleep", "infinity"
} else {
    # 交互式运行时直接进入bash
    $dockerArgs += "/bin/bash"
}

# 显示完整命令
Write-Host ""
Write-Host "执行命令:" -ForegroundColor Cyan
Write-Host "  docker $($dockerArgs -join ' ')" -ForegroundColor White
Write-Host ""

# 执行 Docker 命令
docker $dockerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "容器已启动！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "常用命令:" -ForegroundColor Yellow
    Write-Host "  进入容器: docker exec -it $ContainerName /bin/bash" -ForegroundColor White
    Write-Host "  停止容器: docker stop $ContainerName" -ForegroundColor White
    Write-Host "  启动容器: docker start $ContainerName" -ForegroundColor White
    Write-Host "  删除容器: docker rm -f $ContainerName" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "容器启动失败！" -ForegroundColor Red
    Write-Host "请检查错误信息并重试。" -ForegroundColor Red
    exit 1
}

