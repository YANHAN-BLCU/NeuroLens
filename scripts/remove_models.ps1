# PowerShell 脚本：删除本地模型文件
# 使用方法: .\scripts\remove_models.ps1 [--confirm]

param(
    [switch]$Confirm = $false,
    [string[]]$Models = @()
)

# 默认要删除的模型
$DefaultModels = @(
    "meta-llama_Llama-3.2-3B-Instruct",
    "meta-llama_Llama-Guard-3-1B",
    "Meta-Llama-3-8B-Instruct",
    "Llama-Guard-3-8B"
)

# 如果指定了模型列表，使用指定的；否则使用默认的
if ($Models.Count -eq 0) {
    $Models = $DefaultModels
}

# 可能的模型路径
$ModelPaths = @(
    "F:/models",
    "$env:USERPROFILE\.cache\huggingface\models",
    "$env:HF_HOME\models",
    "$env:TRANSFORMERS_CACHE\models",
    "hf_models"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Remove Local Model Files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 收集要删除的模型路径
$ModelsToDelete = @()

foreach ($modelName in $Models) {
    Write-Host "Searching for model: $modelName" -ForegroundColor Yellow
    
    foreach ($basePath in $ModelPaths) {
        $modelPath = Join-Path $basePath $modelName
        
        # 检查项目内的 hf_models 目录
        if ($basePath -eq "hf_models") {
            # 获取脚本所在目录的父目录（项目根目录）
            if ($PSScriptRoot) {
                $projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
            } else {
                # 如果 PSScriptRoot 不可用，使用当前工作目录
                $projectRoot = Get-Location
                if ($projectRoot.Path -like "*scripts*") {
                    $projectRoot = Split-Path -Parent $projectRoot.Path
                }
            }
            $modelPath = Join-Path (Join-Path $projectRoot "hf_models") $modelName
        }
        
        if (Test-Path $modelPath) {
            $size = (Get-ChildItem -Path $modelPath -Recurse -File -ErrorAction SilentlyContinue | 
                     Measure-Object -Property Length -Sum).Sum / 1GB
            $sizeRounded = [math]::Round($size, 2)
            Write-Host "  Found: $modelPath (Size: $sizeRounded GB)" -ForegroundColor Green
            $ModelsToDelete += @{
                Path = $modelPath
                Name = $modelName
                Size = $size
            }
        }
    }
}

if ($ModelsToDelete.Count -eq 0) {
    Write-Host "No model files found." -ForegroundColor Yellow
    exit 0
}

# 显示要删除的模型
Write-Host ""
Write-Host "Models to be deleted:" -ForegroundColor Yellow
$totalSize = 0
foreach ($model in $ModelsToDelete) {
    $modelSizeRounded = [math]::Round($model.Size, 2)
    Write-Host "  - $($model.Name): $($model.Path) ($modelSizeRounded GB)" -ForegroundColor Red
    $totalSize += $model.Size
}
Write-Host ""
$totalSizeRounded = [math]::Round($totalSize, 2)
Write-Host "Total size: $totalSizeRounded GB" -ForegroundColor Red
Write-Host ""

# 确认删除
if (-not $Confirm) {
    $response = Read-Host "Confirm deletion? (yes/no)"
    if ($response -ne "yes" -and $response -ne "y") {
        Write-Host "Deletion cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# 执行删除
Write-Host ""
Write-Host "Starting deletion..." -ForegroundColor Yellow
$deletedCount = 0
$failedCount = 0

foreach ($model in $ModelsToDelete) {
    try {
        Write-Host "Deleting: $($model.Path)" -ForegroundColor Yellow
        Remove-Item -Path $model.Path -Recurse -Force -ErrorAction Stop
        Write-Host "  [OK] Deleted: $($model.Name)" -ForegroundColor Green
        $deletedCount++
    }
    catch {
        Write-Host "  [FAIL] Failed to delete: $($model.Name) - $_" -ForegroundColor Red
        $failedCount++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deletion Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Successfully deleted: $deletedCount model(s)" -ForegroundColor Green
if ($failedCount -gt 0) {
    Write-Host "Failed to delete: $failedCount model(s)" -ForegroundColor Red
}
$finalSizeRounded = [math]::Round($totalSize, 2)
Write-Host "Space freed: $finalSizeRounded GB" -ForegroundColor Green

