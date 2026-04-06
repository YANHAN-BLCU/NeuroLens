# PowerShell 脚本：运行 SNIP Scorer 计算神经元分数

param(
    [string]$ContainerName = "neurobreak-container",
    [string]$ModelPath = "",
    [string]$DatasetPath = "",
    [string]$OutputPath = "/workspace/outputs/snip_scores",
    [int]$BatchSize = 8,
    [int]$NumSamples = 0,
    [string]$Mode = "safety",  # "safety" 或 "utility"
    [switch]$RunInContainer = $true,
    [switch]$RunLocally = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SNIP Scorer - 神经元分数计算" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "模式: $Mode" -ForegroundColor Yellow
Write-Host "批大小: $BatchSize" -ForegroundColor Yellow
if ($NumSamples -gt 0) {
    Write-Host "样本数: $NumSamples" -ForegroundColor Yellow
} else {
    Write-Host "样本数: 全部" -ForegroundColor Yellow
}
Write-Host ""

# 如果指定了本地运行
if ($RunLocally) {
    $RunInContainer = $false
}

if ($RunInContainer) {
    # 检查容器是否存在
    $containerExists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"
    if (-not $containerExists) {
        Write-Host "Error: Container '$ContainerName' does not exist!" -ForegroundColor Red
        Write-Host "提示: 使用 -RunLocally 参数在本地运行，或先创建容器" -ForegroundColor Yellow
        exit 1
    }
    
    # 检查容器是否运行
    $containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
    if (-not $containerRunning) {
        Write-Host "Warning: Container '$ContainerName' is not running, starting..." -ForegroundColor Yellow
        docker start $ContainerName
        Start-Sleep -Seconds 2
    }
    
    Write-Host "在容器 '$ContainerName' 中运行 SNIP Scorer..." -ForegroundColor Green
    Write-Host ""
    
    # 构建 Python 命令
    $pythonCmd = @"
import sys
import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

# 添加项目路径
sys.path.insert(0, '/workspace')

from engine.neurons.snip_scorer import compute_snip_scores
from engine.neurons.safety_identifier import identify_safety_neurons, default_safety_loss_fn
from engine.neurons.utility_identifier import identify_utility_neurons, default_utility_loss_fn

# 配置
model_path = os.getenv('MODEL_PATH', '$ModelPath')
dataset_path = os.getenv('DATASET_PATH', '$DatasetPath')
output_path = Path('$OutputPath')
batch_size = $BatchSize
num_samples = $NumSamples if $NumSamples > 0 else None
mode = '$Mode'

print(f'[SNIP Scorer] 模型路径: {model_path}')
print(f'[SNIP Scorer] 数据集路径: {dataset_path}')
print(f'[SNIP Scorer] 输出路径: {output_path}')
print(f'[SNIP Scorer] 模式: {mode}')
print()

# 检查路径
if not model_path or not Path(model_path).exists():
    print(f'Error: 模型路径不存在: {model_path}')
    sys.exit(1)

if not dataset_path or not Path(dataset_path).exists():
    print(f'Error: 数据集路径不存在: {dataset_path}')
    sys.exit(1)

# 加载模型和分词器
print('[SNIP Scorer] 加载模型...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[SNIP Scorer] 使用设备: {device}')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto' if torch.cuda.is_available() else None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
print('[SNIP Scorer] 加载数据集...')
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.texts.append(line)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {'text': self.texts[idx]}

dataset = TextDataset(dataset_path)
print(f'[SNIP Scorer] 数据集大小: {len(dataset)}')

# 根据模式选择函数
if mode == 'safety':
    print('[SNIP Scorer] 计算安全神经元 SNIP 分数...')
    neurons = identify_safety_neurons(
        model=model,
        tokenizer=tokenizer,
        benign_dataset=dataset,
        device=device,
        safety_threshold_q=0.005,  # 0.5%
        batch_size=batch_size,
        num_samples=num_samples,
    )
    output_file = output_path / 'safety_neurons.json'
elif mode == 'utility':
    print('[SNIP Scorer] 计算效用神经元 SNIP 分数...')
    neurons = identify_utility_neurons(
        model=model,
        tokenizer=tokenizer,
        utility_dataset=dataset,
        device=device,
        utility_threshold_p=0.001,  # 0.1%
        batch_size=batch_size,
        num_samples=num_samples,
    )
    output_file = output_path / 'utility_neurons.json'
else:
    print(f'Error: 未知模式: {mode}')
    sys.exit(1)

# 保存结果
output_path.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(neurons, f, indent=2, ensure_ascii=False)

print(f'[SNIP Scorer] 结果已保存到: {output_file}')
print(f'[SNIP Scorer] 识别到 {len(neurons)} 个神经元')
"@
    
    # 在容器内运行
    $pythonCmd | docker exec -i $ContainerName python -
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "SNIP Scorer 完成!" -ForegroundColor Green
        Write-Host "结果保存在容器内: $OutputPath" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Error: SNIP Scorer 执行失败!" -ForegroundColor Red
        exit 1
    }
} else {
    # 本地运行
    Write-Host "在本地运行 SNIP Scorer..." -ForegroundColor Green
    Write-Host ""
    Write-Host "提示: 请确保已安装所有依赖并配置好环境" -ForegroundColor Yellow
    Write-Host ""
    
    # 创建临时 Python 脚本
    $tempScript = Join-Path $env:TEMP "run_snip_scorer_$(Get-Date -Format 'yyyyMMdd_HHmmss').py"
    
    $pythonScript = @"
import sys
import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

# 添加项目路径
project_root = Path(r'$PSScriptRoot').Parent.FullName
sys.path.insert(0, str(project_root))

from engine.neurons.snip_scorer import compute_snip_scores
from engine.neurons.safety_identifier import identify_safety_neurons
from engine.neurons.utility_identifier import identify_utility_neurons

# 配置
model_path = r'$ModelPath'
dataset_path = r'$DatasetPath'
output_path = Path(r'$OutputPath')
batch_size = $BatchSize
num_samples = $NumSamples if $NumSamples > 0 else None
mode = '$Mode'

print(f'[SNIP Scorer] 模型路径: {model_path}')
print(f'[SNIP Scorer] 数据集路径: {dataset_path}')
print(f'[SNIP Scorer] 输出路径: {output_path}')
print(f'[SNIP Scorer] 模式: {mode}')
print()

# 检查路径
if not model_path or not Path(model_path).exists():
    print(f'Error: 模型路径不存在: {model_path}')
    sys.exit(1)

if not dataset_path or not Path(dataset_path).exists():
    print(f'Error: 数据集路径不存在: {dataset_path}')
    sys.exit(1)

# 加载模型和分词器
print('[SNIP Scorer] 加载模型...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[SNIP Scorer] 使用设备: {device}')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto' if torch.cuda.is_available() else None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
print('[SNIP Scorer] 加载数据集...')
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.texts.append(line)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {'text': self.texts[idx]}

dataset = TextDataset(dataset_path)
print(f'[SNIP Scorer] 数据集大小: {len(dataset)}')

# 根据模式选择函数
if mode == 'safety':
    print('[SNIP Scorer] 计算安全神经元 SNIP 分数...')
    neurons = identify_safety_neurons(
        model=model,
        tokenizer=tokenizer,
        benign_dataset=dataset,
        device=device,
        safety_threshold_q=0.005,
        batch_size=batch_size,
        num_samples=num_samples,
    )
    output_file = output_path / 'safety_neurons.json'
elif mode == 'utility':
    print('[SNIP Scorer] 计算效用神经元 SNIP 分数...')
    neurons = identify_utility_neurons(
        model=model,
        tokenizer=tokenizer,
        utility_dataset=dataset,
        device=device,
        utility_threshold_p=0.001,
        batch_size=batch_size,
        num_samples=num_samples,
    )
    output_file = output_path / 'utility_neurons.json'
else:
    print(f'Error: 未知模式: {mode}')
    sys.exit(1)

# 保存结果
output_path.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(neurons, f, indent=2, ensure_ascii=False)

print(f'[SNIP Scorer] 结果已保存到: {output_file}')
print(f'[SNIP Scorer] 识别到 {len(neurons)} 个神经元')
"@
    
    $pythonScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    Write-Host "运行 Python 脚本: $tempScript" -ForegroundColor Gray
    python $tempScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "SNIP Scorer 完成!" -ForegroundColor Green
        Write-Host "结果保存在: $OutputPath" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Error: SNIP Scorer 执行失败!" -ForegroundColor Red
    }
    
    # 清理临时文件
    if (Test-Path $tempScript) {
        Remove-Item $tempScript -Force
    }
}
