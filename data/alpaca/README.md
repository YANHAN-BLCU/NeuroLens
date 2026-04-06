# Stanford Alpaca 数据集

## 简介

Stanford Alpaca 是一个用于指令微调的数据集，包含约 52,000 个指令-响应对。

- **来源**: [Stanford Alpaca GitHub](https://github.com/tatsu-lab/stanford_alpaca)
- **格式**: JSON/JSONL
- **样本数**: ~52,000

## 下载方法

### 方法 1: 使用下载脚本（推荐）

#### 本地环境

```bash
# 下载到默认位置 data/alpaca/
python scripts/download_alpaca.py

# 指定输出目录
python scripts/download_alpaca.py --output /path/to/alpaca

# 只下载 JSON，不转换为 JSONL
python scripts/download_alpaca.py --no-convert

# 保留原始 JSON 文件
python scripts/download_alpaca.py --keep-json
```

#### Docker 容器内

**方法 A: 使用 PowerShell 脚本（Windows 推荐）**

```powershell
# 使用默认设置下载
.\scripts\download_alpaca_docker.ps1

# 指定容器名称
.\scripts\download_alpaca_docker.ps1 -ContainerName "your-container-name"

# 保留原始 JSON 文件
.\scripts\download_alpaca_docker.ps1 -KeepJson

# 只下载，不转换格式
.\scripts\download_alpaca_docker.ps1 -NoConvert
```

**方法 B: 直接使用 Docker 命令**

```bash
# 从宿主机直接执行（推荐）
docker exec neurolens python /workspace/scripts/download_alpaca.py --yes

# 或进入容器内执行
docker exec -it neurolens /bin/bash
python scripts/download_alpaca.py --yes

# 指定容器内路径
python scripts/download_alpaca.py --output /workspace/data/alpaca --yes
```

**Docker 注意事项**:
- 使用 `--yes` 参数避免交互式提示（Docker 非交互式环境）
- 数据会下载到容器内的 `/workspace/data/alpaca/`
- 如果挂载了 volume，数据会持久化到宿主机

### 方法 2: 手动下载

1. 访问 [Stanford Alpaca GitHub](https://github.com/tatsu-lab/stanford_alpaca)
2. 下载 `alpaca_data.json` 文件
3. 放置到 `data/alpaca/` 目录

## 数据集格式

### 原始格式 (JSON)

```json
[
  {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced diet..."
  },
  ...
]
```

### 转换后格式 (JSONL)

每行一个 JSON 对象，适配代码库的数据格式要求：

```json
{"input": {"prompt": "Give three tips for staying healthy.\n\n"}}
{"input": {"prompt": "Translate the following sentence to French.\n\nHello, how are you?"}}
```

## 使用示例

### 本地环境

#### 计算效用神经元

```bash
python scripts/run_snip_scorer.py \
    --model-path /path/to/model \
    --dataset-path data/alpaca/alpaca_data.jsonl \
    --output-path outputs/utility_neurons \
    --mode utility \
    --batch-size 8 \
    --num-samples 0
```

#### 快速测试（使用部分样本）

```bash
python scripts/run_snip_scorer.py \
    --model-path /path/to/model \
    --dataset-path data/alpaca/alpaca_data.jsonl \
    --output-path outputs/utility_neurons \
    --mode utility \
    --batch-size 8 \
    --num-samples 1000
```

### Docker 容器内

#### 完整流程（下载 + 计算）

```bash
# 1. 下载数据集（从宿主机执行）
docker exec neurolens python /workspace/scripts/download_alpaca.py --yes

# 2. 计算效用神经元（从宿主机执行）
docker exec neurolens python /workspace/scripts/run_snip_scorer.py \
    --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --dataset-path /workspace/data/alpaca/alpaca_data.jsonl \
    --output-path /workspace/outputs/utility_neurons \
    --mode utility \
    --batch-size 8 \
    --num-samples 0

# 或者进入容器内执行
docker exec -it neurolens /bin/bash
python scripts/download_alpaca.py --yes
python scripts/run_snip_scorer.py \
    --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
    --dataset-path /workspace/data/alpaca/alpaca_data.jsonl \
    --output-path /workspace/outputs/utility_neurons \
    --mode utility \
    --batch-size 8 \
    --num-samples 0
```

## 文件说明

- `alpaca_data.json`: 原始 JSON 格式数据集
- `alpaca_data.jsonl`: 转换后的 JSONL 格式数据集（用于代码库）

## 注意事项

1. **数据集大小**: Alpaca 数据集包含约 52K 样本，完整计算可能需要数小时
2. **内存要求**: 建议至少 32GB RAM
3. **格式转换**: 脚本会自动将原始 JSON 格式转换为代码库需要的 JSONL 格式
4. **网络要求**: 首次下载需要网络连接，文件大小约 4-5 MB
5. **Docker 环境**: 
   - 使用 `--yes` 参数避免交互式提示
   - 确保容器有网络访问权限（下载需要）
   - 数据路径使用容器内路径（如 `/workspace/data/alpaca/`）

## 参考链接

- [Stanford Alpaca GitHub](https://github.com/tatsu-lab/stanford_alpaca)
- [Alpaca 数据集说明](https://github.com/tatsu-lab/stanford_alpaca#data-release)
