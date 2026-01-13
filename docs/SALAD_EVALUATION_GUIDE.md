# SALAD-Bench 评估指南

本指南说明如何使用 Llama-3.2-3B-Instruct 和 Llama-Guard-3-1B 评估 SALAD-Bench 数据集。

## 概述

评估流程：
1. 从 SALAD-Bench 数据集加载样本
2. 使用 Llama-3.2-3B-Instruct 生成响应
3. 使用 Llama-Guard-3-1B 判断响应安全性
4. 生成包含所有信息的联合输出文件

## 前置要求

1. **数据集已下载**：确保 SALAD-Bench 数据已下载到 `data/salad/raw/`
2. **模型已准备**：确保模型在容器内可用（`/cache/` 或 `/workspace/models/`）
3. **Docker 容器运行中**：确保 `neurobreak-container` 容器正在运行

## 使用方法

### 方法 1：在容器内直接运行（推荐）

```bash
# 进入容器
docker exec -it neurobreak-container /bin/bash

# 运行评估脚本
cd /workspace
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/salad_evaluation.jsonl \
    --config base_set \
    --max_samples 100
```

### 方法 2：使用 PowerShell 脚本（Windows）

```powershell
# 在项目根目录运行
.\scripts\run_salad_evaluation.ps1 \
    -ContainerName neurobreak-container \
    -Config base_set \
    -MaxSamples 100
```

### 方法 3：使用 Bash 脚本（Linux/Mac）

```bash
# 在容器内运行
docker exec neurobreak-container bash /workspace/scripts/run_salad_evaluation.sh
```

## 参数说明

- `--data_dir`: SALAD-Bench 数据目录（默认：`data/salad/raw`）
- `--output`: 输出 JSONL 文件路径（必需）
- `--config`: 配置名称，可选值：
  - `base_set`: 基础数据集（21,318 样本）
  - `attack_enhanced_set`: 攻击增强集（5,000 样本）
  - `defense_enhanced_set`: 防御增强集（200 样本）
  - `mcq_set`: 多选题集（3,840 样本）
  - `None`: 加载所有配置
- `--max_samples`: 最大样本数（默认：None，处理所有样本）
- `--start_from`: 从第几个样本开始（用于断点续传）

## 输出格式

输出文件为 JSONL 格式，每行一个 JSON 对象，包含：

```json
{
  "sample_id": "样本ID",
  "config": "配置名称",
  "timestamp": "时间戳",
  "input": {
    "prompt": "输入提示",
    "original_sample": { /* 原始样本数据 */ }
  },
  "inference": {
    "output": "模型生成的响应",
    "tokens": {
      "input": 输入token数,
      "output": 输出token数
    },
    "latency_ms": 生成延迟（毫秒）
  },
  "guard": {
    "verdict": "allow|flag|block",
    "severity": "low|medium|high",
    "rationale": ["判断理由"],
    "categories": [ /* 类别分数 */ ],
    "latency_ms": Guard延迟（毫秒）
  },
  "stats": {
    "total_latency_ms": 总延迟,
    "generation_latency_ms": 生成延迟,
    "guard_latency_ms": Guard延迟
  }
}
```

## 示例

### 评估 base_set 的前 100 个样本

```bash
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/base_set_100.jsonl \
    --config base_set \
    --max_samples 100
```

### 评估所有配置，每个配置 50 个样本

```bash
# base_set
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/base_set.jsonl \
    --config base_set \
    --max_samples 10

# attack_enhanced_set
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/attack_enhanced_set.jsonl \
    --config attack_enhanced_set \
    --max_samples 10

# defense_enhanced_set
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/defense_enhanced_set.jsonl \
    --config defense_enhanced_set \
    --max_samples 10

# mcq_set
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/mcq_set.jsonl \
    --config mcq_set \
    --max_samples 10
```

### 断点续传

如果评估中断，可以从指定位置继续：

```bash
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/base_set.jsonl \
    --config base_set \
    --start_from 100  # 从第 100 个样本开始
```

## 注意事项

1. **显存要求**：确保有足够的 GPU 显存（建议至少 16GB）
2. **时间消耗**：评估大量样本可能需要较长时间
3. **输出文件**：脚本会增量写入，即使中断也不会丢失已处理的数据
4. **模型路径**：脚本会自动检测容器内的模型路径（`/cache/` 或 `/workspace/models/`）

## 故障排除

### 模型未找到

如果出现模型加载错误，检查：
1. 模型是否在 `/cache/` 或 `/workspace/models/` 目录
2. 环境变量是否正确设置

### 数据文件未找到

确保数据已下载并同步到容器：
```bash
docker exec neurobreak-container ls -lh /workspace/data/salad/raw/
```

### 显存不足

减少 `--max_samples` 或使用更小的批次大小。

## 结果分析

评估完成后，可以使用以下命令查看结果：

```bash
# 查看文件行数（样本数）
wc -l /workspace/logs/salad_evaluation.jsonl

# 查看前几个结果
head -n 3 /workspace/logs/salad_evaluation.jsonl | jq .

# 统计 verdict 分布
cat /workspace/logs/salad_evaluation.jsonl | jq -r '.guard.verdict' | sort | uniq -c
```

