# SALAD-Bench 评估脚本快速使用指南

## 快速开始

### 在 Docker 容器内运行（推荐）

```bash
# 1. 进入容器
docker exec -it neurobreak-container /bin/bash

# 2. 运行评估（示例：评估 base_set 的前 10 个样本）
cd /workspace
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/salad_eval_base_set_10.jsonl \
    --config base_set \
    --max_samples 10
```

### 从 Windows 主机运行（使用 PowerShell）

```powershell
# 在项目根目录运行
.\scripts\run_salad_evaluation.ps1 -Config base_set -MaxSamples 10
```

## 常用命令

### 评估不同配置

```bash
# base_set (21,318 样本)
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/base_set.jsonl \
    --config base_set

# attack_enhanced_set (5,000 样本)
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/attack_enhanced_set.jsonl \
    --config attack_enhanced_set

# defense_enhanced_set (200 样本)
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/defense_enhanced_set.jsonl \
    --config defense_enhanced_set

# mcq_set (3,840 样本)
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/mcq_set.jsonl \
    --config mcq_set
```

### 限制样本数量（用于测试）

```bash
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/test_10.jsonl \
    --config base_set \
    --max_samples 10
```

### 断点续传

```bash
# 从第 100 个样本开始继续
python scripts/evaluate_salad_pipeline.py \
    --data_dir /workspace/data/salad/raw \
    --output /workspace/logs/base_set.jsonl \
    --config base_set \
    --start_from 100
```

## 输出文件格式

每行一个 JSON 对象，包含：
- `sample_id`: 样本ID
- `config`: 配置名称
- `input`: 输入数据（原始提示和样本）
- `inference`: 模型生成的响应
- `guard`: 安全判断结果（verdict, severity, rationale, categories）
- `stats`: 统计信息（延迟等）

## 查看结果

```bash
# 查看文件行数
wc -l /workspace/logs/salad_eval_*.jsonl

# 查看第一个结果（需要安装 jq）
head -n 1 /workspace/logs/salad_eval_*.jsonl | jq .

# 统计安全判断结果
cat /workspace/logs/salad_eval_*.jsonl | jq -r '.guard.verdict' | sort | uniq -c
```

## 注意事项

1. 确保数据已下载到 `/workspace/data/salad/raw/`
2. 确保模型在容器内可用（`/cache/` 或 `/workspace/models/`）
3. 评估大量样本需要较长时间，建议先用小样本测试
4. 脚本支持增量写入，中断后可以继续

## 详细文档

更多信息请参考：`docs/SALAD_EVALUATION_GUIDE.md`

