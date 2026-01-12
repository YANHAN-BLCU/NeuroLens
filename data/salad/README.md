# SALAD-Bench 数据集

## 目录结构
- `raw/`: 原始 JSON/CSV 文件
- `processed/`: 统一 schema 后的 JSONL 文件
- `cache/`: Arrow/HDF5 中间结果

## Schema
统一字段：`id, attack_type, prompt, base_prompt, template, reference_label`

## 数据划分
- `analysis`: ≥100/类，用于探针训练和表征分析
- `eval`: ≥100/类，用于评估基线
- `finetune`: 成功越狱样本，用于微调

## 预处理命令
```
python scripts/download_salad.py --split all --output data/salad/raw
python scripts/verify_salad.py --input data/salad/raw
```

