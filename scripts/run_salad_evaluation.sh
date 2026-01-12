#!/bin/bash
# 在 Docker 容器内运行 SALAD-Bench 评估脚本

# 设置默认值
DATA_DIR="${DATA_DIR:-/workspace/data/salad/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/logs}"
CONFIG="${CONFIG:-base_set}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
START_FROM="${START_FROM:-0}"

# 输出文件路径
OUTPUT_FILE="${OUTPUT_DIR}/salad_evaluation_${CONFIG}_$(date +%Y%m%d_%H%M%S).jsonl"

echo "========================================"
echo "SALAD-Bench Evaluation Pipeline"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Config: $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "Start from: $START_FROM"
echo ""

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评估脚本
cd /workspace
python scripts/evaluate_salad_pipeline.py \
    --data_dir "$DATA_DIR" \
    --output "$OUTPUT_FILE" \
    --config "$CONFIG" \
    --max_samples "$MAX_SAMPLES" \
    --start_from "$START_FROM"

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_FILE"

