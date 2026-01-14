#!/bin/bash
# Bash 脚本：删除 Docker 容器内的模型文件
# 使用方法: 
#   在容器内: bash scripts/remove_models.sh
#   从宿主机: docker exec <container_name> bash scripts/remove_models.sh

set -e

# 默认要删除的模型
DEFAULT_MODELS=(
    "meta-llama_Llama-3.2-3B-Instruct"
    "meta-llama_Llama-Guard-3-1B"
    "Meta-Llama-3-8B-Instruct"
    "Llama-Guard-3-8B"
)

# 解析参数
CONFIRM=false
MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --confirm|-y)
            CONFIRM=true
            shift
            ;;
        --models)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 如果未指定模型，使用默认的
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# 可能的模型路径
MODEL_PATHS=(
    "/cache"
    "/workspace/hf_models"
    "/workspace/models"
    "$HOME/.cache/huggingface/models"
    "${HF_HOME:-/workspace/.cache/huggingface}/models"
    "${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}/models"
)

echo "========================================"
echo "删除容器内模型文件"
echo "========================================"
echo ""

# 收集要删除的模型路径
declare -a MODELS_TO_DELETE
declare -a MODEL_NAMES
declare -a MODEL_SIZES

for model_name in "${MODELS[@]}"; do
    echo "查找模型: $model_name"
    
    for base_path in "${MODEL_PATHS[@]}"; do
        model_path="$base_path/$model_name"
        
        if [ -d "$model_path" ] && [ -f "$model_path/config.json" ]; then
            size=$(du -sh "$model_path" 2>/dev/null | cut -f1)
            size_bytes=$(du -sb "$model_path" 2>/dev/null | cut -f1)
            size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc)
            
            echo "  找到: $model_path (大小: ${size_gb} GB)"
            MODELS_TO_DELETE+=("$model_path")
            MODEL_NAMES+=("$model_name")
            MODEL_SIZES+=("$size_gb")
        fi
    done
done

if [ ${#MODELS_TO_DELETE[@]} -eq 0 ]; then
    echo "未找到任何模型文件。"
    exit 0
fi

# 显示要删除的模型
echo ""
echo "将要删除以下模型:"
total_size=0
for i in "${!MODELS_TO_DELETE[@]}"; do
    echo "  - ${MODEL_NAMES[$i]}: ${MODELS_TO_DELETE[$i]} (${MODEL_SIZES[$i]} GB)"
    total_size=$(echo "$total_size + ${MODEL_SIZES[$i]}" | bc)
done
echo ""
echo "总计大小: ${total_size} GB"
echo ""

# 确认删除
if [ "$CONFIRM" != "true" ]; then
    read -p "确认删除以上模型? (yes/no): " response
    if [ "$response" != "yes" ] && [ "$response" != "y" ]; then
        echo "已取消删除操作。"
        exit 0
    fi
fi

# 执行删除
echo ""
echo "开始删除..."
deleted_count=0
failed_count=0

for i in "${!MODELS_TO_DELETE[@]}"; do
    model_path="${MODELS_TO_DELETE[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    
    echo "删除: $model_path"
    if rm -rf "$model_path"; then
        echo "  ✓ 已删除: $model_name"
        ((deleted_count++))
    else
        echo "  ✗ 删除失败: $model_name"
        ((failed_count++))
    fi
done

echo ""
echo "========================================"
echo "删除完成"
echo "========================================"
echo "成功删除: $deleted_count 个模型"
if [ $failed_count -gt 0 ]; then
    echo "删除失败: $failed_count 个模型"
fi
echo "释放空间: ${total_size} GB"

