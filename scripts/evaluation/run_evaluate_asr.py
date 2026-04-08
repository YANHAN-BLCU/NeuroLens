"""
====================================================================
ASR（攻击成功率）评估脚本
====================================================================
功能：对目标模型进行越狱攻击成功率评估
评估流程：
    1. 加载主模型 + 安全分类器
    2. 加载 SALAD-Bench 数据集
    3. 逐样本推理并判定安全性
    4. 输出 JSONL 格式结果

依赖：
    accelerate, transformers, torch, pyyaml

使用方式：
    # 方式一：accelerate launch（推荐，GPU 并行）
    accelerate launch scripts/run_evaluate_asr.py `
        --model "/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct" `
        --classifier "/root/autodl-tmp/neurolens/ms_models/Qwen/Qwen3Guard-Gen-8B" `
        --output "outputs/asr_results.jsonl"

    # 方式二：直接运行（单卡）
    python scripts/run_evaluate_asr.py `
        --model "/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct" `
        --classifier "/root/autodl-tmp/neurolens/ms_models/Qwen/Qwen3Guard-Gen-8B" `
        --output "outputs/asr_results.jsonl"
====================================================================
"""

import argparse
import sys
from pathlib import Path

# 确保 engine 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.assessment import run_evaluation

if __name__ == "__main__":
    run_evaluation()
