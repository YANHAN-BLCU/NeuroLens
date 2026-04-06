#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔记本GPU分步快速测试脚本（使用缓存）

步骤1: 提取隐藏状态（一次性）
步骤2: 使用缓存训练探针（可多次调整参数）

优势:
- 提取隐藏状态只需一次
- 可以快速调整训练参数进行多次实验
- 如果训练失败，可以快速重新开始
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("笔记本GPU分步快速测试（使用缓存）")
    print("=" * 60)
    print()
    
    # 检查数据文件
    data_file = Path("data/salad/raw/base_evaluation.jsonl")
    if not data_file.exists():
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        print("请先下载数据集或检查路径")
        sys.exit(1)
    
    # 缓存文件路径
    cache_file = Path("outputs/hidden_states_cache/quick_test.npz")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 提取隐藏状态
    print("=" * 60)
    print("步骤1: 提取隐藏状态")
    print("=" * 60)
    print("配置:")
    print("  - 样本数: 200")
    print("  - 序列长度: 256")
    print("  - 批大小: 4")
    print("  - 预计时间: 3-5分钟")
    print()
    
    extract_cmd = [
        sys.executable,
        "scripts/extract_hidden_states.py",
        "--data_file", str(data_file),
        "--max_samples", "200",
        "--seed", "42",
        "--max_length", "256",
        "--batch_size", "4",
        "--output", str(cache_file),
    ]
    
    print("执行命令:")
    print(" ".join(extract_cmd))
    print()
    print("开始提取隐藏状态...")
    print()
    
    try:
        subprocess.run(extract_cmd, check=True)
        print()
        print("✅ 隐藏状态提取完成!")
        print(f"缓存文件: {cache_file}")
        print()
    except subprocess.CalledProcessError as e:
        print()
        print("❌ 提取隐藏状态失败")
        print(f"错误代码: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("提取被用户中断")
        sys.exit(1)
    
    # 步骤2: 训练探针
    print("=" * 60)
    print("步骤2: 训练探针")
    print("=" * 60)
    print("配置:")
    print("  - 使用缓存: 是")
    print("  - 训练轮数: 10")
    print("  - 批大小: 16")
    print("  - 预计时间: 10-20分钟")
    print()
    
    train_cmd = [
        sys.executable,
        "scripts/train_linear_probes.py",
        "--hidden_states_cache", str(cache_file),
        "--output_dir", "outputs/probes_quick_test",
        "--num_epochs", "10",
        "--probe_batch_size", "16",
        "--lr", "2e-3",
        "--weight_decay", "1e-2",
        "--seed", "42",
    ]
    
    print("执行命令:")
    print(" ".join(train_cmd))
    print()
    print("开始训练探针...")
    print()
    
    try:
        subprocess.run(train_cmd, check=True)
        print()
        print("=" * 60)
        print("✅ 测试完成!")
        print("=" * 60)
        print(f"结果保存在: outputs/probes_quick_test")
        print()
        print("提示:")
        print("  - 可以再次运行此脚本，调整训练参数（num_epochs, probe_batch_size等）")
        print("  - 隐藏状态已缓存，再次训练会更快")
        print("  - 如果显存不足，可以减小 probe_batch_size")
    except subprocess.CalledProcessError as e:
        print()
        print("❌ 训练失败")
        print(f"错误代码: {e.returncode}")
        print()
        print("提示:")
        print("  - 隐藏状态已缓存，可以修改参数后重新训练")
        print("  - 如果显存不足，减小 probe_batch_size")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("训练被用户中断")
        print("提示: 隐藏状态已缓存，可以稍后继续训练")
        sys.exit(1)


if __name__ == "__main__":
    main()

