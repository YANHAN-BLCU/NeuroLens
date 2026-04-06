#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔记本GPU快速测试脚本

快速验证训练流程，使用最小配置：
- 50个样本
- 序列长度256
- 5轮训练
- 预计时间: 5-10分钟
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("笔记本GPU快速测试")
    print("=" * 60)
    print("配置:")
    print("  - 样本数: 50")
    print("  - 序列长度: 256")
    print("  - 训练轮数: 5")
    print("  - 批大小: 2 (提取), 8 (训练)")
    print("  - 预计时间: 5-10分钟")
    print("=" * 60)
    print()
    
    # 检查数据文件
    data_file = Path("data/salad/raw/base_evaluation.jsonl")
    if not data_file.exists():
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        print("请先下载数据集或检查路径")
        sys.exit(1)
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/train_linear_probes.py",
        "--data_file", str(data_file),
        "--output_dir", "outputs/probes_quick_test",
        "--max_samples", "50",
        "--batch_size", "2",
        "--max_length", "256",
        "--num_epochs", "5",
        "--probe_batch_size", "8",
        "--lr", "2e-3",
        "--weight_decay", "1e-2",
        "--seed", "42",
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    print("开始测试...")
    print()
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("✅ 测试完成!")
        print("=" * 60)
        print(f"结果保存在: outputs/probes_quick_test")
        print()
        print("提示:")
        print("  - 如果测试成功，可以增加样本数进行更完整的测试")
        print("  - 如果显存不足，可以减小 batch_size 和 probe_batch_size")
        print("  - 如果时间允许，可以增加 num_epochs 和 max_samples")
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ 测试失败")
        print("=" * 60)
        print(f"错误代码: {e.returncode}")
        print()
        print("常见问题:")
        print("  1. 显存不足: 减小 batch_size 和 probe_batch_size")
        print("  2. 数据文件不存在: 检查 data/salad/raw/base_evaluation.jsonl")
        print("  3. 模型未下载: 运行 python scripts/download_models.py --all-8b")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("测试被用户中断")
        sys.exit(1)


if __name__ == "__main__":
    main()

