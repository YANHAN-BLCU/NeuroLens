#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据当前划分方式，给定样本数，统计安全/有害样本个数。

逻辑与 train_linear_probes_1000_A.py 保持一致：
- 使用 load_attack_enhanced 加载数据（按行数截断到 max_samples）
- 使用 split_data_optimized 且开启 6:2:2 剩余全作训练集（use_ratio_6_2_2_full_train=True）
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from train_linear_probes_1000_A import (  # noqa: E402
    load_attack_enhanced,
    split_data_optimized,
)


def main():
    parser = argparse.ArgumentParser(
        description="给定样本数，统计整体及各子集的安全/有害样本个数（与 train_linear_probes_1000_A.py 划分方式一致）。"
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        default=Path("data/salad/raw/base_evaluation.jsonl"),
        help="数据集文件路径（base_evaluation.jsonl）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="希望使用的样本数（按行数截断，不足时使用全部可用样本）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保持与训练脚本一致）",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples 必须为正整数")

    print("=" * 80)
    print(f"按当前划分方式统计样本（num_samples={args.num_samples}）")
    print("=" * 80)
    print()

    # 1. 加载数据（与训练脚本一致，使用当前 load_attack_enhanced 实现）
    print(f"[Step 1] 加载数据: {args.data_file} (max_samples={args.num_samples})")
    texts, labels = load_attack_enhanced(args.data_file, max_samples=args.num_samples)

    if len(texts) == 0:
        raise ValueError("未加载到任何样本，请检查数据路径与字段。")

    total = len(texts)
    safe = sum(1 for l in labels if l == 0)
    toxic = sum(1 for l in labels if l == 1)

    print(f"  加载到的有效样本数: {total}")
    print(f"    安全(0): {safe}")
    print(f"    有害(1): {toxic}")
    print()

    # 2. 按 6:2:2 剩余全作训练集方式划分（与 train_linear_probes_1000_A 默认策略一致）
    print("[Step 2] 6:2:2 剩余全作训练集划分（60% 训练、20% 验证、20% 测试）...")
    probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = (
        split_data_optimized(
            texts=texts,
            labels=labels,
            test_ratio=0.20,
            val_ratio=0.20,
            min_test_toxic=100,
            min_val_toxic=100,
            seed=args.seed,
            use_ratio_6_2_2=False,
            probe_val_ratio_in_train=0.3,
            use_ratio_6_2_2_full_train=True,
        )
    )

    n_train = len(probe_train_texts)
    n_train_safe = sum(1 for l in probe_train_labels if l == 0)
    n_train_toxic = sum(1 for l in probe_train_labels if l == 1)

    n_val = len(probe_val_texts)
    n_val_safe = sum(1 for l in probe_val_labels if l == 0)
    n_val_toxic = sum(1 for l in probe_val_labels if l == 1)

    n_test = len(test_texts)
    n_test_safe = sum(1 for l in test_labels if l == 0)
    n_test_toxic = sum(1 for l in test_labels if l == 1)

    print()
    print("[结果] 数据集划分统计（未做过采样前）：")
    print("  集合           | 总样本数 | 安全(0) | 有害(1)")
    print("  ---------------+----------+---------+--------")
    print(f"  探针训练集     | {n_train:8d} | {n_train_safe:7d} | {n_train_toxic:6d}")
    print(f"  探针验证集     | {n_val:8d} | {n_val_safe:7d} | {n_val_toxic:6d}")
    print(f"  测试集         | {n_test:8d} | {n_test_safe:7d} | {n_test_toxic:6d}")
    print()

    print("=" * 80)
    print("说明：")
    print("1. 上述统计完全复用了 train_linear_probes_1000_A.py 中的加载和划分逻辑；")
    print("2. 训练脚本后续还会对训练集做一次安全:有害≈1.5:1 的比例调整（过采样/下采样），")
    print("   本脚本统计的是『划分完成但尚未过采样之前』的真实分布。")
    print("=" * 80)


if __name__ == "__main__":
    main()

