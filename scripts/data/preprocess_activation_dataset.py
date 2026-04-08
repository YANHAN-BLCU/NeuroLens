#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理脚本：将 base_set_train.jsonl 和 attack_enhanced_set_train.jsonl
合并，并关联 outputs/data_set_output/labels/ 目录下的标签，
转换为 run_activation_projection.py 可用的格式。

输出字段：
  - text：统一后的文本字段
    · base_set_train.jsonl: question → text
    · attack_enhanced_set_train.jsonl: augq → text
  - jailbreak_success：布尔标志（由标签文件映射）
    · Safe / Controversial → False（拒绝有害请求 = jailbreak 失败）
    · Unsafe → True（模型被成功越狱）

标签文件来源：
  - base_set_train.jsonl → outputs/data_set_output/labels/base_set_outputs_*.jsonl（共 4 个分片）
  - attack_enhanced_set_train.jsonl → outputs/data_set_output/labels/attack_enhanced_outputs.jsonl

用法：
    python scripts/preprocess_activation_dataset.py
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict


def label_to_jailbreak(label: str) -> bool:
    """
    将标签映射为 jailbreak_success 布尔值。

    - Safe / Controversial：模型拒绝了有害请求 → jailbreak 失败 → False
    - Unsafe：模型被成功越狱 → True
    """
    if label == "Unsafe":
        return True
    return False  # Safe / Controversial 均视为失败


def merge_labels_with_samples(raw_file: str, label_files: list, text_field: str) -> list:
    """
    读取原始数据集，按 original_index 合并标签文件。

    Args:
        raw_file: 原始数据文件路径
        label_files: 标签文件路径列表（已按 original_index 排序合并）
        text_field: 原始数据中的文本字段名

    Returns:
        合并后的样本列表
    """
    # Step 1: 加载所有标签
    label_map = {}
    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                idx = entry.get("original_index")
                label = entry.get("label", "Safe")
                label_map[idx] = label_to_jailbreak(label)

    # Step 2: 读取原始数据并关联标签
    samples = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line.strip())
                text = sample.get(text_field)
                jailbreak_success = label_map.get(line_num - 1)
                if text is None:
                    print(f"  [警告] 第 {line_num} 行缺少文本字段 '{text_field}'，跳过")
                    continue
                if jailbreak_success is None:
                    print(f"  [警告] 第 {line_num} 行缺少标签，跳过")
                    continue
                converted = dict(sample)
                converted["text"] = text
                converted["jailbreak_success"] = jailbreak_success
                samples.append(converted)
            except json.JSONDecodeError as e:
                print(f"  [警告] 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="合并 base_set_train + attack_enhanced_set 并关联标签，"
                    "转换为激活投影分析可用的格式"
    )
    parser.add_argument(
        "--base-set",
        type=str,
        default="data/salad/raw/base_set_train.jsonl",
        help="base_set_train.jsonl 路径"
    )
    parser.add_argument(
        "--base-labels",
        type=str,
        nargs="+",
        default=[
            "outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl",
            "outputs/data_set_output/labels/base_set_outputs_5000_9999.jsonl",
            "outputs/data_set_output/labels/base_set_outputs_10000_14999.jsonl",
            "outputs/data_set_output/labels/base_set_outputs_15000_21316.jsonl",
        ],
        help="base_set_train 对应的标签文件（4 个分片，按 original_index 顺序传入）"
    )
    parser.add_argument(
        "--attack-enhanced-set",
        type=str,
        default="data/salad/raw/attack_enhanced_set_train.jsonl",
        help="attack_enhanced_set_train.jsonl 路径"
    )
    parser.add_argument(
        "--attack-labels",
        type=str,
        default="outputs/data_set_output/labels/attack_enhanced_outputs.jsonl",
        help="attack_enhanced_set 对应的标签文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/salad/raw/activation_projection_dataset.jsonl",
        help="输出文件路径"
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="base_set_train",
        help="base 数据集名称（用于统计输出）"
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default="attack_enhanced_set",
        help="attack 数据集名称（用于统计输出）"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: 处理 base_set_train
    print(f"[1/2] 处理 {args.base_name} ...")
    print(f"       原始数据: {args.base_set}")
    print(f"       标签文件: {args.base_labels}")
    base_samples = merge_labels_with_samples(
        raw_file=args.base_set,
        label_files=args.base_labels,
        text_field="question",
    )
    base_success = sum(1 for s in base_samples if s["jailbreak_success"])
    base_failed = len(base_samples) - base_success
    print(f"       转换完成: {len(base_samples)} 条样本 "
          f"(成功 {base_success}, 失败 {base_failed})")

    # Step 2: 处理 attack_enhanced_set
    print(f"[2/2] 处理 {args.attack_name} ...")
    print(f"       原始数据: {args.attack_enhanced_set}")
    print(f"       标签文件: {args.attack_labels}")
    attack_samples = merge_labels_with_samples(
        raw_file=args.attack_enhanced_set,
        label_files=[args.attack_labels],
        text_field="augq",
    )
    attack_success = sum(1 for s in attack_samples if s["jailbreak_success"])
    attack_failed = len(attack_samples) - attack_success
    print(f"       转换完成: {len(attack_samples)} 条样本 "
          f"(成功 {attack_success}, 失败 {attack_failed})")

    # Step 3: 合并
    merged_samples = base_samples + attack_samples
    total_success = base_success + attack_success
    total_failed = base_failed + attack_failed
    print(f"\n合并完成: 共 {len(merged_samples)} 条样本 "
          f"(成功 {total_success}, 失败 {total_failed})")

    # Step 4: 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in merged_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 5: 显示示例
    print("\n--- 成功 jailbreak 样本示例 ---")
    shown = 0
    for s in merged_samples:
        if s["jailbreak_success"] and shown < 2:
            print(f"  jailbreak_success: {s['jailbreak_success']}  |  "
                  f"text: {s.get('text', '')[:80]}...")
            shown += 1
    print("\n--- 失败 jailbreak 样本示例 ---")
    shown = 0
    for s in merged_samples:
        if not s["jailbreak_success"] and shown < 2:
            print(f"  jailbreak_success: {s['jailbreak_success']}  |  "
                  f"text: {s.get('text', '')[:80]}...")
            shown += 1

    print("\n生成完成后，可使用以下命令运行第五步：")
    print(f"  python scripts/run_activation_projection.py \\")
    print(f"      --model-path <MODEL_PATH> \\")
    print(f"      --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz \\")
    print(f"      --dataset-path {output_path} \\")
    print(f"      --target-neurons-path outputs/neurons/dedicated_safety_neurons.json \\")
    print(f"      --output-path outputs/neurons \\")
    print(f"      --output-filename activation_projection.json \\")
    print(f"      --batch-size 4 \\")
    print(f"      --num-samples 500 \\")
    print(f"      --load-in-4bit \\")
    print(f"      --clear-cache")


if __name__ == "__main__":
    main()
