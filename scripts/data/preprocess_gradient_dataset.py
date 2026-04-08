#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理脚本：将 base_set_train.jsonl 和 attack_enhanced_set_train.jsonl
合并转换为 run_gradient_dependency.py 可用的格式。

输出字段统一为 `text`：
- base_set_train.jsonl: question → text
- attack_enhanced_set_train.jsonl: augq → text

用法：
    python scripts/preprocess_gradient_dataset.py
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_and_convert(file_path: str, text_field: str, output_field: str = "text") -> list:
    """
    读取 JSONL 文件，将指定字段转换为统一的 text 字段。

    Args:
        file_path: JSONL 文件路径
        text_field: 源文本字段名
        output_field: 目标字段名（默认为 text）

    Returns:
        转换后的样本列表
    """
    samples = []
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line.strip())
                text = sample.get(text_field)
                if text:
                    # 保留原始字段，添加 text 字段
                    converted = dict(sample)
                    converted[output_field] = text
                    samples.append(converted)
            except json.JSONDecodeError as e:
                print(f"  [警告] 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="合并 base_set_train.jsonl 和 attack_enhanced_set_train.jsonl "
                    "为梯度依赖分析可用的格式"
    )
    parser.add_argument(
        "--base-set",
        type=str,
        default="data/salad/raw/base_set_train.jsonl",
        help="base_set_train.jsonl 路径（默认: data/salad/raw/base_set_train.jsonl）"
    )
    parser.add_argument(
        "--attack-enhanced-set",
        type=str,
        default="data/salad/raw/attack_enhanced_set_train.jsonl",
        help="attack_enhanced_set_train.jsonl 路径（默认: data/salad/raw/attack_enhanced_set_train.jsonl）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/salad/raw/gradient_dependency_dataset.jsonl",
        help="输出文件路径（默认: data/salad/raw/gradient_dependency_dataset.jsonl）"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: 处理 base_set_train.jsonl（question → text）
    print(f"[1/2] 处理 base_set_train.jsonl ...")
    base_samples = load_and_convert(args.base_set, text_field="question")
    print(f"      转换完成: {len(base_samples)} 条样本")

    # Step 2: 处理 attack_enhanced_set_train.jsonl（augq → text）
    print(f"[2/2] 处理 attack_enhanced_set_train.jsonl ...")
    attack_samples = load_and_convert(args.attack_enhanced_set, text_field="augq")
    print(f"      转换完成: {len(attack_samples)} 条样本")

    # Step 3: 合并
    merged_samples = base_samples + attack_samples
    print(f"\n合并完成: 共 {len(merged_samples)} 条样本 "
          f"({len(base_samples)} base + {len(attack_samples)} attack)")

    # Step 4: 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in merged_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 5: 显示示例
    print("\n--- 示例输出（前2条）---")
    for i, sample in enumerate(merged_samples[:2]):
        text_preview = sample.get("text", "")[:80]
        print(f"  [{i}] text: {text_preview}...")

    print("\n生成完成后，可使用以下命令运行第七步：")
    print(f"  python scripts/run_gradient_dependency.py \\")
    print(f"      --model-path <MODEL_PATH> \\")
    print(f"      --dataset-path {output_path} \\")
    print(f"      --target-neurons-path outputs/neurons/dedicated_safety_neurons.json \\")
    print(f"      --output-path outputs/neurons/gradient_dependency")


if __name__ == "__main__":
    main()
