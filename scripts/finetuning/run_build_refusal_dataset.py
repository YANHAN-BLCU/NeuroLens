#!/usr/bin/env python3
"""
Refusal-guided 数据集构建脚本

用途：
- 从 SALAD 评估日志（JSONL）+ refusal templates 构建 fine-tuning 数据集
- 或者从已保存的 SALAD taxonomy 文件 + refusal templates 构建 fine-tuning 数据集

示例用法（从评估日志构建）：
    python scripts/run_build_refusal_dataset.py ^
        --evaluation-log logs/base_evaluation.jsonl ^
        --refusal-templates outputs/tmp_refusal_templates/refusal_templates.json ^
        --output outputs/refusal_guided_datasets/base_eval_dataset.jsonl

示例用法（从 taxonomy 构建）：
    python scripts/run_build_refusal_dataset.py ^
        --taxonomy-path outputs/salad_taxonomy/base_evaluation_taxonomy.json ^
        --refusal-templates outputs/tmp_refusal_templates/refusal_templates.json ^
        --output outputs/refusal_guided_datasets/base_eval_from_taxonomy.jsonl
"""

import os
import sys
import argparse
from pathlib import Path


# ---- 处理 Python 路径，使得可以在任何工作目录下直接运行脚本 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.exists() and (PROJECT_ROOT / "engine").exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    workspace_path = os.getenv("WORKSPACE_PATH", "/workspace")
    if os.path.exists(workspace_path):
        sys.path.insert(0, workspace_path)
    else:
        cwd = Path.cwd()
        if (cwd / "engine").exists():
            sys.path.insert(0, str(cwd))
        else:
            sys.path.insert(0, "/workspace")


from engine.fine_tuning.refusal_templates import load_refusal_templates  # noqa: E402
from engine.fine_tuning.dataset_builder import (  # noqa: E402
    build_refusal_guided_dataset,
    build_dataset_from_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 refusal-guided fine-tuning 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--evaluation-log",
        type=str,
        help="评估日志文件路径（JSONL 格式），用于从原始 SALAD 评估结果构建数据集",
    )
    source_group.add_argument(
        "--taxonomy-path",
        type=str,
        help="已保存的 SALAD taxonomy 文件路径（JSON/JSONL），用于从分类结果构建数据集",
    )

    parser.add_argument(
        "--refusal-templates",
        type=str,
        required=True,
        help="Refusal templates JSON 文件路径（由 save_refusal_templates 生成）",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出数据集路径（建议使用 .jsonl 后缀）",
    )

    parser.add_argument(
        "--only-successful",
        action="store_true",
        help="仅使用 successful jailbreak 样本（默认：对于 taxonomy，仅显式为 False 的才会被过滤）",
    )

    parser.add_argument(
        "--min-templates-per-prompt",
        type=int,
        default=1,
        help="每个 prompt 使用的最少 template 数量（仅 evaluation-log 模式生效，默认 1）",
    )

    parser.add_argument(
        "--max-templates-per-prompt",
        type=int,
        default=3,
        help="每个 prompt 使用的最多 template 数量（仅 evaluation-log 模式生效，默认 3）",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )

    # 仅在 taxonomy 模式下生效的采样控制参数
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=None,
        help="每个 generic bucket 的最大样本数（taxonomy 模式：>0 时对大类下采样）",
    )
    parser.add_argument(
        "--min-samples-per-category",
        type=int,
        default=None,
        help="每个 generic bucket 的最小样本数（taxonomy 模式：>0 时对冷门类过采样，有放回抽样）",
    )
    parser.add_argument(
        "--no-upsampling-rare",
        action="store_true",
        help="禁用对冷门类别的过采样（即使设置了 --min-samples-per-category）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[Dataset Script] 加载 refusal templates...")
    refusal_templates = load_refusal_templates(args.refusal_templates)
    if not refusal_templates:
        raise ValueError(
            f"[Dataset Script] 无法从 {args.refusal_templates} 加载任何 refusal templates，"
            "请先使用提取脚本生成模板文件。"
        )

    if args.evaluation_log:
        print(f"[Dataset Script] 从评估日志构建数据集: {args.evaluation_log}")
        dataset = build_refusal_guided_dataset(
            evaluation_log_path=args.evaluation_log,
            refusal_templates=refusal_templates,
            output_path=str(output_path),
            only_successful_jailbreaks=args.only_successful,
            min_templates_per_prompt=args.min_templates_per_prompt,
            max_templates_per_prompt=args.max_templates_per_prompt,
            seed=args.seed,
        )
    else:
        print(f"[Dataset Script] 从 taxonomy 构建数据集: {args.taxonomy_path}")
        dataset = build_dataset_from_taxonomy(
            taxonomy_path=args.taxonomy_path,
            refusal_templates=refusal_templates,
            output_path=str(output_path),
            only_successful=args.only_successful,
            max_samples_per_category=args.max_samples_per_category,
            min_samples_per_category=args.min_samples_per_category,
            upsample_rare_categories=not args.no_upsampling_rare,
            seed=args.seed,
        )

    print(f"[Dataset Script] 数据集构建完成，样本数: {len(dataset)}")
    print(f"[Dataset Script] 已保存到: {output_path}")


if __name__ == "__main__":
    main()

