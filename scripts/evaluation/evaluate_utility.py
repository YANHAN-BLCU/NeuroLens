#!/usr/bin/env python3
"""
Utility 评估脚本

用于评估模型在通用任务上的 Utility（效用），基于 Wanda 论文方法。

使用方法:
    # 评估单个模型
    python scripts/evaluate_utility.py --model-path <模型路径> --output <输出目录>

    # 评估多个模型进行对比
    python scripts/evaluate_utility.py --model-path <模型1路径> --output <输出目录>
    python scripts/evaluate_utility.py --model-path <模型2路径> --output <输出目录>

    # 对比评估
    python scripts/evaluate_utility.py --compare --model-paths <模型1> <模型2> --output <输出目录>

依赖:
    pip install lm-eval torch transformers tqdm

论文引用:
    @inproceedings{sun2024simple,
        title={A Simple and Effective Pruning Approach for Large Language Models},
        author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
        booktitle={ICLR},
        year={2024}
    }
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.assessment.utility_evaluator import (
    evaluate_utility,
    evaluate_zero_shot_tasks,
    compute_wikitext_perplexity,
    quick_evaluate,
    compare_models,
    ZERO_SHOT_TASKS,
    PAPER_BASELINE,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Utility 评估脚本 - 基于 Wanda 论文的零样本任务评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估单个模型
    python scripts/evaluate_utility.py --model-path D:/NeuroLens-master/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct --output outputs/utility_evaluation

    # 指定评估任务
    python scripts/evaluate_utility.py --model-path <模型路径> --tasks hellaswag winogrande arc_easy --output <输出目录>

    # 跳过 WikiText 困惑度评估
    python scripts/evaluate_utility.py --model-path <模型路径> --no-wikitext --output <输出目录>

    # 对比多个模型
    python scripts/evaluate_utility.py --compare --model-paths <模型1> <模型2> --output <输出目录>

评估任务:
    hellaswag      - HellaSwag 常识推理
    winogrande    - WinoGrande 常识推理
    arc_easy      - ARC Easy 科学问答
    arc_challenge - ARC Challenge 科学问答
    obqa          - OpenBookQA 科学问答
    boolq         - BoolQ 文本蕴含
    rte           - RTE 文本蕴含

默认评估所有 7 个任务。
        """
    )

    # 模型参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型路径或 HuggingFace 模型 ID"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比模式：评估多个模型"
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        type=str,
        help="对比模式下多个模型的路径"
    )

    # 评估参数
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        help=f"要评估的任务（默认：{', '.join(ZERO_SHOT_TASKS)}）"
    )
    parser.add_argument(
        "--no-wikitext",
        action="store_true",
        help="跳过 WikiText 困惑度评估"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批大小（默认：8）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个任务的最大样本数（默认：全部）"
    )

    # 输出参数
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/utility_evaluation",
        help="输出目录（默认：outputs/utility_evaluation）"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="保存结果到 JSON 文件"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（最小输出）"
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（默认：cuda if available else cpu）"
    )

    return parser.parse_args()


def print_banner():
    """打印横幅"""
    print("=" * 70)
    print("  Utility 评估脚本 - 基于 Wanda 论文 (ICLR 2024)")
    print("=" * 70)
    print()


def print_results_summary(results: dict):
    """打印结果摘要"""
    print("\n" + "=" * 70)
    print("  评估结果摘要")
    print("=" * 70)

    print(f"\n模型: {results.get('model', 'unknown')}")
    print(f"时间: {results.get('timestamp', 'unknown')}")

    print("\n零样本任务准确率:")
    zero_shot = results.get("zero_shot", {})
    for task in ZERO_SHOT_TASKS:
        if task in zero_shot:
            actual = zero_shot[task]
            paper = PAPER_BASELINE.get(task, 0.0)
            diff = actual - paper
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"  {task:20s}: {actual:.4f}  (论文基准: {paper:.4f}, 差异: {diff_str})")

    if "mean" in zero_shot:
        paper_mean = PAPER_BASELINE.get("mean", 0.0)
        diff = zero_shot["mean"] - paper_mean
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"  {'-'*20}: {'-'*8}  {'-'*12}  {'-'*12}")
        print(f"  {'平均准确率':20s}: {zero_shot['mean']:.4f}  (论文基准: {paper_mean:.4f}, 差异: {diff_str})")

    if results.get("wiki_perplexity"):
        ppl = results["wiki_perplexity"]
        paper_ppl = PAPER_BASELINE.get("wiki_perplexity", 0.0)
        diff = ppl - paper_ppl
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        print(f"\nWikiText 困惑度: {ppl:.4f}  (论文基准: {paper_ppl:.4f}, 差异: {diff_str})")

    print(f"\nUtility 分数: {results.get('utility_score', 0.0):.4f}")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    args = parse_args()

    if not args.quiet:
        print_banner()

    verbose = args.verbose and not args.quiet

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对比模式
    if args.compare:
        if not args.model_paths:
            print("错误: 对比模式需要提供 --model-paths 参数")
            sys.exit(1)

        if verbose:
            print(f"对比模式: 评估 {len(args.model_paths)} 个模型")

        comparison_results = compare_models(
            model_paths=args.model_paths,
            output_dir=str(output_dir),
        )

        # 保存对比结果
        comparison_file = output_dir / "comparison_results.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n对比结果已保存到: {comparison_file}")

        return

    # 单模型评估
    if not args.model_path:
        print("错误: 需要提供 --model-path 参数")
        print("使用 --help 查看帮助信息")
        sys.exit(1)

    if verbose:
        print(f"模型路径: {args.model_path}")

        if args.tasks:
            print(f"评估任务: {', '.join(args.tasks)}")
        else:
            print(f"评估任务: {', '.join(ZERO_SHOT_TASKS)} (全部)")

        if not args.no_wikitext:
            print("WikiText 困惑度: 是")
        else:
            print("WikiText 困惑度: 否")

        print(f"输出目录: {output_dir}")
        print()

    # 评估
    results = evaluate_utility(
        model_path=args.model_path,
        tasks=args.tasks if args.tasks else ZERO_SHOT_TASKS,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=str(output_dir),
        save_results=args.save_results,
        verbose=verbose,
    )

    if verbose:
        print_results_summary(results)


if __name__ == "__main__":
    main()
