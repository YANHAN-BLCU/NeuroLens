"""
====================================================================
Utility（效用能力）评估脚本
====================================================================
功能：评估模型在通用基准任务上的效用能力
评估内容：
    1. 零样本任务评估（HellaSwag, WinoGrande, ARC, OBQA, BoolQ, RTE）
    2. WikiText 困惑度计算

依赖：
    transformers, torch, datasets（可选）

使用方式：
    # 完整 Utility 评估
    python scripts/run_evaluate_utility.py \
        --model "/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"

    # 只评估零样本任务
    python scripts/run_evaluate_utility.py \
        --model "/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct" \
        --tasks hellaswag winogrande arc_easy arc_challenge boolq rte

    # 指定输出目录
    python scripts/run_evaluate_utility.py \
        --model "/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct" \
        --output_dir "outputs/utility"

    # 检查数据集是否已下载
    python scripts/run_evaluate_utility.py --check_datasets
====================================================================
"""

import argparse
import sys
from pathlib import Path

# 确保 engine 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.assessment.utility_evaluator import (
    evaluate_utility,
    evaluate_zero_shot_tasks,
    compute_wikitext_perplexity,
    ZERO_SHOT_TASKS,
    _get_local_data_path,
    _get_wikitext_path,
)


def check_datasets():
    """检查所有评估数据集是否已下载到本地"""
    print("=" * 60)
    print("检查数据集下载状态")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "data" / "utility"

    results = {}

    # 检查零样本任务数据集
    for task in ZERO_SHOT_TASKS:
        path = _get_local_data_path(task)
        status = "✓ 已下载" if (path and path.exists()) else "✗ 未找到"
        if path:
            results[task] = (status, str(path))
        else:
            results[task] = (status, "未定位到本地文件")

    # 检查 WikiText
    wiki_path = _get_wikitext_path()
    wiki_status = "✓ 已下载" if wiki_path else "✗ 未找到"
    wiki_detail = str(wiki_path) if wiki_path else "未定位到本地文件"

    # 汇总输出
    print(f"\n数据集根目录: {base}\n")
    print(f"{'任务':<20} {'状态':<12} {'路径'}")
    print("-" * 70)
    for task, (status, path) in results.items():
        print(f"{task:<20} {status:<12} {path}")
    print("-" * 70)
    print(f"{'wikitext':<20} {wiki_status:<12} {wiki_detail}")
    print("-" * 70)

    all_ok = all(
        status == "✓ 已下载"
        for status, _ in results.values()
    ) and wiki_status == "✓ 已下载"

    if all_ok:
        print("\n所有数据集已就绪，可以运行评估。")
    else:
        missing = [
            task for task, (status, _) in results.items()
            if status == "✗ 未找到"
        ]
        if wiki_status == "✗ 未找到":
            missing.append("wikitext")
        print(f"\n缺少数据集: {', '.join(missing)}")
        print("请运行: python scripts/download_utility_datasets.py --mirror")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Utility 评估脚本")
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型路径或 HuggingFace ID（使用 --check_datasets 时可省略）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/utility",
        help="结果输出目录（默认: outputs/utility）"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=None,
        help=f"评估任务列表，默认全部: {' '.join(ZERO_SHOT_TASKS)}"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="批大小（默认: 8）"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="每个任务最大样本数（None=全部，默认: None）"
    )
    parser.add_argument(
        "--wikitext_path", type=str, default=None,
        help="WikiText 验证集路径（默认: 自动查找）"
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="不保存结果到文件"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式，减少输出"
    )
    parser.add_argument(
        "--check_datasets", action="store_true",
        help="仅检查数据集是否已下载，不运行评估"
    )

    args = parser.parse_args()

    # 数据集检查模式
    if args.check_datasets:
        check_datasets()
        return

    if args.model is None:
        parser.error("--model 是必填参数（除非使用 --check_datasets）")

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("Utility 评估")
        print("=" * 60)
        print(f"模型: {args.model}")
        print(f"任务: {args.tasks or ZERO_SHOT_TASKS}")
        print(f"输出目录: {args.output_dir}")
        print("=" * 60)

    results = evaluate_utility(
        model_path=args.model,
        tasks=args.tasks,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir if not args.no_save else None,
        save_results=not args.no_save,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        zero_shot = results.get("zero_shot", {})
        mean_acc = zero_shot.get("mean", 0.0)
        wiki_ppl = results.get("wiki_perplexity")
        utility_score = results.get("utility_score", 0.0)

        print(f"零样本平均准确率: {mean_acc:.4f}")
        if wiki_ppl is not None:
            print(f"WikiText 困惑度: {wiki_ppl:.4f}")
        print(f"综合 Utility 分数: {utility_score:.4f}")

        print("=" * 60)


if __name__ == "__main__":
    main()
