"""
====================================================================
评估报告生成脚本
====================================================================
功能：根据评估结果生成汇总报告（Markdown 格式）

报告类型：
    1. ASR 报告：根据越狱评估结果计算攻击成功率
    2. Utility 报告：根据效用评估结果计算综合分数
    3. 综合报告：同时包含 ASR 和 Utility

依赖：
    无特殊依赖

使用方式：
    # ASR 报告（从 JSONL 文件）
    python scripts/run_evaluate_report.py `
        --input "outputs/asr_results.jsonl" `
        --output "outputs/reports/asr_report.md"

    # 支持 glob 模式批量读取
    python scripts/run_evaluate_report.py `
        --input "outputs/asr_results_*.jsonl" `
        --output "outputs/reports/asr_report.md"

    # 指定模型名称（报告中显示）
    python scripts/run_evaluate_report.py `
        --input "outputs/asr_results.jsonl" `
        --output "outputs/reports/asr_report.md" `
        --model "Meta-Llama-3-8B-Instruct"

    # 生成 Utility 报告（从 JSON 文件）
    python scripts/run_evaluate_report.py `
        --input "outputs/utility/utility_results_*.json" `
        --output "outputs/reports/utility_report.md" `
        --report_type utility `
        --model "Meta-Llama-3-8B-Instruct"
====================================================================
"""

import argparse
import json
import sys
from pathlib import Path

# 确保 engine 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.assessment.report import (
    load_results,
    calculate_metrics,
    generate_report,
    generate_asr_report,
    generate_utility_report,
)


def load_utility_results(input_paths):
    """加载 Utility 评估结果（JSON 文件列表）"""
    results = []
    for path in input_paths:
        path = Path(path)
        if path.is_file() and path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        elif "*" in str(path):
            parent = path.parent
            pattern = path.name
            for file in parent.glob(pattern):
                if file.suffix == ".json":
                    with open(file, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
    return results


def main():
    parser = argparse.ArgumentParser(description="评估报告生成脚本")
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入文件路径（JSONL 格式，支持 glob 如 'outputs/*.jsonl'）"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出 Markdown 报告路径"
    )
    parser.add_argument(
        "--model", type=str, default="",
        help="模型名称（报告中显示）"
    )
    parser.add_argument(
        "--classifier", type=str, default="",
        help="分类器名称（仅 ASR 报告中显示）"
    )
    parser.add_argument(
        "--report_type", type=str, default="asr",
        choices=["asr", "utility", "auto"],
        help="报告类型: asr（默认）/ utility / auto（自动检测）"
    )

    args = parser.parse_args()

    # 解析输入路径（支持 glob 模式）
    input_path = Path(args.input)
    if "*" in str(input_path):
        input_paths = list(input_path.parent.glob(input_path.name))
    else:
        input_paths = [input_path]

    print(f"发现 {len(input_paths)} 个输入文件")

    # 根据类型生成报告
    is_jsonl = all(p.suffix == ".jsonl" for p in input_paths if p.is_file())

    if args.report_type == "asr" or (args.report_type == "auto" and is_jsonl):
        # ========== ASR 报告 ==========
        print("生成 ASR 报告...")
        results = load_results(input_paths)
        print(f"加载了 {len(results)} 条评估结果")

        if not results:
            print("错误: 没有找到有效结果")
            sys.exit(1)

        report = generate_asr_report(
            results=results,
            output_path=args.output,
            model_name=args.model,
            title=f"ASR Evaluation Report: {args.model}",
        )
        print(f"报告已保存: {args.output}")
        print(f"\n整体 ASR: {calculate_metrics(results)['overall_asr']:.2f}%")

    elif args.report_type == "utility" or (args.report_type == "auto" and not is_jsonl):
        # ========== Utility 报告 ==========
        print("生成 Utility 报告...")
        results = load_utility_results(input_paths)
        print(f"加载了 {len(results)} 个评估结果")

        if not results:
            print("错误: 没有找到有效结果")
            sys.exit(1)

        # 如果有多个结果，取第一个（通常只有一个）
        utility_results = results[0]

        report = generate_utility_report(
            utility_results=utility_results,
            output_path=args.output,
            model_name=args.model,
            title=f"Utility Evaluation Report: {args.model}",
        )
        print(f"报告已保存: {args.output}")

    else:
        print("错误: 无法自动判断报告类型，请使用 --report_type 手动指定")
        sys.exit(1)


if __name__ == "__main__":
    main()
