#!/usr/bin/env python3
"""
Metrics aggregation script for security evaluation

Usage:
    python engine/assessment/report.py \
      --input logs/baseline/security_*.jsonl \
      --output reports/baseline_security.md
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(input_paths: List[Path]) -> List[Dict]:
    """Load evaluation results from JSONL files"""
    results = []
    for path in input_paths:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        elif "*" in str(path):
            # Handle glob patterns
            parent = path.parent
            pattern = path.name
            for file in parent.glob(pattern):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregated metrics"""
    metrics = {
        "total_samples": len(results),
        "successful": 0,
        "failed": 0,
        "attack_types": Counter(),
        "asr_by_type": defaultdict(lambda: {"total": 0, "unsafe": 0}),
        "latencies": [],
        "token_counts": [],
    }
    
    for result in results:
        if "error" in result:
            metrics["failed"] += 1
            continue
        
        metrics["successful"] += 1
        attack_type = result.get("attack_type", "unknown")
        metrics["attack_types"][attack_type] += 1
        
        # ASR calculation (unsafe = jailbreak successful)
        if result.get("guard_label") == "unsafe":
            metrics["asr_by_type"][attack_type]["unsafe"] += 1
        metrics["asr_by_type"][attack_type]["total"] += 1
        
        # Latency and tokens
        if "latency_ms" in result:
            metrics["latencies"].append(result["latency_ms"])
        if "num_tokens" in result:
            metrics["token_counts"].append(result["num_tokens"])
    
    # Calculate overall ASR
    total_unsafe = sum(stats["unsafe"] for stats in metrics["asr_by_type"].values())
    total_samples = sum(stats["total"] for stats in metrics["asr_by_type"].values())
    metrics["overall_asr"] = (total_unsafe / total_samples * 100) if total_samples > 0 else 0
    
    # Calculate per-type ASR
    metrics["asr_by_type"] = {
        atype: {
            "asr": (stats["unsafe"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            "total": stats["total"],
            "unsafe": stats["unsafe"],
        }
        for atype, stats in metrics["asr_by_type"].items()
    }
    
    # Statistics
    metrics["avg_latency_ms"] = statistics.mean(metrics["latencies"]) if metrics["latencies"] else 0
    metrics["median_latency_ms"] = statistics.median(metrics["latencies"]) if metrics["latencies"] else 0
    metrics["avg_tokens"] = statistics.mean(metrics["token_counts"]) if metrics["token_counts"] else 0
    
    return metrics


def generate_report(metrics: Dict, output_path: Path, model_name: str = "", classifier_name: str = ""):
    """Generate markdown report"""
    report = f"""# Security Evaluation Baseline Report

## Configuration
- Model: {model_name}
- Classifier: {classifier_name}
- Total Samples: {metrics['total_samples']}
- Successful Evaluations: {metrics['successful']}
- Failed Evaluations: {metrics['failed']}

## Overall Metrics
- **Overall ASR**: {metrics['overall_asr']:.2f}%
- **Average Latency**: {metrics['avg_latency_ms']:.2f} ms
- **Median Latency**: {metrics['median_latency_ms']:.2f} ms
- **Average Tokens**: {metrics['avg_tokens']:.0f}

## Attack Success Rate by Type

| Attack Type | Total Samples | Unsafe | ASR (%) |
|------------|---------------|--------|---------|
"""

    for attack_type, stats in sorted(metrics["asr_by_type"].items()):
        report += f"| {attack_type} | {stats['total']} | {stats['unsafe']} | {stats['asr']:.2f} |\n"

    report += f"""
## Sample Distribution by Attack Type

"""
    for attack_type, count in metrics["attack_types"].most_common():
        report += f"- {attack_type}: {count} samples\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report generated: {output_path}")


def generate_asr_report(
    results: List[Dict],
    output_path: Optional[Path] = None,
    model_name: str = "",
    title: str = "ASR Evaluation Report",
) -> str:
    """生成 ASR 评估报告（Markdown 格式）

    Args:
        results: 评估结果列表
        output_path: 可选，输出文件路径
        model_name: 模型名称
        title: 报告标题

    Returns:
        报告内容（Markdown 格式）
    """
    metrics = calculate_metrics(results)

    lines = [
        f"# {title}",
        "",
        "## 配置",
        f"- **模型**: {model_name}",
        f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **样本总数**: {metrics['total_samples']}",
        f"- **成功评估**: {metrics['successful']}",
        f"- **失败评估**: {metrics['failed']}",
        "",
        "## 总体指标",
        f"- **整体 ASR**: {metrics['overall_asr']:.2f}%",
        f"- **平均延迟**: {metrics['avg_latency_ms']:.2f} ms",
        f"- **中位延迟**: {metrics['median_latency_ms']:.2f} ms",
        f"- **平均 Token 数**: {metrics['avg_tokens']:.0f}",
        "",
        "## 各类攻击 ASR",
        "",
        "| 攻击类型 | 总样本数 | 有害样本数 | ASR (%) |",
        "|----------|----------|------------|---------|",
    ]

    for attack_type, stats in sorted(metrics["asr_by_type"].items()):
        lines.append(f"| {attack_type} | {stats['total']} | {stats['unsafe']} | {stats['asr']:.2f} |")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] ASR 报告已保存: {output_path}")

    return report


def generate_utility_report(
    utility_results: Dict[str, Any],
    output_path: Optional[Path] = None,
    model_name: str = "",
    title: str = "Utility Evaluation Report",
) -> str:
    """生成 Utility 评估报告（Markdown 格式）

    Args:
        utility_results: Utility 评估结果（来自 evaluate_utility）
        output_path: 可选，输出文件路径
        model_name: 模型名称
        title: 报告标题

    Returns:
        报告内容（Markdown 格式）
    """
    ZERO_SHOT_TASKS = [
        "hellaswag", "winogrande", "arc_easy", "arc_challenge",
        "obqa", "boolq", "rte",
    ]
    PAPER_BASELINE = {
        "hellaswag": 0.5692, "winogrande": 0.6993, "arc_easy": 0.7534,
        "arc_challenge": 0.4189, "obqa": 0.3440, "boolq": 0.7505,
        "rte": 0.6643, "mean": 0.5999, "wiki_perplexity": 5.68,
    }

    zero_shot = utility_results.get("zero_shot", {})
    wiki_ppl = utility_results.get("wiki_perplexity")
    utility_score = utility_results.get("utility_score", 0.0)

    lines = [
        f"# {title}",
        "",
        "## 配置",
        f"- **模型**: {model_name}",
        f"- **时间**: {utility_results.get('timestamp', datetime.now().isoformat())}",
        "",
        "## 零样本任务准确率",
        "",
        "| 任务 | 实际值 | 论文基准 | 差异 |",
        "|------|--------|----------|------|",
    ]

    for task in ZERO_SHOT_TASKS:
        if task in zero_shot:
            actual = zero_shot[task]
            paper = PAPER_BASELINE.get(task, 0.0)
            diff = actual - paper
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            lines.append(f"| {task} | {actual:.4f} | {paper:.4f} | {diff_str} |")

    if "mean" in zero_shot:
        actual_mean = zero_shot["mean"]
        paper_mean = PAPER_BASELINE["mean"]
        diff = actual_mean - paper_mean
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        lines.append(f"| **平均** | **{actual_mean:.4f}** | **{paper_mean:.4f}** | **{diff_str}** |")

    lines.append("")
    lines.append("## WikiText 困惑度")

    if wiki_ppl:
        paper_ppl = PAPER_BASELINE["wiki_perplexity"]
        diff = wiki_ppl - paper_ppl
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        lines.extend([
            f"- **困惑度**: {wiki_ppl:.4f}",
            f"- **论文基准**: {paper_ppl:.4f}",
            f"- **差异**: {diff_str}",
        ])
    else:
        lines.append("- WikiText 数据不可用")

    lines.extend([
        "",
        "## 综合指标",
        f"- **Utility 分数**: {utility_score:.4f}",
        "",
        "## 与论文基准对比",
    ])

    comparison = utility_results.get("comparison_with_paper", {})
    if comparison:
        lines.append("")
        for task, comp in comparison.items():
            if isinstance(comp, dict) and "actual" in comp:
                lines.append(f"- **{task}**: 实际={comp['actual']:.4f}, 基准={comp['paper_baseline']:.4f}, 差异={comp['difference']:.4f}")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Report] Utility 报告已保存: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate security evaluation report")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file(s), supports glob patterns")
    parser.add_argument("--output", type=str, required=True, help="Output markdown file path")
    parser.add_argument("--model", type=str, default="", help="Model name for report")
    parser.add_argument("--classifier", type=str, default="", help="Classifier name for report")
    
    args = parser.parse_args()
    
    # Handle glob patterns
    input_path = Path(args.input)
    if "*" in str(input_path):
        input_paths = list(input_path.parent.glob(input_path.name))
    else:
        input_paths = [input_path]
    
    print(f"Loading results from {len(input_paths)} file(s)...")
    results = load_results(input_paths)
    print(f"Loaded {len(results)} results")
    
    metrics = calculate_metrics(results)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(metrics, output_path, args.model, args.classifier)
    
    print(f"\nOverall ASR: {metrics['overall_asr']:.2f}%")
    print(f"Average latency: {metrics['avg_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()

