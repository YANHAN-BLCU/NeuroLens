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
from pathlib import Path
from typing import Dict, List


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

