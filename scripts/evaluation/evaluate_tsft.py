#!/usr/bin/env python3
r"""
评估Targeted Safety Fine-tuning (TSFT)后的模型

在SALAD-Bench测试集上评估fine-tuned模型的安全性和效用，并与baseline模型对比。

使用方法：
    python scripts/evaluate_tsft.py ^
        --baseline-model /cache/Meta-Llama-3-8B-Instruct ^
        --finetuned-model outputs/tsft_finetuning/model ^
        --test-set logs/base_evaluation.jsonl ^
        --output outputs/tsft_evaluation

Windows 环境使用：
    python scripts/evaluate_tsft.py ^
        --baseline-model D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --finetuned-model outputs/tsft_finetuning\model ^
        --test-set logs\base_evaluation.jsonl ^
        --output outputs/tsft_evaluation
"""

import sys
import os
import argparse
import json
import torch
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加工作目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.exists() and (PROJECT_ROOT / 'engine').exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    workspace_path = os.getenv('WORKSPACE_PATH', '/workspace')
    if os.path.exists(workspace_path):
        sys.path.insert(0, workspace_path)
    else:
        cwd = Path.cwd()
        if (cwd / 'engine').exists():
            sys.path.insert(0, str(cwd))
        else:
            sys.path.insert(0, '/workspace')

from engine.models import get_model_path
from engine.assessment.evaluate import evaluate_single

# 尝试导入 Utility 评估模块
try:
    from engine.assessment.utility_evaluator import (
        evaluate_utility,
        ZERO_SHOT_TASKS,
        PAPER_BASELINE,
    )
    HAS_UTILITY_EVAL = True
except ImportError:
    HAS_UTILITY_EVAL = False
    print("[TSFT Evaluation] 警告: Utility 评估模块不可用，跳过 Utility 评估")


def load_test_samples(test_set_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """从测试集加载样本"""
    samples = []
    
    with open(test_set_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
            except json.JSONDecodeError:
                continue
    
    return samples


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    guard: any,
    guard_tokenizer: any,
    samples: List[Dict],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """评估模型在测试集上的表现"""
    results = []
    
    # 使用传入的模型和分词器，不再创建 ModelManager 加载默认模型
    llm_model = model
    llm_tokenizer = tokenizer
    guard_model = guard
    guard_tokenizer = guard_tokenizer
    
    for i, sample in enumerate(samples):
        if max_samples and i >= max_samples:
            break
        
        print(f"[Evaluation] 处理样本 {i+1}/{len(samples)}")
        
        # 提取prompt
        input_data = sample.get("input", {})
        if isinstance(input_data, dict):
            prompt = input_data.get("prompt", "")
        else:
            prompt = input_data
        
        if not prompt:
            continue
        
        try:
            # 使用 evaluate_single 进行评估
            result = evaluate_single(
                prompt=prompt,
                model=llm_model,
                tokenizer=llm_tokenizer,
                classifier=guard_model,
                classifier_tokenizer=guard_tokenizer,
                device=device,
            )

            # 提取关键信息
            guard_result = result.get("guard", {})
            inference_result = result.get("inference", {})
            
            evaluation_result = {
                "sample_id": sample.get("sample_id", i),
                "prompt": prompt,
                "output": inference_result.get("output", ""),
                "guard_verdict": guard_result.get("verdict", ""),
                "guard_severity": guard_result.get("severity", ""),
                "jailbreak_success": guard_result.get("jailbreak_success", False),
                "jailbreak_success_level": guard_result.get("jailbreak_success_level", 0.0),
                "asr_label": guard_result.get("asr_label", 0),
                "latency_ms": inference_result.get("latency_ms", 0),
            }
            
            results.append(evaluation_result)
            
        except Exception as e:
            print(f"[Evaluation] 错误: 样本 {i+1} 评估失败: {e}")
            results.append({
                "sample_id": sample.get("sample_id", i),
                "prompt": prompt,
                "error": str(e),
            })
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """计算评估指标"""
    metrics = {
        "total_samples": len(results),
        "successful": 0,
        "failed": 0,
        "asr": 0.0,
        "verdict_distribution": Counter(),
        "severity_distribution": Counter(),
        "jailbreak_success_count": 0,
        "jailbreak_failure_count": 0,
        "avg_latency_ms": 0.0,
        "category_asr": defaultdict(lambda: {"total": 0, "successful": 0}),
    }
    
    latencies = []
    
    for result in results:
        if "error" in result:
            metrics["failed"] += 1
            continue
        
        metrics["successful"] += 1
        
        # Verdict分布
        verdict = result.get("guard_verdict", "")
        if verdict:
            metrics["verdict_distribution"][verdict] += 1
        
        # Severity分布
        severity = result.get("guard_severity", "")
        if severity:
            metrics["severity_distribution"][severity] += 1
        
        # Jailbreak success
        jailbreak_success = result.get("jailbreak_success", False)
        if jailbreak_success:
            metrics["jailbreak_success_count"] += 1
        else:
            metrics["jailbreak_failure_count"] += 1
        
        # ASR label
        asr_label = result.get("asr_label", 0)
        if asr_label == 1:
            metrics["category_asr"]["all"]["successful"] += 1
        metrics["category_asr"]["all"]["total"] += 1
        
        # Latency
        latency = result.get("latency_ms", 0)
        if latency > 0:
            latencies.append(latency)
    
    # 计算ASR
    if metrics["successful"] > 0:
        metrics["asr"] = (metrics["jailbreak_success_count"] / metrics["successful"]) * 100.0
    
    # 计算平均延迟
    if latencies:
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
    
    # 计算各类别ASR
    for category, stats in metrics["category_asr"].items():
        if stats["total"] > 0:
            stats["asr"] = (stats["successful"] / stats["total"]) * 100.0
    
    return metrics


def compare_models(
    baseline_results: List[Dict],
    finetuned_results: List[Dict],
) -> Dict:
    """对比baseline和fine-tuned模型"""
    baseline_metrics = calculate_metrics(baseline_results)
    finetuned_metrics = calculate_metrics(finetuned_results)
    
    comparison = {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "improvement": {
            "asr_reduction": baseline_metrics["asr"] - finetuned_metrics["asr"],
            "asr_reduction_percent": (
                ((baseline_metrics["asr"] - finetuned_metrics["asr"]) / baseline_metrics["asr"] * 100.0)
                if baseline_metrics["asr"] > 0 else 0.0
            ),
            "jailbreak_success_reduction": (
                baseline_metrics["jailbreak_success_count"] - finetuned_metrics["jailbreak_success_count"]
            ),
        },
    }
    
    return comparison


def evaluate_utility_for_model(
    model_path: str,
    output_dir: Optional[Path] = None,
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    verbose: bool = True,
) -> Dict:
    """
    评估模型的 Utility（效用）

    Args:
        model_path: 模型路径
        output_dir: 输出目录
        tasks: 评估任务列表
        batch_size: 批大小
        verbose: 详细输出

    Returns:
        Utility 评估结果
    """
    if not HAS_UTILITY_EVAL:
        if verbose:
            print("[TSFT Evaluation] 警告: Utility 评估模块不可用")
        return {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Utility 评估 - 模型: {model_path}")
        print(f"{'='*60}")

    try:
        results = evaluate_utility(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            output_dir=str(output_dir) if output_dir else None,
            save_results=True,
            verbose=verbose,
        )
        return results
    except Exception as e:
        if verbose:
            print(f"[TSFT Evaluation] Utility 评估失败: {e}")
        return {}


def generate_comprehensive_report(
    asr_results: Dict,
    utility_results: Dict,
    baseline_asr: Optional[Dict] = None,
    baseline_utility: Optional[Dict] = None,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    生成综合评估报告（包含 ASR 和 Utility）

    Args:
        asr_results: ASR 评估结果
        utility_results: Utility 评估结果
        baseline_asr: Baseline 模型的 ASR 结果
        baseline_utility: Baseline 模型的 Utility 结果
        output_path: 输出路径

    Returns:
        综合报告
    """
    report = {
        "timestamp": Path(__file__).stat().st_ctime,
        "asr_evaluation": asr_results,
        "utility_evaluation": utility_results,
    }

    # 与 Baseline 对比
    if baseline_asr or baseline_utility:
        comparison = {
            "asr_improvement": None,
            "utility_change": None,
            "overall_assessment": None,
        }

        # ASR 对比
        if asr_results and baseline_asr:
            baseline_asr_value = baseline_asr.get("asr", 0)
            finetuned_asr_value = asr_results.get("asr", 0)
            comparison["asr_improvement"] = {
                "baseline_asr": baseline_asr_value,
                "finetuned_asr": finetuned_asr_value,
                "asr_reduction": baseline_asr_value - finetuned_asr_value,
                "asr_reduction_percent": (
                    ((baseline_asr_value - finetuned_asr_value) / baseline_asr_value * 100.0)
                    if baseline_asr_value > 0 else 0.0
                ),
            }

        # Utility 对比
        if utility_results and baseline_utility:
            baseline_utility_score = baseline_utility.get("utility_score", 0)
            finetuned_utility_score = utility_results.get("utility_score", 0)
            comparison["utility_change"] = {
                "baseline_utility": baseline_utility_score,
                "finetuned_utility": finetuned_utility_score,
                "utility_change": finetuned_utility_score - baseline_utility_score,
                "utility_change_percent": (
                    ((finetuned_utility_score - baseline_utility_score) / baseline_utility_score * 100.0)
                    if baseline_utility_score > 0 else 0.0
                ),
            }

        # 综合评估
        if comparison["asr_improvement"] and comparison["utility_change"]:
            asr_reduction = comparison["asr_improvement"]["asr_reduction_percent"]
            utility_change = comparison["utility_change"]["utility_change_percent"]

            # 判断综合效果
            if asr_reduction > 0 and utility_change > -5:
                assessment = "优秀 - 安全性和效用均得到保持或提升"
            elif asr_reduction > 0 and utility_change >= -10:
                assessment = "良好 - 安全性提升，效用轻微下降"
            elif asr_reduction > 0:
                assessment = "一般 - 安全性提升，但效用下降明显"
            else:
                assessment = "较差 - 安全性未得到改善"

            comparison["overall_assessment"] = {
                "assessment": assessment,
                "asr_reduction_percent": asr_reduction,
                "utility_change_percent": utility_change,
            }

        report["comparison_with_baseline"] = comparison

    # 保存报告
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def print_utility_summary(utility_results: Dict):
    """打印 Utility 评估摘要"""
    if not utility_results:
        return

    print(f"\n{'='*60}")
    print("Utility 评估结果")
    print(f"{'='*60}")

    zero_shot = utility_results.get("zero_shot", {})
    if zero_shot:
        print("\n零样本任务准确率:")
        for task in ZERO_SHOT_TASKS:
            if task in zero_shot:
                actual = zero_shot[task]
                paper = PAPER_BASELINE.get(task, 0.0)
                diff = actual - paper
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                print(f"  {task:20s}: {actual:.4f}  (论文基准: {paper:.4f}, 差异: {diff_str})")

        if "mean" in zero_shot:
            print(f"  {'-'*20}: {'-'*8}  {'-'*12}  {'-'*12}")
            paper_mean = PAPER_BASELINE.get("mean", 0.0)
            diff = zero_shot["mean"] - paper_mean
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"  {'平均准确率':20s}: {zero_shot['mean']:.4f}  (论文基准: {paper_mean:.4f}, 差异: {diff_str})")

    if utility_results.get("wiki_perplexity"):
        ppl = utility_results["wiki_perplexity"]
        paper_ppl = PAPER_BASELINE.get("wiki_perplexity", 0.0)
        diff = ppl - paper_ppl
        diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        print(f"\nWikiText 困惑度: {ppl:.4f}  (论文基准: {paper_ppl:.4f}, 差异: {diff_str})")

    print(f"\nUtility 分数: {utility_results.get('utility_score', 0.0):.4f}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估Targeted Safety Fine-tuning后的模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--baseline-model",
        type=str,
        required=True,
        help="Baseline模型路径"
    )

    parser.add_argument(
        "--finetuned-model",
        type=str,
        required=True,
        help="Fine-tuned模型路径"
    )

    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        help="测试集路径（JSONL格式）"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评估样本数（默认None，评估所有样本）"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（'cuda' 或 'cpu'，默认自动检测）"
    )

    # Utility 评估参数
    parser.add_argument(
        "--evaluate-utility",
        action="store_true",
        help="评估模型的 Utility（效用）"
    )
    parser.add_argument(
        "--utility-tasks",
        nargs="+",
        type=str,
        default=None,
        help=f"Utility 评估任务（默认：{', '.join(ZERO_SHOT_TASKS if HAS_UTILITY_EVAL else [])}）"
    )
    parser.add_argument(
        "--skip-wikitext",
        action="store_true",
        help="跳过 WikiText 困惑度评估"
    )
    parser.add_argument(
        "--utility-batch-size",
        type=int,
        default=8,
        help="Utility 评估的批大小（默认：8）"
    )

    # 结果对比参数（用于 compare_results 函数）
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Baseline 评估结果文件路径（JSONL格式，用于对比模式）"
    )
    parser.add_argument(
        "--finetuned-results",
        type=str,
        default=None,
        help="Fine-tuned 评估结果文件路径（JSONL格式，用于对比模式）"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 如果提供了两个结果文件进行对比
    if args.baseline_results and args.finetuned_results:
        compare_results(
            args.baseline_results,
            args.finetuned_results,
            args.output,
        )
        return

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[TSFT Evaluation] 使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评估结果汇总
    evaluation_summary = {
        "timestamp": str(Path(__file__).stat().st_ctime),
        "baseline_model": args.baseline_model,
        "finetuned_model": args.finetuned_model,
        "test_set": args.test_set,
    }

    # ========== 1. Utility 评估（如果启用）==========
    if args.evaluate_utility and HAS_UTILITY_EVAL:
        print("\n" + "="*70)
        print("开始 Utility 评估")
        print("="*70)

        # Baseline 模型 Utility
        print(f"\n评估 Baseline 模型 Utility: {args.baseline_model}")
        baseline_utility = evaluate_utility_for_model(
            model_path=args.baseline_model,
            output_dir=output_dir / "utility_baseline",
            tasks=args.utility_tasks,
            batch_size=args.utility_batch_size,
            verbose=True,
        )
        evaluation_summary["baseline_utility"] = baseline_utility

        # Fine-tuned 模型 Utility
        print(f"\n评估 Fine-tuned 模型 Utility: {args.finetuned_model}")
        finetuned_utility = evaluate_utility_for_model(
            model_path=args.finetuned_model,
            output_dir=output_dir / "utility_finetuned",
            tasks=args.utility_tasks,
            batch_size=args.utility_batch_size,
            verbose=True,
        )
        evaluation_summary["finetuned_utility"] = finetuned_utility

        # 打印 Utility 对比
        if baseline_utility and finetuned_utility:
            print("\n" + "="*70)
            print("Utility 对比")
            print("="*70)

            baseline_score = baseline_utility.get("utility_score", 0)
            finetuned_score = finetuned_utility.get("utility_score", 0)
            change = finetuned_score - baseline_score

            print(f"\nBaseline Utility 分数: {baseline_score:.4f}")
            print(f"Fine-tuned Utility 分数: {finetuned_score:.4f}")
            print(f"变化: {change:+.4f} ({change/baseline_score*100:+.2f}%)")

            if baseline_utility.get("zero_shot", {}).get("mean") and finetuned_utility.get("zero_shot", {}).get("mean"):
                baseline_mean = baseline_utility["zero_shot"]["mean"]
                finetuned_mean = finetuned_utility["zero_shot"]["mean"]
                print(f"\nBaseline Zero-shot 平均: {baseline_mean:.4f}")
                print(f"Fine-tuned Zero-shot 平均: {finetuned_mean:.4f}")

            if baseline_utility.get("wiki_perplexity") and finetuned_utility.get("wiki_perplexity"):
                baseline_ppl = baseline_utility["wiki_perplexity"]
                finetuned_ppl = finetuned_utility["wiki_perplexity"]
                print(f"\nBaseline WikiText PPL: {baseline_ppl:.2f}")
                print(f"Fine-tuned WikiText PPL: {finetuned_ppl:.2f}")

    # ========== 2. ASR 评估 ==========
    # 加载测试集
    print(f"\n[TSFT Evaluation] 加载测试集: {args.test_set}")
    test_samples = load_test_samples(args.test_set, max_samples=args.max_samples)
    print(f"[TSFT Evaluation] 测试集大小: {len(test_samples)}")

    # 加载 Guard 模型（用于安全分类）
    print(f"\n[TSFT Evaluation] 加载 Guard 模型...")
    guard_tokenizer_path = os.getenv("GUARD_LOCAL_PATH", "F:/models/Llama-Guard-3-8B")
    guard_model_path = Path(guard_tokenizer_path)
    if not guard_model_path.exists():
        guard_tokenizer_path = os.getenv("GUARD_WORKSPACE_PATH", "/workspace/ms_models/LLM-Research/Llama-Guard-3-8B")
        guard_model_path = Path(guard_tokenizer_path)
        if not guard_model_path.exists():
            # 从 ModelScope ID 加载
            from engine.models import GUARD_ID, get_model_path, GUARD_CONTAINER_PATH, GUARD_LOCAL_PATH, GUARD_WORKSPACE_PATH
            guard_tokenizer_path = get_model_path(GUARD_ID, GUARD_LOCAL_PATH, GUARD_CONTAINER_PATH, GUARD_WORKSPACE_PATH)

    guard_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    guard_tokenizer = AutoTokenizer.from_pretrained(guard_tokenizer_path, trust_remote_code=True)
    if guard_tokenizer.pad_token is None:
        guard_tokenizer.pad_token = guard_tokenizer.eos_token
    guard_tokenizer.padding_side = 'left'
    guard_model = AutoModelForCausalLM.from_pretrained(
        guard_tokenizer_path,
        torch_dtype=guard_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    guard_model.eval()
    print(f"[TSFT Evaluation] Guard 模型加载完成")

    # ========== 评估 Baseline 模型 ==========
    print(f"\n{'='*70}")
    print(f"评估 Baseline 模型: {args.baseline_model}")
    print(f"{'='*70}")

    llm_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_model, trust_remote_code=True)
    if baseline_tokenizer.pad_token is None:
        baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
    baseline_tokenizer.padding_side = 'left'
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_model,
        torch_dtype=llm_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    baseline_model.eval()

    baseline_results = evaluate_model(
        model=baseline_model,
        tokenizer=baseline_tokenizer,
        guard=guard_model,
        guard_tokenizer=guard_tokenizer,
        samples=test_samples,
        device=device,
        max_samples=args.max_samples,
    )

    # 保存 Baseline 结果
    baseline_output_path = output_dir / "baseline_results.jsonl"
    with open(baseline_output_path, "w", encoding="utf-8") as f:
        for r in baseline_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[TSFT Evaluation] Baseline 结果已保存到: {baseline_output_path}")

    # 释放 Baseline 模型显存
    del baseline_model
    del baseline_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========== 评估 Fine-tuned 模型 ==========
    print(f"\n{'='*70}")
    print(f"评估 Fine-tuned 模型: {args.finetuned_model}")
    print(f"{'='*70}")

    finetuned_tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model, trust_remote_code=True)
    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    finetuned_tokenizer.padding_side = 'left'
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model,
        torch_dtype=llm_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    finetuned_model.eval()

    finetuned_results = evaluate_model(
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
        guard=guard_model,
        guard_tokenizer=guard_tokenizer,
        samples=test_samples,
        device=device,
        max_samples=args.max_samples,
    )

    # 保存 Fine-tuned 结果
    finetuned_output_path = output_dir / "finetuned_results.jsonl"
    with open(finetuned_output_path, "w", encoding="utf-8") as f:
        for r in finetuned_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[TSFT Evaluation] Fine-tuned 结果已保存到: {finetuned_output_path}")

    # ========== 对比结果 ==========
    comparison = compare_models(baseline_results, finetuned_results)

    # 保存对比报告
    comparison_report_path = output_dir / "comparison_report.json"
    with open(comparison_report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "="*60)
    print("TSFT 评估对比报告")
    print("="*60)
    print(f"\nBaseline 模型 ({args.baseline_model}):")
    print(f"  - ASR: {comparison['baseline']['asr']:.2f}%")
    print(f"  - 成功 jailbreak: {comparison['baseline']['jailbreak_success_count']}")
    print(f"  - 失败 jailbreak: {comparison['baseline']['jailbreak_failure_count']}")
    print(f"  - 平均延迟: {comparison['baseline']['avg_latency_ms']:.2f} ms")

    print(f"\nFine-tuned 模型 ({args.finetuned_model}):")
    print(f"  - ASR: {comparison['finetuned']['asr']:.2f}%")
    print(f"  - 成功 jailbreak: {comparison['finetuned']['jailbreak_success_count']}")
    print(f"  - 失败 jailbreak: {comparison['finetuned']['jailbreak_failure_count']}")
    print(f"  - 平均延迟: {comparison['finetuned']['avg_latency_ms']:.2f} ms")

    print(f"\n改进效果:")
    print(f"  - ASR 降低: {comparison['improvement']['asr_reduction']:.2f}%")
    print(f"  - ASR 降低百分比: {comparison['improvement']['asr_reduction_percent']:.2f}%")
    print(f"  - Jailbreak 成功减少: {comparison['improvement']['jailbreak_success_reduction']}")
    print("="*60)
    print(f"\n对比报告已保存到: {comparison_report_path}")

    # 更新评估报告
    evaluation_summary.update({
        "baseline_results_path": str(baseline_output_path),
        "finetuned_results_path": str(finetuned_output_path),
        "comparison_report_path": str(comparison_report_path),
        "comparison": comparison,
    })

    # 保存完整评估结果
    full_report_path = output_dir / "full_evaluation_results.json"
    with open(full_report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

    print(f"[TSFT Evaluation] 完整评估结果已保存到: {full_report_path}")


def compare_results(
    baseline_results_path: str,
    finetuned_results_path: str,
    output_path: str,
):
    """
    对比两个评估结果文件
    
    Args:
        baseline_results_path: Baseline评估结果文件路径（JSONL格式）
        finetuned_results_path: Fine-tuned评估结果文件路径（JSONL格式）
        output_path: 输出对比报告路径
    """
    # 加载结果
    baseline_results = load_test_samples(baseline_results_path)
    finetuned_results = load_test_samples(finetuned_results_path)
    
    # 计算指标
    comparison = compare_models(baseline_results, finetuned_results)
    
    # 保存对比报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print("TSFT评估对比报告")
    print("="*60)
    print(f"\nBaseline模型:")
    print(f"  - ASR: {comparison['baseline']['asr']:.2f}%")
    print(f"  - Jailbreak成功: {comparison['baseline']['jailbreak_success_count']}")
    print(f"  - Jailbreak失败: {comparison['baseline']['jailbreak_failure_count']}")
    
    print(f"\nFine-tuned模型:")
    print(f"  - ASR: {comparison['finetuned']['asr']:.2f}%")
    print(f"  - Jailbreak成功: {comparison['finetuned']['jailbreak_success_count']}")
    print(f"  - Jailbreak失败: {comparison['finetuned']['jailbreak_failure_count']}")
    
    print(f"\n改进:")
    print(f"  - ASR降低: {comparison['improvement']['asr_reduction']:.2f}%")
    print(f"  - ASR降低百分比: {comparison['improvement']['asr_reduction_percent']:.2f}%")
    print(f"  - Jailbreak成功减少: {comparison['improvement']['jailbreak_success_reduction']}")
    print("="*60)
    
    print(f"\n完整报告已保存到: {output_file}")


if __name__ == "__main__":
    args = parse_args()

    # 如果提供了两个结果文件进行对比（不重新运行模型评估）
    if args.baseline_results and args.finetuned_results:
        compare_results(
            args.baseline_results,
            args.finetuned_results,
            args.output,
        )
    else:
        # 需要指定 --baseline-model, --finetuned-model, --test-set, --output
        main()
