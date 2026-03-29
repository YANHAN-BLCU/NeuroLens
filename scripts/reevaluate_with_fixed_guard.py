#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选择性重评估脚本 - 修复 toxic 假阳性问题

策略：
1. 对所有样本进行二次 Guard 审核
2. 比较新旧结果，标记有变化的样本
3. 生成修复报告
4. 可选：直接修复或仅标记

使用方法：
    python scripts/reevaluate_with_fixed_guard.py --input logs/base_evaluation.jsonl --output fixed_evaluation.jsonl
    python scripts/reevaluate_with_fixed_guard.py --input logs/base_evaluation.jsonl --compare-only  # 仅比较，不修改
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from engine.models import ModelManager
except ImportError:
    print("错误: 无法导入 ModelManager，请确保项目路径正确")
    sys.exit(1)


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)


def load_data(file_path: Path) -> Tuple[List[dict], dict]:
    """加载数据集"""
    samples = []
    stats = {
        "total": 0,
        "verdict_distribution": {},
        "asr_label_distribution": {},
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                samples.append(record)
                stats["total"] += 1
                
                guard = record.get('guard', {})
                verdict = guard.get('verdict', 'UNKNOWN')
                asr_label = guard.get('asr_label', -1)
                
                stats["verdict_distribution"][verdict] = stats["verdict_distribution"].get(verdict, 0) + 1
                stats["asr_label_distribution"][asr_label] = stats["asr_label_distribution"].get(asr_label, 0) + 1
            except json.JSONDecodeError:
                continue
    
    return samples, stats


def compare_results(old_guard: dict, new_guard: dict) -> dict:
    """比较新旧 Guard 结果"""
    comparison = {
        "verdict_changed": old_guard.get('verdict') != new_guard.get('verdict'),
        "asr_label_changed": old_guard.get('asr_label') != new_guard.get('asr_label'),
        "severity_changed": old_guard.get('severity') != new_guard.get('severity'),
        "details": {}
    }
    
    # 判断变化类型
    old_asr = old_guard.get('asr_label', -1)
    new_asr = new_guard.get('asr_label', -1)
    
    if old_asr == 1 and new_asr == 0:
        comparison["change_type"] = "FALSE_POSITIVE_FIXED"  # 假阳性修复
        comparison["details"]["description"] = "原标记 toxic，现修复为 safe"
    elif old_asr == 0 and new_asr == 1:
        comparison["change_type"] = "FALSE_NEGATIVE_FIXED"  # 假阴性修复
        comparison["details"]["description"] = "原标记 safe，现修复为 toxic"
    elif old_asr == 1 and new_asr == 0.5:
        comparison["change_type"] = "SEVERITY_REDUCED"  # 严重程度降低
        comparison["details"]["description"] = "原 asr_label=1，现 asr_label=0.5（软标签）"
    else:
        comparison["change_type"] = "OTHER"
        comparison["details"]["description"] = f"asr_label: {old_asr} -> {new_asr}"
    
    return comparison


def reevaluate_samples(
    samples: List[dict],
    model_manager: ModelManager,
    sample_rate: float = 1.0,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[dict], dict]:
    """
    使用改进的 Guard 逻辑重新评估样本
    
    Args:
        samples: 原始样本列表
        model_manager: ModelManager 实例
        sample_rate: 采样率 (0.0-1.0)，用于快速验证
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        (修复后的样本列表, 统计信息)
    """
    set_seed(seed)
    
    results = []
    stats = {
        "processed": 0,
        "fixed_false_positive": 0,  # 修复的假阳性
        "fixed_false_negative": 0,  # 修复的假阴性
        "severity_reduced": 0,      # 严重程度降低
        "unchanged": 0,             # 无变化
        "verdict_changes": {},      # verdict 变化统计
        "asr_label_changes": {},    # asr_label 变化统计
    }
    
    # 随机采样（如果需要快速验证）
    if sample_rate < 1.0:
        indices = list(range(len(samples)))
        random.shuffle(indices)
        selected_indices = indices[:int(len(samples) * sample_rate)]
        samples = [samples[i] for i in selected_indices]
        if verbose:
            print(f"[Info] 采样 {sample_rate*100:.1f}% 样本进行验证，共 {len(samples)} 条")
    
    total = len(samples)
    
    for idx, record in enumerate(samples):
        # 提取需要重新审核的内容
        inference = record.get('inference', {})
        output = inference.get('output', '')
        
        if not output:
            results.append(record)
            continue
        
        # 调用改进的 Guard
        new_guard_result = model_manager.moderate(output)
        old_guard = record.get('guard', {})
        
        # 比较结果
        comparison = compare_results(old_guard, new_guard_result)
        
        # 统计
        stats["processed"] += 1
        if comparison["verdict_changed"]:
            old_v = old_guard.get('verdict', 'UNKNOWN')
            new_v = new_guard_result.get('verdict', 'UNKNOWN')
            key = f"{old_v} -> {new_v}"
            stats["verdict_changes"][key] = stats["verdict_changes"].get(key, 0) + 1
        
        if comparison["asr_label_changed"]:
            old_a = old_guard.get('asr_label', -1)
            new_a = new_guard_result.get('asr_label', -1)
            key = f"{old_a} -> {new_a}"
            stats["asr_label_changes"][key] = stats["asr_label_changes"].get(key, 0) + 1
        
        change_type = comparison.get("change_type", "OTHER")
        if change_type == "FALSE_POSITIVE_FIXED":
            stats["fixed_false_positive"] += 1
        elif change_type == "FALSE_NEGATIVE_FIXED":
            stats["fixed_false_negative"] += 1
        elif change_type == "SEVERITY_REDUCED":
            stats["severity_reduced"] += 1
        else:
            stats["unchanged"] += 1
        
        # 创建新记录（保留原始结构，更新 guard 字段）
        new_record = record.copy()
        new_record['guard'] = new_guard_result
        new_record['reevaluation'] = {
            "timestamp": datetime.utcnow().isoformat(),
            "original_guard": old_guard,
            "comparison": comparison
        }
        
        results.append(new_record)
        
        # 打印进度
        if verbose and (idx + 1) % 100 == 0:
            print(f"  进度: {idx+1}/{total} ({((idx+1)/total*100):.1f}%) | "
                  f"修复假阳性: {stats['fixed_false_positive']} | "
                  f"修复假阴性: {stats['fixed_false_negative']}")
    
    return results, stats


def print_comparison_report(original_stats: dict, new_stats: dict) -> None:
    """打印对比报告"""
    print()
    print("=" * 70)
    print("重新评估报告")
    print("=" * 70)
    print()
    
    print("【处理统计】")
    print(f"  处理样本数: {new_stats['processed']}")
    print(f"  修复假阳性: {new_stats['fixed_false_positive']} "
          f"({new_stats['fixed_false_positive']/max(new_stats['processed'],1)*100:.2f}%)")
    print(f"  修复假阴性: {new_stats['fixed_false_negative']} "
          f"({new_stats['fixed_false_negative']/max(new_stats['processed'],1)*100:.2f}%)")
    print(f"  严重程度降低: {new_stats['severity_reduced']} "
          f"({new_stats['severity_reduced']/max(new_stats['processed'],1)*100:.2f}%)")
    print(f"  无变化: {new_stats['unchanged']} "
          f"({new_stats['unchanged']/max(new_stats['processed'],1)*100:.2f}%)")
    print()
    
    print("【Verdict 变化】")
    for change, count in sorted(new_stats['verdict_changes'].items(), key=lambda x: -x[1]):
        print(f"  {change}: {count}")
    print()
    
    print("【ASR Label 变化】")
    for change, count in sorted(new_stats['asr_label_changes'].items(), key=lambda x: -x[1]):
        print(f"  {change}: {count}")
    print()
    
    # 计算修复后的分布
    print("【修复后 Verdict 分布预估】")
    original_verdict = original_stats.get('verdict_distribution', {})
    # 估算修复后的分布（不准确，但供参考）
    estimated_changes = {
        'UNKNOWN -> SAFE': new_stats['fixed_false_positive'],
        'UNKNOWN -> allow': new_stats['fixed_false_positive'],
        'flag -> SAFE': new_stats['fixed_false_positive'],
        'flag -> allow': new_stats['fixed_false_positive'],
    }
    
    print(f"  原始 allow: {original_verdict.get('allow', 0)}")
    print(f"  原始 flag: {original_verdict.get('flag', 0)}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="使用改进的 Guard 逻辑重新评估数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整重新评估并保存
  python scripts/reevaluate_with_fixed_guard.py --input logs/base_evaluation.jsonl --output fixed.jsonl
  
  # 仅采样 10%% 进行验证（快速测试）
  python scripts/reevaluate_with_fixed_guard.py --input logs/base_evaluation.jsonl --sample-rate 0.1 --verbose
  
  # 仅比较，不修改数据
  python scripts/reevaluate_with_fixed_guard.py --input logs/base_evaluation.jsonl --compare-only
        """
    )
    
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="输入文件路径 (JSONL 格式)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="输出文件路径 (可选)")
    parser.add_argument("--sample-rate", "-s", type=float, default=1.0,
                        help="采样率 0.0-1.0，默认 1.0 (全部)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--compare-only", action="store_true",
                        help="仅比较，不修改数据")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细信息")
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.input.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    if args.sample_rate < 0.0 or args.sample_rate > 1.0:
        print("错误: 采样率必须在 0.0-1.0 之间")
        sys.exit(1)
    
    print("=" * 70)
    print("Guard 重新评估工具")
    print("=" * 70)
    print()
    print(f"输入文件: {args.input}")
    print(f"采样率: {args.sample_rate*100:.1f}%")
    print(f"模式: {'仅比较' if args.compare_only else '评估并修复'}")
    print()
    
    # 加载原始数据
    print("[1/4] 加载原始数据...")
    samples, original_stats = load_data(args.input)
    print(f"  加载 {original_stats['total']} 条样本")
    print(f"  Verdict 分布: {original_stats['verdict_distribution']}")
    
    # 加载模型
    print("[2/4] 加载 Guard 模型...")
    try:
        model_manager = ModelManager()
        print("  模型加载成功")
    except Exception as e:
        print(f"  错误: 模型加载失败: {e}")
        sys.exit(1)
    
    # 重新评估
    print("[3/4] 重新评估样本...")
    if args.compare_only:
        print("  [仅比较模式] 使用改进逻辑评估样本...")
    else:
        print("  [修复模式] 使用改进逻辑评估并修复样本...")
    
    new_samples, new_stats = reevaluate_samples(
        samples,
        model_manager,
        sample_rate=args.sample_rate,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # 打印报告
    print("[4/4] 生成报告...")
    print_comparison_report(original_stats, new_stats)
    
    # 保存结果
    if not args.compare_only and args.output:
        print(f"保存修复后的数据到: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            for record in new_samples:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print("保存完成!")
    
    # 提示下一步
    if args.compare_only:
        print()
        print("【建议】如果修复效果满意，可以运行以下命令进行完整修复:")
        print(f"  python scripts/reevaluate_with_fixed_guard.py --input {args.input} --output fixed_{args.input.name}")
    else:
        print()
        print("【后续步骤】")
        print(f"1. 查看修复后的数据: head -10 {args.output}")
        print("2. 使用修复后的数据重新训练:")
        print(f"   python scripts/train_probes_balanced.py --data_file {args.output}")
        print("3. 比较修复前后的训练效果")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

