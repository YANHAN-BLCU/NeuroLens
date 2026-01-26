#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同训练集比例（train_safe_ratio）的效果

功能：
1. 使用部分数据（通过max_samples）进行快速测试
2. 保持数据划分一致（相同的seed和max_samples）
3. 测试多个train_safe_ratio值（1.0, 2.0, 3.0）
4. 比较训练结果（准确率、数据利用率等）
5. 生成对比报告

使用方法：
    python scripts/test_train_ratios.py \
        --max_samples 2000 \
        --test_ratios 1.0 2.0 3.0 \
        --num_epochs 20 \
        --output_dir outputs/ratio_tests
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm


def run_training(
    train_safe_ratio: float,
    max_samples: int,
    test_ratio: float,
    val_ratio: float,
    num_epochs: int,
    seed: int,
    data_file: Path,
    output_dir: Path,
    lr: float = 0.002,
    probe_batch_size: int = 32,
    batch_size: int = 8,
) -> Dict:
    """运行一次训练，返回结果"""
    
    # 使用 tqdm.write 避免与进度条冲突
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"测试 train_safe_ratio = {train_safe_ratio}")
    tqdm.write(f"{'='*60}\n")
    
    # 创建输出目录
    ratio_output_dir = output_dir / f"ratio_{train_safe_ratio}"
    ratio_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/train_linear_probes.py",
        "--use_optimized_split",
        "--train_safe_ratio", str(train_safe_ratio),
        "--max_samples", str(max_samples),
        "--test_ratio", str(test_ratio),
        "--val_ratio", str(val_ratio),
        "--num_epochs", str(num_epochs),
        "--seed", str(seed),
        "--data_file", str(data_file),
        "--output_dir", str(ratio_output_dir),
        "--lr", str(lr),
        "--probe_batch_size", str(probe_batch_size),
        "--batch_size", str(batch_size),
    ]
    
    tqdm.write(f"执行命令: {' '.join(cmd)}\n")
    
    # 运行训练
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # 读取训练日志（实际路径是 output_dir/probes/llama-3-8b/training_log.json）
        probes_dir = ratio_output_dir / "probes" / "llama-3-8b"
        log_file = probes_dir / "training_log.json"
        summary_file = probes_dir / "summary.json"
        
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 提取关键指标
            metrics = extract_metrics(log_data)
            metrics['train_safe_ratio'] = train_safe_ratio
            metrics['output_dir'] = str(ratio_output_dir)
            
            return metrics
        elif summary_file.exists():
            # 备选方案：如果 training_log.json 不存在，尝试读取 summary.json
            tqdm.write(f"警告: training_log.json 不存在，尝试读取 summary.json")
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # 从 summary.json 提取指标
            metrics = {
                'train_safe_ratio': train_safe_ratio,
                'num_layers': summary_data.get('num_layers', 0),
                'layers_passed': len(summary_data.get('valid_layers', [])),
                'layers_failed': len(summary_data.get('invalid_layers', [])),
                'avg_train_acc': 0.0,
                'avg_val_acc': 0.0,
                'best_val_acc': 0.0,
                'worst_val_acc': 1.0,
                'output_dir': str(ratio_output_dir),
            }
            
            # 从 summary 中提取准确率信息
            layers = summary_data.get('layers', {})
            val_accs = []
            for layer_data in layers.values():
                val_acc = layer_data.get('val_acc', 0.0)
                if val_acc > 0:
                    val_accs.append(val_acc)
                    metrics['best_val_acc'] = max(metrics['best_val_acc'], val_acc)
                    metrics['worst_val_acc'] = min(metrics['worst_val_acc'], val_acc)
            
            if val_accs:
                metrics['avg_val_acc'] = sum(val_accs) / len(val_accs)
            
            return metrics
        else:
            # 检查输出目录是否存在
            if not probes_dir.exists():
                tqdm.write(f"错误: 输出目录不存在: {probes_dir}")
                tqdm.write(f"可能原因: 训练在数据划分阶段就失败了（如训练集为空）")
            else:
                tqdm.write(f"警告: 训练日志文件不存在: {log_file}")
                tqdm.write(f"检查目录: {probes_dir}")
                tqdm.write(f"目录内容: {list(probes_dir.iterdir()) if probes_dir.exists() else '目录不存在'}")
            
            return {
                'train_safe_ratio': train_safe_ratio,
                'error': '训练日志不存在',
            }
            
    except subprocess.CalledProcessError as e:
        tqdm.write(f"错误: 训练失败")
        tqdm.write(f"stdout: {e.stdout}")
        tqdm.write(f"stderr: {e.stderr}")
        return {
            'train_safe_ratio': train_safe_ratio,
            'error': str(e),
        }


def extract_metrics(log_data: Dict) -> Dict:
    """从训练日志中提取关键指标"""
    
    metrics = {
        'num_layers': log_data.get('num_layers', 0),
        'layers_passed': 0,  # 满足要求的层数
        'layers_failed': 0,  # 未满足要求的层数
        'avg_train_acc': 0.0,
        'avg_val_acc': 0.0,
        'avg_train_loss': 0.0,
        'avg_val_loss': 0.0,
        'best_val_acc': 0.0,
        'worst_val_acc': 1.0,
        'layer_results': {},
    }
    
    layers = log_data.get('layers', {})
    if not layers:
        return metrics
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    for layer_idx, layer_data in layers.items():
        layer_metrics = layer_data.get('metrics', {})
        training_history = layer_data.get('training_history', {})
        
        # 统计满足要求的层数
        if layer_metrics.get('meets_requirement', False):
            metrics['layers_passed'] += 1
        else:
            metrics['layers_failed'] += 1
        
        # 提取准确率和损失
        train_acc = layer_metrics.get('train_acc', 0.0)
        val_acc = layer_metrics.get('val_acc', 0.0)
        train_loss = layer_metrics.get('train_loss', 0.0)
        val_loss = layer_metrics.get('val_loss', 0.0)
        
        if train_acc > 0:
            train_accs.append(train_acc)
        if val_acc > 0:
            val_accs.append(val_acc)
            metrics['best_val_acc'] = max(metrics['best_val_acc'], val_acc)
            metrics['worst_val_acc'] = min(metrics['worst_val_acc'], val_acc)
        if train_loss > 0:
            train_losses.append(train_loss)
        if val_loss > 0:
            val_losses.append(val_loss)
        
        # 保存每层结果
        metrics['layer_results'][layer_idx] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'meets_requirement': layer_metrics.get('meets_requirement', False),
        }
    
    # 计算平均值
    if train_accs:
        metrics['avg_train_acc'] = sum(train_accs) / len(train_accs)
    if val_accs:
        metrics['avg_val_acc'] = sum(val_accs) / len(val_accs)
    if train_losses:
        metrics['avg_train_loss'] = sum(train_losses) / len(train_losses)
    if val_losses:
        metrics['avg_val_loss'] = sum(val_losses) / len(val_losses)
    
    return metrics


def analyze_data_split(
    max_samples: int,
    test_ratio: float,
    val_ratio: float,
    train_safe_ratio: float,
) -> Dict:
    """分析数据划分情况（不实际运行训练）"""
    
    # 估算数据分布（基于实际数据）
    # 假设安全:有害 = 82.3%:17.7%
    total_samples = max_samples
    safe_ratio = 0.823
    toxic_ratio = 0.177
    
    total_safe = int(total_samples * safe_ratio)
    total_toxic = int(total_samples * toxic_ratio)
    
    # 第一步：划分测试集
    test_size = int(total_samples * test_ratio)
    test_safe = int(test_size * safe_ratio)
    test_toxic = int(test_size * toxic_ratio)
    
    remaining_safe = total_safe - test_safe
    remaining_toxic = total_toxic - test_toxic
    
    # 第二步：划分验证集
    val_size = int(total_samples * val_ratio)
    val_safe = int(val_size * safe_ratio)
    val_toxic = int(val_size * toxic_ratio)
    
    remaining_safe -= val_safe
    remaining_toxic -= val_toxic
    
    # 第三步：训练集
    train_toxic = remaining_toxic  # 使用所有剩余有害样本
    train_safe = min(int(train_toxic * train_safe_ratio), remaining_safe)
    
    train_total = train_safe + train_toxic
    used_total = test_size + val_size + train_total
    utilization = used_total / total_samples if total_samples > 0 else 0.0
    
    return {
        'total_samples': total_samples,
        'test_size': test_size,
        'test_safe': test_safe,
        'test_toxic': test_toxic,
        'val_size': val_size,
        'val_safe': val_safe,
        'val_toxic': val_toxic,
        'train_size': train_total,
        'train_safe': train_safe,
        'train_toxic': train_toxic,
        'train_ratio': train_safe / train_toxic if train_toxic > 0 else 0.0,
        'utilization': utilization,
        'unused_safe': remaining_safe - train_safe,
    }


def generate_report(
    results: List[Dict],
    data_analysis: List[Dict],
    output_file: Path,
):
    """生成对比报告"""
    
    report = []
    report.append("# 训练集比例对比测试报告\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 数据划分对比
    report.append("## 1. 数据划分对比\n\n")
    report.append("| 比例 | 测试集 | 验证集 | 训练集 | 训练比例 | 数据利用率 |\n")
    report.append("|------|--------|--------|--------|----------|------------|\n")
    
    for analysis in data_analysis:
        ratio = analysis['train_ratio']
        report.append(
            f"| {ratio:.1f}:1 | "
            f"{analysis['test_size']} | "
            f"{analysis['val_size']} | "
            f"{analysis['train_size']} | "
            f"{ratio:.1f}:1 | "
            f"{analysis['utilization']:.1%} |\n"
        )
    
    report.append("\n")
    
    # 训练结果对比
    report.append("## 2. 训练结果对比\n\n")
    report.append("| 比例 | 通过层数 | 失败层数 | 平均训练准确率 | 平均验证准确率 | 最佳验证准确率 | 最差验证准确率 |\n")
    report.append("|------|----------|----------|----------------|----------------|----------------|----------------|\n")
    
    for result in results:
        if 'error' in result:
            continue
        
        ratio = result['train_safe_ratio']
        report.append(
            f"| {ratio:.1f}:1 | "
            f"{result.get('layers_passed', 0)} | "
            f"{result.get('layers_failed', 0)} | "
            f"{result.get('avg_train_acc', 0.0):.3f} | "
            f"{result.get('avg_val_acc', 0.0):.3f} | "
            f"{result.get('best_val_acc', 0.0):.3f} | "
            f"{result.get('worst_val_acc', 1.0):.3f} |\n"
        )
    
    report.append("\n")
    
    # 详细分析
    report.append("## 3. 详细分析\n\n")
    
    for result in results:
        if 'error' in result:
            report.append(f"### 比例 {result['train_safe_ratio']:.1f}:1 - 训练失败\n\n")
            report.append(f"错误: {result['error']}\n\n")
            continue
        
        ratio = result['train_safe_ratio']
        report.append(f"### 比例 {ratio:.1f}:1\n\n")
        report.append(f"- **通过层数**: {result.get('layers_passed', 0)}/{result.get('num_layers', 0)}\n")
        report.append(f"- **平均训练准确率**: {result.get('avg_train_acc', 0.0):.3f}\n")
        report.append(f"- **平均验证准确率**: {result.get('avg_val_acc', 0.0):.3f}\n")
        report.append(f"- **过拟合程度**: {result.get('avg_train_acc', 0.0) - result.get('avg_val_acc', 0.0):.3f}\n")
        report.append(f"- **输出目录**: {result.get('output_dir', 'N/A')}\n\n")
    
    # 推荐
    report.append("## 4. 推荐\n\n")
    
    # 找出最佳比例
    best_ratio = None
    best_score = -1
    
    for result in results:
        if 'error' in result:
            continue
        
        # 综合评分：通过层数 + 验证准确率 - 过拟合程度
        score = (
            result.get('layers_passed', 0) * 10 +
            result.get('avg_val_acc', 0.0) * 100 -
            (result.get('avg_train_acc', 0.0) - result.get('avg_val_acc', 0.0)) * 50
        )
        
        if score > best_score:
            best_score = score
            best_ratio = result['train_safe_ratio']
    
    if best_ratio:
        report.append(f"**推荐使用比例**: {best_ratio:.1f}:1\n\n")
        report.append(f"理由:\n")
        report.append(f"- 综合评分最高\n")
        report.append(f"- 平衡了通过层数和验证准确率\n")
        report.append(f"- 过拟合程度较低\n")
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"\n报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="测试不同训练集比例（train_safe_ratio）的效果"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="最大样本数（用于快速测试，默认2000）",
    )
    parser.add_argument(
        "--test_ratios",
        type=float,
        nargs='+',
        default=[1.0, 2.0, 3.0],
        help="要测试的训练集比例列表（默认: 1.0 2.0 3.0）",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例（默认0.15）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例（默认0.15）",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="训练轮数（用于快速测试，默认20）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（保证数据划分一致，默认42）",
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        default=Path("data/salad/raw/base_evaluation.jsonl"),
        help="数据文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/ratio_tests"),
        help="输出目录",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="学习率（默认0.002）",
    )
    parser.add_argument(
        "--probe_batch_size",
        type=int,
        default=32,
        help="探针训练批大小（默认32）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="提取隐藏态时的批大小（默认8）",
    )
    
    args = parser.parse_args()
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cpu":
        print("警告: 未检测到GPU，训练速度可能较慢")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析数据划分
    print("\n" + "="*60)
    print("数据划分分析")
    print("="*60 + "\n")
    
    data_analysis = []
    for ratio in tqdm(args.test_ratios, desc="分析数据划分", unit="比例"):
        analysis = analyze_data_split(
            args.max_samples,
            args.test_ratio,
            args.val_ratio,
            ratio,
        )
        data_analysis.append(analysis)
        
        print(f"比例 {ratio:.1f}:1:")
        print(f"  测试集: {analysis['test_size']} (安全={analysis['test_safe']}, 有害={analysis['test_toxic']})")
        print(f"  验证集: {analysis['val_size']} (安全={analysis['val_safe']}, 有害={analysis['val_toxic']})")
        print(f"  训练集: {analysis['train_size']} (安全={analysis['train_safe']}, 有害={analysis['train_toxic']})")
        print(f"  数据利用率: {analysis['utilization']:.1%}")
        print()
    
    # 运行训练
    print("\n" + "="*60)
    print("开始运行训练测试")
    print("="*60 + "\n")
    
    results = []
    for ratio in tqdm(args.test_ratios, desc="训练测试进度", unit="比例"):
        result = run_training(
            train_safe_ratio=ratio,
            max_samples=args.max_samples,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            num_epochs=args.num_epochs,
            seed=args.seed,
            data_file=args.data_file,
            output_dir=args.output_dir,
            lr=args.lr,
            probe_batch_size=args.probe_batch_size,
            batch_size=args.batch_size,
        )
        results.append(result)
    
    # 生成报告
    print("\n" + "="*60)
    print("生成对比报告...")
    print("="*60 + "\n")
    
    report_file = args.output_dir / "comparison_report.md"
    with tqdm(total=1, desc="生成报告", unit="报告") as pbar:
        generate_report(results, data_analysis, report_file)
        pbar.update(1)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"报告已保存到: {report_file}")
    print(f"各比例的训练结果保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()

