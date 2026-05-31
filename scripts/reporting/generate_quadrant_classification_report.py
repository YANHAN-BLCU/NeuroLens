#!/usr/bin/env python3
r"""
象限分类分析报告生成脚本

根据象限分类结果生成详细的 Markdown 报告。

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/generate_quadrant_classification_report.py \
        --input-path /workspace/outputs/quadrant_classification/quadrant_classification.json \
        --output-path /workspace/outputs/quadrant_classification

Windows 环境使用（本地运行）：
    python scripts/generate_quadrant_classification_report.py ^
        --input-path outputs/quadrant_classification/quadrant_classification.json ^
        --output-path outputs/quadrant_classification
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

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


def load_quadrant_classification(json_path: str) -> Dict[Tuple[int, int], Dict]:
    """
    从JSON文件中加载象限分类结果
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        Dict[(layer_idx, neuron_idx), Dict]，键为元组，值为数据字典
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[Report Generation] 文件不存在: {json_path}")
    
    print(f"[Report Generation] 加载象限分类数据: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为元组键格式
    result = {}
    
    for key, value in data.items():
        if key == '_statistics' or key == '_metadata':
            continue
        
        if isinstance(value, dict) and 'layer_idx' in value and 'neuron_idx' in value:
            layer_idx = int(value['layer_idx'])
            neuron_idx = int(value['neuron_idx'])
            result[(layer_idx, neuron_idx)] = value
    
    print(f"[Report Generation] 成功加载 {len(result)} 个神经元的分类结果")
    return result


def calculate_statistics(quadrant_results: Dict[Tuple[int, int], Dict]) -> Dict:
    """
    计算象限统计信息
    
    Args:
        quadrant_results: 象限分类结果
    
    Returns:
        统计信息字典
    """
    from collections import defaultdict
    
    stats = defaultdict(lambda: {
        'count': 0,
        'alignments': [],
        'activation_projections': [],
        'activation_diffs': [],
        'successful_means': [],
        'failed_means': [],
        'neurons': [],
    })
    
    # 按层统计
    layer_stats = defaultdict(lambda: {
        'S+A+': 0, 'S-A+': 0, 'S+A-': 0, 'S-A-': 0
    })
    
    for (layer_idx, neuron_idx), data in quadrant_results.items():
        quadrant = data['quadrant']
        stats[quadrant]['count'] += 1
        stats[quadrant]['alignments'].append(data['alignment'])
        stats[quadrant]['activation_projections'].append(data['activation_projection'])
        stats[quadrant]['activation_diffs'].append(data['activation_diff'])
        stats[quadrant]['successful_means'].append(data.get('successful_mean', data['activation_projection']))
        stats[quadrant]['failed_means'].append(data.get('failed_mean', 0.0))
        stats[quadrant]['neurons'].append((layer_idx, neuron_idx))
        
        # 按层统计
        if layer_idx not in layer_stats:
            layer_stats[layer_idx] = {'S+A+': 0, 'S-A+': 0, 'S+A-': 0, 'S-A-': 0}
        layer_stats[layer_idx][quadrant] += 1
    
    # 计算统计量
    result = {}
    total = len(quadrant_results)
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        if quadrant not in stats or stats[quadrant]['count'] == 0:
            result[quadrant] = {
                'count': 0,
                'percentage': 0.0,
                'avg_alignment': 0.0,
                'std_alignment': 0.0,
                'avg_activation_projection': 0.0,
                'std_activation_projection': 0.0,
                'avg_activation_diff': 0.0,
                'std_activation_diff': 0.0,
                'avg_successful_mean': 0.0,
                'avg_failed_mean': 0.0,
                'neurons': [],
            }
            continue
        
        stat = stats[quadrant]
        alignments = np.array(stat['alignments'])
        activation_projections = np.array(stat['activation_projections'])
        activation_diffs = np.array(stat['activation_diffs'])
        
        result[quadrant] = {
            'count': stat['count'],
            'percentage': (stat['count'] / total * 100) if total > 0 else 0.0,
            'avg_alignment': float(np.mean(alignments)),
            'std_alignment': float(np.std(alignments)),
            'min_alignment': float(np.min(alignments)),
            'max_alignment': float(np.max(alignments)),
            'avg_activation_projection': float(np.mean(activation_projections)),
            'std_activation_projection': float(np.std(activation_projections)),
            'min_activation_projection': float(np.min(activation_projections)),
            'max_activation_projection': float(np.max(activation_projections)),
            'avg_activation_diff': float(np.mean(activation_diffs)),
            'std_activation_diff': float(np.std(activation_diffs)),
            'min_activation_diff': float(np.min(activation_diffs)),
            'max_activation_diff': float(np.max(activation_diffs)),
            'avg_successful_mean': float(np.mean(stat['successful_means'])),
            'avg_failed_mean': float(np.mean(stat['failed_means'])),
            'neurons': stat['neurons'],
        }
    
    return {
        'quadrant_stats': result,
        'layer_stats': dict(layer_stats),
        'total': total,
    }


def find_top_neurons(quadrant_results: Dict[Tuple[int, int], Dict], 
                     quadrant: str, 
                     metric: str, 
                     top_n: int = 5,
                     ascending: bool = False) -> List[Tuple[Tuple[int, int], Dict]]:
    """
    找到每个象限中特定指标最高/最低的神经元
    
    Args:
        quadrant_results: 象限分类结果
        quadrant: 象限名称
        metric: 指标名称（'alignment', 'activation_projection', 'activation_diff'）
        top_n: 返回前N个
        ascending: 是否升序排列（False表示降序，取最大值）
    
    Returns:
        神经元列表，每个元素为 ((layer_idx, neuron_idx), data)
    """
    quadrant_neurons = [
        ((layer_idx, neuron_idx), data)
        for (layer_idx, neuron_idx), data in quadrant_results.items()
        if data['quadrant'] == quadrant
    ]
    
    if not quadrant_neurons:
        return []
    
    # 按指标排序
    sorted_neurons = sorted(
        quadrant_neurons,
        key=lambda x: x[1].get(metric, 0.0),
        reverse=not ascending
    )
    
    return sorted_neurons[:top_n]


def generate_report(quadrant_results: Dict[Tuple[int, int], Dict], 
                    output_path: Path) -> str:
    """
    生成象限分类分析报告
    
    Args:
        quadrant_results: 象限分类结果
        output_path: 输出目录路径
    
    Returns:
        报告文件路径
    """
    print("[Report Generation] 开始生成报告...")
    
    # 计算统计信息
    stats = calculate_statistics(quadrant_results)
    quadrant_stats = stats['quadrant_stats']
    layer_stats = stats['layer_stats']
    total = stats['total']
    
    # 生成报告内容
    report_lines = []
    
    # 标题
    report_lines.append("# 功能象限分类分析报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. 执行摘要
    report_lines.append("## 1. 执行摘要")
    report_lines.append("")
    report_lines.append("本报告分析了神经元的功能象限分类结果，根据参数对齐（S）和激活投影（A）将神经元分为四个功能象限。")
    report_lines.append("")
    report_lines.append("### 关键指标")
    report_lines.append("")
    report_lines.append(f"- **总神经元数**: {total}")
    report_lines.append(f"- **分析层数**: {len(layer_stats)}")
    report_lines.append("")
    
    # 象限分布摘要
    report_lines.append("### 象限分布摘要")
    report_lines.append("")
    report_lines.append("| 象限 | 数量 | 百分比 | 功能描述 |")
    report_lines.append("|------|------|--------|----------|")
    
    quadrant_descriptions = {
        'S+A+': '毒性特征增强（Toxic feature enhancement）',
        'S-A+': '良性特征抑制（Benign feature suppression）',
        'S+A-': '毒性特征抑制（Toxic feature suppression）',
        'S-A-': '良性特征增强（Benign feature enhancement）',
    }
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        stat = quadrant_stats[quadrant]
        report_lines.append(
            f"| {quadrant} | {stat['count']} | {stat['percentage']:.2f}% | {quadrant_descriptions[quadrant]} |"
        )
    
    report_lines.append("")
    
    # 2. 总体统计
    report_lines.append("## 2. 总体统计")
    report_lines.append("")
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        stat = quadrant_stats[quadrant]
        if stat['count'] == 0:
            continue
        
        report_lines.append(f"### 2.{['S+A+', 'S-A+', 'S+A-', 'S-A-'].index(quadrant) + 1} {quadrant} 象限统计")
        report_lines.append("")
        report_lines.append(f"**功能描述**: {quadrant_descriptions[quadrant]}")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 神经元数量 | {stat['count']} ({stat['percentage']:.2f}%) |")
        report_lines.append(f"| 平均参数对齐 (S) | {stat['avg_alignment']:.6f} |")
        report_lines.append(f"| 参数对齐标准差 | {stat['std_alignment']:.6f} |")
        report_lines.append(f"| 参数对齐范围 | [{stat['min_alignment']:.6f}, {stat['max_alignment']:.6f}] |")
        report_lines.append(f"| 平均激活投影 (A) | {stat['avg_activation_projection']:.6f} |")
        report_lines.append(f"| 激活投影标准差 | {stat['std_activation_projection']:.6f} |")
        report_lines.append(f"| 激活投影范围 | [{stat['min_activation_projection']:.6f}, {stat['max_activation_projection']:.6f}] |")
        report_lines.append(f"| 平均激活差异 | {stat['avg_activation_diff']:.6f} |")
        report_lines.append(f"| 激活差异标准差 | {stat['std_activation_diff']:.6f} |")
        report_lines.append(f"| 激活差异范围 | [{stat['min_activation_diff']:.6f}, {stat['max_activation_diff']:.6f}] |")
        report_lines.append(f"| 平均成功样本激活 | {stat['avg_successful_mean']:.6f} |")
        report_lines.append(f"| 平均失败样本激活 | {stat['avg_failed_mean']:.6f} |")
        report_lines.append("")
    
    # 3. 按层分析
    report_lines.append("## 3. 按层分析")
    report_lines.append("")
    report_lines.append("### 3.1 层分布统计")
    report_lines.append("")
    report_lines.append("| 层索引 | S+A+ | S-A+ | S+A- | S-A- | 总计 |")
    report_lines.append("|--------|------|------|------|------|------|")
    
    for layer_idx in sorted(layer_stats.keys()):
        layer_stat = layer_stats[layer_idx]
        total_in_layer = sum(layer_stat.values())
        report_lines.append(
            f"| {layer_idx} | {layer_stat['S+A+']} | {layer_stat['S-A+']} | "
            f"{layer_stat['S+A-']} | {layer_stat['S-A-']} | {total_in_layer} |"
        )
    
    report_lines.append("")
    
    # 4. 关键神经元分析
    report_lines.append("## 4. 关键神经元分析")
    report_lines.append("")
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        stat = quadrant_stats[quadrant]
        if stat['count'] == 0:
            continue
        
        report_lines.append(f"### 4.{['S+A+', 'S-A+', 'S+A-', 'S-A-'].index(quadrant) + 1} {quadrant} 象限关键神经元")
        report_lines.append("")
        
        # 参数对齐最高的神经元
        top_alignment = find_top_neurons(quadrant_results, quadrant, 'alignment', top_n=3, ascending=False)
        if top_alignment:
            report_lines.append("#### 参数对齐最高的神经元（S值最大）")
            report_lines.append("")
            for i, ((layer_idx, neuron_idx), data) in enumerate(top_alignment, 1):
                report_lines.append(f"**{i}. 神经元**: `layer_{layer_idx}_neuron_{neuron_idx}`")
                report_lines.append(f"- 层索引: {layer_idx}")
                report_lines.append(f"- 神经元索引: {neuron_idx}")
                report_lines.append(f"- 参数对齐 (S): {data['alignment']:.6f}")
                report_lines.append(f"- 激活投影 (A): {data['activation_projection']:.6f}")
                report_lines.append(f"- 激活差异: {data['activation_diff']:.6f}")
                report_lines.append("")
        
        # 激活投影最高的神经元
        top_activation = find_top_neurons(quadrant_results, quadrant, 'activation_projection', top_n=3, ascending=False)
        if top_activation:
            report_lines.append("#### 激活投影最高的神经元（A值最大）")
            report_lines.append("")
            for i, ((layer_idx, neuron_idx), data) in enumerate(top_activation, 1):
                report_lines.append(f"**{i}. 神经元**: `layer_{layer_idx}_neuron_{neuron_idx}`")
                report_lines.append(f"- 层索引: {layer_idx}")
                report_lines.append(f"- 神经元索引: {neuron_idx}")
                report_lines.append(f"- 参数对齐 (S): {data['alignment']:.6f}")
                report_lines.append(f"- 激活投影 (A): {data['activation_projection']:.6f}")
                report_lines.append(f"- 激活差异: {data['activation_diff']:.6f}")
                report_lines.append("")
        
        # 激活差异最大的神经元
        top_diff = find_top_neurons(quadrant_results, quadrant, 'activation_diff', top_n=3, ascending=False)
        if top_diff:
            report_lines.append("#### 激活差异最大的神经元")
            report_lines.append("")
            for i, ((layer_idx, neuron_idx), data) in enumerate(top_diff, 1):
                report_lines.append(f"**{i}. 神经元**: `layer_{layer_idx}_neuron_{neuron_idx}`")
                report_lines.append(f"- 层索引: {layer_idx}")
                report_lines.append(f"- 神经元索引: {neuron_idx}")
                report_lines.append(f"- 参数对齐 (S): {data['alignment']:.6f}")
                report_lines.append(f"- 激活投影 (A): {data['activation_projection']:.6f}")
                report_lines.append(f"- 激活差异: {data['activation_diff']:.6f}")
                report_lines.append(f"- 成功样本平均激活: {data.get('successful_mean', 0.0):.6f}")
                report_lines.append(f"- 失败样本平均激活: {data.get('failed_mean', 0.0):.6f}")
                report_lines.append("")
    
    # 5. 象限特征分析
    report_lines.append("## 5. 象限特征分析")
    report_lines.append("")
    
    report_lines.append("### 5.1 象限定义回顾")
    report_lines.append("")
    report_lines.append("根据论文5.4节，神经元的功能象限分类基于：")
    report_lines.append("")
    report_lines.append("- **S_i^k (参数对齐)**: 参数方向与毒性向量的余弦相似度")
    report_lines.append("  - S+ (S_i^k > 0): 参数对齐为正，表示神经元参数方向促进有害内容生成")
    report_lines.append("  - S- (S_i^k ≤ 0): 参数对齐为负，表示神经元参数方向有助于防御性转向")
    report_lines.append("")
    report_lines.append("- **A_i^k (激活投影)**: 激活向量在归一化毒性向量上的投影")
    report_lines.append("  - A+ (A_i^k > 0): 激活投影为正，在成功jailbreak时激活更强，促进毒性")
    report_lines.append("  - A- (A_i^k ≤ 0): 激活投影为负，在失败jailbreak时激活更强，抑制毒性")
    report_lines.append("")
    
    report_lines.append("### 5.2 象限功能特征")
    report_lines.append("")
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        stat = quadrant_stats[quadrant]
        if stat['count'] == 0:
            continue
        
        report_lines.append(f"#### {quadrant}: {quadrant_descriptions[quadrant]}")
        report_lines.append("")
        report_lines.append(f"- **神经元数量**: {stat['count']} ({stat['percentage']:.2f}%)")
        report_lines.append(f"- **平均参数对齐**: {stat['avg_alignment']:.6f}")
        report_lines.append(f"- **平均激活投影**: {stat['avg_activation_projection']:.6f}")
        report_lines.append(f"- **平均激活差异**: {stat['avg_activation_diff']:.6f}")
        report_lines.append("")
        
        if quadrant == 'S+A+':
            report_lines.append("**特征**: 参数和激活都促进毒性，是最危险的神经元类型。")
        elif quadrant == 'S-A+':
            report_lines.append("**特征**: 参数方向抑制毒性，但激活时促进毒性（可能是防御机制失效）。")
        elif quadrant == 'S+A-':
            report_lines.append("**特征**: 参数方向促进毒性，但激活时抑制毒性（可能是防御机制生效）。")
        elif quadrant == 'S-A-':
            report_lines.append("**特征**: 参数和激活都抑制毒性，是最安全的神经元类型。")
        
        report_lines.append("")
    
    # 6. 结论与建议
    report_lines.append("## 6. 结论与建议")
    report_lines.append("")
    
    report_lines.append("### 6.1 主要发现")
    report_lines.append("")
    
    # 找出最多的象限
    max_quadrant = max(quadrant_stats.items(), key=lambda x: x[1]['count'])
    report_lines.append(f"1. **象限分布**: {max_quadrant[0]} 象限包含最多的神经元（{max_quadrant[1]['count']}个，{max_quadrant[1]['percentage']:.2f}%）。")
    report_lines.append("")
    
    # S+A+ 象限分析
    s_plus_a_plus = quadrant_stats['S+A+']
    if s_plus_a_plus['count'] > 0:
        report_lines.append(f"2. **危险神经元 (S+A+)**: 发现 {s_plus_a_plus['count']} 个 S+A+ 象限神经元，这些神经元的参数和激活都促进毒性，需要特别关注。")
        report_lines.append("")
    
    # S-A- 象限分析
    s_minus_a_minus = quadrant_stats['S-A-']
    if s_minus_a_minus['count'] > 0:
        report_lines.append(f"3. **安全神经元 (S-A-)**: 发现 {s_minus_a_minus['count']} 个 S-A- 象限神经元，这些神经元的参数和激活都抑制毒性，是防御机制的重要组成部分。")
        report_lines.append("")
    
    report_lines.append("### 6.2 建议")
    report_lines.append("")
    report_lines.append("1. **重点关注 S+A+ 象限**: 这些神经元是毒性生成的关键，建议进行深入分析或考虑干预措施。")
    report_lines.append("")
    report_lines.append("2. **保护 S-A- 象限**: 这些神经元是防御机制的核心，应该避免在微调或干预中削弱它们的功能。")
    report_lines.append("")
    report_lines.append("3. **分析矛盾象限**: S-A+ 和 S+A- 象限表现出参数和激活的矛盾，值得进一步研究其机制。")
    report_lines.append("")
    report_lines.append("4. **层特异性分析**: 不同层的神经元可能表现出不同的象限分布模式，建议按层进行深入分析。")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("**报告结束**")
    report_lines.append("")
    
    # 保存报告
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "quadrant_classification_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[Report Generation] 报告已保存到: {report_file}")
    return str(report_file)


def main():
    parser = argparse.ArgumentParser(
        description='生成象限分类分析报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='象限分类结果文件路径（JSON格式）'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print("[Report Generation] 开始加载数据...")
    quadrant_results = load_quadrant_classification(args.input_path)
    
    if len(quadrant_results) == 0:
        print("[Report Generation] 错误: 没有找到任何分类结果")
        return 1
    
    # 生成报告
    report_file = generate_report(quadrant_results, args.output_path)
    
    print(f"\n[Report Generation] 完成！报告已保存到: {report_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
