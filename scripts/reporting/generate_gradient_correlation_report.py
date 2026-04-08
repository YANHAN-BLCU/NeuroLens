#!/usr/bin/env python3
r"""
梯度关联分析报告生成脚本

根据梯度关联结果生成详细的 Markdown 报告。

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/generate_gradient_correlation_report.py \
        --input-path /workspace/outputs/gradient_correlation/gradient_correlation.json \
        --output-path /workspace/outputs/gradient_correlation

Windows 环境使用（本地运行）：
    python scripts/generate_gradient_correlation_report.py ^
        --input-path outputs/gradient_correlation/gradient_correlation.json ^
        --output-path outputs/gradient_correlation
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

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


def load_gradient_correlation(json_path: str) -> Dict[Tuple[int, int], Dict]:
    """
    从JSON文件中加载梯度关联结果
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        Dict[(layer_idx, neuron_idx), Dict]，键为元组，值为数据字典
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[Report Generation] 文件不存在: {json_path}")
    
    print(f"[Report Generation] 加载梯度关联数据: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为元组键格式
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict) and 'layer_idx' in value and 'neuron_idx' in value:
            layer_idx = int(value['layer_idx'])
            neuron_idx = int(value['neuron_idx'])
            result[(layer_idx, neuron_idx)] = value
    
    print(f"[Report Generation] 成功加载 {len(result)} 个目标神经元的梯度关联结果")
    return result


def calculate_statistics(gradient_results: Dict[Tuple[int, int], Dict]) -> Dict:
    """
    计算梯度关联统计信息
    
    Args:
        gradient_results: 梯度关联结果
    
    Returns:
        统计信息字典
    """
    stats = {
        'total_target_neurons': len(gradient_results),
        'neurons_with_upstream': 0,
        'neurons_without_upstream': 0,
        'total_upstream_connections': 0,
        'avg_upstream_per_neuron': 0.0,
        'max_upstream_count': 0,
        'min_upstream_count': float('inf'),
        'layer_stats': defaultdict(lambda: {
            'count': 0,
            'total_upstream': 0,
            'neurons': [],
        }),
        'upstream_layer_distribution': defaultdict(int),
        'gradient_strength_stats': {
            'max': 0.0,
            'min': float('inf'),
            'avg': 0.0,
            'all_strengths': [],
        },
    }
    
    for (layer_idx, neuron_idx), data in gradient_results.items():
        upstream_neurons = data.get('upstream_neurons', [])
        gradient_strengths = data.get('gradient_strengths', [])
        
        stats['layer_stats'][layer_idx]['count'] += 1
        stats['layer_stats'][layer_idx]['neurons'].append((layer_idx, neuron_idx))
        
        if upstream_neurons:
            stats['neurons_with_upstream'] += 1
            stats['total_upstream_connections'] += len(upstream_neurons)
            stats['layer_stats'][layer_idx]['total_upstream'] += len(upstream_neurons)
            
            if len(upstream_neurons) > stats['max_upstream_count']:
                stats['max_upstream_count'] = len(upstream_neurons)
            if len(upstream_neurons) < stats['min_upstream_count']:
                stats['min_upstream_count'] = len(upstream_neurons)
            
            # 统计上游神经元所在的层分布
            for upstream_neuron in upstream_neurons:
                if isinstance(upstream_neuron, dict):
                    upstream_layer = upstream_neuron.get('layer_idx', -1)
                elif isinstance(upstream_neuron, (list, tuple)) and len(upstream_neuron) >= 1:
                    upstream_layer = upstream_neuron[0]
                else:
                    continue
                stats['upstream_layer_distribution'][upstream_layer] += 1
            
            # 统计梯度强度
            for strength in gradient_strengths:
                stats['gradient_strength_stats']['all_strengths'].append(strength)
                if strength > stats['gradient_strength_stats']['max']:
                    stats['gradient_strength_stats']['max'] = strength
                if strength < stats['gradient_strength_stats']['min']:
                    stats['gradient_strength_stats']['min'] = strength
        else:
            stats['neurons_without_upstream'] += 1
    
    # 计算平均值
    if stats['neurons_with_upstream'] > 0:
        stats['avg_upstream_per_neuron'] = stats['total_upstream_connections'] / stats['neurons_with_upstream']
    
    if stats['gradient_strength_stats']['all_strengths']:
        strengths = stats['gradient_strength_stats']['all_strengths']
        stats['gradient_strength_stats']['avg'] = sum(strengths) / len(strengths)
    
    if stats['min_upstream_count'] == float('inf'):
        stats['min_upstream_count'] = 0
    
    if stats['gradient_strength_stats']['min'] == float('inf'):
        stats['gradient_strength_stats']['min'] = 0.0
    
    return stats


def generate_report(
    gradient_results: Dict[Tuple[int, int], Dict],
    stats: Dict,
    output_path: Path,
) -> str:
    """
    生成Markdown报告
    
    Args:
        gradient_results: 梯度关联结果
        stats: 统计信息
        output_path: 输出目录路径
    
    Returns:
        报告文件路径
    """
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "gradient_correlation_report.md"
    
    lines = []
    lines.append("# 梯度关联分析报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. 执行摘要")
    lines.append("")
    lines.append("本报告展示了神经元之间的梯度依赖关系分析结果。")
    lines.append("梯度关联（G_{i,j}）衡量了上游神经元 j 对目标神经元 i 的激活影响强度。")
    lines.append("")
    lines.append("### 1.1 总体统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 目标神经元总数 | {stats['total_target_neurons']} |")
    lines.append(f"| 有上游关联的神经元数 | {stats['neurons_with_upstream']} |")
    lines.append(f"| 无上游关联的神经元数 | {stats['neurons_without_upstream']} |")
    lines.append(f"| 总上游连接数 | {stats['total_upstream_connections']} |")
    lines.append(f"| 平均每个神经元的上游连接数 | {stats['avg_upstream_per_neuron']:.2f} |")
    lines.append(f"| 最大上游连接数 | {stats['max_upstream_count']} |")
    lines.append(f"| 最小上游连接数 | {stats['min_upstream_count']} |")
    lines.append("")
    
    lines.append("### 1.2 梯度强度统计")
    lines.append("")
    strength_stats = stats['gradient_strength_stats']
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 最大梯度强度 | {strength_stats['max']:.6f} |")
    lines.append(f"| 最小梯度强度 | {strength_stats['min']:.6f} |")
    lines.append(f"| 平均梯度强度 | {strength_stats['avg']:.6f} |")
    lines.append("")
    
    lines.append("## 2. 按层分析")
    lines.append("")
    lines.append("### 2.1 各层统计")
    lines.append("")
    lines.append("| 层索引 | 目标神经元数 | 总上游连接数 | 平均上游连接数 |")
    lines.append("|--------|-------------|-------------|---------------|")
    
    for layer_idx in sorted(stats['layer_stats'].keys()):
        layer_stat = stats['layer_stats'][layer_idx]
        avg_upstream = (layer_stat['total_upstream'] / layer_stat['count'] 
                       if layer_stat['count'] > 0 else 0.0)
        lines.append(f"| {layer_idx} | {layer_stat['count']} | "
                    f"{layer_stat['total_upstream']} | {avg_upstream:.2f} |")
    lines.append("")
    
    lines.append("### 2.2 上游神经元层分布")
    lines.append("")
    lines.append("| 上游层索引 | 连接数 |")
    lines.append("|-----------|--------|")
    
    for upstream_layer in sorted(stats['upstream_layer_distribution'].keys()):
        count = stats['upstream_layer_distribution'][upstream_layer]
        lines.append(f"| {upstream_layer} | {count} |")
    lines.append("")
    
    lines.append("## 3. 关键神经元分析")
    lines.append("")
    lines.append("### 3.1 上游连接数最多的神经元")
    lines.append("")
    
    # 按上游连接数排序
    neurons_by_upstream_count = sorted(
        gradient_results.items(),
        key=lambda x: len(x[1].get('upstream_neurons', [])),
        reverse=True
    )
    
    lines.append("| 排名 | 神经元 | 层索引 | 神经元索引 | 上游连接数 | 最大梯度强度 |")
    lines.append("|------|--------|--------|-----------|-----------|-------------|")
    
    for rank, ((layer_idx, neuron_idx), data) in enumerate(neurons_by_upstream_count[:20], 1):
        upstream_count = len(data.get('upstream_neurons', []))
        max_strength = max(data.get('gradient_strengths', [0.0]) or [0.0])
        neuron_name = f"layer_{layer_idx}_neuron_{neuron_idx}"
        lines.append(f"| {rank} | {neuron_name} | {layer_idx} | {neuron_idx} | "
                    f"{upstream_count} | {max_strength:.6f} |")
    lines.append("")
    
    lines.append("### 3.2 梯度强度最高的神经元")
    lines.append("")
    
    # 按最大梯度强度排序
    neurons_by_strength = sorted(
        gradient_results.items(),
        key=lambda x: max(x[1].get('gradient_strengths', [0.0]) or [0.0]),
        reverse=True
    )
    
    lines.append("| 排名 | 神经元 | 层索引 | 神经元索引 | 最大梯度强度 | 上游连接数 |")
    lines.append("|------|--------|--------|-----------|-------------|-----------|")
    
    for rank, ((layer_idx, neuron_idx), data) in enumerate(neurons_by_strength[:20], 1):
        max_strength = max(data.get('gradient_strengths', [0.0]) or [0.0])
        upstream_count = len(data.get('upstream_neurons', []))
        neuron_name = f"layer_{layer_idx}_neuron_{neuron_idx}"
        lines.append(f"| {rank} | {neuron_name} | {layer_idx} | {neuron_idx} | "
                    f"{max_strength:.6f} | {upstream_count} |")
    lines.append("")
    
    lines.append("## 4. 详细神经元信息")
    lines.append("")
    lines.append("### 4.1 所有目标神经元的上游关联")
    lines.append("")
    
    # 按层和神经元索引排序
    sorted_neurons = sorted(gradient_results.items(), key=lambda x: (x[0][0], x[0][1]))
    
    for (layer_idx, neuron_idx), data in sorted_neurons:
        neuron_name = f"layer_{layer_idx}_neuron_{neuron_idx}"
        upstream_neurons = data.get('upstream_neurons', [])
        gradient_strengths = data.get('gradient_strengths', [])
        
        lines.append(f"#### {neuron_name}")
        lines.append("")
        lines.append(f"- **层索引**: {layer_idx}")
        lines.append(f"- **神经元索引**: {neuron_idx}")
        lines.append(f"- **上游连接数**: {len(upstream_neurons)}")
        lines.append("")
        
        if upstream_neurons:
            lines.append("**上游神经元列表（按梯度强度排序）**:")
            lines.append("")
            lines.append("| 排名 | 上游神经元 | 层索引 | 神经元索引 | 梯度强度 |")
            lines.append("|------|-----------|--------|-----------|---------|")
            
            for rank, (upstream_neuron, strength) in enumerate(
                zip(upstream_neurons, gradient_strengths), 1
            ):
                if isinstance(upstream_neuron, dict):
                    upstream_layer = upstream_neuron.get('layer_idx', -1)
                    upstream_neuron_idx = upstream_neuron.get('neuron_idx', -1)
                elif isinstance(upstream_neuron, (list, tuple)) and len(upstream_neuron) >= 2:
                    upstream_layer = upstream_neuron[0]
                    upstream_neuron_idx = upstream_neuron[1]
                else:
                    continue
                
                upstream_name = f"layer_{upstream_layer}_neuron_{upstream_neuron_idx}"
                lines.append(f"| {rank} | {upstream_name} | {upstream_layer} | "
                            f"{upstream_neuron_idx} | {strength:.6f} |")
            lines.append("")
        else:
            lines.append("**无上游关联**")
            lines.append("")
    
    lines.append("## 5. 结论与建议")
    lines.append("")
    lines.append("### 5.1 主要发现")
    lines.append("")
    lines.append(f"- 共分析了 {stats['total_target_neurons']} 个目标神经元")
    lines.append(f"- {stats['neurons_with_upstream']} 个神经元（{stats['neurons_with_upstream']/stats['total_target_neurons']*100:.1f}%）具有上游关联")
    lines.append(f"- 平均每个神经元有 {stats['avg_upstream_per_neuron']:.2f} 个上游连接")
    lines.append("")
    
    lines.append("### 5.2 建议")
    lines.append("")
    lines.append("- 重点关注上游连接数多且梯度强度高的神经元，这些神经元可能是关键的信息汇聚点")
    lines.append("- 分析上游神经元的分布模式，识别是否存在特定的层间连接模式")
    lines.append("- 结合其他分析（如参数对齐、激活投影）来全面理解神经元的功能")
    lines.append("")
    
    # 写入文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"[Report Generation] 报告已生成: {report_file}")
    return str(report_file)


def main():
    parser = argparse.ArgumentParser(
        description="生成梯度关联分析报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="梯度关联结果JSON文件路径"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    try:
        gradient_results = load_gradient_correlation(args.input_path)
    except Exception as e:
        print(f"[Report Generation] 加载数据失败: {e}")
        return
    
    # 计算统计信息
    print("[Report Generation] 计算统计信息...")
    stats = calculate_statistics(gradient_results)
    
    # 生成报告
    print("[Report Generation] 生成报告...")
    output_path = Path(args.output_path)
    report_file = generate_report(gradient_results, stats, output_path)
    
    print(f"[Report Generation] 完成！报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
