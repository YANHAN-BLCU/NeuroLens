#!/usr/bin/env python3
r"""
功能象限分类运行脚本

根据论文5.4节要求，将神经元分为四个功能象限：S+A+, S-A+, S+A-, S-A-

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/run_quadrant_classification.py \
        --parameter-alignment-path /workspace/outputs/parameter_alignment/parameter_alignment.json \
        --activation-projection-path /workspace/outputs/activation_projection/activation_projection.json \
        --output-path /workspace/outputs/quadrant_classification

Windows 环境使用（本地运行）：
    python scripts/run_quadrant_classification.py ^
        --parameter-alignment-path outputs/parameter_alignment/parameter_alignment.json ^
        --activation-projection-path outputs/activation_projection/activation_projection.json ^
        --output-path outputs/quadrant_classification

自定义阈值：
    python scripts/run_quadrant_classification.py ^
        --parameter-alignment-path outputs/parameter_alignment/parameter_alignment.json ^
        --activation-projection-path outputs/activation_projection/activation_projection.json ^
        --output-path outputs/quadrant_classification ^
        --threshold-s 0.1 ^
        --threshold-a 0.05

过滤特定象限：
    python scripts/run_quadrant_classification.py ^
        --parameter-alignment-path outputs/parameter_alignment/parameter_alignment.json ^
        --activation-projection-path outputs/activation_projection/activation_projection.json ^
        --output-path outputs/quadrant_classification ^
        --filter-quadrants S+A+ S-A+

注意：
- 脚本会自动检测项目根目录，支持 Docker 环境和本地环境
- 默认阈值 threshold_s=0.0, threshold_a=0.0（基于论文定义）
- 可以使用 --filter-quadrants 参数只保留特定象限的神经元
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

# 添加工作目录到路径
# 支持两种方式：1) 从脚本位置推断项目根目录 2) 使用 /workspace（Docker 环境）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.exists() and (PROJECT_ROOT / 'engine').exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    # Docker 环境或使用绝对路径
    workspace_path = os.getenv('WORKSPACE_PATH', '/workspace')
    if os.path.exists(workspace_path):
        sys.path.insert(0, workspace_path)
    else:
        # 尝试当前工作目录
        cwd = Path.cwd()
        if (cwd / 'engine').exists():
            sys.path.insert(0, str(cwd))
        else:
            # 最后尝试 /workspace
            sys.path.insert(0, '/workspace')

from engine.neurons.quadrant_classification import (
    classify_neuron_quadrants,
    get_quadrant_statistics,
    save_quadrant_classification,
    filter_neurons_by_quadrant,
)


def load_json_to_dict(json_path: str, data_type: str = "unknown") -> Dict[Tuple[int, int], Dict]:
    """
    从JSON文件中加载数据并转换为元组键格式
    
    Args:
        json_path: JSON文件路径
        data_type: 数据类型（用于日志），如 "parameter_alignment" 或 "activation_projection"
    
    Returns:
        Dict[(layer_idx, neuron_idx), Dict]，键为元组，值为数据字典
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[Quadrant Classification] 文件不存在: {json_path}")
    
    print(f"[Quadrant Classification] 加载 {data_type} 数据: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"[Quadrant Classification] JSON 解析失败 ({json_path}): {e}")
    
    # 转换为元组键格式
    result = {}
    
    # 跳过统计信息键（如果存在）
    skip_keys = ['_statistics', '_metadata']
    
    for key, value in data.items():
        if key in skip_keys:
            continue
        
        # 支持两种格式：
        # 1. "layer_X_neuron_Y": {layer_idx, neuron_idx, ...}
        # 2. 直接包含 layer_idx/neuron_idx 的值
        
        layer_idx = None
        neuron_idx = None
        
        # 首先尝试从值中获取
        if isinstance(value, dict):
            if 'layer_idx' in value and 'neuron_idx' in value:
                layer_idx = int(value['layer_idx'])
                neuron_idx = int(value['neuron_idx'])
            elif 'layer' in value and 'neuron' in value:
                layer_idx = int(value['layer'])
                neuron_idx = int(value['neuron'])
        
        # 如果值中没有，尝试从键名解析
        if layer_idx is None or neuron_idx is None:
            if '_' in key:
                try:
                    parts = key.split('_')
                    # 格式1: layer_X_neuron_Y
                    if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'neuron':
                        layer_idx = int(parts[1])
                        neuron_idx = int(parts[3])
                    # 格式2: X_Y (下划线分隔，如 "31_4062")
                    elif len(parts) == 2:
                        layer_idx = int(parts[0])
                        neuron_idx = int(parts[1])
                except (ValueError, IndexError):
                    print(f"[Quadrant Classification] 警告: 无法解析键名: {key}")
                    continue
        
        if layer_idx is not None and neuron_idx is not None:
            result[(layer_idx, neuron_idx)] = value
        else:
            print(f"[Quadrant Classification] 警告: 无法从键 '{key}' 中提取层和神经元索引")
    
    print(f"[Quadrant Classification] 成功加载 {len(result)} 个神经元（{data_type}）")
    return result


def print_statistics(stats: Dict[str, Dict]):
    """
    打印象限统计信息
    
    Args:
        stats: 象限统计信息，来自 get_quadrant_statistics()
    """
    print("\n" + "="*80)
    print("[Quadrant Classification] 象限统计信息")
    print("="*80)
    
    # 计算总数
    total = sum(stat['count'] for stat in stats.values())
    
    print(f"\n总神经元数: {total}")
    print("\n象限分布:")
    print("-" * 80)
    print(f"{'象限':<10} {'数量':<10} {'百分比':<10} {'平均对齐':<15} {'平均激活投影':<15}")
    print("-" * 80)
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        stat = stats.get(quadrant, {})
        count = stat.get('count', 0)
        percentage = stat.get('percentage', 0.0)
        avg_alignment = stat.get('avg_alignment', 0.0)
        avg_activation_projection = stat.get('avg_activation_projection', 0.0)
        
        print(f"{quadrant:<10} {count:<10} {percentage:>6.2f}%   {avg_alignment:>13.6f}   {avg_activation_projection:>13.6f}")
    
    print("-" * 80)
    
    # 打印详细统计
    print("\n详细统计:")
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        stat = stats.get(quadrant, {})
        if stat.get('count', 0) > 0:
            print(f"\n{quadrant}:")
            print(f"  数量: {stat.get('count', 0)} ({stat.get('percentage', 0.0):.2f}%)")
            print(f"  平均参数对齐 (S): {stat.get('avg_alignment', 0.0):.6f}")
            print(f"  平均激活投影 (A): {stat.get('avg_activation_projection', 0.0):.6f}")
            print(f"  平均激活差异: {stat.get('avg_activation_diff', 0.0):.6f}")
            print(f"  平均成功样本激活: {stat.get('avg_successful_mean', 0.0):.6f}")
            print(f"  平均失败样本激活: {stat.get('avg_failed_mean', 0.0):.6f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='功能象限分类 - 根据参数对齐和激活投影结果将神经元分为四个象限',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--parameter-alignment-path',
        type=str,
        required=True,
        help='参数对齐结果文件路径（JSON格式）'
    )
    parser.add_argument(
        '--activation-projection-path',
        type=str,
        required=True,
        help='激活投影结果文件路径（JSON格式）'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='quadrant_classification.json',
        help='输出文件名（默认: quadrant_classification.json）'
    )
    parser.add_argument(
        '--threshold-s',
        type=float,
        default=0.0,
        help='参数对齐阈值（默认: 0.0）。如果 cosine_similarity > threshold_s，则为 S+'
    )
    parser.add_argument(
        '--threshold-a',
        type=float,
        default=0.0,
        help='激活投影阈值（默认: 0.0）。如果 activation_projection > threshold_a，则为 A+'
    )
    parser.add_argument(
        '--filter-quadrants',
        type=str,
        nargs='+',
        default=None,
        choices=['S+A+', 'S-A+', 'S+A-', 'S-A-'],
        help='只保留指定象限的神经元（可选，可指定多个象限）'
    )
    parser.add_argument(
        '--print-statistics',
        action='store_true',
        help='打印详细的象限统计信息'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print("[Quadrant Classification] 开始加载数据...")
    parameter_alignment = load_json_to_dict(
        args.parameter_alignment_path,
        data_type="parameter_alignment"
    )
    activation_projection = load_json_to_dict(
        args.activation_projection_path,
        data_type="activation_projection"
    )
    
    # 执行象限分类
    print("\n[Quadrant Classification] 开始象限分类...")
    quadrant_results = classify_neuron_quadrants(
        parameter_alignment=parameter_alignment,
        activation_projection=activation_projection,
        threshold_s=args.threshold_s,
        threshold_a=args.threshold_a,
    )
    
    if len(quadrant_results) == 0:
        print("[Quadrant Classification] 错误: 没有成功分类任何神经元")
        return 1
    
    # 过滤象限（如果指定）
    if args.filter_quadrants:
        print(f"\n[Quadrant Classification] 过滤象限: {args.filter_quadrants}")
        quadrant_results = filter_neurons_by_quadrant(
            quadrant_results,
            quadrants=args.filter_quadrants
        )
    
    # 获取统计信息
    stats = get_quadrant_statistics(quadrant_results)
    
    # 打印统计信息
    if args.print_statistics:
        print_statistics(stats)
    else:
        # 简单打印
        print("\n[Quadrant Classification] 象限分布:")
        total = sum(stat['count'] for stat in stats.values())
        for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
            stat = stats.get(quadrant, {})
            count = stat.get('count', 0)
            percentage = (count / total * 100) if total > 0 else 0.0
            print(f"  {quadrant}: {count} ({percentage:.1f}%)")
    
    # 保存结果
    output_path = Path(args.output_path)
    output_file = save_quadrant_classification(
        quadrant_results=quadrant_results,
        output_path=output_path,
        filename=args.output_filename,
    )
    
    print(f"\n[Quadrant Classification] 完成！结果已保存到: {output_file}")
    print(f"[Quadrant Classification] 共分类 {len(quadrant_results)} 个神经元")
    
    return 0


if __name__ == '__main__':
    exit(main())
