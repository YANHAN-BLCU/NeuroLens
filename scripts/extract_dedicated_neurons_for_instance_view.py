"""
提取专用安全神经元数据，用于Instance View显示
索引从1-525
"""
import json
from pathlib import Path
from typing import Dict, List

def extract_dedicated_neurons_for_instance_view(
    input_file: str = "outputs/dedicated_safety_neurons.json",
    output_file: str = "outputs/dedicated_safety_neurons_for_instance_view.json",
    max_index: int = 525
) -> None:
    """
    提取专用安全神经元数据，添加索引（1-525），用于Instance View显示
    
    Args:
        input_file: 输入的专用安全神经元JSON文件
        output_file: 输出的JSON文件路径
        max_index: 最大索引数（默认525）
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取专用安全神经元
    dedicated_neurons = data.get('dedicated_safety_neurons', {})
    
    # 转换为列表并按rank排序
    neuron_list = []
    for key, neuron_data in dedicated_neurons.items():
        neuron_list.append({
            'key': key,
            **neuron_data
        })
    
    # 按rank排序
    neuron_list.sort(key=lambda x: x.get('rank', float('inf')))
    
    # 只取前max_index个，并添加索引（从1开始）
    indexed_neurons = []
    for idx, neuron in enumerate(neuron_list[:max_index], start=1):
        indexed_neurons.append({
            'index': idx,
            'layer': neuron['layer'],
            'neuron': neuron['neuron'],
            'key': neuron['key'],
            'score': neuron['score'],
            'rank': neuron['rank'],
            'percentile': neuron['percentile']
        })
    
    # 创建输出数据结构
    output_data = {
        'metadata': {
            'total_neurons': len(indexed_neurons),
            'index_range': f'1-{len(indexed_neurons)}',
            'source_file': input_file,
            'note': '专用安全神经元列表，索引从1开始，按rank排序'
        },
        'neurons': indexed_neurons,
        # 同时创建一个按key索引的快速查找字典
        'neurons_by_key': {
            neuron['key']: neuron for neuron in indexed_neurons
        },
        # 按layer分组的神经元
        'neurons_by_layer': {}
    }
    
    # 按layer分组
    for neuron in indexed_neurons:
        layer = neuron['layer']
        if layer not in output_data['neurons_by_layer']:
            output_data['neurons_by_layer'][layer] = []
        output_data['neurons_by_layer'][layer].append(neuron)
    
    # 保存输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 成功提取 {len(indexed_neurons)} 个专用安全神经元")
    print(f"[FILE] 输出文件: {output_file}")
    print(f"[RANGE] 索引范围: 1-{len(indexed_neurons)}")
    print(f"[LAYERS] 层数分布: {len(output_data['neurons_by_layer'])} 层")
    
    # 打印统计信息
    layer_counts = {layer: len(neurons) for layer, neurons in output_data['neurons_by_layer'].items()}
    print(f"[STATS] 各层神经元数量: {dict(sorted(layer_counts.items()))}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='提取专用安全神经元数据用于Instance View')
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/dedicated_safety_neurons.json',
        help='输入的专用安全神经元JSON文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/dedicated_safety_neurons_for_instance_view.json',
        help='输出的JSON文件路径'
    )
    parser.add_argument(
        '--max-index',
        type=int,
        default=525,
        help='最大索引数（默认525）'
    )
    
    args = parser.parse_args()
    
    extract_dedicated_neurons_for_instance_view(
        input_file=args.input,
        output_file=args.output,
        max_index=args.max_index
    )
