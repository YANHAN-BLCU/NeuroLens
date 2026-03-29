#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查激活投影数据文件是否有问题
"""
import json
import sys
import io
from pathlib import Path
from collections import defaultdict

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_activation_projection_data(json_path):
    """检查激活投影数据"""
    print(f"正在检查文件: {json_path}")
    print("=" * 80)
    
    # 1. 检查文件是否存在
    if not Path(json_path).exists():
        print(f"[ERROR] 错误: 文件不存在: {json_path}")
        return False
    
    # 2. 尝试解析 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("[OK] JSON 格式有效")
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析错误: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 读取文件错误: {e}")
        return False
    
    # 3. 检查数据结构
    if not isinstance(data, dict):
        print(f"[ERROR] 错误: 根对象应该是字典，实际是 {type(data)}")
        return False
    
    total_neurons = len(data)
    print(f"[OK] 总神经元数量: {total_neurons}")
    
    if total_neurons == 0:
        print("[WARNING] 警告: 数据为空")
        return True
    
    # 4. 检查每个神经元的数据结构
    required_fields = [
        'layer_idx', 'neuron_idx', 'successful_mean', 'failed_mean',
        'successful_std', 'failed_std', 'successful_count', 'failed_count',
        'activation_projection', 'activation_diff'
    ]
    
    issues = []
    layer_neuron_counts = defaultdict(int)
    layer_stats = defaultdict(lambda: {'neurons': set(), 'counts': []})
    
    for key, neuron_data in data.items():
        # 检查键格式
        if not key.startswith('layer_'):
            issues.append(f"[WARNING] 键格式异常: {key}")
            continue
        
        # 检查必需字段
        for field in required_fields:
            if field not in neuron_data:
                issues.append(f"[ERROR] 缺失字段 '{field}' 在 {key}")
        
        # 检查字段类型
        if 'layer_idx' in neuron_data:
            layer_idx = neuron_data['layer_idx']
            neuron_idx = neuron_data.get('neuron_idx', -1)
            layer_neuron_counts[layer_idx] += 1
            layer_stats[layer_idx]['neurons'].add(neuron_idx)
            
            # 检查计数一致性
            successful_count = neuron_data.get('successful_count', 0)
            failed_count = neuron_data.get('failed_count', 0)
            layer_stats[layer_idx]['counts'].append((successful_count, failed_count))
    
    # 5. 检查计数一致性
    print("\n检查计数一致性...")
    all_successful_counts = set()
    all_failed_counts = set()
    
    for key, neuron_data in data.items():
        sc = neuron_data.get('successful_count', 0)
        fc = neuron_data.get('failed_count', 0)
        all_successful_counts.add(sc)
        all_failed_counts.add(fc)
    
    if len(all_successful_counts) > 1:
        print(f"[WARNING] 警告: successful_count 不一致: {sorted(all_successful_counts)}")
    else:
        print(f"[OK] successful_count 一致: {list(all_successful_counts)[0]}")
    
    if len(all_failed_counts) > 1:
        print(f"[WARNING] 警告: failed_count 不一致: {sorted(all_failed_counts)}")
    else:
        print(f"[OK] failed_count 一致: {list(all_failed_counts)[0]}")
    
    # 6. 检查 activation_diff 计算是否正确
    print("\n检查 activation_diff 计算...")
    diff_errors = 0
    for key, neuron_data in data.items():
        successful_mean = neuron_data.get('successful_mean', 0)
        failed_mean = neuron_data.get('failed_mean', 0)
        activation_diff = neuron_data.get('activation_diff', 0)
        
        expected_diff = successful_mean - failed_mean
        actual_diff = activation_diff
        
        # 允许浮点数误差
        if abs(expected_diff - actual_diff) > 1e-10:
            diff_errors += 1
            if diff_errors <= 5:  # 只显示前5个错误
                issues.append(f"[ERROR] activation_diff 计算错误在 {key}: 期望 {expected_diff}, 实际 {actual_diff}")
    
    if diff_errors == 0:
        print("[OK] activation_diff 计算正确")
    else:
        print(f"[ERROR] 发现 {diff_errors} 个 activation_diff 计算错误")
    
    # 7. 检查 activation_projection 是否等于 successful_mean
    print("\n检查 activation_projection 是否等于 successful_mean...")
    projection_errors = 0
    for key, neuron_data in data.items():
        successful_mean = neuron_data.get('successful_mean', 0)
        activation_projection = neuron_data.get('activation_projection', 0)
        
        if abs(successful_mean - activation_projection) > 1e-10:
            projection_errors += 1
            if projection_errors <= 5:
                issues.append(f"[ERROR] activation_projection 不等于 successful_mean 在 {key}")
    
    if projection_errors == 0:
        print("[OK] activation_projection 等于 successful_mean")
    else:
        print(f"[ERROR] 发现 {projection_errors} 个 activation_projection 不一致")
    
    # 8. 检查数值范围
    print("\n检查数值范围...")
    successful_means = [neuron_data.get('successful_mean', 0) for neuron_data in data.values()]
    failed_means = [neuron_data.get('failed_mean', 0) for neuron_data in data.values()]
    activation_diffs = [neuron_data.get('activation_diff', 0) for neuron_data in data.values()]
    
    print(f"successful_mean 范围: [{min(successful_means):.6f}, {max(successful_means):.6f}]")
    print(f"failed_mean 范围: [{min(failed_means):.6f}, {max(failed_means):.6f}]")
    print(f"activation_diff 范围: [{min(activation_diffs):.6f}, {max(activation_diffs):.6f}]")
    
    # 检查是否有 NaN 或 Inf
    import math
    nan_inf_count = 0
    for key, neuron_data in data.items():
        for field in ['successful_mean', 'failed_mean', 'successful_std', 'failed_std', 
                      'activation_projection', 'activation_diff']:
            val = neuron_data.get(field, 0)
            if math.isnan(val) or math.isinf(val):
                nan_inf_count += 1
                issues.append(f"[ERROR] {key} 的 {field} 是 NaN 或 Inf: {val}")
                break
    
    if nan_inf_count == 0:
        print("[OK] 没有发现 NaN 或 Inf 值")
    else:
        print(f"[ERROR] 发现 {nan_inf_count} 个包含 NaN 或 Inf 的神经元")
    
    # 9. 检查层分布
    print("\n检查层分布...")
    if layer_neuron_counts:
        layers = sorted(layer_neuron_counts.keys())
        print(f"层范围: {min(layers)} - {max(layers)}")
        print(f"包含神经元的层数: {len(layers)}")
        print(f"每层神经元数量: {dict(sorted(layer_neuron_counts.items()))}")
    
    # 10. 汇总问题
    print("\n" + "=" * 80)
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:20]:  # 只显示前20个问题
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... 还有 {len(issues) - 20} 个问题未显示")
        return False
    else:
        print("[OK] 数据检查通过，未发现明显问题")
        return True

if __name__ == '__main__':
    json_path = Path(__file__).parent.parent / 'outputs' / 'activation_projection_after' / 'activation_projection.json'
    
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    
    success = check_activation_projection_data(json_path)
    sys.exit(0 if success else 1)
