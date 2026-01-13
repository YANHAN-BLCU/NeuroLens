#!/usr/bin/env python3
"""
分析 SALAD 评估结果
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def analyze_file(file_path: Path) -> Dict:
    """分析单个结果文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return {}
    
    # 统计信息
    stats = {
        'total_samples': len(results),
        'successful': 0,
        'errors': 0,
        'guard_verdicts': defaultdict(int),
        'guard_severities': defaultdict(int),
        'avg_latency_ms': 0,
        'avg_generation_latency_ms': 0,
        'avg_guard_latency_ms': 0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'avg_output_length': 0,
    }
    
    latencies = []
    gen_latencies = []
    guard_latencies = []
    output_lengths = []
    
    for r in results:
        if 'error' in r:
            stats['errors'] += 1
            continue
        
        stats['successful'] += 1
        
        # Guard verdicts
        if 'guard' in r:
            verdict = r['guard'].get('verdict', 'unknown')
            severity = r['guard'].get('severity', 'unknown')
            stats['guard_verdicts'][verdict] += 1
            stats['guard_severities'][severity] += 1
        
        # Latencies
        if 'stats' in r:
            if 'total_latency_ms' in r['stats']:
                latencies.append(r['stats']['total_latency_ms'])
            if 'generation_latency_ms' in r['stats']:
                gen_latencies.append(r['stats']['generation_latency_ms'])
            if 'guard_latency_ms' in r['stats']:
                guard_latencies.append(r['stats']['guard_latency_ms'])
        
        # Tokens
        if 'inference' in r and 'tokens' in r['inference']:
            stats['total_input_tokens'] += r['inference']['tokens'].get('input', 0)
            stats['total_output_tokens'] += r['inference']['tokens'].get('output', 0)
            output_lengths.append(len(r['inference'].get('output', '')))
    
    # 计算平均值
    if latencies:
        stats['avg_latency_ms'] = sum(latencies) / len(latencies)
    if gen_latencies:
        stats['avg_generation_latency_ms'] = sum(gen_latencies) / len(gen_latencies)
    if guard_latencies:
        stats['avg_guard_latency_ms'] = sum(guard_latencies) / len(guard_latencies)
    if output_lengths:
        stats['avg_output_length'] = sum(output_lengths) / len(output_lengths)
    
    return stats

def main():
    base_dir = Path(__file__).parent.parent
    log_dir = base_dir / 'logs'
    
    files = {
        'base_set': log_dir / 'base_set.jsonl',
        'attack_enhanced_set': log_dir / 'attack_enhanced_set.jsonl',
        'defense_enhanced_set': log_dir / 'defense_enhanced_set.jsonl',
        'mcq_set': log_dir / 'mcq_set.jsonl',
    }
    
    all_stats = {}
    for name, path in files.items():
        if path.exists():
            all_stats[name] = analyze_file(path)
    
    # 输出统计结果
    print(json.dumps(all_stats, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()

