#!/usr/bin/env python3
"""
详细评估 SALAD 评估结果
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import argparse

def analyze_results(file_path: Path) -> Dict:
    """详细分析评估结果"""
    results = []
    errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                    if 'error' in result:
                        errors.append({
                            'line': line_num,
                            'sample_id': result.get('sample_id', 'unknown'),
                            'error': result.get('error', 'unknown')
                        })
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
    
    if not results:
        return {'error': 'No valid results found'}
    
    # 统计信息
    stats = {
        'total_samples': len(results),
        'successful': 0,
        'errors': len(errors),
        'error_details': errors,
        'guard_verdicts': defaultdict(int),
        'guard_severities': defaultdict(int),
        'categories': defaultdict(lambda: {'count': 0, 'scores': []}),
        'latencies': {
            'total': [],
            'generation': [],
            'guard': []
        },
        'tokens': {
            'input': [],
            'output': []
        },
        'output_lengths': [],
        'samples_by_category': defaultdict(list),
    }
    
    # 分析每个结果
    for r in results:
        if 'error' in r:
            continue
        
        stats['successful'] += 1
        
        # Guard verdicts
        if 'guard' in r:
            guard = r['guard']
            verdict = guard.get('verdict', 'unknown')
            severity = guard.get('severity', 'unknown')
            stats['guard_verdicts'][verdict] += 1
            stats['guard_severities'][severity] += 1
            
            # 分析类别
            if 'categories' in guard:
                for cat in guard['categories']:
                    cat_id = cat.get('id', 'unknown')
                    score = cat.get('score', 0)
                    stats['categories'][cat_id]['count'] += 1
                    stats['categories'][cat_id]['scores'].append(score)
            
            # 按原始类别分组
            if 'input' in r and 'original_sample' in r['input']:
                orig = r['input']['original_sample']
                category_3 = orig.get('3-category', 'unknown')
                stats['samples_by_category'][category_3].append({
                    'sample_id': r.get('sample_id'),
                    'verdict': verdict,
                    'severity': severity
                })
        
        # Latencies
        if 'stats' in r:
            s = r['stats']
            if 'total_latency_ms' in s:
                stats['latencies']['total'].append(s['total_latency_ms'])
            if 'generation_latency_ms' in s:
                stats['latencies']['generation'].append(s['generation_latency_ms'])
            if 'guard_latency_ms' in s:
                stats['latencies']['guard'].append(s['guard_latency_ms'])
        
        # Tokens
        if 'inference' in r and 'tokens' in r['inference']:
            tokens = r['inference']['tokens']
            if 'input' in tokens:
                stats['tokens']['input'].append(tokens['input'])
            if 'output' in tokens:
                stats['tokens']['output'].append(tokens['output'])
            
            # Output length
            if 'output' in r['inference']:
                stats['output_lengths'].append(len(r['inference']['output']))
    
    # 计算统计值
    def calc_stats(values):
        if not values:
            return {}
        sorted_vals = sorted(values)
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': sorted_vals[len(sorted_vals) // 2],
            'p95': sorted_vals[int(len(sorted_vals) * 0.95)] if len(sorted_vals) > 0 else None,
            'p99': sorted_vals[int(len(sorted_vals) * 0.99)] if len(sorted_vals) > 0 else None,
        }
    
    # 计算类别平均分数
    category_stats = {}
    for cat_id, data in stats['categories'].items():
        if data['scores']:
            category_stats[cat_id] = {
                'count': data['count'],
                'avg_score': sum(data['scores']) / len(data['scores']),
                'max_score': max(data['scores']),
                'min_score': min(data['scores'])
            }
    
    # 按类别统计 verdict
    category_verdicts = defaultdict(lambda: defaultdict(int))
    for cat, samples in stats['samples_by_category'].items():
        for sample in samples:
            category_verdicts[cat][sample['verdict']] += 1
    
    return {
        'summary': {
            'total_samples': stats['total_samples'],
            'successful': stats['successful'],
            'errors': stats['errors'],
            'success_rate': stats['successful'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
        },
        'guard_analysis': {
            'verdicts': dict(stats['guard_verdicts']),
            'severities': dict(stats['guard_severities']),
            'verdict_rate': {
                'allow': stats['guard_verdicts']['allow'] / stats['successful'] * 100 if stats['successful'] > 0 else 0,
                'block': stats['guard_verdicts']['block'] / stats['successful'] * 100 if stats['successful'] > 0 else 0,
            }
        },
        'category_analysis': category_stats,
        'category_verdicts': {k: dict(v) for k, v in category_verdicts.items()},
        'performance': {
            'latency': {
                'total': calc_stats(stats['latencies']['total']),
                'generation': calc_stats(stats['latencies']['generation']),
                'guard': calc_stats(stats['latencies']['guard'])
            },
            'tokens': {
                'input': calc_stats(stats['tokens']['input']),
                'output': calc_stats(stats['tokens']['output']),
                'total_input': sum(stats['tokens']['input']),
                'total_output': sum(stats['tokens']['output']),
            },
            'output_length': calc_stats(stats['output_lengths'])
        },
        'errors': errors if errors else None
    }

def print_report(analysis: Dict, output_format: str = 'text'):
    """打印评估报告"""
    if output_format == 'json':
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        return
    
    # 文本格式报告
    print("=" * 80)
    print("SALAD 评估结果报告")
    print("=" * 80)
    print()
    
    # 摘要
    summary = analysis['summary']
    print("[摘要统计]")
    print("-" * 80)
    print(f"总样本数:     {summary['total_samples']}")
    print(f"成功处理:     {summary['successful']} ({summary['success_rate']:.1f}%)")
    print(f"错误数量:     {summary['errors']}")
    print()
    
    # Guard 分析
    guard = analysis['guard_analysis']
    print("[安全判断分析]")
    print("-" * 80)
    print("判断结果:")
    for verdict, count in guard['verdicts'].items():
        rate = guard['verdict_rate'].get(verdict, 0)
        print(f"  {verdict:10s}: {count:4d} ({rate:5.1f}%)")
    print()
    print("严重程度:")
    for severity, count in guard['severities'].items():
        print(f"  {severity:10s}: {count:4d}")
    print()
    
    # 类别分析
    if analysis['category_analysis']:
        print("[类别分析]")
        print("-" * 80)
        for cat_id, stats in sorted(analysis['category_analysis'].items()):
            print(f"{cat_id:20s}: 平均分数 {stats['avg_score']:.3f} (范围: {stats['min_score']:.3f}-{stats['max_score']:.3f}, 出现 {stats['count']} 次)")
        print()
    
    # 按类别统计 verdict
    if analysis['category_verdicts']:
        print("[按类别统计判断结果]")
        print("-" * 80)
        for cat, verdicts in sorted(analysis['category_verdicts'].items()):
            print(f"\n{cat}:")
            for verdict, count in verdicts.items():
                print(f"  {verdict:10s}: {count:4d}")
        print()
    
    # 性能分析
    perf = analysis['performance']
    print("[性能分析]")
    print("-" * 80)
    
    def print_perf_stats(name, stats):
        if stats:
            print(f"{name}:")
            print(f"  平均: {stats['mean']:.2f} ms")
            print(f"  中位数: {stats['median']:.2f} ms")
            print(f"  最小值: {stats['min']:.2f} ms")
            print(f"  最大值: {stats['max']:.2f} ms")
            if stats.get('p95'):
                print(f"  P95: {stats['p95']:.2f} ms")
    
    print_perf_stats("总延迟", perf['latency']['total'])
    print()
    print_perf_stats("生成延迟", perf['latency']['generation'])
    print()
    print_perf_stats("Guard延迟", perf['latency']['guard'])
    print()
    
    tokens = perf['tokens']
    if tokens['input']:
        print("Token 统计:")
        print(f"  总输入 tokens: {tokens['total_input']}")
        print(f"  总输出 tokens: {tokens['total_output']}")
        if tokens['input']['mean']:
            print(f"  平均输入 tokens: {tokens['input']['mean']:.1f}")
            print(f"  平均输出 tokens: {tokens['output']['mean']:.1f}")
        print()
    
    if perf['output_length']:
        print("输出长度:")
        print(f"  平均: {perf['output_length']['mean']:.0f} 字符")
        print(f"  中位数: {perf['output_length']['median']:.0f} 字符")
        print(f"  范围: {perf['output_length']['min']:.0f} - {perf['output_length']['max']:.0f} 字符")
        print()
    
    # 错误信息
    if analysis.get('errors'):
        print("[错误详情]")
        print("-" * 80)
        for err in analysis['errors']:
            print(f"样本 {err['sample_id']} (行 {err['line']}): {err['error']}")
        print()
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='评估 SALAD 评估结果')
    parser.add_argument('input', type=str, help='输入 JSONL 文件路径')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    analysis = analyze_results(input_path)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if args.format == 'json':
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            else:
                # 保存文本报告
                import io
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                print_report(analysis, args.format)
                sys.stdout = old_stdout
                f.write(buf.getvalue())
        print(f"Report saved to: {output_path}")
    else:
        print_report(analysis, args.format)

if __name__ == '__main__':
    main()

