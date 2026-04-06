#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据集结构和分布
"""

import json
from pathlib import Path

def analyze_dataset():
    input_file = Path("logs/base_evaluation.jsonl")

    total = 0
    verdict_counts = {}
    asr_label_counts = {}
    severity_counts = {}
    
    # 假阳性分析
    false_positive_samples = []
    true_positive_samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
                guard = record.get('guard', {})
                verdict = guard.get('verdict', 'UNKNOWN')
                asr_label = guard.get('asr_label', -1)
                severity = guard.get('severity', 'UNKNOWN')
                rationale = guard.get('rationale', [])
                
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                asr_label_counts[asr_label] = asr_label_counts.get(asr_label, 0) + 0
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # 记录一些样本用于分析
                if verdict == 'flag' and asr_label == 1 and len(false_positive_samples) < 5:
                    false_positive_samples.append({
                        'sample_id': record.get('sample_id'),
                        'verdict': verdict,
                        'severity': severity,
                        'rationale': rationale[:2] if rationale else [],
                        'output_preview': record.get('inference', {}).get('output', '')[:200] + '...'
                    })
                    
            except json.JSONDecodeError:
                continue

    print('='*60)
    print('数据集统计')
    print('='*60)
    print(f'总样本数: {total}')
    print()
    print('Verdict 分布:')
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        pct = c/total*100
        print(f'  {v}: {c} ({pct:.2f}%)')
    print()
    print('ASR Label 分布:')
    for a, c in sorted(asr_label_counts.items()):
        pct = c/total*100
        print(f'  {a}: {c} ({pct:.2f}%)')
    print()
    print('Severity 分布:')
    for s, c in sorted(severity_counts.items(), key=lambda x: -x[1]):
        pct = c/total*100
        print(f'  {s}: {c} ({pct:.2f}%)')
    print()
    
    # 预估问题规模
    flag_asr1 = verdict_counts.get('flag', 0)
    block_asr1 = verdict_counts.get('block', 0)
    toxic_total = flag_asr1 + block_asr1
    safe_total = verdict_counts.get('allow', 0)
    
    print('='*60)
    print('问题诊断')
    print('='*60)
    print(f'标记为 toxic 的样本: {toxic_total} ({toxic_total/total*100:.2f}%)')
    print(f'标记为 safe 的样本: {safe_total} ({safe_total/total*100:.2f}%)')
    print()
    print(f'预期比例: toxic ~17.7%, safe ~82.3%')
    print(f'实际 toxic 比例: {toxic_total/total*100:.2f}%')
    
    if toxic_total/total > 0.25:
        print('⚠️ 警告: toxic 比例过高，可能存在大量假阳性')
    elif toxic_total/total < 0.10:
        print('⚠️ 警告: toxic 比例过低，可能存在假阴性')

if __name__ == '__main__':
    analyze_dataset()

