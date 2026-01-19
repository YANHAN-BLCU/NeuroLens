#!/usr/bin/env python3
"""验证 ASR 字段是否已添加"""

import json
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else "logs/base_evaluation.jsonl"

with open(file_path, 'r', encoding='utf-8') as f:
    total = 0
    has_fields = 0
    missing_fields = 0
    
    for i, line in enumerate(f, 1):
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
            total += 1
            
            if "guard" in data:
                guard = data["guard"]
                if "jailbreak_success" in guard and "jailbreak_success_level" in guard and "asr_label" in guard:
                    has_fields += 1
                else:
                    missing_fields += 1
                    if missing_fields <= 5:
                        print(f"第 {i} 行缺少字段: {list(guard.keys())}")
            
            if i >= 100:
                break
        except:
            pass
    
    print(f"总记录数: {total}")
    print(f"有 ASR 字段: {has_fields}")
    print(f"缺少 ASR 字段: {missing_fields}")

