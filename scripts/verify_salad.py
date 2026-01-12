#!/usr/bin/env python3
"""
验证 SALAD-Bench 数据集完整性

使用方法：
    python scripts/verify_salad.py --input data/salad/raw
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


def verify_salad_data(input_dir: Path) -> Dict:
    """
    验证 SALAD-Bench 数据
    
    Args:
        input_dir: 输入目录
    
    Returns:
        验证结果字典
    """
    results = {
        "total_files": 0,
        "attack_types": Counter(),
        "missing_fields": Counter(),
        "samples_by_type": Counter(),
        "file_stats": {},  # 每个文件的统计信息
    }
    
    print(f"验证数据目录: {input_dir}")
    
    if not input_dir.exists():
        print(f"[ERROR] 目录不存在: {input_dir}")
        return results
    
    # 查找所有 JSON/JSONL 文件
    json_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.jsonl"))
    
    if not json_files:
        print(f"[WARNING] 未找到数据文件")
        return results
    
    results["total_files"] = len(json_files)
    
    # 不同配置的预期字段
    config_fields = {
        "base_set": ["qid", "question", "source", "1-category", "2-category", "3-category"],
        "attack_enhanced_set": ["qid", "baseq", "augq", "method", "1-category", "2-category", "3-category"],
        "defense_enhanced_set": ["qid", "baseq", "daugq", "dmethod", "1-category", "2-category", "3-category"],
        "mcq_set": ["baseq", "mcq", "choices", "gt", "1-category", "2-category", "3-category"],
    }
    
    for json_file in json_files:
        print(f"处理文件: {json_file.name}")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                if json_file.suffix == ".jsonl":
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            
            # 确定配置类型
            config_type = None
            for config in config_fields.keys():
                if config in json_file.name:
                    config_type = config
                    break
            
            # 获取该配置的预期字段
            expected_fields = config_fields.get(config_type, [])
            file_missing = Counter()
            file_sample_count = 0
            
            for sample in data:
                file_sample_count += 1
                # 统计类别（使用3-category或1-category作为攻击类型）
                attack_type = sample.get("3-category", sample.get("1-category", sample.get("source", "unknown")))
                results["attack_types"][attack_type] += 1
                results["samples_by_type"][attack_type] += 1
                
                # 检查该配置的预期字段
                if config_type:
                    for field in expected_fields:
                        if field not in sample:
                            file_missing[field] += 1
                            results["missing_fields"][f"{config_type}.{field}"] += 1
            
            # 保存文件统计信息
            results["file_stats"][json_file.name] = {
                "samples": file_sample_count,
                "config": config_type or "unknown",
                "missing_fields": dict(file_missing),
                "all_fields": list(data[0].keys()) if data else []
            }
        
        except Exception as e:
            print(f"[WARNING] 处理文件 {json_file.name} 时出错: {e}")
    
    return results


def print_summary(results: Dict):
    """打印验证摘要"""
    print(f"\n{'='*60}")
    print("验证摘要")
    print(f"{'='*60}")
    print(f"总文件数: {results['total_files']}")
    
    print(f"\n攻击类型统计:")
    for attack_type, count in results["attack_types"].most_common():
        print(f"  {attack_type}: {count} 样本")
    
    print(f"\n文件详情:")
    for filename, stats in results["file_stats"].items():
        print(f"  {filename}:")
        print(f"    配置: {stats['config']}")
        print(f"    样本数: {stats['samples']}")
        print(f"    字段: {', '.join(stats['all_fields'][:10])}{'...' if len(stats['all_fields']) > 10 else ''}")
        if stats['missing_fields']:
            print(f"    缺失字段: {dict(stats['missing_fields'])}")
        else:
            print(f"    [OK] 所有预期字段都存在")
    
    print(f"\n缺失字段统计 (按配置):")
    if results["missing_fields"]:
        for field, count in results["missing_fields"].most_common():
            print(f"  {field}: {count} 次缺失")
    else:
        print("  [OK] 所有必需字段都存在")
    
    print(f"\n攻击类型统计 (Top 20):")
    for attack_type, count in results["attack_types"].most_common(20):
        print(f"  {attack_type}: {count} 样本")


def main():
    parser = argparse.ArgumentParser(description="验证 SALAD-Bench 数据集")
    parser.add_argument(
        "--input",
        type=str,
        default="data/salad/raw",
        help="输入目录",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    results = verify_salad_data(input_dir)
    print_summary(results)


if __name__ == "__main__":
    main()

