#!/usr/bin/env python3
"""
SALAD-Bench 数据集评估脚本
将数据集对接到 Llama-3.2-3B-Instruct 模型，并使用 Llama-Guard-3-1B 判断安全性

Usage:
    python scripts/evaluate_salad_pipeline.py \
      --data_dir data/salad/raw \
      --output logs/salad_evaluation.jsonl \
      --max_samples 100 \
      --config base_set
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加 engine 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.models import ModelManager


def load_salad_data(data_dir: Path, config: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict]:
    """
    加载 SALAD-Bench 数据集
    
    Args:
        data_dir: 数据目录路径
        config: 配置名称 (base_set, attack_enhanced_set, defense_enhanced_set, mcq_set)
               如果为 None，则加载所有配置
        max_samples: 最大样本数
    
    Returns:
        样本列表
    """
    samples = []
    
    if config:
        # 加载指定配置
        jsonl_file = data_dir / f"{config}_train.jsonl"
        if jsonl_file.exists():
            print(f"Loading {config} from {jsonl_file}")
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        # 添加配置信息
                        sample["_config"] = config
                        samples.append(sample)
                        if max_samples and len(samples) >= max_samples:
                            break
        else:
            print(f"Warning: {jsonl_file} not found")
    else:
        # 加载所有配置
        configs = ["base_set", "attack_enhanced_set", "defense_enhanced_set", "mcq_set"]
        for cfg in configs:
            jsonl_file = data_dir / f"{cfg}_train.jsonl"
            if jsonl_file.exists():
                print(f"Loading {cfg} from {jsonl_file}")
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            sample["_config"] = cfg
                            samples.append(sample)
                            if max_samples and len(samples) >= max_samples:
                                break
                if max_samples and len(samples) >= max_samples:
                    break
    
    print(f"Loaded {len(samples)} samples")
    return samples


def extract_prompt(sample: Dict) -> str:
    """
    从样本中提取提示文本
    
    不同配置有不同的字段：
    - base_set: question
    - attack_enhanced_set: baseq 或 augq
    - defense_enhanced_set: baseq 或 daugq
    - mcq_set: baseq 或 mcq
    """
    config = sample.get("_config", "base_set")
    
    if config == "base_set":
        return sample.get("question", "")
    elif config == "attack_enhanced_set":
        # 优先使用 augq（增强后的问题），否则使用 baseq
        return sample.get("augq", sample.get("baseq", ""))
    elif config == "defense_enhanced_set":
        # 优先使用 daugq（防御增强后的问题），否则使用 baseq
        return sample.get("daugq", sample.get("baseq", ""))
    elif config == "mcq_set":
        # 优先使用 mcq（多选题格式），否则使用 baseq
        return sample.get("mcq", sample.get("baseq", ""))
    else:
        # 通用回退：尝试常见字段
        for field in ["question", "baseq", "augq", "daugq", "mcq", "prompt"]:
            if field in sample and sample[field]:
                return sample[field]
        return ""


def evaluate_sample(
    model_manager: ModelManager,
    sample: Dict,
) -> Dict:
    """
    评估单个样本
    
    Args:
        model_manager: 模型管理器
        sample: 样本字典
    
    Returns:
        评估结果字典
    """
    # 提取提示
    prompt = extract_prompt(sample)
    if not prompt:
        return {
            "sample_id": sample.get("qid", sample.get("id", "unknown")),
            "error": "Empty prompt",
            "timestamp": datetime.now().isoformat(),
        }
    
    start_time = time.time()
    
    try:
        # 1. 使用 Llama-3.2-3B-Instruct 生成响应
        output_text, input_tokens, output_tokens, latency_ms = model_manager.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        
        generation_time = time.time() - start_time
        
        # 2. 使用 Llama-Guard-3-1B 判断安全性
        guard_start = time.time()
        guard_result = model_manager.moderate(
            text=output_text,
            threshold=0.5,
        )
        guard_time = time.time() - guard_start
        
        # 3. 构建结果
        result = {
            "sample_id": sample.get("qid", sample.get("id", "unknown")),
            "config": sample.get("_config", "unknown"),
            "timestamp": datetime.now().isoformat(),
            
            # 原始数据
            "input": {
                "prompt": prompt,
                "original_sample": {k: v for k, v in sample.items() if not k.startswith("_")},
            },
            
            # 模型响应
            "inference": {
                "output": output_text,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                },
                "latency_ms": latency_ms,
            },
            
            # 安全判断
            "guard": {
                "verdict": guard_result["verdict"],  # "allow" | "flag" | "block"
                "severity": guard_result["severity"],  # "low" | "medium" | "high"
                "rationale": guard_result["rationale"],
                "categories": guard_result["categories"],
                "latency_ms": guard_time * 1000,
            },
            
            # 统计信息
            "stats": {
                "total_latency_ms": (time.time() - start_time) * 1000,
                "generation_latency_ms": generation_time * 1000,
                "guard_latency_ms": guard_time * 1000,
            },
        }
        
        return result
        
    except Exception as e:
        return {
            "sample_id": sample.get("qid", sample.get("id", "unknown")),
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "latency_ms": (time.time() - start_time) * 1000,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SALAD-Bench dataset with Llama-3.2-3B-Instruct and Llama-Guard-3-1B")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/salad/raw",
        help="SALAD-Bench 数据目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["base_set", "attack_enhanced_set", "defense_enhanced_set", "mcq_set"],
        default=None,
        help="配置名称（如果为 None，则加载所有配置）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小（当前仅支持 1）",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="从第几个样本开始（用于断点续传）",
    )
    
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型管理器
    print("Initializing ModelManager...")
    model_manager = ModelManager()
    
    # 预热模型（懒加载）
    print("Warming up models...")
    _ = model_manager.load_llm()
    _ = model_manager.load_guard()
    print("Models loaded successfully")
    
    # 加载数据
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Loading data from {data_dir}")
    samples = load_salad_data(data_dir, config=args.config, max_samples=args.max_samples)
    
    if not samples:
        print("Error: No samples loaded")
        sys.exit(1)
    
    # 处理断点续传
    if args.start_from > 0:
        samples = samples[args.start_from:]
        print(f"Starting from sample {args.start_from}, {len(samples)} samples remaining")
    
    # 准备输出文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 评估
    print(f"\nStarting evaluation...")
    print(f"Total samples: {len(samples)}")
    print(f"Output file: {output_path}")
    print("-" * 60)
    
    results = []
    success_count = 0
    error_count = 0
    
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(samples)} (Success: {success_count}, Errors: {error_count})")
        
        result = evaluate_sample(model_manager, sample)
        results.append(result)
        
        if "error" in result:
            error_count += 1
        else:
            success_count += 1
        
        # 增量写入
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    
    if success_count > 0:
        # 统计安全判断结果
        verdicts = {}
        for r in results:
            if "guard" in r:
                verdict = r["guard"]["verdict"]
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        print(f"\nGuard verdicts:")
        for verdict, count in sorted(verdicts.items()):
            print(f"  {verdict}: {count} ({count/success_count*100:.1f}%)")
        
        # 平均延迟
        latencies = [r.get("stats", {}).get("total_latency_ms", 0) for r in results if "stats" in r]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"\nAverage latency: {avg_latency:.2f} ms")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

