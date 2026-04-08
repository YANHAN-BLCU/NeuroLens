"""
从 SALAD 数据集中提取安全部分用于安全神经元识别

支持从以下数据源提取安全样本：
1. base_evaluation.jsonl - 评估日志中安全响应的样本（guard.verdict == "allow"）
2. defense_enhanced_set_train.jsonl - 防御增强的样本（daugq 字段）
3. mcq_set_train.jsonl - 多选题中的安全答案（gt == "A" 的样本）

生成统一的 jsonl 文件到 logs 目录，格式为：
- {"text": "..."} - 简单文本格式
- {"input": {"prompt": "..."}, "output": "..."} - 带输入输出的格式（如果有输出）

示例用法：
    python scripts/extract_salad_safety_samples.py \
        --input_paths \
            logs/base_evaluation.jsonl \
            data/salad/raw/defense_enhanced_set_train.jsonl \
            data/salad/raw/mcq_set_train.jsonl \
        --output_path logs/salad_safety_samples.jsonl \
        --max_samples_per_file 10000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def extract_from_evaluation(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从评估日志中提取安全样本（guard.verdict == "allow"）"""
    guard = obj.get("guard", {})
    verdict = guard.get("verdict")
    
    if verdict != "allow":
        return None
    
    input_data = obj.get("input", {})
    inference = obj.get("inference", {})
    
    # 提取 prompt 和 output
    if isinstance(input_data, dict):
        prompt = input_data.get("prompt")
    else:
        prompt = input_data
    
    output = inference.get("output")
    
    if prompt and isinstance(prompt, str) and prompt.strip():
        if output and isinstance(output, str) and output.strip():
            # 有输出：返回 prompt + output 格式
            return {
                "input": {"prompt": prompt.strip()},
                "output": output.strip(),
            }
        else:
            # 只有 prompt：返回 text 格式
            return {"text": prompt.strip()}
    
    return None


def extract_from_defense(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从防御增强数据集中提取安全样本（使用 daugq 字段）"""
    daugq = obj.get("daugq")
    if daugq and isinstance(daugq, str) and daugq.strip():
        return {"text": daugq.strip()}
    return None


def extract_from_mcq(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从多选题数据集中提取安全样本（gt == "A"）"""
    gt = obj.get("gt")
    baseq = obj.get("baseq")
    
    if gt == "A" and baseq and isinstance(baseq, str) and baseq.strip():
        return {"text": baseq.strip()}
    
    return None


def detect_source_type(file_path: Path) -> str:
    """根据文件名自动检测数据源类型"""
    filename = file_path.name.lower()
    if "evaluation" in filename:
        return "evaluation"
    elif "defense" in filename:
        return "defense"
    elif "mcq" in filename:
        return "mcq"
    else:
        return "auto"


def extract_safe_samples(
    file_path: Path,
    source_type: str = "auto",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从文件中提取安全样本"""
    if source_type == "auto":
        source_type = detect_source_type(file_path)
    
    samples = []
    extract_fn = None
    
    if source_type == "evaluation":
        extract_fn = extract_from_evaluation
    elif source_type == "defense":
        extract_fn = extract_from_defense
    elif source_type == "mcq":
        extract_fn = extract_from_mcq
    else:
        # 自动检测：尝试所有方法
        def auto_extract(obj):
            # 按优先级尝试
            sample = extract_from_evaluation(obj)
            if sample:
                return sample
            sample = extract_from_defense(obj)
            if sample:
                return sample
            sample = extract_from_mcq(obj)
            if sample:
                return sample
            return None
        extract_fn = auto_extract
    
    print(f"[提取] 处理文件: {file_path.name} (类型: {source_type})")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            
            sample = extract_fn(obj)
            if sample is not None:
                samples.append(sample)
    
    print(f"[提取] 从 {file_path.name} 提取了 {len(samples)} 个安全样本")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="从 SALAD 数据集中提取安全部分用于安全神经元识别"
    )
    
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
        help="输入文件路径（可以指定多个文件）",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/salad_safety_samples.jsonl",
        help="输出文件路径（默认: logs/salad_safety_samples.jsonl）",
    )
    
    parser.add_argument(
        "--source_types",
        type=str,
        nargs="*",
        default=None,
        choices=["auto", "defense", "mcq", "evaluation"],
        help="对应的数据源类型列表（None 表示自动检测）",
    )
    
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="每个文件的最大样本数（None 表示全部）",
    )
    
    parser.add_argument(
        "--max_total_samples",
        type=int,
        default=None,
        help="总最大样本数（None 表示全部）",
    )
    
    args = parser.parse_args()
    
    # 准备数据源类型
    if args.source_types is None:
        source_types = ["auto"] * len(args.input_paths)
    else:
        if len(args.source_types) != len(args.input_paths):
            print(f"[错误] source_types 数量 ({len(args.source_types)}) 与 input_paths 数量 ({len(args.input_paths)}) 不匹配")
            return
        source_types = args.source_types
    
    # 提取所有安全样本
    all_samples = []
    for file_path_str, source_type in zip(args.input_paths, source_types):
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print(f"[警告] 文件不存在: {file_path}")
            continue
        
        samples = extract_safe_samples(
            file_path=file_path,
            source_type=source_type,
            max_samples=args.max_samples_per_file,
        )
        
        remaining = (
            args.max_total_samples - len(all_samples)
            if args.max_total_samples is not None
            else None
        )
        
        if remaining is not None and remaining <= 0:
            break
        
        if remaining is not None and len(samples) > remaining:
            samples = samples[:remaining]
        
        all_samples.extend(samples)
    
    print(f"\n[总计] 总共提取了 {len(all_samples)} 个安全样本")
    
    # 保存到输出文件
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"[保存] 结果已保存到: {output_path}")
    
    # 显示统计信息
    text_only_count = sum(1 for s in all_samples if "text" in s and "input" not in s)
    prompt_output_count = sum(1 for s in all_samples if "input" in s and "output" in s)
    
    print(f"\n[统计] 样本格式分布:")
    print(f"  - 仅文本格式 ({{text: ...}}): {text_only_count}")
    print(f"  - 输入输出格式 ({{input: {{prompt: ...}}, output: ...}}): {prompt_output_count}")


if __name__ == "__main__":
    main()
