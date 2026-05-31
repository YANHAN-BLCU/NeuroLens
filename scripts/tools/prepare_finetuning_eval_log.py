#!/usr/bin/env python3
"""
prepare_finetuning_eval_log.py

将 NeuroLens 项目现有的分离式数据文件（模型输出 + 标签 + 原始 prompt）
合并为一个模拟评估日志（JSONL），输出格式与 fine_tuning 模块期望的格式完全兼容。

支持的输入数据集：
  1. base_set — 基础有害请求（question 字段，需按 original_index 关联标签）
  2. attack_enhanced_set — 攻击增强请求（augq 字段）

输出格式（每行一个 JSON 对象）：
{
    "guard": {
        "verdict": "allow" | "deny" | "controversial",   // 模型安全判决
        "jailbreak_success": true | false,                // jailbreak 是否成功（deny=True）
        "categories": [{"id": "illegal", "score": 0.95}] // 类别评分（可选）
    },
    "inference": {
        "output": "模型生成的安全拒绝回复...",
        "model": "meta-llama/Llama-3-8B-Instruct"
    },
    "input": {
        "prompt": "原始有害请求 prompt...",
        "original_sample": {
            "question": "...",
            "3-category": "O19: Illegal Drugs and Regulated/Controlled Substances",
            "2-category": "O14: Illegal Activities",
            "1-category": "O5: Malicious Use"
        }
    }
}

使用示例（base_set）：
    python scripts/prepare_finetuning_eval_log.py `
        --dataset-type base_set `
        --model-outputs outputs/data_set_output/base_set_outputs_0_4999.jsonl `
        --labels outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl `
        --raw-prompts data/salad/raw/base_set_train.jsonl `
        --output outputs/simulated_eval/base_set_eval_0_4999.jsonl

使用示例（attack_enhanced_set）：
    python scripts/prepare_finetuning_eval_log.py `
        --dataset-type attack_enhanced_set `
        --model-outputs outputs/data_set_output/attack_enhanced_outputs.jsonl `
        --labels outputs/data_set_output/labels/attack_enhanced_outputs.jsonl `
        --raw-prompts data/salad/raw/attack_enhanced_set_train.jsonl `
        --output outputs/simulated_eval/attack_enhanced_eval.jsonl

合并 base_set 全部 4 个分片（需先合并标签文件）：
    # 1. 合并标签
    Get-Content outputs/data_set_output/labels/base_set_outputs_0_4999.jsonl, `
                         outputs/data_set_output/labels/base_set_outputs_5000_9999.jsonl, `
                         outputs/data_set_output/labels/base_set_outputs_10000_14999.jsonl, `
                         outputs/data_set_output/labels/base_set_outputs_15000_21316.jsonl `
        | Set-Content outputs/data_set_output/labels/base_set_outputs_merged.jsonl

    # 2. 合并模型输出
    Get-Content outputs/data_set_output/base_set_outputs_0_4999.jsonl, `
                         outputs/data_set_output/base_set_outputs_5000_9999.jsonl, `
                         outputs/data_set_output/base_set_outputs_10000_14999.jsonl, `
                         outputs/data_set_output/base_set_outputs_15000_21316.jsonl `
        | Set-Content outputs/data_set_output/base_set_outputs_merged.jsonl

    # 3. 生成模拟评估日志
    python scripts/prepare_finetuning_eval_log.py `
        --dataset-type base_set `
        --model-outputs outputs/data_set_output/base_set_outputs_merged.jsonl `
        --labels outputs/data_set_output/labels/base_set_outputs_merged.jsonl `
        --raw-prompts data/salad/raw/base_set_train.jsonl `
        --output outputs/simulated_eval/base_set_eval_full.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional


# ============================================================================
# 数据集配置
# ============================================================================

DATASET_CONFIG = {
    "base_set": {
        "prompt_field": "question",
        "id_field": "qid",
        "output_field": "generated_output",
        "index_field": "original_index",
        "categories": ["3-category", "2-category", "1-category"],
    },
    "attack_enhanced_set": {
        "prompt_field": "augq",
        "id_field": "aid",
        "output_field": "generated_output",
        "index_field": "original_index",
        "categories": ["3-category", "2-category", "1-category"],
    },
}


def _normalize_label(label: str) -> str:
    """将项目的标签格式（Safe/Unsafe/Controversial）映射为评估日志格式。"""
    mapping = {
        "safe": "allow",
        "unsafe": "deny",
        "controversial": "controversial",
    }
    return mapping.get(label.strip().lower(), "controversial")


def _label_to_jailbreak_success(label: str) -> bool:
    """将标签映射为 jailbreak_success 布尔值。deny=True 表示被攻破。"""
    return label.strip().lower() == "unsafe"


def _iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterator[Dict]:
    """迭代读取 JSONL 文件。"""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 0):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON 解析失败（第 {i} 行）: {e}", file=sys.stderr)
                continue


def _load_indexed_labels(labels_path: Path) -> Dict[int, str]:
    """加载标签文件，返回 {original_index: label} 字典。"""
    label_map: Dict[int, str] = {}
    for obj in _iter_jsonl(labels_path):
        idx = obj.get("original_index")
        label = obj.get("label")
        if idx is not None and label is not None:
            label_map[int(idx)] = str(label).strip()
    print(f"[Labels] 加载了 {len(label_map)} 条标签（范围: {min(label_map)} ~ {max(label_map)}）")
    return label_map


def _load_raw_prompts(
    raw_path: Path,
    prompt_field: str,
    id_field: str,
    categories: List[str],
) -> Dict[int, Dict]:
    """
    加载原始 prompt 文件，返回 {id: prompt_data} 字典。

    base_set: key = qid, prompt_field = "question"
    attack_enhanced_set: key = aid, prompt_field = "augq"
    """
    prompts: Dict[int, Dict] = {}
    for obj in _iter_jsonl(raw_path):
        idx = obj.get(id_field)
        if idx is None:
            continue
        idx = int(idx)
        prompt_data = {
            "prompt": obj.get(prompt_field, ""),
        }
        for cat in categories:
            if cat in obj:
                prompt_data[cat] = obj[cat]
        prompts[idx] = prompt_data
    print(f"[Raw Prompts] 加载了 {len(prompts)} 条 prompt（id_field={id_field}, prompt_field={prompt_field}）")
    return prompts


def build_eval_log_entry(
    original_index: int,
    generated_output: str,
    label: str,
    raw_prompt_data: Optional[Dict],
) -> Dict:
    """
    将一条记录转换为模拟评估日志格式。

    Args:
        original_index: 样本在原始数据集中的索引
        generated_output: 模型生成的回复
        label: 安全标签（Safe / Unsafe / Controversial）
        raw_prompt_data: 原始 prompt 数据（含 prompt 文本和类别信息）

    Returns:
        符合 fine_tuning 模块期望的评估日志对象
    """
    verdict = _normalize_label(label)
    jailbreak_success = _label_to_jailbreak_success(label)

    # 构造 input 部分
    if raw_prompt_data:
        input_obj = {
            "prompt": raw_prompt_data.get("prompt", ""),
            "original_sample": {},
        }
        for cat in ["3-category", "2-category", "1-category"]:
            if cat in raw_prompt_data:
                input_obj["original_sample"][cat] = raw_prompt_data[cat]
    else:
        input_obj = {
            "prompt": "",
            "original_sample": {},
        }

    # 构造完整的评估日志对象
    entry = {
        "guard": {
            "verdict": verdict,
            "jailbreak_success": jailbreak_success,
            "categories": [],  # 原始标签文件中无类别评分，留空
        },
        "inference": {
            "output": generated_output,
            "model": "meta-llama/Llama-3-8B-Instruct",
        },
        "input": input_obj,
    }

    return entry


def merge_dataset(
    model_outputs_path: Path,
    labels_path: Path,
    raw_prompts_path: Path,
    dataset_type: str,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> int:
    """
    将分离的数据文件合并为一个模拟评估日志。

    Args:
        model_outputs_path: 模型输出文件路径（base_set_outputs_*.jsonl）
        labels_path: 标签文件路径（labels/*.jsonl）
        raw_prompts_path: 原始 prompt 文件路径（base_set_train.jsonl / attack_enhanced_set_train.jsonl）
        dataset_type: 数据集类型（"base_set" 或 "attack_enhanced_set"）
        output_path: 输出文件路径
        max_samples: 最大处理样本数（用于测试）

    Returns:
        写入的记录数
    """
    config = DATASET_CONFIG.get(dataset_type)
    if not config:
        raise ValueError(f"未知的数据集类型: {dataset_type}，支持的类型: {list(DATASET_CONFIG.keys())}")

    prompt_field = config["prompt_field"]
    id_field = config["id_field"]
    categories = config["categories"]

    # 加载标签（{original_index: label}）
    label_map = _load_indexed_labels(labels_path)

    # 加载原始 prompt（{qid/aid: {prompt, categories...}}）
    raw_prompts = _load_raw_prompts(raw_prompts_path, prompt_field, id_field, categories)

    # 遍历模型输出文件，按 original_index 关联标签和原始 prompt
    written = 0
    skipped_no_label = 0
    skipped_no_prompt = 0
    skipped_no_output = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for obj in _iter_jsonl(model_outputs_path, max_samples):
            idx = obj.get("original_index")
            if idx is None:
                continue
            idx = int(idx)

            generated_output = obj.get("generated_output", "").strip()
            if not generated_output:
                skipped_no_output += 1
                continue

            # 关联标签
            label = label_map.get(idx)
            if label is None:
                skipped_no_label += 1
                continue

            # 关联原始 prompt
            raw_prompt_data = raw_prompts.get(idx)

            # 构建评估日志条目
            entry = build_eval_log_entry(
                original_index=idx,
                generated_output=generated_output,
                label=label,
                raw_prompt_data=raw_prompt_data,
            )

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    # 打印统计信息
    print(f"\n[Done] 模拟评估日志已生成: {output_path}")
    print(f"  成功写入:   {written} 条")
    if skipped_no_label:
        print(f"  跳过（无标签）:   {skipped_no_label} 条")
    if skipped_no_prompt:
        print(f"  跳过（无 prompt）: {skipped_no_prompt} 条")
    if skipped_no_output:
        print(f"  跳过（无输出）:   {skipped_no_output} 条")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 NeuroLens 分离数据文件合并为模拟评估日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=["base_set", "attack_enhanced_set"],
        help="数据集类型（决定 prompt 字段名和关联方式）",
    )

    parser.add_argument(
        "--model-outputs",
        type=str,
        required=True,
        help="模型输出文件路径（outputs/data_set_output/base_set_outputs_*.jsonl）",
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="标签文件路径（outputs/data_set_output/labels/*.jsonl）",
    )

    parser.add_argument(
        "--raw-prompts",
        type=str,
        required=True,
        help="原始 prompt 文件路径（data/salad/raw/base_set_train.jsonl 或 attack_enhanced_set_train.jsonl）",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径（outputs/simulated_eval/*.jsonl）",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试，默认处理全部）",
    )

    return parser.parse_args()


def main() -> int:
    # Windows 终端下强制 UTF-8 输出
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    args = parse_args()

    model_outputs_path = Path(args.model_outputs)
    labels_path = Path(args.labels)
    raw_prompts_path = Path(args.raw_prompts)
    output_path = Path(args.output)

    # 检查输入文件是否存在
    for name, path in [
        ("模型输出", model_outputs_path),
        ("标签", labels_path),
        ("原始 Prompt", raw_prompts_path),
    ]:
        if not path.exists():
            print(f"[Error] 文件不存在: {name} = {path}", file=sys.stderr)
            return 1

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Config] 数据集类型: {args.dataset_type}")
    print(f"[Config] 模型输出: {model_outputs_path}")
    print(f"[Config] 标签文件: {labels_path}")
    print(f"[Config] 原始 Prompt: {raw_prompts_path}")
    print(f"[Config] 输出文件: {output_path}")
    if args.max_samples:
        print(f"[Config] 最大样本数: {args.max_samples}")

    try:
        written = merge_dataset(
            model_outputs_path=model_outputs_path,
            labels_path=labels_path,
            raw_prompts_path=raw_prompts_path,
            dataset_type=args.dataset_type,
            output_path=output_path,
            max_samples=args.max_samples,
        )
        print(f"\n[Success] 共写入 {written} 条评估日志记录")
        return 0
    except Exception as e:
        print(f"[Error] 合并失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
