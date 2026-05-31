r"""
====================================================================
应用 Delta 权重并生成模型输出
====================================================================
功能：
    1. 加载原始基础模型
    2. 应用 Delta 权重生成微调后模型
    3. 在源数据集上推理，输出模型生成文本

使用方式：
    python scripts/apply_delta_extract_and_label.py \
        --base_model "/root/autodl-tmp/neurolens/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct" \
        --delta_weights "/root/autodl-tmp/neurolens/outputs/vatsft_pipeline/model/stage1_safety/delta_weights.pt" \
        --output_dir "outputs/finetuned_extraction"

====================================================================
"""

import argparse
import json
import sys
import torch
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from tqdm import tqdm

# 确保 engine 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from engine.fine_tuning.tsft import load_delta_weights


# ──────────────────────────────────────────────────────────────────
# 配置和常量
# ──────────────────────────────────────────────────────────────────

GEN_TEMPERATURE = 0.3
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 512

@dataclass
class DatasetConfig:
    name: str
    file_path: str
    input_field: str

DATASET_CONFIGS = {
    "base_set": DatasetConfig(
        name="base_set",
        file_path="data/salad/raw/base_set_train.jsonl",
        input_field="question",
    ),
    "attack_enhanced": DatasetConfig(
        name="attack_enhanced",
        file_path="data/salad/raw/attack_enhanced_set_train.jsonl",
        input_field="augq",
    ),
}

# 提取标签的正则表达式
SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)

# 源数据缓存（避免重复加载）
_prompts_cache: Dict[str, Dict[int, str]] = {}


# ──────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────

def extract_label(content: str) -> str:
    """从模型输出中提取 Safe / Controversial / Unsafe，解析失败返回 unknown。"""
    m = SAFE_PATTERN.search(content)
    if m:
        return m.group(1).capitalize()
    return "unknown"

def load_jsonl(file_path: Path, max_lines: Optional[int] = None) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON解析错误 (line {i+1}): {e}")
    return data


def extract_input_text(sample: Dict, config: DatasetConfig) -> Optional[str]:
    """从样本中提取输入文本"""
    field = config.input_field
    value = sample.get(field)
    return str(value) if value is not None else None


# ──────────────────────────────────────────────────────────────────
# 加载和应用 Delta 权重
# ──────────────────────────────────────────────────────────────────

def load_finetuned_model(base_model_path: str, delta_weights_path: str, device=None):
    """加载应用了 Delta 权重的微调后模型

    Args:
        base_model_path: 原始基础模型路径
        delta_weights_path: Delta 权重文件路径
        device: 加载设备

    Returns:
        应用了 delta 权重后的模型和分词器
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("步骤 1: 加载微调后模型")
    print("=" * 60)

    # 加载分词器
    print(f"[1/2] 加载分词器: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # 加载 Delta 权重并应用
    print(f"[2/2] 应用 Delta 权重: {delta_weights_path}")
    model = load_delta_weights(base_model_path, delta_weights_path, device)

    print(f"[完成] 模型已加载到设备: {device}")
    return model, tokenizer


def get_delta_weights_info(delta_weights_path: str) -> dict:
    """获取 Delta 权重文件的统计信息"""
    try:
        delta_state = torch.load(delta_weights_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"[错误] 加载 Delta 权重文件失败: {e}")
        raise

    # 计算参数统计
    num_layers = 0
    total_elements = 0
    l2_norm = 0.0
    layer_names = list(delta_state.keys())

    for name, tensor in delta_state.items():
        num_elements = tensor.numel()
        num_layers += 1
        total_elements += num_elements
        l2_norm += (tensor ** 2).sum().item()

    l2_norm = l2_norm ** 0.5

    # 获取文件大小
    file_size = Path(delta_weights_path).stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    return {
        'num_layers': num_layers,
        'total_elements': total_elements,
        'l2_norm': l2_norm,
        'file_size_mb': file_size_mb,
        'layer_names': layer_names[:5] if len(layer_names) > 5 else layer_names,
    }


# ──────────────────────────────────────────────────────────────────
# 推理和输出生成
# ──────────────────────────────────────────────────────────────────

def generate_outputs(
    model,
    tokenizer,
    texts: List[str],
    indices: List[int],
    device: torch.device,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> Tuple[List[str], List[int]]:
    """
    批量生成推理输出。

    返回：(outputs, indices)
      - outputs: List[str]，生成文本
      - indices: List[int]，对应原数据集的索引
    """
    num_samples = len(texts)
    all_outputs: List[str] = []
    all_indices: List[int] = []

    with tqdm(total=num_samples, desc="生成输出", unit="条", dynamic_ncols=True, leave=False) as pbar:
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_texts = texts[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]
            batch_n = len(batch_texts)

            # 构建 prompt
            formatted_texts = [
                f"<|begin_of_text|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
                for text in batch_texts
            ]

            inputs = tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=False,
                max_length=16384,
            ).to(device)

            attention_mask = inputs["attention_mask"]
            batch_input_lens = attention_mask.sum(dim=1).tolist()

            # 推理
            with torch.inference_mode():
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=GEN_TEMPERATURE > 0,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                    repetition_penalty=GEN_REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Handle both Tensor and GenerateOutput formats
            generated_ids = gen_outputs.sequences if hasattr(gen_outputs, 'sequences') else gen_outputs
            eos_id = tokenizer.eos_token_id
            assistant_header_ids = tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n",
                add_special_tokens=False
            )
            batch_outputs = []

            for b in range(batch_n):
                seq = generated_ids[b]
                inp_len = batch_input_lens[b]
                seq_len = seq.shape[0]

                gen_slice = seq[inp_len:]
                gen_eos_pos = (gen_slice == eos_id).nonzero(as_tuple=True)[0]
                if len(gen_eos_pos) > 0:
                    eos_in_gen = gen_eos_pos[0].item()
                    if eos_in_gen == 0:
                        last_pos = seq_len - 1
                    else:
                        last_pos = min(inp_len + eos_in_gen - 1, seq_len - 1)
                else:
                    last_pos = seq_len - 1

                # 定位生成内容
                gen_seq = seq[inp_len:].tolist()
                gen_start_offset = 0
                for offset in range(len(gen_seq) - len(assistant_header_ids) + 1):
                    if gen_seq[offset:offset + len(assistant_header_ids)] == assistant_header_ids:
                        gen_start_offset = offset + len(assistant_header_ids)
                        break
                gen_start_pos = inp_len + gen_start_offset

                # 提取生成文本
                out_ids = seq[gen_start_pos:last_pos + 1]
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                batch_outputs.append(out_text)

            all_outputs.extend(batch_outputs)
            all_indices.extend(batch_indices)

            del inputs, gen_outputs, generated_ids

            pbar.update(batch_n)

    return all_outputs, all_indices


def run_inference(
    model,
    tokenizer,
    output_dir: Path,
    dataset_configs: Dict[str, DatasetConfig],
    batch_size: int = 4,
    max_samples: Optional[int] = None,
):
    """运行推理生成输出

    Args:
        model: 微调后的模型
        tokenizer: 分词器
        output_dir: 输出目录
        dataset_configs: 数据集配置字典
        batch_size: 批大小
        max_samples: 最大样本数
    """
    print("\n" + "=" * 60)
    print("步骤 2: 生成推理输出")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inference_results = {}

    for dataset_name, config in dataset_configs.items():
        file_path = Path(config.file_path)
        if not file_path.exists():
            logging.warning(f"数据集文件不存在: {file_path}")
            continue

        print(f"\n处理数据集: {dataset_name}")

        # 加载数据
        raw_data = load_jsonl(file_path, max_lines=max_samples)
        if not raw_data:
            print(f"  [警告] 无数据")
            continue

        # 提取输入文本
        inputs = []
        for i, sample in enumerate(raw_data):
            text = extract_input_text(sample, config)
            if text:
                inputs.append((i, text))

        if not inputs:
            print(f"  [警告] 无有效样本")
            continue

        indices = [item[0] for item in inputs]
        texts = [item[1] for item in inputs]
        n_total = len(indices)

        print(f"  样本数: {n_total}")

        # 生成输出
        outputs, output_indices = generate_outputs(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            indices=indices,
            device=device,
            batch_size=batch_size,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        # 保存推理输出
        output_output_dir = output_dir / "outputs"
        output_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_output_dir / f"{dataset_name}_outputs.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, out in zip(output_indices, outputs):
                f.write(json.dumps({"original_index": idx, "generated_output": out}, ensure_ascii=False) + "\n")
        print(f"  推理输出已保存: {output_file.name}")

        inference_results[dataset_name] = {
            "output_file": str(output_file),
            "num_samples": n_total,
        }

    return inference_results


# ──────────────────────────────────────────────────────────────────
# 安全标注
# ──────────────────────────────────────────────────────────────────

def moderate_batch(prompts: List[str], responses: List[str], model, tokenizer, batch_size: int = 16) -> List[str]:
    """批量使用 Qwen3Guard 进行 response moderation"""
    labels = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_responses = responses[batch_start:batch_end]

        messages_list = [
            [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(batch_prompts, batch_responses)
        ]

        texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in messages_list]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        for i, gen_id in enumerate(generated_ids):
            output_ids = gen_id[len(inputs.input_ids[i]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True)
            labels.append(extract_label(content))

    return labels


def moderate(prompt: str, response: str, model, tokenizer) -> str:
    """使用 Qwen3Guard 进行单条 response moderation"""
    return moderate_batch([prompt], [response], model, tokenizer, batch_size=1)[0]


def run_labeling(
    output_dir: Path,
    classifier_path: str,
    dataset_configs: Dict[str, DatasetConfig],
):
    """运行安全标注

    Args:
        output_dir: 输出目录
        classifier_path: Qwen3Guard 模型路径
        dataset_configs: 数据集配置字典
    """
    print("\n" + "=" * 60)
    print("步骤 3: 使用 Qwen3Guard 标注安全性")
    print("=" * 60)

    # 加载 Qwen3Guard
    print(f"[Guard] 加载模型: {classifier_path}")
    guard_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    if guard_tokenizer.pad_token is None:
        guard_tokenizer.pad_token = guard_tokenizer.eos_token
    guard_tokenizer.padding_side = 'left'

    guard_model = AutoModelForCausalLM.from_pretrained(
        classifier_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    guard_model.eval()
    print("[Guard] 模型加载完成")

    output_output_dir = output_dir / "outputs"
    label_output_dir = output_dir / "labels"
    label_output_dir.mkdir(parents=True, exist_ok=True)

    labeling_results = {}

    for dataset_name, config in dataset_configs.items():
        output_file = output_output_dir / f"{dataset_name}_outputs.jsonl"
        label_file = label_output_dir / f"{dataset_name}_labels.jsonl"

        if not output_file.exists():
            print(f"  [跳过] {dataset_name} - 推理输出文件不存在")
            continue

        if label_file.exists():
            print(f"  [跳过] {dataset_name} - 标签文件已存在")
            continue

        print(f"\n  处理: {dataset_name}")

        # 从缓存或加载源数据
        source_file = str(Path(config.file_path))
        if source_file not in _prompts_cache:
            prompts = load_jsonl(Path(config.file_path), max_lines=None)
            _prompts_cache[source_file] = {i: sample.get(config.input_field, "") for i, sample in enumerate(prompts)}
        prompt_dict = _prompts_cache[source_file]

        # 读取推理输出
        outputs = []
        with open(output_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        outputs.append(item)
                    except json.JSONDecodeError:
                        pass

        # 批量标注
        output_indices = [item["original_index"] for item in outputs]
        output_texts = [item["generated_output"] for item in outputs]
        output_prompts = [prompt_dict.get(idx, "") for idx in output_indices]

        labels = moderate_batch(output_prompts, output_texts, guard_model, guard_tokenizer, batch_size=16)

        # 写入标签文件
        with open(label_file, "w", encoding="utf-8") as fout:
            for idx, label in zip(output_indices, labels):
                fout.write(json.dumps({"original_index": idx, "label": label}, ensure_ascii=False) + "\n")

        print(f"  标签已保存: {label_file.name} ({len(labels)} 条)")
        labeling_results[dataset_name] = str(label_file)

    return labeling_results


# ──────────────────────────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────────────────────────

def generate_report(
    output_dir: Path,
    model_name: str,
    classifier_name: str,
    delta_weights_info: dict,
    inference_results: dict,
    labeling_results: dict = None,
):
    """生成综合报告"""
    print("\n" + "=" * 60)
    print("步骤 4: 生成报告")
    print("=" * 60)

    report_file = output_dir / "evaluation_summary.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 微调模型评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**模型**: {model_name}\n\n")
        if classifier_name != "N/A":
            f.write(f"**分类器**: {classifier_name}\n\n")
        f.write("---\n\n")

        # Delta 权重信息
        if delta_weights_info:
            f.write("## Delta 权重参数\n\n")
            f.write(f"- **修改层数**: {delta_weights_info['num_layers']}\n")
            f.write(f"- **修改参数量**: {delta_weights_info['total_elements']:,}\n")
            f.write(f"- **L2 范数**: {delta_weights_info['l2_norm']:.4f}\n")
            f.write(f"- **文件大小**: {delta_weights_info['file_size_mb']:.2f} MB\n\n")

        # 推理结果
        if inference_results:
            f.write("## 推理结果\n\n")
            for dataset_name, result in inference_results.items():
                f.write(f"### {dataset_name}\n\n")
                f.write(f"- **样本数**: {result['num_samples']}\n")
                f.write(f"- **推理输出**: {result['output_file']}\n\n")

        # 标注结果
        if labeling_results:
            f.write("## 安全标注结果\n\n")
            for dataset_name, label_file in labeling_results.items():
                f.write(f"- **{dataset_name}**: {label_file}\n")
            f.write("\n")

    print(f"  报告已生成: {report_file}")



def main():
    parser = argparse.ArgumentParser(
        description="应用 Delta 权重并生成模型推理输出"
    )

    # 模型和权重路径
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="原始基础模型路径"
    )
    parser.add_argument(
        "--delta_weights", type=str, required=True,
        help="Delta 权重文件路径 (.pt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/finetuned_extraction",
        help="输出目录"
    )

    # 数据集选择
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=["base_set", "attack_enhanced"],
        help="要处理的数据集"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="每个数据集最大样本数"
    )

    # 推理参数
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="推理批大小"
    )

    # 流程控制
    parser.add_argument(
        "--info_only", action="store_true",
        help="仅显示 Delta 权重信息"
    )

    # 报告参数
    parser.add_argument(
        "--model_name", type=str, default="",
        help="模型显示名称"
    )

    # 分类器参数
    parser.add_argument(
        "--classifier", type=str, default=None,
        help="Qwen3Guard 分类器模型路径 (用于安全标注)"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置显示名称
    model_name = args.model_name or Path(args.base_model).name

    # ========== 步骤 0: 分析 Delta 权重 ==========
    print("=" * 60)
    print("步骤 0: 分析 Delta 权重")
    print("=" * 60)
    delta_weights_info = get_delta_weights_info(args.delta_weights)
    print(f"  修改层数: {delta_weights_info['num_layers']}")
    print(f"  修改参数量: {delta_weights_info['total_elements']:,}")
    print(f"  L2 范数: {delta_weights_info['l2_norm']:.4f}")
    print(f"  文件大小: {delta_weights_info['file_size_mb']:.2f} MB")

    # ========== 步骤 1: 加载模型 ==========
    model, tokenizer = load_finetuned_model(
        args.base_model, args.delta_weights
    )

    # 如果只是查看信息
    if args.info_only:
        print("\n[完成] Delta 权重信息已加载")
        return

    # 选择要处理的数据集
    selected_configs = {
        name: DATASET_CONFIGS[name]
        for name in args.datasets
        if name in DATASET_CONFIGS
    }

    # ========== 步骤 2: 生成推理输出 ==========
    inference_results = run_inference(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        dataset_configs=selected_configs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # ========== 步骤 3: 安全标注（如果指定了分类器） ==========
    labeling_results = {}
    if args.classifier:
        labeling_results = run_labeling(
            output_dir=output_dir,
            classifier_path=args.classifier,
            dataset_configs=selected_configs,
        )

    # ========== 步骤 4: 生成报告 ==========
    generate_report(
        output_dir=output_dir,
        model_name=model_name,
        classifier_name=Path(args.classifier).name if args.classifier else "N/A",
        delta_weights_info=delta_weights_info,
        inference_results=inference_results,
        labeling_results=labeling_results if labeling_results else None,
    )

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"结果目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
