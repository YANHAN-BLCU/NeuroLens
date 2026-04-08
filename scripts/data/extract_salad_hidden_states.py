#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.pop("NPU_VISIBLE_DEVICES", None)
os.environ.pop("ASCEND_VISIBLE_DEVICES", None)

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 控制每次 generation 并行处理的样本数，影响速度和显存占用
# 设为 0 表示不限制输出长度（实际受限于模型最大上下文和内存）
MAX_NEW_TOKENS = 16384
GEN_TEMPERATURE = 0.3
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.05


@dataclass
class DatasetConfig:
    name: str
    file_path: str
    input_field: str
    max_total: Optional[int] = None


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
    "defense_enhanced": DatasetConfig(
        name="defense_enhanced",
        file_path="data/salad/raw/defense_enhanced_set_train.jsonl",
        input_field="daugq",
    ),
    "mcq_set": DatasetConfig(
        name="mcq_set",
        file_path="data/salad/raw/mcq_set_train.jsonl",
        input_field="mcq",
    ),
}


def load_jsonl(
    file_path: Path,
    max_lines: Optional[int] = None,
    skip: int = 0,
) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if skip and i < skip:
                continue
            if max_lines and (i - skip) >= max_lines:
                break
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON解析错误 (line {i+1}): {e}")
    return data


def extract_input_text(sample: Dict, config: DatasetConfig) -> Optional[str]:
    field = config.input_field
    if "." in field:
        parts = field.split(".")
        obj = sample
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return str(obj) if obj is not None else None
    else:
        value = sample.get(field)
        return str(value) if value is not None else None


def load_existing_hidden_states(
    npz_path: Path,
) -> Tuple[Optional[np.ndarray], List[int]]:
    if not npz_path.exists():
        return None, []

    try:
        data = np.load(npz_path, allow_pickle=True)
        # Check each key safely — numpy arrays raise "ambiguous truth value"
        # when used in `or`/`and` chains, so use explicit None checks.
        raw_hs = None
        for key in ("hidden_states", "generation_hs", "train_hs"):
            val = data.get(key)
            if val is not None:
                raw_hs = val
                break
        # Treat empty or None as no existing data.
        if raw_hs is None or (hasattr(raw_hs, "size") and raw_hs.size == 0):
            hidden_states = None
        else:
            hidden_states = raw_hs

        if "original_indices" in data:
            existing_indices = data["original_indices"].tolist()
        else:
            existing_indices = []

        return hidden_states, existing_indices
    except Exception as e:
        logging.warning(f"加载已有数据失败 {npz_path}: {e}")
        return None, []


def load_existing_output_indices(jsonl_path: Path) -> List[int]:
    if not jsonl_path.exists():
        return []

    indices = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    indices.append(record.get("original_index", -1))
                except json.JSONDecodeError:
                    pass
    return indices


def extract_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    indices: List[int],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 4,
    max_new_tokens: int = 16384,
    desc: str = "",
    output_dir: Optional[Path] = None,
    pooling: str = "last_hidden",
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    流式批次处理：每批次生成后立即写盘，不在内存中累积。

    pooling 可选值：
      - "last_hidden"  (默认)：所有层(1~N)在生成序列末尾 last_pos 位置的向量拼接
      - "last_layer"   ：仅最后一层在生成序列末尾 last_pos 位置的向量
      - "last_input"   ：所有层在输入序列末尾 token 位置的向量拼接

    输出文件：
      - {output_dir}/{desc}_hidden_states.npz   # 追加写入
      - {output_dir}/{desc}_outputs.jsonl        # 追加写入
    """
    num_samples = len(texts)
    all_hidden_list: List[np.ndarray] = []
    all_outputs: List[str] = []
    all_indices: List[int] = []

    num_layers = None
    hidden_dim = None
    total_written = 0  # 跟踪全局已写入样本总数，用于一致性校验

    with tqdm(total=num_samples, desc=f"提取隐藏态{desc}", unit="条") as pbar:
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_texts = texts[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]
            batch_n = len(batch_texts)

            # --- 构建 prompt ---
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

            # 计算每个样本的输入长度（不含左padding）
            padded_len = attention_mask.shape[1]
            batch_input_lens = []
            for b in range(batch_n):
                row_mask = attention_mask[b]
                pad_pos = (row_mask == 0).nonzero(as_tuple=True)[0]
                batch_input_lens.append(pad_pos[0].item() if len(pad_pos) > 0 else padded_len)

            # --- 单遍：generate + output_hidden_states 同时完成，一次 forward 无重复计算 ---
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
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

            generated_ids = gen_outputs.sequences
            raw_hs = gen_outputs.hidden_states

            # 初始化全局维度信息（仅第一批时确定）
            # generate() 返回的 hidden_states 有两种格式（因模型/设备实现而异）：
            #   格式A: tuple of (num_layers+1) tensors, each shape (batch, seq, hidden)
            #          → raw_hs[layer] = 全序列的 hidden state（seq 覆盖输入+生成）
            #   格式B: tuple of (num_steps) tuples, each (num_layers+1) tensors
            #          → raw_hs[step] = tuple of layers, 每步 tensor shape (batch, 1, hidden)
            if num_layers is None:
                first = raw_hs[0]
                if isinstance(first, torch.Tensor):
                    # 格式A: tensor，直接取 hidden_dim
                    num_layers = len(raw_hs) - 1
                    hidden_dim = first.shape[-1]
                elif isinstance(first, tuple):
                    # 格式B: tuple of step-tuples
                    num_layers = len(first) - 1
                    # 任意一步的最后一层都能拿到 hidden_dim
                    hidden_dim = first[num_layers].shape[-1]
                else:
                    raise ValueError(f"hidden_states 格式未知: type={type(first)}")

            # 构建每层的完整序列 hidden state
            if isinstance(raw_hs[0], torch.Tensor):
                # 格式A: raw_hs[layer] 已是 (batch, full_seq, hidden)，直接用
                hs_per_layer = raw_hs
            else:
                # 格式B: 拼接 raw_hs[0]（输入序列）+ raw_hs[1..N]（每步1个token）
                # raw_hs[0]: tuple of (num_layers+1) tensors, each (batch, input_len, hidden)
                # raw_hs[s] (s>=1): tuple of (num_layers+1) tensors, each (batch, 1, hidden)
                # → 拼接后: hs_per_layer[layer] = (batch, full_seq, hidden)
                inp_layer_tensors = raw_hs[0]           # tuple of num_layers+1 tensors
                gen_layer_tensors = raw_hs[1:]          # tuple of num_steps-1 tuples

                hs_per_layer = []
                for layer_idx in range(num_layers + 1):
                    inp_t = inp_layer_tensors[layer_idx]        # (batch, input_len, hidden)
                    gen_tensors = [step_tensors[layer_idx] for step_tensors in gen_layer_tensors]
                    # cat dim=1: (batch, input_len, hidden) + (batch, 1, hidden)*N
                    if gen_tensors:
                        full_t = torch.cat([inp_t] + gen_tensors, dim=1)
                    else:
                        full_t = inp_t
                    hs_per_layer.append(full_t)

            # --- 解码 + 提取 ---
            eos_id = tokenizer.eos_token_id
            # assistant header token IDs（用于精确定位生成起始位置）
            assistant_header_ids = tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n",
                add_special_tokens=False
            )
            batch_last_positions = []
            batch_outputs = []
            batch_gen_start_positions = []  # 记录每条的实际生成起始位置（调试用）

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

                batch_last_positions.append(last_pos)

                # 精确定位生成内容起始位置：
                # 在 seq[inp_len:] 中查找 assistant header，generation 从其之后开始
                gen_seq = seq[inp_len:].tolist()  # 转 list 便于滑动搜索
                gen_start_offset = 0
                for offset in range(len(gen_seq) - len(assistant_header_ids) + 1):
                    if gen_seq[offset:offset + len(assistant_header_ids)] == assistant_header_ids:
                        gen_start_offset = offset + len(assistant_header_ids)
                        break
                # 如果没找到 assistant header（兼容旧行为），使用 inp_len
                gen_start_pos = inp_len + gen_start_offset

                batch_gen_start_positions.append(gen_start_pos)

                # 仅提取模型的生成内容，不含输入部分
                out_ids = seq[gen_start_pos:last_pos + 1]
                out_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                batch_outputs.append(out_text)

            # hs_per_layer[layer_idx]: tensor (batch_n, full_seq_len, hidden_dim)
            # 注意：Llama 3 等模型可能在 generate 内部追加 BOS token，
            # 导致 generated_ids 比 hidden_states 序列长 1，此时需做边界截断
            hs_len = hs_per_layer[0].shape[1]

            # 索引 [b, last_pos, :] 提取样本 b 在 last_pos 位置的向量
            for b in range(batch_n):
                last_pos = batch_last_positions[b]
                inp_len = batch_input_lens[b]
                gen_start_pos = batch_gen_start_positions[b]

                if pooling == "last_input":
                    # 输入序列末尾 token 的位置
                    target_pos = inp_len - 1
                elif pooling == "last_response":
                    # 生成内容（assistant header 之后）的末尾 token 位置
                    target_pos = last_pos
                else:
                    # 生成序列末尾 token 的位置（默认 last_hidden）
                    target_pos = last_pos
                # 截断到 hidden_states 实际长度（修复 Llama 3 BOS token 偏移问题）
                target_pos = min(target_pos, hs_len - 1)

                if pooling == "last_layer":
                    # 仅取最后一层
                    sample_hs = hs_per_layer[num_layers][b, target_pos, :].float().cpu().numpy()
                else:
                    # 所有层(1~num_layers)拼接，或仅 last_hidden 时每层向量
                    sample_hs = np.stack([
                        hs_per_layer[layer_idx][b, target_pos, :].float().cpu().numpy()
                        for layer_idx in range(1, num_layers + 1)
                    ], axis=0)

                all_hidden_list.append(sample_hs.flatten())

            all_outputs.extend(batch_outputs)
            all_indices.extend(batch_indices)

            # --- 批次结束：立即写盘，释放显存 ---
            if output_dir is not None:
                if pooling == "last_layer":
                    # shape: (batch, hidden_dim) — 不需要 reshape
                    batch_hidden = np.stack(all_hidden_list, axis=0)
                else:
                    batch_hidden = np.stack(all_hidden_list, axis=0)
                    batch_hidden = batch_hidden.reshape(-1, num_layers, hidden_dim)

                npz_path = output_dir / f"{desc}_hidden_states.npz"
                jsonl_path = output_dir / f"{desc}_outputs.jsonl"

                _append_hidden_states(npz_path, batch_hidden, all_indices[-batch_n:])
                _append_outputs(jsonl_path, batch_outputs, all_indices[-batch_n:])
                total_written += batch_n

                # 一致性检查：jsonl 行数应与全局已写入数一致
                if jsonl_path.exists():
                    jsonl_line_count = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
                    if jsonl_line_count != total_written:
                        logging.warning(
                            f"{desc} jsonl/npz 索引不一致 ({jsonl_line_count} 行 vs {total_written} 条)，"
                            "删除 npz 重新开始"
                        )
                        try:
                            npz_path.unlink()
                        except OSError:
                            pass

                # 清空本批次数据，释放显存
                all_hidden_list.clear()
                all_outputs.clear()
                all_indices.clear()
                del batch_hidden
                torch.cuda.empty_cache()

            pbar.update(batch_n)

    # 若未指定 output_dir（兼容旧调用），在最后一次性返回
    if output_dir is None:
        if all_hidden_list:
            all_hidden = np.stack(all_hidden_list, axis=0)
            if pooling == "last_layer":
                final_shape = (-1, hidden_dim)
            else:
                final_shape = (-1, num_layers, hidden_dim)
            all_hidden = all_hidden.reshape(*final_shape)
        else:
            all_hidden = np.array([]).reshape(0, num_layers or 32, hidden_dim or 0)
        return all_hidden, all_outputs, all_indices

    return np.array([]), [], []  # 流式模式下不需要返回值


def _append_hidden_states(npz_path: Path, new_hidden: np.ndarray, new_indices: List[int]):
    """追加写入 hidden_states 到 npz 文件，保留已有数据。"""
    existing_hidden = None
    existing_indices: List[int] = []

    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            raw_hs = None
            for key in ("hidden_states", "generation_hs", "train_hs"):
                val = data.get(key)
                if val is not None:
                    raw_hs = val
                    break
            if raw_hs is not None and hasattr(raw_hs, "size") and raw_hs.size > 0:
                existing_hidden = raw_hs
            ei = data.get("original_indices")
            if ei is not None:
                existing_indices = ei.tolist()
        except Exception:
            pass

    merged_hidden = np.concatenate([existing_hidden, new_hidden], axis=0) if existing_hidden is not None else new_hidden.astype(np.float32)
    merged_indices = existing_indices + new_indices

    np.savez_compressed(
        npz_path,
        hidden_states=merged_hidden.astype(np.float32),
        original_indices=np.array(merged_indices, dtype=np.int32),
    )


def _append_outputs(jsonl_path: Path, new_outputs: List[str], new_indices: List[int]):
    """追加写入输出到 jsonl 文件。"""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for idx, output in zip(new_indices, new_outputs):
            f.write(json.dumps({"original_index": idx, "generated_output": output}, ensure_ascii=False) + "\n")


def process_dataset(
    config: DatasetConfig,
    max_samples: Optional[int],
    skip: int,
    model,
    tokenizer,
    device: torch.device,
    output_dir: Path,
    max_length: int,
    batch_size: int,
    max_new_tokens: int,
    pooling: str = "last_hidden",
) -> bool:
    dataset_name = config.name
    log_prefix = f"[{dataset_name}]"

    file_path = PROJECT_ROOT / config.file_path
    if not file_path.exists():
        logging.error(f"{log_prefix} 文件不存在: {file_path}")
        return False

    raw_data = load_jsonl(file_path, max_lines=max_samples, skip=skip)

    if not raw_data:
        logging.warning(f"{log_prefix} 无数据")
        return False

    inputs = []
    for i, sample in enumerate(raw_data):
        text = extract_input_text(sample, config)
        if text:
            inputs.append((skip + i, text))
        else:
            logging.warning(
                f"{log_prefix} 样本 {skip+i} 缺少字段 '{config.input_field}'，跳过"
            )

    if not inputs:
        logging.warning(f"{log_prefix} 无有效样本")
        return False

    indices = [item[0] for item in inputs]
    texts = [item[1] for item in inputs]

    hidden_path = output_dir / f"{dataset_name}_hidden_states.npz"
    output_path = output_dir / f"{dataset_name}_outputs.jsonl"

    existing_hidden, existing_indices = load_existing_hidden_states(hidden_path)
    existing_output_indices = load_existing_output_indices(output_path)

    all_existing = set(existing_indices) | set(existing_output_indices)
    new_indices = [idx for idx in indices if idx not in all_existing]
    idx_to_text = {idx: text for idx, text in zip(indices, texts)}
    new_texts = [idx_to_text[idx] for idx in new_indices]

    if not new_indices:
        logging.info(f"{log_prefix} 全部已处理，跳过")
        return True

    start = time.time()
    extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=new_texts,
        indices=new_indices,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc=dataset_name,
        output_dir=output_dir,
        pooling=pooling,
    )
    elapsed = time.time() - start
    n_new = len(new_indices)
    logging.info(f"{log_prefix} 完成 {n_new} 条 | {elapsed:.1f}s ({elapsed/n_new:.2f}s/条)")

    size_mb = hidden_path.stat().st_size / 1024 / 1024
    print(f"{log_prefix} 隐藏态已保存: {hidden_path} ({size_mb:.1f}MB)")
    print(f"{log_prefix} 输出已追加: {n_new} 条")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SALAD 数据集生成阶段隐藏态提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=["base_set"],
    )
    parser.add_argument(
        "--max_samples", nargs="+", type=int, default=None,
    )
    parser.add_argument(
        "--skip", nargs="+", type=int, default=None,
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="每批并行生成的样本数，越大越快但显存占用越高（1~8）",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=16384,
        help="每个样本最大生成 token 数（0 表示不限制，直到模型自然停止）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/salad_extraction",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last_hidden",
        choices=["last_hidden", "last_layer", "last_input"],
        help=(
            "隐藏态池化方式："
            "last_hidden (默认) = 所有层在生成末尾的向量拼接;"
            "last_layer = 仅最后一层在生成末尾的向量;"
            "last_input = 所有层在输入序列末尾的向量拼接"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    config_info = {
        "timestamp": datetime.now().isoformat(),
        "datasets": args.datasets,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "pooling": args.pooling,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    try:
        from engine.models import ModelManager
        mm = ModelManager()
        tokenizer, model = mm.load_llm()
        device = next(model.parameters()).device
        logger = logging.getLogger(__name__)
        logger.info(f"LLM 加载成功, Device: {device}")
    except Exception as e:
        logging.error(f"LLM 加载失败: {e}")
        sys.exit(1)

    results = {}
    total_start = time.time()

    for i, dataset_name in enumerate(args.datasets):
        if dataset_name not in DATASET_CONFIGS:
            logging.warning(f"[{dataset_name}] 未知数据集，跳过")
            continue

        config = DATASET_CONFIGS[dataset_name]
        max_samples = (
            args.max_samples[i]
            if args.max_samples and i < len(args.max_samples)
            else config.max_total
        )
        skip = args.skip[i] if args.skip and i < len(args.skip) else 0

        success = process_dataset(
            config=config,
            max_samples=max_samples,
            skip=skip,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            pooling=args.pooling,
        )
        results[dataset_name] = "成功" if success else "失败"

    total_elapsed = time.time() - total_start

    for name, status in results.items():
        status_icon = "OK" if status else "FAIL"
        logging.info(f"  {status_icon} {name}: {status}")
    logging.info(f"总耗时: {total_elapsed:.1f}s | 输出: {output_dir}")


if __name__ == "__main__":
    main()
