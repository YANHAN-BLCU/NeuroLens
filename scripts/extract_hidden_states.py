#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.pop("NPU_VISIBLE_DEVICES", None)
os.environ.pop("ASCEND_VISIBLE_DEVICES", None)

import argparse
import io
import json
import logging
import random
import signal
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


class RealtimeFileHandler(logging.Handler):
    """每次 emit 立即刷盘的文件 handler，保证日志实时持久化。"""
    def __init__(self, path: Path):
        super().__init__()
        self._lock = threading.Lock()
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record):
        try:
            msg = self.format(record) + "\n"
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(msg)
                    f.flush()
        except Exception:
            self.handleError(record)

# 抑制 transformers 警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 控制每次 generation 并行处理的样本数，影响速度和显存占用
# 设为 0 表示不限制输出长度（实际受限于模型最大上下文和内存）
MAX_NEW_TOKENS = 2048

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


class InMemoryHiddenStateStore:
    """
    分批写盘型隐藏态存储。

    每次 flush:
      - overwrite=True  : 文件从空开始写（第一段写入）
      - overwrite=False  : 先读已有 npz/jsonl，再追加新数据

    npz 的 original_indices 始终存储真实 original_index（从 jsonl_batch 解析），
    jsonl 和 npz 条数完全对齐，写完立即验证。
    """

    def __init__(self, npz_path: Path, jsonl_path: Path):
        self.npz_path = npz_path
        self.jsonl_path = jsonl_path

        # 每个元素是 (flattened_hs, original_index_int)
        self.paired_list: List[Tuple[np.ndarray, int]] = []
        self.jsonl_batch: List[str] = []

    def append(
        self,
        hidden_batch: np.ndarray,      # (batch, num_layers, hidden_dim) 或 (batch, hidden_dim)
        outputs: List[str],
        indices: List[int],
    ):
        """
        将一批数据追加到内存缓冲区（不去重，调用方保证不重复）。
        paired_list:  存 (flattened_hs, original_index)
        jsonl_batch: 存 JSON 行（与 paired_list 一一对应）
        """
        if hidden_batch is None or hidden_batch.shape[0] == 0:
            return

        for i in range(hidden_batch.shape[0]):
            # 保持 3D 形状 (num_layers, hidden_dim)，不 flatten
            hs = hidden_batch[i]
            self.paired_list.append((hs, indices[i]))

        for idx, out in zip(indices, outputs):
            self.jsonl_batch.append(
                json.dumps({"original_index": idx, "generated_output": out}, ensure_ascii=False) + "\n"
            )

    def flush(self, overwrite: bool = False):
        """
        直接写盘（直接写入 .hs.npy / .idx.npy / .jsonl，追加时读旧文件合并）。
        overwrite=True  : 覆盖模式（第一段写文件时使用）
        overwrite=False : 追加模式（后续段追加到已有文件）
        """
        if not self.paired_list:
            logging.info("无新数据需要写入")
            return

        n_new = len(self.paired_list)
        hidden_arrays = [hs for hs, _ in self.paired_list]
        real_indices = [idx for _, idx in self.paired_list]

        # new_hidden shape: (n_new, num_layers, hidden_dim) 或 (n_new, hidden_dim)
        # —— 形状由 extract_hidden_states 保证，这里不再 reshape
        new_hidden = np.stack(hidden_arrays, axis=0).astype(np.float32)

        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        npz_hs_file = self.npz_path.with_suffix(".hs.npy")
        npz_idx_file = self.npz_path.with_suffix(".idx.npy")

        # 读取旧数据（追加模式）或直接使用新数据
        if overwrite or (not npz_hs_file.exists() and not npz_idx_file.exists()):
            npz_hidden = new_hidden
            merged_indices = real_indices
        else:
            existing_hs, existing_indices = None, []
            try:
                if npz_hs_file.exists() and npz_idx_file.exists():
                    existing_hs = np.load(npz_hs_file)
                    existing_indices = np.load(npz_idx_file).tolist()
                elif self.npz_path.exists():
                    data = np.load(self.npz_path, allow_pickle=True)
                    for key in ("hidden_states", "generation_hs", "train_hs"):
                        val = data.get(key)
                        if val is not None and val.size > 0:
                            existing_hs = val
                            break
                    ei = data.get("original_indices")
                    if ei is not None:
                        existing_indices = ei.tolist()
            except Exception:
                pass

            npz_hidden = (
                np.concatenate([existing_hs, new_hidden], axis=0)
                if existing_hs is not None
                else new_hidden
            )
            merged_indices = existing_indices + real_indices

        # --- 直接写 .npy 文件（无临时文件，无 rename） ---
        np.save(npz_hs_file, npz_hidden)
        np.save(npz_idx_file, np.array(merged_indices, dtype=np.int32))

        # --- 写 jsonl 文件 ---
        if overwrite or not self.jsonl_path.exists():
            jsonl_lines = self.jsonl_batch
        else:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                existing_lines = f.readlines()
            jsonl_lines = existing_lines + self.jsonl_batch

        if jsonl_lines and not jsonl_lines[-1].endswith("\n"):
            jsonl_lines = list(jsonl_lines)
            jsonl_lines[-1] += "\n"

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.writelines(jsonl_lines)

        mode_str = "覆盖" if overwrite else "追加"
        logging.info(
            f"Flush [{mode_str}]: {n_new} 条 → {npz_hs_file.name} + {npz_idx_file.name}, "
            f"{n_new} 条 → {self.jsonl_path.name}"
        )

        self.paired_list.clear()
        self.jsonl_batch.clear()


def extract_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    indices: List[int],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    pooling: str = "last_hidden",
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    批量生成并提取所有样本的隐藏态，显存优化版。

    所有批次完成后一次性返回，write 由调用方统一管理。
    显存优化：每批次后及时 del 无用 tensor + torch.cuda.empty_cache()。

    pooling 可选值：
      - "last_hidden"  (默认)：所有层(1~N)在生成序列末尾 last_pos 位置的向量拼接

    返回：(hidden_states, outputs, indices)
      - hidden_states: np.ndarray shape (N, num_layers, hidden_dim)
      - outputs: List[str]，生成文本
      - indices: List[int]，对应原数据集的索引
    """
    num_samples = len(texts)
    all_hidden_list: List[np.ndarray] = []
    all_outputs: List[str] = []
    all_indices: List[int] = []

    num_layers = None
    hidden_dim = None

    with tqdm(total=num_samples, desc="提取隐藏态", unit="条", dynamic_ncols=True, leave=False, mininterval=0) as pbar:
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
                gen_seq = seq[inp_len:].tolist()
                gen_start_offset = 0
                for offset in range(len(gen_seq) - len(assistant_header_ids) + 1):
                    if gen_seq[offset:offset + len(assistant_header_ids)] == assistant_header_ids:
                        gen_start_offset = offset + len(assistant_header_ids)
                        break
                # 如果没找到 assistant header（兼容旧行为），使用 inp_len
                gen_start_pos = inp_len + gen_start_offset

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

                # 截断到 hidden_states 实际长度（修复 Llama 3 BOS token 偏移问题）
                target_pos = min(last_pos, hs_len - 1)

                # 所有层(1~num_layers)拼接
                sample_hs = np.stack([
                    hs_per_layer[layer_idx][b, target_pos, :].float().cpu().numpy()
                    for layer_idx in range(1, num_layers + 1)
                ], axis=0)

                all_hidden_list.append(sample_hs)

            all_outputs.extend(batch_outputs)
            all_indices.extend(batch_indices)

            # 显存回收（只清理 tensor 相关引用，numpy 数据留在 all_hidden_list）
            del inputs, gen_outputs, generated_ids, raw_hs, hs_per_layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(batch_n)

    if all_hidden_list:
        all_hidden = np.stack(all_hidden_list, axis=0)
        # all_hidden_list 每个元素 shape = (num_layers, hidden_dim)，stack 后已是 (N, num_layers, hidden_dim)
    else:
        all_hidden = np.array([]).reshape(0, num_layers or 32, hidden_dim or 0)

    return all_hidden, all_outputs, all_indices


def process_dataset(
    config: DatasetConfig,
    max_samples: Optional[int],
    skip: int,
    start: Optional[int],
    end: Optional[int],
    model,
    tokenizer,
    device: torch.device,
    output_dir: Path,
    max_length: int,
    batch_size: int,
    max_new_tokens: int,
    pooling: str = "last_hidden",
    flush_interval: int = 100,
) -> bool:
    """
    每次运行以分段方式处理样本范围 [start, end]（不含 end，为 Python 惯例）。

    每段保存为:
      {dataset}_hidden_states_{start}_{end-1}.hs.npy   (隐藏态)
      {dataset}_hidden_states_{start}_{end-1}.idx.npy  (原始索引)
      {dataset}_outputs_{start}_{end-1}.jsonl           (生成文本)

    overwrite=True（覆盖写）。
    如果同名文件已存在，自动重命名为 _backup_{timestamp}。
    """
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

    # 手动指定范围过滤（start/end 是数据集原始行号）
    if start is not None or end is not None:
        inputs = [(i, t) for i, t in inputs
                  if (start is None or i >= start) and (end is None or i < end)]

    if not inputs:
        logging.warning(f"{log_prefix} 范围内无有效样本")
        return False

    indices = [item[0] for item in inputs]
    texts = [item[1] for item in inputs]
    n_total = len(indices)

    # 范围后缀：使用数据集中真实的索引值
    range_start = indices[0]
    range_end = indices[-1]
    range_str = f"_{range_start}_{range_end}"

    hidden_path = output_dir / f"{dataset_name}_hidden_states{range_str}.npz"
    output_path = output_dir / f"{dataset_name}_outputs{range_str}.jsonl"
    hidden_hs_file = hidden_path.with_suffix(".hs.npy")
    hidden_idx_file = hidden_path.with_suffix(".idx.npy")

    # 检查同名文件是否已存在，存在则备份
    for _path in (hidden_hs_file, hidden_idx_file, output_path):
        if _path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = _path.parent / f"{_path.stem}_backup_{ts}{_path.suffix}"
            _path.rename(backup_path)
            logging.warning(f"目标文件已存在，自动重命名为: {backup_path.name}")

    logging.info(
        f"{log_prefix} 范围 {range_start}~{range_end}，共 {n_total} 条，"
        f"batch_size={batch_size}"
    )

    # 分批推理 + 分批写盘
    seg_hidden_list: List[np.ndarray] = []
    seg_outputs: List[str] = []
    seg_idx_list: List[int] = []

    store = InMemoryHiddenStateStore(hidden_path, output_path)
    is_first_flush = True  # 标记是否为首次写盘

    start_time = time.time()
    n_total_samples = n_total  # 避免与 pbar.total 冲突

    with tqdm(total=n_total_samples, desc=f"{dataset_name}[{range_start}_{range_end}]", unit="条", dynamic_ncols=True) as pbar:
        for batch_off in range(0, n_total_samples, batch_size):
            batch_end = min(batch_off + batch_size, n_total_samples)
            batch_texts = texts[batch_off:batch_end]
            batch_indices = indices[batch_off:batch_end]

            bh, bo, bi = extract_hidden_states(
                model=model,
                tokenizer=tokenizer,
                texts=batch_texts,
                indices=batch_indices,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                pooling=pooling,
            )

            for i in range(bh.shape[0]):
                seg_hidden_list.append(bh[i])
            seg_outputs.extend(bo)
            seg_idx_list.extend(bi)
            pbar.update(len(bi))

            # 实时写盘进度到日志（方便监控日志文件）
            elapsed_batch = time.time() - start_time
            n_done = pbar.n
            n_total_cur = pbar.total
            avg_ms = elapsed_batch / n_done * 1000 if n_done > 0 else 0
            logging.info(
                f"[{log_prefix}] 进度 {n_done}/{n_total_cur} "
                f"({n_done/n_total_cur*100:.1f}%) | batch {len(bi)} 条 | "
                f"累计 {elapsed_batch:.0f}s | 均 {avg_ms:.0f}ms/条"
            )

            # 分批写盘：每 flush_interval 条写一次，降低崩溃丢失风险
            if len(seg_outputs) >= flush_interval:
                seg_hidden_arr = np.stack(seg_hidden_list, axis=0)
                store.append(seg_hidden_arr, seg_outputs, seg_idx_list)
                store.flush(overwrite=is_first_flush)
                is_first_flush = False
                seg_hidden_list.clear()
                seg_outputs.clear()
                seg_idx_list.clear()

        # 剩余数据写盘
        if seg_outputs:
            seg_hidden_arr = np.stack(seg_hidden_list, axis=0)
            store.append(seg_hidden_arr, seg_outputs, seg_idx_list)
            store.flush(overwrite=is_first_flush)

    elapsed = time.time() - start_time

    for _path in (hidden_hs_file, hidden_idx_file, output_path):
        if _path.exists():
            size_mb = _path.stat().st_size / 1024 / 1024
            logging.info(f"  已保存: {_path.name} ({size_mb:.1f}MB)")

    logging.info(
        f"{log_prefix} 完成 {n_total_samples} 条 ({range_start}~{range_end}) "
        f"| {elapsed:.1f}s ({elapsed / n_total_samples:.2f}s/条)"
    )
    return True


def daemonize(stdout_path: Optional[Path] = None, stderr_path: Optional[Path] = None):
    """
    Fork 当前进程为守护进程：父进程退出，子进程在后台运行。
    SIGHUP 信号会被忽略，SSH 断开连接不影响子进程。
    stdout/stderr 重定向到指定文件，避免输出丢失。
    """
    # 第一次 fork：脱离父进程
    try:
        pid = os.fork()
        if pid > 0:
            # 父进程：打印子进程 PID 后退出
            print(f"[daemon] 守护进程已启动，子进程 PID={pid}", flush=True)
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"[daemon] 第一次 fork 失败: {e}\n")
        sys.exit(1)

    # 子进程：脱离控制终端，进入新会话
    os.setsid()

    # 第二次 fork（可选，但推荐）：防止子进程意外获取控制终端
    try:
        pid = os.fork()
        if pid > 0:
            print(f"[daemon] 孙子进程 PID={pid}，子进程退出", flush=True)
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"[daemon] 第二次 fork 失败: {e}\n")
        sys.exit(1)

    # 重定向标准文件描述符
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (sys.stdin, sys.stdout, sys.stderr):
        try:
            fd.flush()
        except Exception:
            pass

    if stdout_path:
        sys.stdout = open(stdout_path, "a", buffering=1, encoding="utf-8")
    else:
        os.dup2(devnull, sys.stdout.fileno())
    if stderr_path:
        sys.stderr = open(stderr_path, "a", buffering=1, encoding="utf-8")
    else:
        os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)

    # 忽略 SIGHUP（SSH 断开时发送的信号）
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


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
        "--start", nargs="+", type=int, default=None,
        help="样本范围起点（对应数据集的行号）",
    )
    parser.add_argument(
        "--end", nargs="+", type=int, default=None,
        help="样本范围终点（对应数据集的行号，含）",
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
        help=(
            "隐藏态池化方式："
            "last_hidden (默认) = 所有层在生成序列末尾的向量拼接"
        ),
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="以后台守护进程模式运行，SSH 断开后继续执行",
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
    stdout_file = output_dir / "daemon_stdout.log"
    stderr_file = output_dir / "daemon_stderr.log"

    if hasattr(args, "daemon") and args.daemon:
        daemonize(stdout_path=stdout_file, stderr_path=stderr_file)

    # 清空旧日志（每次运行重新开始）
    with open(log_file, "w", encoding="utf-8"):
        pass

    # stdout 实时刷新，保证 tqdm 等输出立即可见
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = RealtimeFileHandler(log_file)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root_logger.addHandler(stream_handler)

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
        start = args.start[i] if args.start and i < len(args.start) else None
        end = args.end[i] if args.end and i < len(args.end) else None

        success = process_dataset(
            config=config,
            max_samples=max_samples,
            skip=skip,
            start=start,
            end=end,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            pooling=args.pooling,
            flush_interval=100,
        )
        results[dataset_name] = "成功" if success else "失败"

    total_elapsed = time.time() - total_start

    for name, status in results.items():
        status_icon = "OK" if status else "FAIL"
        logging.info(f"  {status_icon} {name}: {status}")
    logging.info(f"总耗时: {total_elapsed:.1f}s | 输出: {output_dir}")


if __name__ == "__main__":
    main()
