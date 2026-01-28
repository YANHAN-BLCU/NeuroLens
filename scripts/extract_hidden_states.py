#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在服务器上提前从 base_evaluation.jsonl 提取隐藏态并保存，供后续探针训练直接加载使用。

使用与 train_linear_probes 相同的数据加载、划分与提取逻辑，确保缓存可直接用于训练。
提取一次后可多次用 --hidden_states_cache 训练探针，无需重复加载 LLM 与前向。

用法:
  # 提取（需 GPU、LLM）
  python scripts/extract_hidden_states.py \
    --data_file data/salad/raw/base_evaluation.jsonl \
    --max_samples 4000 \
    --output outputs/hidden_states_cache/base_evaluation_n4000_seed42_len512.npz

  # 之后训练探针（可不加载 LLM）
  python scripts/train_linear_probes.py \\
    --hidden_states_cache outputs/hidden_states_cache/base_evaluation_n4000_seed42_len512.npz \\
    --output_dir outputs/probes
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

# 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 train_linear_probes 的加载与划分（保持完全一致）
_spec = importlib.util.spec_from_file_location(
    "train_linear_probes",
    PROJECT_ROOT / "scripts" / "train_linear_probes.py",
)
_tlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tlp)
load_attack_enhanced = _tlp.load_attack_enhanced
split_data_optimized = _tlp.split_data_optimized
set_seed = _tlp.set_seed

from engine.models import ModelManager  # noqa: E402
from engine.probes.linear_probe import extract_hidden_states  # noqa: E402


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="从 base_evaluation 提取隐藏态并保存，供 train_linear_probes --hidden_states_cache 使用。"
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        default=Path("data/salad/raw/base_evaluation.jsonl"),
        help="base_evaluation.jsonl 路径",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数，默认全量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（需与之后 train_linear_probes 的 --seed 一致）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="分词最大长度",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="提取隐藏态时的批大小",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出 .npz 路径；默认: outputs/hidden_states_cache/{data_file.stem}_n{max_samples}_seed{seed}_len{max_length}.npz",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) 加载与划分（与 train_linear_probes 相同）
    print(f"[Data] Loading {args.data_file} ...")
    all_texts, all_labels = load_attack_enhanced(args.data_file, max_samples=args.max_samples)
    if len(all_texts) == 0:
        raise ValueError("未加载到有效样本，请检查数据路径与字段。")
    print(f"[Data] Loaded {len(all_texts)} total, Safe={sum(1 for l in all_labels if l==0)}, Toxic={sum(1 for l in all_labels if l==1)}")

    probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
        split_data_optimized(
            texts=all_texts,
            labels=all_labels,
            seed=args.seed,
            use_ratio_6_2_2_full_train=True,  # 使用 6:2:2 划分
        )

    # 打乱探针训练集（与 train_linear_probes 一致）
    combined = list(zip(probe_train_texts, probe_train_labels))
    random.Random(args.seed).shuffle(combined)
    probe_train_texts, probe_train_labels = [list(x) for x in zip(*combined)]

    print(f"[Final] 探针训练: {len(probe_train_texts)}, 探针验证: {len(probe_val_texts)}, 测试: {len(test_texts)}")

    # 2) 加载 LLM 并提取
    print("[Model] 正在加载 LLM（4-bit 量化）...")
    model_manager = ModelManager()
    tokenizer, model = model_manager.load_llm()
    device = next(model.parameters()).device
    print(f"[Model] Device: {device}")

    print("[Hidden] 提取探针训练集隐藏态...")
    train_hs_list = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=probe_train_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print("[Hidden] 提取探针验证集隐藏态...")
    val_hs_list = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=probe_val_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print("[Hidden] 提取测试集隐藏态...")
    test_hs_list = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    num_layers, hidden_dim = train_hs_list[0].shape
    print(f"[Hidden] num_layers={num_layers}, hidden_dim={hidden_dim}")

    # 3) 转为 float32 数组并保存
    train_hs = np.stack([h.astype(np.float32) for h in train_hs_list], axis=0)  # (N, L, D)
    val_hs = np.stack([h.astype(np.float32) for h in val_hs_list], axis=0)     # (M, L, D)
    test_hs = np.stack([h.astype(np.float32) for h in test_hs_list], axis=0)   # (K, L, D)
    train_labels = np.array(probe_train_labels, dtype=np.int64)
    val_labels = np.array(probe_val_labels, dtype=np.int64)
    test_labels = np.array(test_labels, dtype=np.int64)

    meta = {
        "data_file": str(args.data_file),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "split_method": "6:2:2_full_train",  # 记录划分方法
        "n_train": int(len(train_labels)),
        "n_val": int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "n_train_safe": int(sum(1 for l in probe_train_labels if l == 0)),
        "n_train_toxic": int(sum(1 for l in probe_train_labels if l == 1)),
        "n_val_safe": int(sum(1 for l in probe_val_labels if l == 0)),
        "n_val_toxic": int(sum(1 for l in probe_val_labels if l == 1)),
        "n_test_safe": int(sum(1 for l in test_labels if l == 0)),
        "n_test_toxic": int(sum(1 for l in test_labels if l == 1)),
    }

    # 默认输出路径
    if args.output is None:
        name = f"{args.data_file.stem}_n{args.max_samples or 'full'}_seed{args.seed}_len{args.max_length}.npz"
        out_dir = PROJECT_ROOT / "outputs" / "hidden_states_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / name

    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        train_hs=train_hs,
        train_labels=train_labels,
        val_hs=val_hs,
        val_labels=val_labels,
        test_hs=test_hs,
        test_labels=test_labels,
        num_layers=np.int32(num_layers),
        hidden_dim=np.int32(hidden_dim),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    print(f"[Save] 已保存: {args.output}")
    print(f"       训练: {train_hs.shape}, 验证: {val_hs.shape}, 测试: {test_hs.shape}, 约 {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("之后训练探针（可不加载 LLM）:")
    print(f"  python scripts/train_linear_probes.py --hidden_states_cache {args.output} --output_dir outputs/probes")


if __name__ == "__main__":
    main()
