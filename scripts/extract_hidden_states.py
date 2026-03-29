#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立提取隐藏态脚本

在 GPU 服务器上提前提取隐藏态并保存为 .npz 缓存，
之后在任意环境下运行 train_probes_balanced.py 时跳过 LLM 加载，直接训练。

与 train_probes_balanced.py 完全一致的：
  - 数据加载（load_and_balance_data：1:1 平衡）
  - 数据划分（split_622：6:2:2，每份保持 1:1）
  - 池化方式（last_token：最后一个非 padding token）
  - 缓存格式（.npz 键名、dtype、meta 结构）

用法：
  # 1) 提取隐藏态（需 GPU 和 LLM）
  python scripts/extract_hidden_states.py \\
      --data_file logs/base_evaluation.jsonl \\
      --output outputs/probes/hidden_states_cache.npz

  # 2) 训练探针（无需 LLM，自动检测缓存）
  python "scripts/train_probes_balanced.py" \\
      --output_dir outputs/probes

  # 或指定缓存路径
  python "scripts/train_probes_balanced.py" \\
      --hidden_states_cache outputs/probes/hidden_states_cache.npz
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================================
# 导入：extract_hidden_states & ModelManager（标准 import）
# ======================================================================
from engine.probes.linear_probe_balanced import extract_hidden_states
from engine.models import ModelManager

# 从 train_probes_balanced.py 复用数据函数
# ======================================================================
_train_script_paths = [
    PROJECT_ROOT / "scripts" / "train_probes_balanced.py",
]
_train_mod = None
for _p in _train_script_paths:
    if _p.exists():
        _spec = importlib.util.spec_from_file_location("_train_balanced", _p)
        _train_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_train_mod)
        break

if _train_mod is not None:
    load_and_balance_data = _train_mod.load_and_balance_data
    split_622 = _train_mod.split_622
    set_seed = _train_mod.set_seed
else:
    raise ImportError(
        "找不到 train_probes_balanced.py，"
        f"已搜索: {[str(p) for p in _train_script_paths]}"
    )


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="提取隐藏态缓存（适配 train_probes_balanced.py）"
    )
    parser.add_argument("--data_file", type=Path,
                        default=Path("logs/base_evaluation.jsonl"),
                        help="数据文件路径")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="输出 .npz 路径（默认: outputs/probes/hidden_states_cache.npz）")
    parser.add_argument("--max_toxic_samples", type=int, default=None,
                        help="最大有害样本数（1:1 平衡后安全样本数相同）")
    parser.add_argument("--max_length", type=int, default=512,
                        help="分词最大长度")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="提取时的批大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（需与训练脚本一致）")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- 数据加载 & 1:1 平衡 & 6:2:2 划分 ----
    print("=" * 60)
    print("隐藏态提取（适配 train_probes_balanced.py）")
    print(f"池化: last_token | 平衡: 1:1 | 划分: 6:2:2")
    print(f"数据: {args.data_file}")
    print("=" * 60)

    texts, labels = load_and_balance_data(
        args.data_file, max_toxic=args.max_toxic_samples, seed=args.seed
    )
    if not texts:
        raise ValueError("未加载到有效样本，请检查数据路径")

    tr_t, tr_l, va_t, va_l, te_t, te_l = split_622(texts, labels, seed=args.seed)

    print(f"\n[Split] 训练={len(tr_l)}(S={sum(1 for l in tr_l if l==0)} "
          f"T={sum(1 for l in tr_l if l==1)})  "
          f"验证={len(va_l)}  测试={len(te_l)}")

    # ---- 加载 LLM ----
    print("\n[Model] 加载 LLM...")
    tokenizer, model = ModelManager().load_llm()
    device = next(model.parameters()).device
    print(f"[Model] Device: {device}")

    # ---- 提取隐藏态（last_token） ----
    print(f"\n[Hidden] 提取隐藏态 (last_token)...")
    tr_hs = extract_hidden_states(model, tokenizer, tr_t, device,
                                   max_length=args.max_length,
                                   batch_size=args.batch_size,
                                   pooling_method="last_token", desc="训练集")
    num_layers, hidden_dim = tr_hs[0].shape

    va_hs = extract_hidden_states(model, tokenizer, va_t, device,
                                   max_length=args.max_length,
                                   batch_size=args.batch_size,
                                   pooling_method="last_token", desc="验证集")
    te_hs = extract_hidden_states(model, tokenizer, te_t, device,
                                   max_length=args.max_length,
                                   batch_size=args.batch_size,
                                   pooling_method="last_token", desc="测试集")

    # ---- 转为数组 ----
    to_np = lambda lst: np.stack(
        [h.cpu().numpy() if isinstance(h, torch.Tensor) else h for h in lst]
    )
    train_hs, val_hs, test_hs = to_np(tr_hs), to_np(va_hs), to_np(te_hs)
    train_labels, val_labels, test_labels = tr_l, va_l, te_l

    # ---- 保存（与 train_probes_balanced.py 缓存格式完全一致） ----
    if args.output is None:
        out_path = Path("outputs/probes/hidden_states_cache.npz")
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_tr_s = sum(1 for l in train_labels if l == 0)
    n_tr_t = sum(1 for l in train_labels if l == 1)
    n_va_s = sum(1 for l in val_labels if l == 0)
    n_va_t = sum(1 for l in val_labels if l == 1)
    n_te_s = sum(1 for l in test_labels if l == 0)
    n_te_t = sum(1 for l in test_labels if l == 1)

    cache_meta = json.dumps({
        "data_file": str(args.data_file),
        "pooling": "last_token",
        "max_length": args.max_length,
        "seed": args.seed,
        "balance": "1:1",
        "split": "6:2:2",
        "train": {"total": len(train_labels), "safe": n_tr_s, "toxic": n_tr_t},
        "val":   {"total": len(val_labels),   "safe": n_va_s, "toxic": n_va_t},
        "test":  {"total": len(test_labels),  "safe": n_te_s, "toxic": n_te_t},
        "num_layers": int(num_layers),
        "hidden_dim": int(hidden_dim),
    }, ensure_ascii=False)

    np.savez_compressed(
        out_path,
        train_hs=train_hs,
        val_hs=val_hs,
        test_hs=test_hs,
        train_labels=np.array(train_labels, dtype=np.int64),
        val_labels=np.array(val_labels, dtype=np.int64),
        test_labels=np.array(test_labels, dtype=np.int64),
        num_layers=np.int32(num_layers),
        hidden_dim=np.int32(hidden_dim),
        meta=np.array([cache_meta]),
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[Save] 已保存: {out_path}  ({size_mb:.1f} MB)")
    print(f"       train={train_hs.shape}  val={val_hs.shape}  test={test_hs.shape}")
    print(f"       train: S={n_tr_s} T={n_tr_t} | val: S={n_va_s} T={n_va_t} | test: S={n_te_s} T={n_te_t}")
    print(f"       layers={num_layers}  dim={hidden_dim}")
    print(f"\n后续训练（无需 LLM）:")
        print(f'  python "scripts/train_probes_balanced.py" --output_dir {out_path.parent}')


if __name__ == "__main__":
    main()
