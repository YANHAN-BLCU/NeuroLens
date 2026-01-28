#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：6:2:2 划分 + 训练集安全:有害=1.5:1，记录每层每轮的验证集 / 测试集准确率。

- 数据划分：use_ratio_6_2_2_full_train，train_safe_ratio_622_full=1.5
- 提取 train / val / test 隐藏态，训练时每轮计算 val_acc、test_acc
- 输出：CSV（layer, epoch, train_acc, val_acc, test_acc, train_loss, val_loss）、
  JSON 汇总、每层详情，便于后续改进分析。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# 工程根目录
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from engine.models import ModelManager
from engine.probes.linear_probe import (
    extract_hidden_states,
    save_probes,
    train_layer_probes,
)
from scripts.train_linear_probes import load_attack_enhanced, set_seed, split_data_optimized


def _write_epoch_log(
    results: Dict[int, Dict],
    log_dir: Path,
    run_id: str,
) -> None:
    """每层每轮：layer, epoch, train_acc, val_acc, test_acc, train_loss, val_loss → CSV + JSON."""
    rows: List[Dict[str, Any]] = []
    for layer_idx, layer_data in sorted(results.items()):
        th = layer_data.get("training_history") or {}
        epochs = th.get("epochs") or []
        train_acc = th.get("train_acc") or []
        val_acc = th.get("val_acc") or []
        test_acc = th.get("test_acc") or []
        train_loss = th.get("train_loss") or []
        val_loss = th.get("val_loss") or []
        n = len(epochs)
        for i in range(n):
            e = epochs[i] if i < len(epochs) else i + 1
            ta = train_acc[i] if i < len(train_acc) else None
            va = val_acc[i] if i < len(val_acc) else None
            tta = test_acc[i] if i < len(test_acc) else None
            tl = train_loss[i] if i < len(train_loss) else None
            vl = val_loss[i] if i < len(val_loss) else None
            rows.append({
                "layer": layer_idx,
                "epoch": e,
                "train_acc": ta,
                "val_acc": va,
                "test_acc": tta,
                "train_loss": tl,
                "val_loss": vl,
            })

    log_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "epoch", "train_acc", "val_acc", "test_acc", "train_loss", "val_loss"]
    csv_path = log_dir / f"train_test_1_5_1_epochs_{run_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fieldnames})
    print(f"[Log] 每层每轮日志 CSV: {csv_path}")

    json_path = log_dir / f"train_test_1_5_1_epochs_{run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[Log] 每层每轮日志 JSON: {json_path}")


def _write_summary_log(
    results: Dict[int, Dict],
    log_dir: Path,
    run_id: str,
    split_info: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """汇总：划分信息、超参、每层最终指标."""
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "split": split_info,
        "args": {
            "data_file": str(getattr(args, "data_file", "")),
            "max_samples": getattr(args, "max_samples", None),
            "lr": getattr(args, "lr", None),
            "num_epochs": getattr(args, "num_epochs", None),
            "probe_batch_size": getattr(args, "probe_batch_size", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "seed": getattr(args, "seed", None),
            "train_safe_ratio_622_full": 1.5,
        },
        "layers": {},
    }
    for layer_idx, layer_data in sorted(results.items()):
        m = (layer_data.get("metrics") or {}).copy()
        summary["layers"][str(layer_idx)] = {
            "val_acc": m.get("val_acc"),
            "train_acc": m.get("train_acc"),
            "min_required_acc": m.get("min_required_acc"),
            "meets_requirement": m.get("meets_requirement"),
        }

    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"train_test_1_5_1_summary_{run_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Log] 汇总: {out}")


def _write_per_layer_log(
    results: Dict[int, Dict],
    log_dir: Path,
    run_id: str,
) -> None:
    """每层详情（含 training_history），便于后续计算与改进."""
    per_layer: Dict[str, Any] = {"run_id": run_id, "layers": {}}
    for layer_idx, layer_data in sorted(results.items()):
        m = layer_data.get("metrics") or {}
        th = layer_data.get("training_history") or {}
        per_layer["layers"][str(layer_idx)] = {
            "min_required_acc": m.get("min_required_acc"),
            "meets_requirement": m.get("meets_requirement"),
            "val_acc": m.get("val_acc"),
            "train_acc": m.get("train_acc"),
            "training_history": th,
        }

    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"train_test_1_5_1_per_layer_{run_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(per_layer, f, indent=2, ensure_ascii=False)
    print(f"[Log] 每层详情（含每轮 val/test acc）: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="1.5:1 测试脚本，记录每层每轮 val/test 准确率")
    ap.add_argument("--data_file", type=Path, default=Path("data/salad/raw/base_evaluation.jsonl"), help="数据路径")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/probes_test_1_5_1"), help="输出目录")
    ap.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    ap.add_argument("--lr", type=float, default=3e-3, help="学习率")
    ap.add_argument("--num_epochs", type=int, default=80, help="训练轮数")
    ap.add_argument("--probe_batch_size", type=int, default=64, help="探针 batch 大小")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--max_length", type=int, default=512, help="max length")
    ap.add_argument("--batch_size", type=int, default=8, help="提取隐藏态时的 batch 大小")
    args = ap.parse_args()

    set_seed(args.seed)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[Data] 加载数据...")
    all_texts, all_labels = load_attack_enhanced(args.data_file, max_samples=args.max_samples)
    if not all_texts:
        raise ValueError("未加载到有效样本")
    print(f"[Data] 共 {len(all_texts)} 条，安全={sum(1 for l in all_labels if l==0)}，有害={sum(1 for l in all_labels if l==1)}")

    print("\n[Split] 6:2:2 剩余全作训练集，训练集 安全:有害=1.5:1")
    probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
        split_data_optimized(
            texts=all_texts,
            labels=all_labels,
            test_ratio=0.2,
            val_ratio=0.2,
            balance_train=True,
            train_safe_ratio=1.0,
            balance_val=False,
            balance_test=False,
            min_test_toxic=50,
            min_val_toxic=50,
            seed=args.seed,
            use_doc_ratios=False,
            use_ratio_6_2_2=False,
            probe_val_ratio_in_train=0.3,
            use_ratio_6_2_2_full_train=True,
            train_safe_ratio_622_full=1.5,
        )

    n_train = len(probe_train_texts)
    n_val = len(probe_val_texts)
    n_test = len(test_texts)
    split_info = {
        "train": n_train,
        "train_safe": int(sum(1 for l in probe_train_labels if l == 0)),
        "train_toxic": int(sum(1 for l in probe_train_labels if l == 1)),
        "val": n_val,
        "val_safe": int(sum(1 for l in probe_val_labels if l == 0)),
        "val_toxic": int(sum(1 for l in probe_val_labels if l == 1)),
        "test": n_test,
        "test_safe": int(sum(1 for l in test_labels if l == 0)),
        "test_toxic": int(sum(1 for l in test_labels if l == 1)),
        "train_safe_ratio": 1.5,
    }
    print(f"[Split] 训练 {n_train}，验证 {n_val}，测试 {n_test}\n")

    combined = list(zip(probe_train_texts, probe_train_labels))
    random.Random(args.seed).shuffle(combined)
    probe_train_texts, probe_train_labels = [x[0] for x in combined], [x[1] for x in combined]

    print("[Model] 加载 LLM...")
    model_manager = ModelManager()
    tokenizer, model = model_manager.load_llm()
    device = next(model.parameters()).device
    print(f"[Model] Device: {device}\n")

    print("[Hidden] 提取训练集隐藏态...")
    hidden_states = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=probe_train_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    num_layers, hidden_dim = hidden_states[0].shape
    print(f"[Hidden] 训练 {len(hidden_states)}，num_layers={num_layers}，hidden_dim={hidden_dim}")

    print("[Hidden] 提取验证集隐藏态...")
    val_hidden_states = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=probe_val_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print("[Hidden] 提取测试集隐藏态...")
    test_hidden_states = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print()

    train_indices = list(range(len(probe_train_labels)))
    results = train_layer_probes(
        hidden_states=hidden_states,
        labels=probe_train_labels,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        train_indices=train_indices,
        val_indices=[],
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.probe_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_hidden_states=val_hidden_states,
        val_labels=probe_val_labels,
        test_hidden_states=test_hidden_states,
        test_labels=test_labels,
        ensure_accuracy_requirements=True,
        use_class_weight=True,
    )

    out_probes = Path(args.output_dir) / "probes" / "llama-3-8b"
    save_probes(results=results, output_dir=Path(args.output_dir), model_id="llama-3-8b", filter_threshold=0.75)
    print(f"[Save] 探针已保存至 {out_probes}\n")

    _write_epoch_log(results, log_dir, run_id)
    _write_summary_log(results, log_dir, run_id, split_info, args)
    _write_per_layer_log(results, log_dir, run_id)
    print("\n[Done] 测试脚本跑完，日志已写入。")


if __name__ == "__main__":
    main()
