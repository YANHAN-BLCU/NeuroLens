#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1:1 平衡训练线性探针

公式：P(toxic | h) = softmax(w^T * h + b)
隐藏态：每层取最后一个非 padding token（last_token）
预处理：StandardScaler 标准化（零均值、单位方差）
数据划分：6:2:2（训练60% / 验证20% / 测试20%），每份保持 1:1 安全:有害

超参数（默认值）：
  学习率          1e-3
  权重衰减        0.01
  Dropout         0.1
  最大训练轮数    80
  批大小          32
  优化器          AdamW
  调度器          ReduceLROnPlateau(patience=8, factor=0.5)
  早停            达到目标准确率后立即退出，否则训练满全部轮数

各层目标（验证集总体准确率）：
  浅层 (0-5)    ≥ 76%
  中层 (6-14)   ≥ 85%
  深层 (15-27)  ≥ 90%
  峰值层 (28)   ≥ 93%
  最深层 (29+)  ≥ 90%

输出目录结构（直接在 output_dir 下）：
  {output_dir}/                        默认 outputs/probes/
  ├── hidden_states_cache.npz          隐藏态缓存（提取后自动保存）
  ├── config.json                      训练配置 & 超参数
  ├── summary.json                     各层达标汇总
  ├── training_log.json                所有层的训练日志（每层metrics+每轮曲线）
  └── layer_{i}/
      ├── probe.pt                     线性探针模型权重
      ├── preprocessor.pkl             StandardScaler（推理用）
      ├── metrics.json                 该层最终指标
      └── training_history.json        该层每轮训练指标

注：验证报告(validation_report.json)和毒性向量(toxic_vectors.npz)
    由独立的后处理脚本从上述产物生成，训练脚本不直接输出。

隐藏态缓存（提取后自动保存，后续训练可直接加载跳过LLM）：
  {output_dir}/hidden_states_cache.npz
  内容：
    train_hs       (N_train, num_layers, hidden_dim)  训练集各层最终状态
    val_hs         (N_val,   num_layers, hidden_dim)  验证集各层最终状态
    test_hs        (N_test,  num_layers, hidden_dim)  测试集各层最终状态
    train_labels   (N_train,)                         训练集标签 0=安全 1=有害
    val_labels     (N_val,)                           验证集标签
    test_labels    (N_test,)                          测试集标签
    num_layers     int                                层数
    hidden_dim     int                                隐藏维度
    meta           json字符串                          数据来源 & 统计信息
"""

import argparse
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent))

from engine.models import ModelManager

# 从 engine/probes/linear_probe_balanced.py 导入模型 & 工具
from engine.probes.linear_probe_balanced import (
    LinearProbe, get_layer_target, extract_hidden_states,
)

# ======================================================================
# 评估
# ======================================================================
def compute_acc(preds, targets):
    """返回 (overall_acc, safe_acc, toxic_acc)"""
    preds, targets = np.array(preds), np.array(targets)
    safe_mask, toxic_mask = targets == 0, targets == 1
    safe_acc = (preds[safe_mask] == 0).mean() if safe_mask.sum() > 0 else 0.0
    toxic_acc = (preds[toxic_mask] == 1).mean() if toxic_mask.sum() > 0 else 0.0
    overall = (preds == targets).mean() if len(targets) > 0 else 0.0
    return overall, safe_acc, toxic_acc


@torch.no_grad()
def evaluate_full(probe, data_x, data_y, device, criterion, batch_size=256):
    """
    完整评估：返回 dict 包含 loss / acc / balanced_acc / probs（用于 AUC）

    返回:
        {
            "loss":         float,   # 平均 CrossEntropyLoss
            "overall_acc":  float,   # 总体准确率
            "safe_acc":     float,   # 安全类准确率
            "toxic_acc":    float,   # 有害类准确率
            "balanced_acc": float,   # (safe_acc + toxic_acc) / 2
            "probs":        np.ndarray,  # (N, 2) softmax 概率
            "preds":        np.ndarray,  # (N,)   预测类别
        }
    """
    probe.eval()
    all_preds, all_probs = [], []
    total_loss, n_batch = 0.0, 0

    for i in range(0, len(data_x), batch_size):
        bx = torch.tensor(data_x[i:i+batch_size], dtype=torch.float32, device=device)
        by = torch.tensor(data_y[i:i+batch_size], dtype=torch.long, device=device)
        logits = probe(bx)
        total_loss += criterion(logits, by).item()
        n_batch += 1
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(np.argmax(probs, axis=-1))

    all_probs = np.concatenate(all_probs, axis=0)
    overall, safe_acc, toxic_acc = compute_acc(all_preds, data_y)
    balanced = (safe_acc + toxic_acc) / 2.0

    return {
        "loss":         total_loss / max(n_batch, 1),
        "overall_acc":  overall,
        "safe_acc":     safe_acc,
        "toxic_acc":    toxic_acc,
        "balanced_acc": balanced,
        "probs":        all_probs,
        "preds":        np.array(all_preds),
    }


def _compute_auc(targets, probs_2d):
    """计算 ROC-AUC 和 PR-AUC，异常时返回 (0, 0)"""
    try:
        targets = np.array(targets)
        if len(set(targets)) < 2:
            return 0.0, 0.0
        toxic_probs = probs_2d[:, 1]
        roc = roc_auc_score(targets, toxic_probs)
        prec, rec, _ = precision_recall_curve(targets, toxic_probs)
        pr = sk_auc(rec, prec)
        return float(roc), float(pr)
    except Exception:
        return 0.0, 0.0


# ======================================================================
# 单层训练
# ======================================================================
def train_one_layer(
    layer_idx: int,
    train_x: np.ndarray, train_y: np.ndarray,
    val_x: np.ndarray, val_y: np.ndarray,
    test_x: Optional[np.ndarray], test_y: Optional[np.ndarray],
    num_layers: int, device: torch.device,
    max_epochs: int = 80, batch_size: int = 32,
    lr: float = 1e-3, weight_decay: float = 0.01,
    dropout: float = 0.1, seed: int = 42,
) -> Dict:
    target_acc, layer_type = get_layer_target(layer_idx, num_layers)

    # 标准化
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    if test_x is not None:
        test_x = scaler.transform(test_x)
    input_dim = train_x.shape[1]

    torch.manual_seed(seed + layer_idx)
    probe = LinearProbe(input_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )

    loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                       torch.tensor(train_y, dtype=torch.long)),
        batch_size=batch_size, shuffle=True,
    )

    best_val_acc, best_epoch, best_state = 0.0, 0, None
    best_val_roc_auc, best_val_pr_auc = 0.0, 0.0
    target_reached, total_trained = False, 0

    history = {k: [] for k in [
        "epoch", "lr",
        # ---- loss ----
        "train_loss", "val_loss",
        # ---- 总体准确率 ----
        "train_acc", "val_acc", "test_acc",
        # ---- 安全类准确率 ----
        "train_safe", "val_safe", "test_safe",
        # ---- 有害类准确率 ----
        "train_toxic", "val_toxic", "test_toxic",
        # ---- 平衡准确率 (safe+toxic)/2 ----
        "train_balanced_acc", "val_balanced_acc", "test_balanced_acc",
        # ---- AUC（仅验证集）----
        "val_roc_auc", "val_pr_auc",
    ]}

    pbar = tqdm(range(1, max_epochs + 1), desc=f"Layer {layer_idx:2d}",
                leave=False, ncols=90)
    for epoch in pbar:
        # ---- 训练阶段 ----
        probe.train()
        epoch_loss, n_batch = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        # ---- 完整评估（含 loss / balanced_acc / probs）----
        tr = evaluate_full(probe, train_x, train_y, device, criterion)
        vl = evaluate_full(probe, val_x, val_y, device, criterion)
        if test_x is not None:
            te = evaluate_full(probe, test_x, test_y, device, criterion)
        else:
            te = None

        # 验证集 ROC-AUC / PR-AUC
        val_roc, val_pr = _compute_auc(val_y, vl["probs"])

        scheduler.step(vl["overall_acc"])   # 用验证集总体准确率调度
        total_trained = epoch

        # ---- 记录历史 ----
        history["epoch"].append(epoch)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        # loss
        history["train_loss"].append(epoch_loss / max(n_batch, 1))
        history["val_loss"].append(vl["loss"])
        # 总体准确率
        history["train_acc"].append(tr["overall_acc"])
        history["val_acc"].append(vl["overall_acc"])
        history["test_acc"].append(te["overall_acc"] if te else None)
        # 安全类准确率
        history["train_safe"].append(tr["safe_acc"])
        history["val_safe"].append(vl["safe_acc"])
        history["test_safe"].append(te["safe_acc"] if te else None)
        # 有害类准确率
        history["train_toxic"].append(tr["toxic_acc"])
        history["val_toxic"].append(vl["toxic_acc"])
        history["test_toxic"].append(te["toxic_acc"] if te else None)
        # 平衡准确率
        history["train_balanced_acc"].append(tr["balanced_acc"])
        history["val_balanced_acc"].append(vl["balanced_acc"])
        history["test_balanced_acc"].append(te["balanced_acc"] if te else None)
        # AUC
        history["val_roc_auc"].append(val_roc)
        history["val_pr_auc"].append(val_pr)

        # 更新进度条
        pbar.set_postfix(val=f"{vl['overall_acc']:.2%}", best=f"{best_val_acc:.2%}")

        # 更新最佳（以验证集总体准确率为准）
        if vl["overall_acc"] > best_val_acc:
            best_val_acc, best_epoch = vl["overall_acc"], epoch
            best_val_roc_auc, best_val_pr_auc = val_roc, val_pr
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            pbar.set_postfix(val=f"{vl['overall_acc']:.2%}", best=f"{best_val_acc:.2%}")
            if vl["overall_acc"] >= target_acc:
                target_reached = True
                pbar.close()
                break  # 达标即退出
    else:
        pbar.close()

    # 加载最佳模型 & 最终评估
    if best_state:
        probe.load_state_dict(best_state)
    tr = evaluate_full(probe, train_x, train_y, device, criterion)
    vl = evaluate_full(probe, val_x, val_y, device, criterion)
    te = evaluate_full(probe, test_x, test_y, device, criterion) if test_x is not None else None
    final_roc, final_pr = _compute_auc(val_y, vl["probs"])

    # 打印
    mark = "✓" if target_reached else "✗"
    print(f"\n{'='*80}")
    print(f"[Layer {layer_idx:2d}] {mark} {layer_type}  "
          f"最佳Epoch {best_epoch}/{total_trained}  dim={input_dim}")
    print(f"  {'':8s} {'总体':>6s} {'balanced':>8s} {'安全':>6s} {'有害':>6s}")
    print(f"  验证集 {vl['overall_acc']:>6.2%} {vl['balanced_acc']:>8.2%} "
          f"{vl['safe_acc']:>6.2%} {vl['toxic_acc']:>6.2%}  (≥{target_acc:.0%})")
    if te is not None:
        print(f"  测试集 {te['overall_acc']:>6.2%} {te['balanced_acc']:>8.2%} "
              f"{te['safe_acc']:>6.2%} {te['toxic_acc']:>6.2%}")
    print(f"  训练集 {tr['overall_acc']:>6.2%} {tr['balanced_acc']:>8.2%} "
          f"{tr['safe_acc']:>6.2%} {tr['toxic_acc']:>6.2%}")
    print(f"  AUC    ROC={final_roc:.4f}  PR={final_pr:.4f}")
    print(f"{'='*80}")

    return {
        "model": probe,
        "scaler": scaler,
        "metrics": {
            # 训练集
            "train_acc": tr["overall_acc"],
            "train_safe_acc": tr["safe_acc"],
            "train_toxic_acc": tr["toxic_acc"],
            "train_balanced_acc": tr["balanced_acc"],
            # 验证集
            "val_acc": best_val_acc,
            "val_safe_acc": vl["safe_acc"],
            "val_toxic_acc": vl["toxic_acc"],
            "val_balanced_acc": vl["balanced_acc"],
            "val_roc_auc": final_roc,
            "val_pr_auc": final_pr,
            # 测试集
            "test_acc": te["overall_acc"] if te else None,
            "test_safe_acc": te["safe_acc"] if te else None,
            "test_toxic_acc": te["toxic_acc"] if te else None,
            "test_balanced_acc": te["balanced_acc"] if te else None,
            # 训练过程
            "best_epoch": best_epoch,
            "total_epochs": total_trained,
            "target_acc": target_acc,
            "target_reached": target_reached,
        },
        "training_history": history,
    }


# ======================================================================
# 数据加载 & 1:1 平衡 & 6:2:2 划分
# ======================================================================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_balance_data(file_path: Path, max_toxic: Optional[int] = None, seed: int = 42):
    safe, toxic = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = sample.get("input")
            prompt = (inp.get("prompt", "") or "") if isinstance(inp, dict) else ""
            if not prompt:
                continue
            # 解析标签
            guard = {}
            if isinstance(sample.get("guard"), dict):
                guard = sample["guard"]
            elif isinstance(sample.get("inference"), dict) and isinstance(
                (sample["inference"] or {}).get("guard"), dict
            ):
                guard = sample["inference"]["guard"]
            label = None
            asr = guard.get("asr_label")
            if asr is not None:
                label = int(asr)
            if label is None:
                v = (guard.get("verdict") or "").lower()
                if v == "allow": label = 0
                elif v in ("flag", "block"): label = 1
            if label is None:
                jb = guard.get("jailbreak_success")
                if jb is False: label = 0
                elif jb is True: label = 1
            if label is None:
                continue
            (safe if label == 0 else toxic).append(prompt)

    print(f"[Data] 原始: 安全={len(safe)}, 有害={len(toxic)}")
    n_toxic = min(max_toxic, len(toxic)) if max_toxic else len(toxic)
    n_safe = min(n_toxic, len(safe))
    n_toxic = n_safe  # 保证 1:1
    random.Random(seed).shuffle(safe)
    random.Random(seed + 1).shuffle(toxic)
    texts = safe[:n_safe] + toxic[:n_toxic]
    labels = [0]*n_safe + [1]*n_toxic
    combined = list(zip(texts, labels))
    random.Random(seed + 2).shuffle(combined)
    texts, labels = zip(*combined) if combined else ([], [])
    print(f"[Data] 1:1 平衡: 安全={n_safe}, 有害={n_toxic}, 总={n_safe+n_toxic}")
    return list(texts), list(labels)


def split_622(texts, labels, seed=42):
    """6:2:2 划分，每份保持 1:1"""
    safe_idx = [i for i, l in enumerate(labels) if l == 0]
    toxic_idx = [i for i, l in enumerate(labels) if l == 1]
    random.Random(seed).shuffle(safe_idx)
    random.Random(seed + 1).shuffle(toxic_idx)

    def split3(idx):
        n = len(idx)
        return idx[:int(n*0.6)], idx[int(n*0.6):int(n*0.8)], idx[int(n*0.8):]

    def gather(indices):
        random.Random(seed + 2).shuffle(indices)
        return [texts[i] for i in indices], [labels[i] for i in indices]

    tr_s, va_s, te_s = split3(safe_idx)
    tr_t, va_t, te_t = split3(toxic_idx)
    tr_t_, tr_l = gather(tr_s + tr_t)
    va_t_, va_l = gather(va_s + va_t)
    te_t_, te_l = gather(te_s + te_t)
    print(f"[Split] 训练={len(tr_t_)}(S{len(tr_s)}+T{len(tr_t)}) "
          f"验证={len(va_t_)}(S{len(va_s)}+T{len(va_t)}) "
          f"测试={len(te_t_)}(S{len(te_s)}+T{len(te_t)})")
    return tr_t_, tr_l, va_t_, va_l, te_t_, te_l


# ======================================================================
# 保存
# ======================================================================
def save_results(results: Dict, output_dir: Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    met, unmet = [], []

    for layer_idx, r in sorted(results.items()):
        d = out / f"layer_{layer_idx}"
        d.mkdir(exist_ok=True)
        torch.save(r["model"].state_dict(), d / "probe.pt")
        with open(d / "preprocessor.pkl", "wb") as f:
            pickle.dump({"scaler": r["scaler"]}, f)
        with open(d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(r["metrics"], f, indent=2, ensure_ascii=False)
        if r["training_history"]:
            with open(d / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(r["training_history"], f, indent=2, ensure_ascii=False)
        (met if r["metrics"]["target_reached"] else unmet).append(layer_idx)

    # 汇总
    summary = {
        "probe_formula": "softmax(w^T * h + b)",
        "pooling": "last_token",
        "num_layers": len(results),
        "met_layers": met, "unmet_layers": unmet,
        "layers": {
            str(k): {
                "val_acc": v["metrics"]["val_acc"],
                "val_balanced_acc": v["metrics"].get("val_balanced_acc"),
                "val_roc_auc": v["metrics"].get("val_roc_auc"),
                "val_pr_auc": v["metrics"].get("val_pr_auc"),
                "target_acc": v["metrics"]["target_acc"],
                "target_reached": v["metrics"]["target_reached"],
                "best_epoch": v["metrics"]["best_epoch"],
            } for k, v in sorted(results.items())
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 完整训练日志
    log = {
        "num_layers": len(results),
        "layers": {
            str(k): {"metrics": v["metrics"], "history": v["training_history"]}
            for k, v in sorted(results.items())
        },
    }
    with open(out / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n[Save] 探针权重 + 指标: {out}")
    print(f"[Save] 汇总: {out / 'summary.json'}")
    print(f"[Save] 训练日志: {out / 'training_log.json'}")
    print(f"  达标: {len(met)} 层 | 未达标: {len(unmet)} 层")
    return out


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="1:1 平衡训练线性探针 softmax(w^T*h+b)")
    parser.add_argument("--data_file", type=Path, default=Path("logs/base_evaluation.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/probes"))
    parser.add_argument("--max_toxic_samples", type=int, default=None)
    parser.add_argument("--hidden_states_cache", type=Path, default=None,
                        help="预提取的隐藏态 .npz（跳过 LLM 加载）")
    # 超参数
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("线性探针训练  P(toxic|h) = softmax(w^T * h + b)")
    print(f"隐藏态: last_token | 预处理: StandardScaler")
    print(f"超参: lr={args.lr} wd={args.weight_decay} dropout={args.dropout} "
          f"epochs={args.num_epochs} bs={args.batch_size}")
    print(f"早停: 达标即退出，未达标训练满 {args.num_epochs} 轮")
    print("=" * 60)

    # ---- 加载隐藏态 ----
    # 优先用 --hidden_states_cache，否则尝试 output_dir 下的默认路径
    cache_file = None
    if args.hidden_states_cache and Path(args.hidden_states_cache).exists():
        cache_file = Path(args.hidden_states_cache)
    elif (Path(args.output_dir) / "hidden_states_cache.npz").exists():
        cache_file = Path(args.output_dir) / "hidden_states_cache.npz"

    if cache_file is not None:
        print(f"\n[Cache] 加载 {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        train_hs = data["train_hs"]
        val_hs = data["val_hs"]
        train_labels = data["train_labels"].tolist()
        val_labels = data["val_labels"].tolist()
        num_layers, hidden_dim = int(data["num_layers"]), int(data["hidden_dim"])
        test_hs = data["test_hs"] if "test_hs" in data else None
        test_labels = data["test_labels"].tolist() if "test_labels" in data else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from_cache = True
        # 打印元信息
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"][0]))
                print(f"[Cache] 来源: {meta.get('data_file', '?')}  池化: {meta.get('pooling', '?')}")
            except Exception:
                pass
        n_tr = len(train_labels)
        n_va = len(val_labels)
        n_te = len(test_labels) if test_labels else 0
        print(f"[Cache] train={n_tr}(S={sum(1 for l in train_labels if l==0)} "
              f"T={sum(1 for l in train_labels if l==1)})  "
              f"val={n_va}  test={n_te}  layers={num_layers}  dim={hidden_dim}")
    else:
        from_cache = False
        texts, labels = load_and_balance_data(
            args.data_file, max_toxic=args.max_toxic_samples, seed=args.seed)
        if not texts:
            raise ValueError("未加载到有效样本")
        tr_t, tr_l, va_t, va_l, te_t, te_l = split_622(texts, labels, seed=args.seed)

        print("\n[Model] 加载 LLM...")
        tokenizer, model = ModelManager().load_llm()
        device = next(model.parameters()).device

        print(f"[Hidden] 提取隐藏态 (last_token)...")
        tr_hs = extract_hidden_states(model, tokenizer, tr_t, device,
                                       max_length=args.max_length, batch_size=args.batch_size,
                                       pooling_method="last_token", desc="训练集")
        num_layers, hidden_dim = tr_hs[0].shape
        va_hs = extract_hidden_states(model, tokenizer, va_t, device,
                                       max_length=args.max_length, batch_size=args.batch_size,
                                       pooling_method="last_token", desc="验证集")
        te_hs = extract_hidden_states(model, tokenizer, te_t, device,
                                       max_length=args.max_length, batch_size=args.batch_size,
                                       pooling_method="last_token", desc="测试集")
        to_np = lambda lst: np.stack([h.cpu().numpy() if isinstance(h, torch.Tensor) else h for h in lst])
        train_hs, val_hs, test_hs = to_np(tr_hs), to_np(va_hs), to_np(te_hs)
        train_labels, val_labels, test_labels = tr_l, va_l, te_l

        # 保存隐藏态缓存（各层各集最终状态 + 标签 + 元信息）
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(args.output_dir) / "hidden_states_cache.npz"
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
            cache_path,
            train_hs=train_hs,         # (N_train, num_layers, hidden_dim)
            val_hs=val_hs,             # (N_val,   num_layers, hidden_dim)
            test_hs=test_hs,           # (N_test,  num_layers, hidden_dim)
            train_labels=np.array(train_labels, dtype=np.int64),
            val_labels=np.array(val_labels, dtype=np.int64),
            test_labels=np.array(test_labels, dtype=np.int64),
            num_layers=np.int32(num_layers),
            hidden_dim=np.int32(hidden_dim),
            meta=np.array([cache_meta]),
        )
        print(f"[Cache] 隐藏态已保存: {cache_path}")
        print(f"        train={train_hs.shape} val={val_hs.shape} test={test_hs.shape}")
        print(f"        train: S={n_tr_s} T={n_tr_t} | val: S={n_va_s} T={n_va_t} | test: S={n_te_s} T={n_te_t}")

    # ---- 逐层训练 ----
    print(f"\n[Train] {num_layers} 层线性探针\n")
    results = {}
    tr_y = np.array(train_labels)
    va_y = np.array(val_labels)
    te_y = np.array(test_labels) if test_labels is not None else None

    for i in tqdm(range(num_layers), desc="逐层训练", ncols=90):
        results[i] = train_one_layer(
            layer_idx=i,
            train_x=train_hs[:, i, :], train_y=tr_y,
            val_x=val_hs[:, i, :], val_y=va_y,
            test_x=test_hs[:, i, :] if test_hs is not None else None,
            test_y=te_y,
            num_layers=num_layers, device=device,
            max_epochs=args.num_epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            dropout=args.dropout, seed=args.seed,
        )

    # ---- 保存 ----
    probes_dir = save_results(results, args.output_dir)

    n_s = sum(1 for l in train_labels if l == 0)
    n_t = sum(1 for l in train_labels if l == 1)
    config = {
        "probe_formula": "softmax(w^T * h + b)",
        "pooling": "last_token",
        "preprocess": "StandardScaler",
        "lr": args.lr, "weight_decay": args.weight_decay,
        "dropout": args.dropout, "max_epochs": args.num_epochs,
        "batch_size": args.batch_size, "seed": args.seed,
        "max_length": args.max_length,
        "balance": "1:1",
        "split": "6:2:2",
        "train_samples": f"{len(train_labels)} (S={n_s}, T={n_t})",
        "val_samples": len(val_labels),
        "test_samples": len(test_labels) if test_labels else 0,
        "from_cache": from_cache,
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(probes_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] 训练完成")


if __name__ == "__main__":
    main()
