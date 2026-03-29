#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成层间语义演化数据（Layer View 所需）

=== 原理 ===

探针训练时，为每层学到一个线性分类器：
    P(toxic | h) = softmax(W · scaler(h) + b)

其中 scaler(h) = (h - mean) / scale 是 StandardScaler 标准化，
W 是 (2, hidden_dim) 的权重矩阵，
毒性向量 w_toxic = W[1] - W[0]，代表"有害方向"（在标准化空间中）。

投影时必须先标准化隐藏状态：
    h_scaled = (h - mean) / scale
    projection = h_scaled · w_toxic / ||w_toxic||

这个投影值越大 → 该样本在这一层越偏向"有害"语义。

对所有样本按 safe / toxic 分组统计投影值的分布，
就能看出模型在不同层如何逐步区分安全和有害内容。

=== 运行方式 ===

模式 A（推荐，无需 GPU）：使用已提取的隐藏状态缓存
    python scripts/generate_layer_evolution.py \
        --hidden_states_cache outputs/probes/hidden_states_cache.npz \
        --toxic_vectors outputs/toxic_vectors/toxic_vectors.npz

模式 B（需 GPU）：直接从模型提取
    python scripts/generate_layer_evolution.py \
        --data_file logs/base_evaluation.jsonl \
        --toxic_vectors outputs/toxic_vectors/toxic_vectors.npz

=== 输出 ===

outputs/layer_evolution/semantic_evolution.json  — 每层完整统计
outputs/layer_evolution/streamgraph_data.json    — 流图数据
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# 第一步：加载隐藏状态
# ======================================================================
#
# hidden_states_cache.npz 由 extract_hidden_states.py 在 GPU 上提取产出
# 格式:
#   train_hs:     shape (N_train, 33, 4096)  — 训练集每个样本在 33 层的隐藏状态
#   val_hs:       shape (N_val,   33, 4096)  — 验证集
#   test_hs:      shape (N_test,  33, 4096)  — 测试集
#   train_labels: shape (N_train,)           — 标签，0=safe, 1=toxic
#   val_labels:   shape (N_val,)
#   test_labels:  shape (N_test,)
#
def load_hidden_states_from_cache(cache_path):
    """加载并合并 train/val/test 隐藏状态"""
    print(f"[Cache] 加载: {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)

    parts_hs = []
    parts_labels = []
    for split in ["train", "val", "test"]:
        hs_key = f"{split}_hs"
        lb_key = f"{split}_labels"
        if hs_key in cache and lb_key in cache:
            parts_hs.append(cache[hs_key])
            parts_labels.append(cache[lb_key])
            print(f"  {split}: hs={cache[hs_key].shape}, labels={cache[lb_key].shape}")

    # 合并所有 split → (N_total, 33, 4096)
    all_hs = np.concatenate(parts_hs, axis=0)
    all_labels = np.concatenate(parts_labels, axis=0)
    num_layers = all_hs.shape[1]

    n_safe = int(np.sum(all_labels == 0))
    n_toxic = int(np.sum(all_labels == 1))
    print(f"[Cache] 合并: {all_hs.shape}, safe={n_safe}, toxic={n_toxic}")
    return all_hs, all_labels, num_layers


# ======================================================================
# 第二步：加载毒性向量
# ======================================================================
#
# toxic_vectors.npz 由 extract_toxic_vectors.py 从探针权重中提取
# 格式:
#   vectors:       shape (33, 4096) — 每层的毒性方向向量
#   biases:        shape (33,)      — 每层的偏置
#   layer_0~32:    shape (4096,)    — 与 vectors 相同，按层独立存储
#   layer_0_bias~: 标量              — 与 biases 相同，按层独立存储
#
def load_toxic_vectors(toxic_vec_path):
    """加载每层的毒性向量和偏置"""
    print(f"[Toxic] 加载: {toxic_vec_path}")
    data = np.load(toxic_vec_path, allow_pickle=True)
    keys = list(data.keys())

    vectors = {}  # layer_idx → w_toxic (4096,)
    bias = {}     # layer_idx → scalar

    # 优先使用批量格式
    if "vectors" in data:
        tv = data["vectors"]  # (33, 4096)
        for i in range(tv.shape[0]):
            vectors[i] = tv[i]
        print(f"  vectors: {tv.shape}")

    if "biases" in data:
        for i, b in enumerate(data["biases"]):
            bias[i] = float(b)
        print(f"  biases: {data['biases'].shape}")

    # 补充单层格式
    for key in keys:
        if key.startswith("layer_") and "_bias" not in key and key not in ("layer_indices", "layer_count"):
            try:
                idx = int(key.split("_")[1])
                if idx not in vectors:
                    vectors[idx] = data[key]
            except ValueError:
                continue

    for key in keys:
        if key.endswith("_bias") and key.startswith("layer_"):
            try:
                idx = int(key.replace("layer_", "").replace("_bias", ""))
                if idx not in bias:
                    bias[idx] = float(data[key])
            except ValueError:
                continue

    print(f"  共 {len(vectors)} 层毒性向量")
    return vectors, bias


# ======================================================================
# 第三步：逐层计算投影统计
# ======================================================================
#
# 核心公式 (对每层 layer_i):
#
#   w = toxic_vectors[layer_i]          # 该层的毒性方向 (4096,)
#   h = all_hs[:, layer_i, :]           # 所有样本在该层的隐藏状态 (N, 4096)
#   proj = h · w / ||w||                # 投影到毒性方向上 → (N,) 标量
#
#   safe_proj  = proj[labels == 0]      # 安全样本的投影值
#   toxic_proj = proj[labels == 1]      # 有害样本的投影值
#
# 然后对两组分别计算箱线图统计量
#
def compute_boxplot_stats(arr):
    """
    计算一组投影值的箱线图统计量

    返回:
      min    — 最小投影值（该组样本中最"安全"的一端）
      q1     — 第25百分位（下四分位数）
      median — 中位数（典型投影值）
      q3     — 第75百分位（上四分位数）
      max    — 最大投影值（该组样本中最"有害"的一端）
      mean   — 均值
      std    — 标准差（该组投影值的离散程度）
      count  — 样本数量
    """
    if len(arr) == 0:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0,
                "mean": 0, "std": 0, "count": 0}
    return {
        "min":    float(np.percentile(arr, 0)),
        "q1":     float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q3":     float(np.percentile(arr, 75)),
        "max":    float(np.percentile(arr, 100)),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "count":  int(len(arr)),
    }


def load_preprocessors(probes_dir):
    """
    加载每层的 StandardScaler 预处理器

    探针训练时对隐藏状态做了 StandardScaler 标准化:
      h_scaled = (h - mean) / scale
    投影计算时必须使用相同的变换，否则结果不在同一坐标系。

    来源: outputs/probes/probes/layer_{i}/preprocessor.pkl
    """
    preprocessors = {}
    for base in [
        Path(probes_dir) / "probes",
        Path(probes_dir),
    ]:
        if not base.exists():
            continue
        for layer_dir in sorted(base.iterdir()):
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            pp_file = layer_dir / "preprocessor.pkl"
            if not pp_file.exists():
                continue
            try:
                idx = int(layer_dir.name.split("_")[1])
                with open(pp_file, "rb") as f:
                    data = pickle.load(f)
                scaler = data["scaler"] if isinstance(data, dict) else data
                preprocessors[idx] = scaler
            except (ValueError, Exception):
                continue
        if preprocessors:
            break

    if preprocessors:
        print(f"[Scaler] 加载了 {len(preprocessors)} 层的 StandardScaler")
    return preprocessors


def load_probe_metrics(probes_dir):
    """
    加载每层的探针准确率指标

    来源: outputs/probes/probes/llama-3-8b/layer_{i}/metrics.json
    或:   outputs/probes/probes/layer_{i}/metrics.json
    """
    metrics = {}
    for base in [
        Path(probes_dir) / "probes" / "llama-3-8b",
        Path(probes_dir) / "probes",
        Path(probes_dir),
    ]:
        if not base.exists():
            continue
        for layer_dir in sorted(base.iterdir()):
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            mf = layer_dir / "metrics.json"
            if not mf.exists():
                continue
            try:
                idx = int(layer_dir.name.split("_")[1])
                with open(mf, "r", encoding="utf-8") as f:
                    metrics[idx] = json.load(f)
            except (ValueError, json.JSONDecodeError):
                continue
        if metrics:
            break
    return metrics


def compute_evolution(all_hs, all_labels, toxic_vectors, toxic_bias,
                      probe_metrics=None, preprocessors=None):
    """
    逐层计算投影统计，生成语义演化数据

    关键：投影前必须对隐藏状态做 StandardScaler 标准化（与探针训练一致）。
    探针训练: scaler.fit(train_h) → h_scaled = scaler.transform(h) → W @ h_scaled + b
    所以 w_toxic = W[1]-W[0] 在标准化空间中，投影也必须在标准化空间计算。

    对每一层输出:
      --- 实际标签（固定，用于流图分组） ---
      safe_count             — 实际安全样本数（每层相同）
      toxic_count            — 实际有害样本数（每层相同）
      safe_ratio             — 实际安全比例
      mean_projection_safe   — 安全样本投影均值
      mean_projection_toxic  — 有害样本投影均值

      --- 探针预测（每层不同） ---
      probe_predicted_safe   — 探针预测为安全的样本数
      probe_predicted_toxic  — 探针预测为有害的样本数
      probe_accuracy         — 探针准确率

      --- 箱线图 + 边界 ---
      safe / toxic           — 两组投影值的箱线图统计
      decision_boundary      — 探针决策边界位置
      probe_separability     — |mean_safe - mean_toxic|
    """
    num_layers = all_hs.shape[1]
    safe_mask = all_labels == 0
    toxic_mask = all_labels == 1

    evolution = {}
    streamgraph = []

    for layer_idx in range(num_layers):
        if layer_idx not in toxic_vectors:
            continue

        h = all_hs[:, layer_idx, :].astype(np.float32)  # (N, 4096)

        # ---- 标准化（与探针训练一致） ----
        if preprocessors and layer_idx in preprocessors:
            scaler = preprocessors[layer_idx]
            h = scaler.transform(h.astype(np.float64)).astype(np.float32)

        w = toxic_vectors[layer_idx].astype(np.float32)  # (4096,)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-8:
            continue

        proj = h @ w / w_norm  # (N,)

        safe_proj = proj[safe_mask]
        toxic_proj = proj[toxic_mask]

        b = toxic_bias.get(layer_idx, 0.0)
        decision_boundary = -b / w_norm if w_norm > 1e-8 else 0.0

        separability = float(np.abs(np.mean(safe_proj) - np.mean(toxic_proj)))

        safe_count = int(len(safe_proj))
        toxic_count = int(len(toxic_proj))
        total = safe_count + toxic_count

        # ---- 探针预测（基于标准化后的投影 + 决策边界） ----
        probe_pred_toxic_mask = proj > decision_boundary
        probe_predicted_toxic = int(np.sum(probe_pred_toxic_mask))
        probe_predicted_safe = total - probe_predicted_toxic

        correct_safe = int(np.sum(~probe_pred_toxic_mask[safe_mask]))
        correct_toxic = int(np.sum(probe_pred_toxic_mask[toxic_mask]))
        probe_accuracy = round((correct_safe + correct_toxic) / total, 4) if total > 0 else 0

        metrics_acc = {}
        if probe_metrics and layer_idx in probe_metrics:
            m = probe_metrics[layer_idx]
            metrics_acc = {
                "val_acc": m.get("val_acc"),
                "test_acc": m.get("test_acc_best", m.get("test_acc")),
                "val_roc_auc": m.get("val_roc_auc"),
            }

        layer_data = {
            "layer": layer_idx,
            # 实际标签（流图分组依据，每层相同）
            "safe_count": safe_count,
            "toxic_count": toxic_count,
            "safe_ratio": round(safe_count / total, 4) if total > 0 else 0,
            "mean_projection_safe": float(np.mean(safe_proj)) if safe_count > 0 else 0,
            "mean_projection_toxic": float(np.mean(toxic_proj)) if toxic_count > 0 else 0,
            # 探针预测（每层不同）
            "probe_predicted_safe": probe_predicted_safe,
            "probe_predicted_toxic": probe_predicted_toxic,
            "probe_accuracy": probe_accuracy,
            "probe_metrics": metrics_acc if metrics_acc else None,
            # 箱线图统计
            "safe": compute_boxplot_stats(safe_proj),
            "toxic": compute_boxplot_stats(toxic_proj),
            "decision_boundary": float(decision_boundary),
            "probe_separability": separability,
        }

        evolution[f"layer_{layer_idx}"] = layer_data
        streamgraph.append({
            "layer": layer_idx,
            "success": layer_data["toxic"],
            "fail": layer_data["safe"],
        })

        if layer_idx % 8 == 0 or layer_idx == num_layers - 1:
            print(f"  layer_{layer_idx}: sep={separability:.4f}, "
                  f"probe_acc={probe_accuracy:.4f}, "
                  f"pred_safe={probe_predicted_safe}, pred_toxic={probe_predicted_toxic}")

    return evolution, streamgraph


# ======================================================================
# 主入口
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="生成层间语义演化数据")
    parser.add_argument("--hidden_states_cache", type=Path, default=None,
                        help="隐藏状态缓存 (.npz)，模式 A")
    parser.add_argument("--data_file", type=Path, default=None,
                        help="评估数据文件，模式 B（需 GPU）")
    parser.add_argument("--toxic_vectors", type=Path,
                        default=Path("outputs/toxic_vectors/toxic_vectors.npz"),
                        help="毒性向量文件")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/layer_evolution"),
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # 自动检测缓存
    if args.hidden_states_cache is None:
        default_cache = Path("outputs/probes/hidden_states_cache.npz")
        if default_cache.exists():
            args.hidden_states_cache = default_cache
            print(f"[Auto] 发现缓存: {default_cache}")

    # 加载毒性向量
    if not args.toxic_vectors.exists():
        alt_path = Path("outputs/probes/probes/llama-3-8b/toxic_vectors.npz")
        if alt_path.exists():
            args.toxic_vectors = alt_path
        else:
            print(f"[Error] 找不到毒性向量: {args.toxic_vectors}")
            sys.exit(1)

    toxic_vectors, toxic_bias = load_toxic_vectors(args.toxic_vectors)

    # 加载探针指标 + 预处理器
    probes_dir = Path("outputs/probes")
    probe_metrics = load_probe_metrics(probes_dir)
    if probe_metrics:
        print(f"[Probe] 加载了 {len(probe_metrics)} 层的探针指标")
    preprocessors = load_preprocessors(probes_dir)

    # ---- 模式 A：从缓存 ----
    if args.hidden_states_cache is not None and args.hidden_states_cache.exists():
        print(f"\n{'='*60}")
        print(f"模式 A：从缓存计算（无需 GPU）")
        print(f"{'='*60}")
        all_hs, all_labels, _ = load_hidden_states_from_cache(args.hidden_states_cache)
        evolution, streamgraph = compute_evolution(
            all_hs, all_labels, toxic_vectors, toxic_bias,
            probe_metrics, preprocessors
        )

    # ---- 模式 B：从模型 ----
    elif args.data_file is not None and args.data_file.exists():
        print(f"\n{'='*60}")
        print(f"模式 B：从模型提取（需 GPU）")
        print(f"{'='*60}")

        import importlib.util
        import torch
        from engine.models import ModelManager
        from engine.probes.linear_probe_balanced import extract_hidden_states

        train_script = PROJECT_ROOT / "scripts" / "train_probes_balanced.py"
        spec = importlib.util.spec_from_file_location("_train", train_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.set_seed(42)
        texts, labels = mod.load_and_balance_data(args.data_file, seed=42)

        tokenizer, model = ModelManager().load_llm()
        device = next(model.parameters()).device

        hs_list = extract_hidden_states(
            model, tokenizer, texts, device,
            max_length=args.max_length, batch_size=args.batch_size,
            pooling_method="last_token", desc="全部样本"
        )
        all_hs = np.stack([h.cpu().numpy() if isinstance(h, torch.Tensor) else h for h in hs_list])
        all_labels = np.array(labels)
        evolution, streamgraph = compute_evolution(
            all_hs, all_labels, toxic_vectors, toxic_bias,
            probe_metrics, preprocessors
        )

        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        print("[Error] 请提供 --hidden_states_cache 或 --data_file")
        print("  python scripts/extract_hidden_states.py \\")
        print("      --data_file logs/base_evaluation.jsonl \\")
        print("      --output outputs/probes/hidden_states_cache.npz")
        sys.exit(1)

    # ---- 保存 ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evo_path = args.output_dir / "semantic_evolution.json"
    with open(evo_path, "w", encoding="utf-8") as f:
        json.dump(evolution, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] {evo_path}")

    stream_path = args.output_dir / "streamgraph_data.json"
    with open(stream_path, "w", encoding="utf-8") as f:
        json.dump(streamgraph, f, ensure_ascii=False, indent=2)
    print(f"[Save] {stream_path}")

    print(f"\n完成！共 {len(evolution)} 层")


if __name__ == "__main__":
    main()
