#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隐藏态可分性诊断分析

目的：量化证明 safe / toxic 两类隐藏态在特征空间中是否可分。

分析指标（逐层）：
  1. Fisher 判别比 (FDR)      — 类间方差 / 类内方差，>1 表示有区分度
  2. 类心距离比 (CDR)          — 类心间距 / 平均类内散布，越大越可分
  3. 余弦相似度分析             — safe-toxic 对 vs safe-safe 对 vs toxic-toxic 对
  4. 线性探针准确率 (快速 SVM)  — 线性 SVM 的交叉验证 balanced accuracy
  5. PCA / t-SNE 2D 可视化     — 直观观察两类是否重叠

Usage:
  # 使用已有的隐藏态缓存
  python scripts/analyze_separability.py --cache outputs/probes/hidden_states_cache.npz

  # 从 base_evaluation.jsonl 提取（需 GPU + LLM）
  python scripts/analyze_separability.py --data_file logs/base_evaluation.jsonl --max_samples 1000

  # 从 attack_enhanced 的缓存分析
  python scripts/analyze_separability.py --cache outputs/probes_attack_enhanced/hidden_states_cache.npz
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# 指标计算
# ======================================================================
def _standardize(X: np.ndarray) -> np.ndarray:
    """零均值单位方差标准化（与探针训练一致）"""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


def fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray, normalize: bool = True) -> float:
    """
    Fisher 判别比 (FDR)

    FDR = ||μ_0 - μ_1||^2 / (σ_0^2 + σ_1^2)

    先对数据做 StandardScaler 标准化（与探针训练一致），
    让各维度贡献均等，避免高方差维度主导结果。

    判定标准:
    - FDR < 0.1 : 严重混叠，两类几乎完全重叠
    - 0.1 ≤ FDR < 0.5 : 混叠，线性分类器接近随机
    - 0.5 ≤ FDR < 2.0 : 弱信号，线性分类器可能达到 60-70%
    - 2.0 ≤ FDR < 4.0 : 可分，线性分类器可达 75-85%
    - FDR ≥ 4.0 : 强可分，线性分类器可达 85%+

    Returns:
        float: Fisher 判别比
    """
    mask0 = y == 0
    mask1 = y == 1
    if mask0.sum() < 2 or mask1.sum() < 2:
        return 0.0

    if normalize:
        X = _standardize(X)

    mu0 = X[mask0].mean(axis=0)
    mu1 = X[mask1].mean(axis=0)

    # 类间距离（类心之间的欧氏距离平方）
    between = np.sum((mu0 - mu1) ** 2)

    # 类内方差（各类到各自类心的平均距离平方）
    within0 = np.mean(np.sum((X[mask0] - mu0) ** 2, axis=1))
    within1 = np.mean(np.sum((X[mask1] - mu1) ** 2, axis=1))
    within = within0 + within1

    return float(between / max(within, 1e-10))


def centroid_distance_ratio(X: np.ndarray, y: np.ndarray, normalize: bool = True) -> Tuple[float, float, float]:
    """
    类心距离比 (CDR)

    CDR = d(μ_0, μ_1) / ((spread_0 + spread_1) / 2)

    标准化后计算，含义:
    - CDR < 0.3 : 类心几乎重合
    - 0.3 ≤ CDR < 1.0 : 类心有一定距离但小于类内散布
    - CDR ≥ 1.0 : 类心距离大于类内散布，存在可分结构

    Returns:
        (cdr, centroid_dist, avg_spread)
    """
    mask0 = y == 0
    mask1 = y == 1
    if mask0.sum() < 2 or mask1.sum() < 2:
        return 0.0, 0.0, 0.0

    if normalize:
        X = _standardize(X)

    mu0 = X[mask0].mean(axis=0)
    mu1 = X[mask1].mean(axis=0)

    centroid_dist = np.linalg.norm(mu0 - mu1)
    spread0 = np.mean(np.linalg.norm(X[mask0] - mu0, axis=1))
    spread1 = np.mean(np.linalg.norm(X[mask1] - mu1, axis=1))
    avg_spread = (spread0 + spread1) / 2.0

    cdr = centroid_dist / max(avg_spread, 1e-10)
    return float(cdr), float(centroid_dist), float(avg_spread)


def cosine_similarity_analysis(X: np.ndarray, y: np.ndarray, n_pairs: int = 2000) -> dict:
    """
    余弦相似度分析

    随机采样 n_pairs 对样本，计算:
    - safe-safe 对的平均余弦相似度
    - toxic-toxic 对的平均余弦相似度
    - safe-toxic 对的平均余弦相似度

    如果 safe-toxic ≈ safe-safe ≈ toxic-toxic，说明两类混在一起。
    如果 safe-toxic << safe-safe, toxic-toxic，说明两类分开了。
    """
    from numpy.linalg import norm

    mask0 = np.where(y == 0)[0]
    mask1 = np.where(y == 1)[0]

    same_group = False  # 标记是否是同组采样（需避免自身配对）

    def sample_cosines(idx_a, idx_b, n, is_same_group=False):
        """采样 n 对，计算余弦相似度"""
        cosines = []
        for _ in range(n):
            i = np.random.choice(idx_a)
            j = np.random.choice(idx_b)
            # 同组采样时避免自身配对
            if is_same_group and i == j:
                for _ in range(5):  # 最多重试 5 次
                    j = np.random.choice(idx_b)
                    if j != i:
                        break
                if i == j:
                    continue  # 放弃这一对
            a, b = X[i], X[j]
            na, nb = norm(a), norm(b)
            if na > 0 and nb > 0:
                cosines.append(float(np.dot(a, b) / (na * nb)))
        return cosines

    ss = sample_cosines(mask0, mask0, n_pairs, is_same_group=True) if len(mask0) >= 2 else []
    tt = sample_cosines(mask1, mask1, n_pairs, is_same_group=True) if len(mask1) >= 2 else []
    st = sample_cosines(mask0, mask1, n_pairs, is_same_group=False) if len(mask0) >= 1 and len(mask1) >= 1 else []

    def stats(arr):
        if not arr:
            return {"mean": 0, "std": 0}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    return {
        "safe_safe": stats(ss),
        "toxic_toxic": stats(tt),
        "safe_toxic": stats(st),
        "gap": float(np.mean(ss) + np.mean(tt)) / 2 - float(np.mean(st)) if ss and tt and st else 0,
    }


def linear_svm_accuracy(X: np.ndarray, y: np.ndarray, pre_scaled: bool = False) -> float:
    """
    线性 SVM 5 折交叉验证 balanced accuracy

    这是对"线性可分性"最直接的度量。

    Args:
        X: 特征矩阵
        y: 标签
        pre_scaled: 如果为 True，跳过内部 StandardScaler（数据已标准化）
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    import warnings

    if len(np.unique(y)) < 2 or len(y) < 10:
        return 0.5

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Liblinear.*")
            if pre_scaled:
                clf = LinearSVC(max_iter=5000, class_weight='balanced')
            else:
                clf = make_pipeline(
                    StandardScaler(),
                    LinearSVC(max_iter=5000, class_weight='balanced')
                )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')
            return float(scores.mean())
    except Exception:
        return 0.5


# ======================================================================
# 可视化
# ======================================================================
def visualize_layer(
    X: np.ndarray, y: np.ndarray,
    layer_idx: int, output_dir: Path,
    method: str = "pca",
) -> Path:
    """PCA 或 t-SNE 2D 可视化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        X2d = reducer.fit_transform(X)
        explained = reducer.explained_variance_ratio_
        title = f"Layer {layer_idx} - PCA (var={explained[0]:.1%}+{explained[1]:.1%})"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp = max(5, min(30, len(X) // 3))  # perplexity 需 < n_samples/3 且 >= 5
        X2d = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)
        title = f"Layer {layer_idx} - t-SNE (perp={perp})"
    else:
        raise ValueError(f"Unknown method: {method}")

    mask0 = y == 0
    mask1 = y == 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(X2d[mask0, 0], X2d[mask0, 1], c='#2196F3', alpha=0.4, s=15, label=f'Safe (n={mask0.sum()})')
    ax.scatter(X2d[mask1, 0], X2d[mask1, 1], c='#F44336', alpha=0.6, s=15, label=f'Toxic (n={mask1.sum()})')
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"layer_{layer_idx}_{method}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ======================================================================
# 数据加载
# ======================================================================
def load_from_cache(cache_path: Path, max_per_class: int = 0):
    """
    从 hidden_states_cache.npz 加载。

    Args:
        cache_path: 缓存文件路径
        max_per_class: 每类最大样本数（0=不限制）。
                       用于控制内存：设为 500 则总共最多加载 1000 条。

    Returns:
        (hs, labels, num_layers)
        hs: (N, num_layers, hidden_dim) — 如果设了 max_per_class 则 N 会减小
    """
    data = np.load(cache_path, allow_pickle=True)
    # 合并 train + val + test
    all_hs = [data["train_hs"]]
    all_labels = [data["train_labels"]]
    if "val_hs" in data:
        all_hs.append(data["val_hs"])
        all_labels.append(data["val_labels"])
    if "test_hs" in data:
        all_hs.append(data["test_hs"])
        all_labels.append(data["test_labels"])
    hs = np.concatenate(all_hs, axis=0)       # (N, num_layers, hidden_dim)
    labels = np.concatenate(all_labels, axis=0)  # (N,)
    num_layers = int(data["num_layers"])

    # 可选：按类别采样以控制内存
    if max_per_class > 0:
        rng = np.random.RandomState(42)
        safe_idx = np.where(labels == 0)[0]
        toxic_idx = np.where(labels == 1)[0]
        if len(safe_idx) > max_per_class:
            safe_idx = rng.choice(safe_idx, max_per_class, replace=False)
        if len(toxic_idx) > max_per_class:
            toxic_idx = rng.choice(toxic_idx, max_per_class, replace=False)
        keep = np.sort(np.concatenate([safe_idx, toxic_idx]))
        print(f"[Sample] 采样: safe={len(safe_idx)}, toxic={len(toxic_idx)}, "
              f"total={len(keep)} (原始 {len(labels)})")
        hs = hs[keep]
        labels = labels[keep]

    return hs, labels, num_layers


def load_from_evaluation(file_path: Path, max_samples: int = 1000, seed: int = 42):
    """从 evaluation JSONL 加载 prompt + label，需要后续提取隐藏态"""
    safe, toxic = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = sample.get("input", {})
            prompt = inp.get("prompt", "") if isinstance(inp, dict) else ""
            if not prompt:
                continue
            guard = sample.get("guard", {})
            asr = guard.get("asr_label")
            if asr is not None:
                label = int(asr)
            else:
                v = (guard.get("verdict") or "").lower()
                if v == "allow":
                    label = 0
                elif v in ("flag", "block"):
                    label = 1
                else:
                    continue
            (safe if label == 0 else toxic).append(prompt)

    print(f"[Data] safe={len(safe)}, toxic={len(toxic)}")

    # 采样
    rng = random.Random(seed)
    n_per_class = min(max_samples // 2, len(safe), len(toxic))
    rng.shuffle(safe)
    rng.shuffle(toxic)
    safe = safe[:n_per_class]
    toxic = toxic[:n_per_class]

    texts = safe + toxic
    labels = [0] * len(safe) + [1] * len(toxic)
    print(f"[Data] 采样: safe={len(safe)}, toxic={len(toxic)}")
    return texts, np.array(labels)


# ======================================================================
# 主分析
# ======================================================================
def analyze_all_layers(hs, labels, num_layers, output_dir, key_layers=None, do_viz=True):
    """
    逐层分析可分性

    Args:
        hs: (N, num_layers, hidden_dim)
        labels: (N,)
        num_layers: 层数
        output_dir: 输出目录
        key_layers: 重点分析的层（如 [0, 5, 10, 15, 20, 25, 28, 31]）
        do_viz: 是否生成可视化
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if key_layers is None:
        # 默认选择有代表性的层
        if num_layers >= 33:
            key_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32]
        elif num_layers >= 10:
            step = max(1, num_layers // 8)
            key_layers = list(range(0, num_layers, step))
            if num_layers - 1 not in key_layers:
                key_layers.append(num_layers - 1)
        else:
            key_layers = list(range(num_layers))

    print(f"\n{'='*80}")
    print(f"隐藏态可分性分析")
    print(f"样本: N={len(labels)} (safe={int((labels==0).sum())}, toxic={int((labels==1).sum())})")
    print(f"层数: {num_layers}  隐藏维度: {hs.shape[2]}")
    print(f"重点层: {key_layers}")
    print(f"{'='*80}\n")

    results = {}

    # 全部层的快速指标
    print(f"{'Layer':>6s} {'FDR':>8s} {'CDR':>8s} {'CosGap':>8s} {'SVM_bal':>8s} {'判定':>6s}")
    print("-" * 60)

    key_layers_set = set(key_layers)

    for i in range(num_layers):
        X_raw = hs[:, i, :]
        y = labels

        # 每层只做一次标准化，复用结果
        X_scaled = _standardize(X_raw)

        fdr = fisher_discriminant_ratio(X_scaled, y, normalize=False)
        cdr, cdist, cspread = centroid_distance_ratio(X_scaled, y, normalize=False)

        # 余弦分析在原始空间做（余弦相似度本身是尺度不变的）
        cos = cosine_similarity_analysis(X_raw, y, n_pairs=500)
        cos_gap = cos["gap"]

        # SVM 仅在 key_layers 上执行（较慢）
        svm_acc = None
        if i in key_layers_set:
            import sys as _sys
            _sys.stdout.write(f"  [Layer {i:2d}] SVM 5-fold CV 运行中...")
            _sys.stdout.flush()
            svm_acc = linear_svm_accuracy(X_scaled, y, pre_scaled=True)
            _sys.stdout.write(f" {svm_acc:.2%}\n")
            _sys.stdout.flush()

        # 判定
        if fdr > 2.0:
            verdict = "可分"
        elif fdr > 0.5:
            verdict = "弱信号"
        else:
            verdict = "混叠"

        svm_str = f"{svm_acc:.2%}" if svm_acc is not None else "   -   "
        print(f"{i:>6d} {fdr:>8.4f} {cdr:>8.4f} {cos_gap:>8.4f} {svm_str:>8s} {verdict:>6s}")

        results[i] = {
            "fisher_discriminant_ratio": fdr,
            "centroid_distance_ratio": cdr,
            "centroid_distance": cdist,
            "avg_class_spread": cspread,
            "cosine_gap": cos_gap,
            "cosine_detail": cos,
            "svm_balanced_acc": svm_acc,
            "verdict": verdict,
        }

    # 重点层可视化
    if do_viz:
        print(f"\n[Viz] 生成重点层可视化...")
        viz_dir = output_dir / "visualizations"
        for i in key_layers:
            X = hs[:, i, :]
            # 标准化后再可视化
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            path = visualize_layer(X_scaled, labels, i, viz_dir, method="pca")
            print(f"  Layer {i}: {path}")
            # t-SNE 较慢，仅对 3 个代表层做
            if i in key_layers[:1] + key_layers[-1:] + [key_layers[len(key_layers)//2]]:
                path = visualize_layer(X_scaled, labels, i, viz_dir, method="tsne")
                print(f"  Layer {i}: {path}")

    # ================================================================
    # 汇总 + 综合判定
    # ================================================================
    print(f"\n{'='*80}")
    print("一、各指标汇总")
    print(f"{'='*80}")

    fdrs = [results[i]["fisher_discriminant_ratio"] for i in range(num_layers)]
    best_fdr_layer = int(np.argmax(fdrs))
    best_fdr = fdrs[best_fdr_layer]

    print(f"\n[指标1] Fisher 判别比 (FDR) — 类间距离² / 类内方差")
    print(f"  最佳层: Layer {best_fdr_layer}, FDR = {best_fdr:.4f}")
    print(f"  平均 FDR: {np.mean(fdrs):.4f}")
    print(f"  FDR >= 2.0 (可分):    {sum(1 for f in fdrs if f >= 2.0):3d} / {num_layers} 层")
    print(f"  0.5 <= FDR < 2.0 (弱): {sum(1 for f in fdrs if 0.5 <= f < 2.0):3d} / {num_layers} 层")
    print(f"  FDR < 0.5 (混叠):     {sum(1 for f in fdrs if f < 0.5):3d} / {num_layers} 层")

    # 层趋势：深层是否优于浅层？
    if num_layers >= 10:
        shallow_fdr = np.mean(fdrs[:num_layers//3])
        deep_fdr = np.mean(fdrs[2*num_layers//3:])
        has_layer_trend = deep_fdr > shallow_fdr * 1.5
        print(f"  浅层平均 FDR: {shallow_fdr:.4f}  深层平均 FDR: {deep_fdr:.4f}")
        print(f"  层趋势: {'深层 > 浅层 (符合理论预期)' if has_layer_trend else '无明显趋势 (异常)'}")
    else:
        has_layer_trend = None

    svm_results = {i: results[i]["svm_balanced_acc"] for i in key_layers if results[i]["svm_balanced_acc"] is not None}
    best_svm = 0.5
    best_svm_layer = -1
    if svm_results:
        best_svm_layer = max(svm_results, key=svm_results.get)
        best_svm = svm_results[best_svm_layer]
        print(f"\n[指标2] 线性 SVM 5-fold balanced accuracy (最直接的线性可分证据)")
        for layer, acc in sorted(svm_results.items()):
            mark = " <-- best" if layer == best_svm_layer else ""
            print(f"  Layer {layer:3d}: {acc:.2%}{mark}")
        print(f"  随机基线: 50.00%")

    cos_gaps = [results[i]["cosine_gap"] for i in range(num_layers)]
    best_cos_layer = int(np.argmax(cos_gaps))
    best_cos_gap = max(cos_gaps)
    print(f"\n[指标3] 余弦相似度间隙 — (类内均值 - 类间均值)")
    print(f"  最大间隙: Layer {best_cos_layer}, gap = {best_cos_gap:.6f}")
    print(f"  平均间隙: {np.mean(cos_gaps):.6f}")
    if best_cos_gap < 0.005:
        print(f"  解读: gap < 0.005 → safe-toxic 对和 safe-safe 对的相似度几乎相同 → 两类混在一起")
    elif best_cos_gap < 0.02:
        print(f"  解读: gap 在 0.005-0.02 → 存在微弱的方向差异")
    else:
        print(f"  解读: gap > 0.02 → 两类在方向上有可检测的差异")

    cdrs = [results[i]["centroid_distance_ratio"] for i in range(num_layers)]
    best_cdr_layer = int(np.argmax(cdrs))
    best_cdr = cdrs[best_cdr_layer]
    print(f"\n[指标4] 类心距离比 (CDR) — 类心间距 / 平均类内散布")
    print(f"  最佳层: Layer {best_cdr_layer}, CDR = {best_cdr:.4f}")
    if best_cdr < 0.3:
        print(f"  解读: CDR < 0.3 → 类心几乎重合")
    elif best_cdr < 1.0:
        print(f"  解读: CDR 在 0.3-1.0 → 类心有距离但小于类内散布")
    else:
        print(f"  解读: CDR >= 1.0 → 类心距离大于类内散布")

    # ================================================================
    # 综合判定
    # ================================================================
    print(f"\n{'='*80}")
    print("二、综合判定：是否线性可分？")
    print(f"{'='*80}")

    # 评分系统 (0-100)
    score = 0
    reasons = []

    # SVM 得分（权重 40%）—— 最可靠的指标
    if best_svm >= 0.85:
        svm_score = 40
        reasons.append(f"SVM balanced acc = {best_svm:.1%} (强线性可分)")
    elif best_svm >= 0.75:
        svm_score = 30
        reasons.append(f"SVM balanced acc = {best_svm:.1%} (中等线性可分)")
    elif best_svm >= 0.65:
        svm_score = 15
        reasons.append(f"SVM balanced acc = {best_svm:.1%} (弱线性可分)")
    elif best_svm >= 0.55:
        svm_score = 5
        reasons.append(f"SVM balanced acc = {best_svm:.1%} (接近随机)")
    else:
        svm_score = 0
        reasons.append(f"SVM balanced acc = {best_svm:.1%} (不可分)")
    score += svm_score

    # FDR 得分（权重 30%）
    if best_fdr >= 4.0:
        fdr_score = 30
        reasons.append(f"FDR = {best_fdr:.3f} (强分离)")
    elif best_fdr >= 2.0:
        fdr_score = 22
        reasons.append(f"FDR = {best_fdr:.3f} (可分)")
    elif best_fdr >= 0.5:
        fdr_score = 10
        reasons.append(f"FDR = {best_fdr:.3f} (弱信号)")
    elif best_fdr >= 0.1:
        fdr_score = 3
        reasons.append(f"FDR = {best_fdr:.3f} (微弱)")
    else:
        fdr_score = 0
        reasons.append(f"FDR = {best_fdr:.3f} (混叠)")
    score += fdr_score

    # 余弦间隙得分（权重 15%）
    if best_cos_gap >= 0.02:
        cos_score = 15
        reasons.append(f"余弦间隙 = {best_cos_gap:.4f} (方向可分)")
    elif best_cos_gap >= 0.005:
        cos_score = 8
        reasons.append(f"余弦间隙 = {best_cos_gap:.4f} (微弱方向差异)")
    else:
        cos_score = 0
        reasons.append(f"余弦间隙 = {best_cos_gap:.4f} (方向混叠)")
    score += cos_score

    # 层趋势得分（权重 15%）—— 深层是否优于浅层
    if has_layer_trend is True:
        trend_score = 15
        reasons.append("深层 FDR > 浅层 (符合安全机制理论)")
    elif has_layer_trend is False:
        trend_score = 0
        reasons.append("无层趋势 (信号可能是噪声而非安全机制)")
    else:
        trend_score = 7  # 层数太少无法判断
        reasons.append("层数不足，无法判断趋势")
    score += trend_score

    # 输出最终判定
    print(f"\n  综合得分: {score}/100")
    print(f"  各项得分: SVM={svm_score}/40  FDR={fdr_score}/30  "
          f"余弦={cos_score}/15  层趋势={trend_score}/15")
    print(f"\n  证据链:")
    for i, reason in enumerate(reasons, 1):
        print(f"    {i}. {reason}")

    print(f"\n  {'─'*60}")
    if score >= 70:
        verdict_final = "线性可分"
        print(f"  结论: ✅ 线性可分 (得分 {score}/100)")
        print(f"  → 线性探针应能有效区分 safe/toxic")
        print(f"  → 预期 balanced accuracy: 80%+")
    elif score >= 40:
        verdict_final = "弱线性可分"
        print(f"  结论: ⚠️ 弱线性可分 (得分 {score}/100)")
        print(f"  → 线性探针能学到部分信号，但上限有限")
        print(f"  → 预期 balanced accuracy: 60-75%")
        print(f"  → 建议: 考虑更强信号的数据集或非线性探针")
    else:
        verdict_final = "不可分"
        print(f"  结论: ❌ 不可线性分 (得分 {score}/100)")
        print(f"  → safe 和 toxic 的隐藏态在特征空间中严重混叠")
        print(f"  → 线性探针无法有效区分，balanced accuracy 接近随机")
        print(f"  → 原因: 数据集的 safe/toxic 标签与隐藏态之间缺乏统计相关性")
        print(f"  → 建议: 切换到 attack_enhanced_set 提供更强的可分信号")
    print(f"  {'─'*60}")

    # 保存结果
    save_data = {
        "num_samples": int(len(labels)),
        "n_safe": int((labels == 0).sum()),
        "n_toxic": int((labels == 1).sum()),
        "num_layers": num_layers,
        "hidden_dim": int(hs.shape[2]),
        "verdict": verdict_final,
        "score": score,
        "score_breakdown": {
            "svm": svm_score,
            "fdr": fdr_score,
            "cosine": cos_score,
            "trend": trend_score,
        },
        "summary": {
            "best_fdr_layer": best_fdr_layer,
            "best_fdr": best_fdr,
            "mean_fdr": float(np.mean(fdrs)),
            "best_svm_layer": best_svm_layer,
            "best_svm_acc": best_svm,
            "best_cos_gap_layer": best_cos_layer,
            "best_cos_gap": best_cos_gap,
            "best_cdr_layer": best_cdr_layer,
            "best_cdr": best_cdr,
            "has_layer_trend": has_layer_trend,
            "fdr_separable_layers": sum(1 for f in fdrs if f >= 2.0),
            "fdr_weak_layers": sum(1 for f in fdrs if 0.5 <= f < 2.0),
            "fdr_mixed_layers": sum(1 for f in fdrs if f < 0.5),
        },
        "layers": {str(i): results[i] for i in range(num_layers)},
    }

    report_path = output_dir / "separability_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] 报告: {report_path}")
    if do_viz:
        print(f"[Save] 可视化: {output_dir / 'visualizations'}")

    return save_data


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="隐藏态可分性诊断分析")
    parser.add_argument("--cache", type=Path, default=None,
                        help="隐藏态缓存 .npz 文件路径")
    parser.add_argument("--data_file", type=Path, default=None,
                        help="evaluation JSONL 文件路径（需要 GPU 提取隐藏态）")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="从 JSONL 采样的最大样本数（每类各一半）")
    parser.add_argument("--max_length", type=int, default=512,
                        help="隐藏态提取最大序列长度")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/separability_analysis"),
                        help="输出目录")
    parser.add_argument("--no_viz", action="store_true",
                        help="跳过可视化（仅计算数值指标）")
    parser.add_argument("--max_per_class", type=int, default=500,
                        help="每类最大采样数（控制内存，默认 500 即总共 ~1000 条）。"
                             "设为 0 表示不限制（需要 >10GB 内存）。")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cache and args.cache.exists():
        # 从缓存加载
        print(f"[Cache] 加载隐藏态: {args.cache}")
        hs, labels, num_layers = load_from_cache(args.cache, max_per_class=args.max_per_class)
        mem_mb = hs.nbytes / 1024 / 1024
        print(f"[Cache] shape={hs.shape}, layers={num_layers}, 内存≈{mem_mb:.0f}MB")

    elif args.data_file and args.data_file.exists():
        # 从 JSONL 提取
        print(f"[Data] 从 {args.data_file} 加载...")
        texts, labels = load_from_evaluation(args.data_file, args.max_samples, args.seed)

        print(f"\n[Model] 加载 LLM...")
        from engine.models import ModelManager
        from engine.probes.linear_probe_balanced import extract_hidden_states
        import torch

        tokenizer, model = ModelManager().load_llm()
        device = next(model.parameters()).device

        print(f"[Hidden] 提取隐藏态 (max_length={args.max_length})...")
        hs_list = extract_hidden_states(
            model, tokenizer, texts, device,
            max_length=args.max_length, batch_size=8,
            pooling_method="last_token", desc="分析集"
        )
        hs = np.stack([h if isinstance(h, np.ndarray) else h.cpu().numpy() for h in hs_list])
        num_layers = hs.shape[1]

        # 保存缓存
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = args.output_dir / "analysis_hidden_states.npz"
        np.savez_compressed(
            cache_path,
            train_hs=hs, train_labels=labels,
            val_hs=hs[:0], val_labels=labels[:0],  # empty
            num_layers=np.int32(num_layers),
            hidden_dim=np.int32(hs.shape[2]),
        )
        print(f"[Cache] 已保存: {cache_path}")

    else:
        print("错误: 请提供 --cache 或 --data_file")
        print("  --cache: 已有的 hidden_states_cache.npz")
        print("  --data_file: evaluation JSONL (如 logs/base_evaluation.jsonl)")
        sys.exit(1)

    analyze_all_layers(
        hs, labels, num_layers,
        output_dir=args.output_dir,
        do_viz=not args.no_viz,
    )


if __name__ == "__main__":
    main()
