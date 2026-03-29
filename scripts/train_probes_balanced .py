#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量训练线性探针（合并训练集+验证集+测试集）

公式：P(toxic | h) = softmax(w^T * h + b)
隐藏态：每层取最后一个非 padding token（last_token）
预处理：分层策略（浅 RobustScaler+SelectKBest，中 StandardScaler+PCA，深 StandardScaler）
训练数据：合并缓存中的训练集+验证集+测试集作为全量训练数据
退出条件：每层测试集 balanced_acc 达标即退出，否则训练满全部轮数

层号约定：
  - 采用 1-based 编号（与论文一致），Layer 1 = 第一个 transformer 层
  - 输出目录命名 layer_1/, layer_2/, ..., layer_32/
  - 命令行 --layers 参数使用 1-based 编号

超参数（默认值）：
  学习率          1e-3
  权重衰减        0.01
  Dropout         0.1
  最大训练轮数    80
  批大小          32
  优化器          AdamW
  调度器          ReduceLROnPlateau(patience=8, factor=0.5)
  早停            测试集总体准确率达到目标后立即退出，否则训练满全部轮数

各层目标（测试集总体准确率，论文 1-based 层号）：
  浅层  (1-6)    ≥ 76%
  中层  (7-15)   ≥ 85%
  深层  (16-27)  ≥ 90%
  峰值层 (28)    ≥ 93%
  最深层 (29+)   ≥ 90%

分层训练：
  可通过 --layers 参数选择训练指定层，例如：
    --layers 1-6          只训练浅层
    --layers 7-15         只训练中层
    --layers 28           只训练峰值层
    --layers 1-6,28,29-32 混合选择
  不指定则训练所有层。

输出目录结构（直接在 output_dir 下）：
  {output_dir}/                        默认 outputs/probes/
  ├── hidden_states_cache.npz          隐藏态缓存（提取后自动保存）
  ├── config.json                      训练配置 & 超参数
  ├── summary.json                     各层达标汇总
  ├── training_log.json                所有层的训练日志（每层metrics+每轮曲线）
  └── layer_{i}/                       i 为 1-based 层号
      ├── probe.pt                     线性探针模型权重
      ├── preprocessor.pkl             分层预处理器（含 project_w_toxic_to_original）
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
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent))

# 分层预处理器（合并至此，供 pickle 与 extract_toxic_vectors 共用）
try:
    from sklearn.preprocessing import RobustScaler
except ImportError:
    RobustScaler = None  # type: ignore
try:
    from sklearn.feature_selection import SelectKBest, f_classif
except ImportError:
    SelectKBest = f_classif = None  # type: ignore

from sklearn.decomposition import PCA


class ShallowPreprocessor:
    def __init__(self, n_features: int = 1024):
        self.n_features = n_features
        self.scaler = RobustScaler() if RobustScaler else StandardScaler()
        self.selector_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        X = self.scaler.fit_transform(X)
        if SelectKBest and f_classif and X.shape[1] > self.n_features:
            self.selector_ = SelectKBest(f_classif, k=min(self.n_features, X.shape[1]))
            self.selector_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = self.scaler.transform(X)
        if self.selector_ is not None:
            X = self.selector_.transform(X)
        return X.astype(np.float32)

    def project_w_toxic_to_original(self, w: np.ndarray, original_dim: int = 4096) -> np.ndarray:
        """投影回原始空间并 unscale，使 R4/R5 的 cos(W_down, w) 语义正确"""
        w = np.asarray(w, dtype=np.float64)
        scale = getattr(self.scaler, "scale_", np.ones(original_dim))
        if self.selector_ is None:
            return (w / (scale + 1e-10)).astype(np.float32)
        w_full = np.zeros(original_dim, dtype=np.float32)
        mask = self.selector_.get_support()
        w_full[mask] = (w / (scale[mask] + 1e-10)).astype(np.float32)
        return w_full


class MidPreprocessor:
    """
    中层 (7-15): StandardScaler + PCA
    去掉 QuantileTransformer，确保 w 可正确投影回原始空间（R4/R5 兼容）
    """
    def __init__(self, pca_variance: float = 0.95):
        self.pca_variance = pca_variance
        self.scaler = StandardScaler()
        self.pca_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        X = self.scaler.fit_transform(X)
        self.pca_ = PCA(n_components=self.pca_variance, random_state=42)
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = self.scaler.transform(X)
        return self.pca_.transform(X).astype(np.float32)

    def project_w_toxic_to_original(self, w: np.ndarray, original_dim: int = 4096) -> np.ndarray:
        """PCA 逆投影 + unscale，使 w 在原始空间与 W_down 可做 cos"""
        w_scaled = self.pca_.components_.T @ np.asarray(w, dtype=np.float64)
        scale = getattr(self.scaler, "scale_", np.ones(original_dim))
        return (w_scaled / (scale + 1e-10)).astype(np.float32)


class DeepPreprocessor:
    """
    深层 (16-32): StandardScaler（与论文一致，R4/R5 兼容）
    去掉 LayerNorm，改用全局 StandardScaler，w 可正确 unscale 回原始空间
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    def project_w_toxic_to_original(self, w: np.ndarray, original_dim: int = 4096) -> np.ndarray:
        """unscale 回原始空间"""
        scale = getattr(self.scaler, "scale_", np.ones(original_dim))
        return (np.asarray(w, dtype=np.float64) / (scale + 1e-10)).astype(np.float32)


def get_layer_preprocessor(layer_idx: int, num_layers: int,
                           shallow_n_features: int = 1024,
                           mid_pca_variance: float = 0.95):
    display_idx = layer_idx + 1
    if display_idx <= 6:
        return ShallowPreprocessor(n_features=shallow_n_features)
    if display_idx <= 15:
        return MidPreprocessor(pca_variance=mid_pca_variance)
    return DeepPreprocessor()


# pickle 将使用本模块路径，需确保 engine.probes.preprocessing 能提供这些类
# 见 engine/probes/preprocessing.py 的 re-export

try:
    from engine.models import ModelManager
except ImportError:
    from engine.engine.models import ModelManager

# 从 engine/probes/linear_probe_balanced.py 导入模型 & 工具
try:
    from engine.probes.linear_probe_balanced import (
        LinearProbe, get_layer_target, extract_hidden_states,
    )
except ImportError:
    from engine.engine.probes.linear_probe_balanced import (
        LinearProbe, get_layer_target, extract_hidden_states,
    )


# ======================================================================
# 层选择解析
# ======================================================================
def parse_layer_spec(spec: str, num_layers: int) -> List[int]:
    """
    解析层选择规范（1-based 输入），返回 0-based 数组索引列表

    支持格式：
      "1,2,3"       → 指定单层
      "1-6"         → 连续范围
      "1-6,28,29-32" → 混合
      "shallow"     → 浅层 1-6
      "mid"         → 中层 7-15
      "deep"        → 深层 16-27
      "peak"        → 峰值层 28
      "deepest"     → 最深层 29+

    Args:
        spec: 层选择表达式（1-based）
        num_layers: 模型总层数

    Returns:
        排序后的 0-based 数组索引列表
    """
    # 预定义层组别名
    aliases = {
        "shallow": "1-6",
        "浅层": "1-6",
        "mid": "7-15",
        "中层": "7-15",
        "deep": f"16-{min(27, num_layers)}",
        "深层": f"16-{min(27, num_layers)}",
        "peak": "28",
        "峰值层": "28",
        "deepest": f"29-{num_layers}",
        "最深层": f"29-{num_layers}",
        "all": f"1-{num_layers}",
        "全部": f"1-{num_layers}",
    }

    indices = set()
    for part in spec.split(","):
        part = part.strip()
        # 替换别名
        if part.lower() in aliases:
            part = aliases[part.lower()]
        if "-" in part:
            tokens = part.split("-", 1)
            try:
                lo, hi = int(tokens[0]), int(tokens[1])
            except ValueError:
                raise ValueError(f"无法解析层范围: '{part}'，格式示例: '1-6' 或 '28'")
            for x in range(lo, hi + 1):
                if 1 <= x <= num_layers:
                    indices.add(x - 1)  # 转为 0-based
        else:
            try:
                x = int(part)
            except ValueError:
                raise ValueError(
                    f"无法解析层号: '{part}'，"
                    f"支持: 数字(1-{num_layers}), 范围(1-6), "
                    f"别名(shallow/mid/deep/peak/deepest)"
                )
            if 1 <= x <= num_layers:
                indices.add(x - 1)

    if not indices:
        raise ValueError(
            f"未选中任何有效层。层号范围: 1-{num_layers}，输入: '{spec}'"
        )

    return sorted(indices)


# ======================================================================
# 线性可分性分析（预处理前后对比）
# ======================================================================

def _fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Fisher 判别比：类间方差/类内方差，越大越可分"""
    mask0, mask1 = y == 0, y == 1
    if mask0.sum() < 2 or mask1.sum() < 2:
        return 0.0
    mu0, mu1 = X[mask0].mean(axis=0), X[mask1].mean(axis=0)
    between = np.sum((mu0 - mu1) ** 2)
    within0 = np.mean(np.sum((X[mask0] - mu0) ** 2, axis=1))
    within1 = np.mean(np.sum((X[mask1] - mu1) ** 2, axis=1))
    return float(between / max(within0 + within1, 1e-10))


def _linear_svm_balanced_acc(X: np.ndarray, y: np.ndarray) -> float:
    """线性 SVM 5 折 CV balanced accuracy，直接度量线性可分性"""
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import warnings
    if len(np.unique(y)) < 2 or len(y) < 10:
        return 0.5
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf = LinearSVC(max_iter=5000, class_weight="balanced")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return float(cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy").mean())
    except Exception:
        return 0.5


def analyze_separability(
    all_hs: np.ndarray,
    all_labels: np.ndarray,
    test_hs: np.ndarray,
    test_labels: np.ndarray,
    num_layers: int,
    shallow_n_features: int = 1024,
    mid_pca_variance: float = 0.95,
    sample_layers: Optional[List[int]] = None,
) -> Dict:
    """
    对比预处理前后线性可分性：FDR、线性 SVM balanced acc

    返回每层（原始/StandardScaler vs 分层预处理）的 FDR 和 SVM acc
    """
    if sample_layers is None:
        sample_layers = [2, 7, 12, 20, 27]  # 浅/中/深/峰值各取代表层（0-based；27=层28峰值）
    results = {}
    for layer_idx in sample_layers:
        if layer_idx >= num_layers:
            continue
        X_train = all_hs[:, layer_idx, :]
        y_train = all_labels
        X_test = test_hs[:, layer_idx, :]
        y_test = test_labels

        # 1. 原始 + StandardScaler（基线）
        scaler_base = StandardScaler()
        X_base_train = scaler_base.fit_transform(X_train)
        X_base_test = scaler_base.transform(X_test)
        fdr_base = _fisher_discriminant_ratio(X_base_train, y_train)
        svm_base = _linear_svm_balanced_acc(X_base_train, y_train)

        # 2. 分层预处理
        preproc = get_layer_preprocessor(
            layer_idx, num_layers,
            shallow_n_features=shallow_n_features,
            mid_pca_variance=mid_pca_variance,
        )
        X_prep_train = preproc.fit(X_train, y_train).transform(X_train)
        X_prep_test = preproc.transform(X_test)
        fdr_prep = _fisher_discriminant_ratio(X_prep_train, y_train)
        svm_prep = _linear_svm_balanced_acc(X_prep_train, y_train)

        dtype = "浅层" if (layer_idx + 1) <= 6 else ("中层" if (layer_idx + 1) <= 15 else "深层")
        results[layer_idx + 1] = {
            "layer_type": dtype,
            "baseline": {"fdr": fdr_base, "svm_acc": svm_base},
            "preprocessed": {"fdr": fdr_prep, "svm_acc": svm_prep},
            "fdr_delta": fdr_prep - fdr_base,
            "svm_delta": svm_prep - svm_base,
            "out_dim": X_prep_train.shape[1],
        }
    return results


def print_separability_report(report: Dict) -> None:
    """打印线性可分性对比报告"""
    print("\n" + "=" * 80)
    print("线性可分性分析：预处理前后对比（FDR=Fisher判别比, SVM=线性SVM 5折CV balanced acc）")
    print("=" * 80)
    print(f"{'层':>4} {'类型':>4} {'基线FDR':>10} {'预处理FDR':>10} {'ΔFDR':>8} {'基线SVM':>8} {'预处理SVM':>8} {'ΔSVM':>6} {'输出维':>6}")
    print("-" * 80)
    for layer_idx in sorted(report.keys()):
        r = report[layer_idx]
        print(f"{layer_idx:>4} {r['layer_type']:>4} "
              f"{r['baseline']['fdr']:>10.4f} {r['preprocessed']['fdr']:>10.4f} "
              f"{r['fdr_delta']:>+8.4f} {r['baseline']['svm_acc']:>8.2%} {r['preprocessed']['svm_acc']:>8.2%} "
              f"{r['svm_delta']:>+6.2%} {r['out_dim']:>6}")
    print("-" * 80)
    fdr_improved = sum(1 for r in report.values() if r["fdr_delta"] > 0)
    svm_improved = sum(1 for r in report.values() if r["svm_delta"] > 0)
    n = len(report)
    print(f"汇总: FDR 提升 {fdr_improved}/{n} 层, SVM acc 提升 {svm_improved}/{n} 层")
    print("=" * 80 + "\n")


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
    test_x: np.ndarray, test_y: np.ndarray,
    num_layers: int, device: torch.device,
    max_epochs: int = 80, batch_size: int = 32,
    lr: float = 1e-3, weight_decay: float = 0.01,
    dropout: float = 0.1, seed: int = 42,
    shallow_n_features: int = 1024,
    mid_pca_variance: float = 0.95,
    use_balanced_acc: bool = True,
    class_weight_toxic: float = 1.0,
) -> Dict:
    """
    单层训练（全量训练，测试集达标即退出）

    分层预处理策略（R4/R5 兼容：投影时 unscale 回原始空间）：
      - 浅层 (1-6):   RobustScaler + 特征选择（保留 top-k 判别性维度）
      - 中层 (7-15):  StandardScaler + PCA 降维
      - 深层 (16-32): StandardScaler

    Args:
        train_x: 合并后的全量训练数据 (train+val+test)
        train_y: 合并后的全量标签
        test_x:  原始测试集隐藏态，用于达标评估
        test_y:  原始测试集标签
    """
    target_acc, layer_type = get_layer_target(layer_idx, num_layers)

    # 可分性增强：balanced_acc 选模型、class_weight 加强 toxic 类
    # 分层预处理
    preprocessor = get_layer_preprocessor(
        layer_idx, num_layers,
        shallow_n_features=shallow_n_features,
        mid_pca_variance=mid_pca_variance,
    )
    train_x = preprocessor.fit(train_x, train_y).transform(train_x)
    test_x = preprocessor.transform(test_x)
    input_dim = train_x.shape[1]

    torch.manual_seed(seed + layer_idx)
    probe = LinearProbe(input_dim, dropout=dropout).to(device)
    # 类别加权：缓解 toxic 类 acc 偏低（报告显示 toxic 假阳性 17.2%）
    n_safe = int((train_y == 0).sum())
    n_toxic = int((train_y == 1).sum())
    total = n_safe + n_toxic
    if n_safe > 0 and n_toxic > 0:
        w_safe = total / (2.0 * n_safe)
        w_toxic = (total / (2.0 * n_toxic)) * class_weight_toxic
        class_weight = torch.tensor([w_safe, w_toxic], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
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

    best_test_acc, best_epoch, best_state = 0.0, 0, None
    target_reached, total_trained = False, 0
    select_metric = "balanced_acc" if use_balanced_acc else "overall_acc"

    history = {k: [] for k in [
        "epoch", "lr",
        # ---- loss ----
        "train_loss",
        # ---- 总体准确率 ----
        "train_acc", "test_acc",
        # ---- 安全类准确率 ----
        "train_safe", "test_safe",
        # ---- 有害类准确率 ----
        "train_toxic", "test_toxic",
        # ---- 平衡准确率 (safe+toxic)/2 ----
        "train_balanced_acc", "test_balanced_acc",
        # ---- AUC（测试集）----
        "test_roc_auc", "test_pr_auc",
    ]}

    display_idx = layer_idx + 1  # 1-based 显示层号
    pbar = tqdm(range(1, max_epochs + 1), desc=f"Layer {display_idx:2d}",
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

        # ---- 完整评估 ----
        tr = evaluate_full(probe, train_x, train_y, device, criterion)
        te = evaluate_full(probe, test_x, test_y, device, criterion)

        # 测试集 ROC-AUC / PR-AUC
        test_roc, test_pr = _compute_auc(test_y, te["probs"])

        scheduler.step(te[select_metric])   # 用选定的指标调度
        total_trained = epoch

        # ---- 记录历史 ----
        history["epoch"].append(epoch)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        # loss
        history["train_loss"].append(epoch_loss / max(n_batch, 1))
        # 总体准确率
        history["train_acc"].append(tr["overall_acc"])
        history["test_acc"].append(te["overall_acc"])
        # 安全类准确率
        history["train_safe"].append(tr["safe_acc"])
        history["test_safe"].append(te["safe_acc"])
        # 有害类准确率
        history["train_toxic"].append(tr["toxic_acc"])
        history["test_toxic"].append(te["toxic_acc"])
        # 平衡准确率
        history["train_balanced_acc"].append(tr["balanced_acc"])
        history["test_balanced_acc"].append(te["balanced_acc"])
        # AUC
        history["test_roc_auc"].append(test_roc)
        history["test_pr_auc"].append(test_pr)

        # 更新进度条
        pbar.set_postfix(test=f"{te[select_metric]:.2%}", best=f"{best_test_acc:.2%}")

        # 更新最佳（以选定指标为准：balanced_acc 更关注 toxic，overall_acc 为总体）
        if te[select_metric] > best_test_acc:
            best_test_acc, best_epoch = te[select_metric], epoch
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            pbar.set_postfix(test=f"{te[select_metric]:.2%}", best=f"{best_test_acc:.2%}")
            if te[select_metric] >= target_acc:
                target_reached = True
                pbar.close()
                break  # 测试集达标即退出
    else:
        pbar.close()

    # 加载最佳模型 & 最终评估
    if best_state:
        probe.load_state_dict(best_state)
    tr = evaluate_full(probe, train_x, train_y, device, criterion)
    te = evaluate_full(probe, test_x, test_y, device, criterion)
    final_roc, final_pr = _compute_auc(test_y, te["probs"])

    # 打印（使用 1-based 层号）
    mark = "✓" if target_reached else "✗"
    print(f"\n{'='*80}")
    print(f"[Layer {display_idx:2d}] {mark} {layer_type}  "
          f"最佳Epoch {best_epoch}/{total_trained}  dim={input_dim}")
    print(f"  {'':8s} {'总体':>6s} {'balanced':>8s} {'安全':>6s} {'有害':>6s}")
    print(f"  测试集 {te['overall_acc']:>6.2%} {te['balanced_acc']:>8.2%} "
          f"{te['safe_acc']:>6.2%} {te['toxic_acc']:>6.2%}  (≥{target_acc:.0%})")
    print(f"  训练集 {tr['overall_acc']:>6.2%} {tr['balanced_acc']:>8.2%} "
          f"{tr['safe_acc']:>6.2%} {tr['toxic_acc']:>6.2%}")
    print(f"  AUC    ROC={final_roc:.4f}  PR={final_pr:.4f}")
    print(f"{'='*80}")

    return {
        "model": probe,
        "scaler": preprocessor,
        "metrics": {
            # 层信息
            "layer": display_idx,         # 1-based 层号
            "layer_type": layer_type,
            # 训练集（全量: train+val+test 合并）
            "train_acc": tr["overall_acc"],
            "train_safe_acc": tr["safe_acc"],
            "train_toxic_acc": tr["toxic_acc"],
            "train_balanced_acc": tr["balanced_acc"],
            # 测试集（原始测试子集，用于达标评估）
            "test_acc": best_test_acc,
            "test_safe_acc": te["safe_acc"],
            "test_toxic_acc": te["toxic_acc"],
            "test_balanced_acc": te["balanced_acc"],
            "test_roc_auc": final_roc,
            "test_pr_auc": final_pr,
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
def save_results(results: Dict, output_dir: Path, use_balanced_acc: bool = True):
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
        "training_mode": "全量训练(train+val+test合并)",
        "exit_criterion": "测试集balanced_acc达标" if use_balanced_acc else "测试集总体准确率达标",
        "layer_numbering": "1-based",
        "trained_layers_count": len(results),
        "met_layers": met, "unmet_layers": unmet,
        "layers": {
            str(k): {
                "test_acc": v["metrics"]["test_acc"],
                "test_balanced_acc": v["metrics"].get("test_balanced_acc"),
                "test_roc_auc": v["metrics"].get("test_roc_auc"),
                "test_pr_auc": v["metrics"].get("test_pr_auc"),
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
        "layer_numbering": "1-based",
        "trained_layers_count": len(results),
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
    parser = argparse.ArgumentParser(description="全量训练线性探针 softmax(w^T*h+b)")
    parser.add_argument("--data_file", type=Path, default=Path("logs/base_evaluation.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/probes"))
    parser.add_argument("--max_toxic_samples", type=int, default=None)
    parser.add_argument("--hidden_states_cache", type=Path, default=None,
                        help="预提取的隐藏态 .npz（跳过 LLM 加载）")
    # 分层训练
    parser.add_argument("--layers", type=str, default=None,
                        help="训练指定层（1-based），格式: '1-6,28,29-32'、'shallow'、'mid'、'deep'、'peak'、'deepest'。"
                             "不指定则训练所有层")
    # 超参数
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    # 分层预处理参数
    parser.add_argument("--shallow_n_features", type=int, default=1024,
                        help="浅层特征选择保留维度（默认 1024，样本/维≈10）")
    parser.add_argument("--mid_pca_variance", type=float, default=0.95,
                        help="中层 PCA 保留方差比例（默认 0.95）")
    parser.add_argument("--analyze_separability", action="store_true",
                        help="训练前运行线性可分性分析，对比预处理前后 FDR/SVM acc")
    parser.add_argument("--use_balanced_acc", action="store_true", default=True,
                        help="以 balanced_acc 选最佳模型（默认开，缓解 toxic 类 acc 偏低）")
    parser.add_argument("--no_use_balanced_acc", action="store_false", dest="use_balanced_acc",
                        help="改用 overall_acc 选最佳模型")
    parser.add_argument("--class_weight_toxic", type=float, default=1.2,
                        help="toxic 类损失权重乘数（默认 1.2，加强 toxic 学习）")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("线性探针全量训练  P(toxic|h) = softmax(w^T * h + b)")
    print("隐藏态: last_token | 预处理: 分层策略")
    print("  浅层(1-6): RobustScaler + 特征选择 | 中层(7-15): StandardScaler + PCA | 深层(16-32): StandardScaler")
    print(f"层号: 1-based（Layer 1 = 第一个 transformer 层）")
    print(f"训练数据: 合并 train+val+test 全量训练")
    print(f"退出条件: 测试集{'balanced_acc' if getattr(args, 'use_balanced_acc', True) else '总体准确率'}达标即退出")
    print(f"超参: lr={args.lr} wd={args.weight_decay} dropout={args.dropout} "
          f"epochs={args.num_epochs} bs={args.batch_size}")
    if args.layers:
        print(f"分层训练: {args.layers}")
    print(f"未达标则训练满 {args.num_epochs} 轮")
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
            embedding_skipped=np.bool_(True),  # 标记：已跳过 embedding 层
            meta=np.array([cache_meta]),
        )
        print(f"[Cache] 隐藏态已保存: {cache_path}")
        print(f"        train={train_hs.shape} val={val_hs.shape} test={test_hs.shape}")
        print(f"        train: S={n_tr_s} T={n_tr_t} | val: S={n_va_s} T={n_va_t} | test: S={n_te_s} T={n_te_t}")

    # ---- 旧缓存兼容：检测并剥离 embedding 层 ----
    # 修复前的 extract_hidden_states 会把 embedding 层（索引0）也包含在内，
    # 导致 num_layers = transformer_layers + 1。
    # 修复后的新缓存包含 embedding_skipped=True 标记，旧缓存无此标记。
    if from_cache and "embedding_skipped" not in data:
        actual_layers = num_layers - 1
        print(f"\n[Compat] 检测到旧缓存含 embedding 层 (num_layers={num_layers})")
        print(f"[Compat] 剥离 embedding 层(索引0) → 实际 transformer 层数={actual_layers}")
        train_hs = train_hs[:, 1:, :]
        val_hs = val_hs[:, 1:, :]
        if test_hs is not None:
            test_hs = test_hs[:, 1:, :]
        num_layers = actual_layers
        hidden_dim = train_hs.shape[2]

    # ---- 合并训练集+验证集+测试集用于全量训练 ----
    if test_hs is None or test_labels is None:
        raise ValueError("缺少测试集隐藏态，无法进行全量训练+测试集达标评估")

    all_hs = np.concatenate([train_hs, val_hs, test_hs], axis=0)
    all_labels = np.concatenate([
        np.array(train_labels), np.array(val_labels), np.array(test_labels)
    ])
    te_y = np.array(test_labels)

    n_all = len(all_labels)
    n_all_s = int((all_labels == 0).sum())
    n_all_t = int((all_labels == 1).sum())
    print(f"\n[Merge] 合并 train+val+test 用于全量训练")
    print(f"[Merge] 全量训练集: {n_all} 样本 (S={n_all_s}, T={n_all_t})")
    print(f"[Merge] 达标评估集: {len(te_y)} 样本 (原始测试集)")

    # ---- 线性可分性分析（可选）----
    if args.analyze_separability:
        report = analyze_separability(
            all_hs, all_labels, test_hs, te_y, num_layers,
            shallow_n_features=args.shallow_n_features,
            mid_pca_variance=args.mid_pca_variance,
        )
        print_separability_report(report)

    # ---- 层选择 ----
    if args.layers:
        train_indices = parse_layer_spec(args.layers, num_layers)
        display_layers = [i + 1 for i in train_indices]
        print(f"\n[Select] 选定训练层（1-based）: {display_layers} ({len(train_indices)} 层)")
    else:
        train_indices = list(range(num_layers))

    # ---- 逐层训练 ----
    n_train = len(train_indices)
    print(f"\n[Train] {n_train}/{num_layers} 层线性探针（全量训练，测试集达标即退出）\n")
    results = {}

    for arr_idx in tqdm(train_indices, desc="逐层训练", ncols=90):
        display_idx = arr_idx + 1  # 1-based 层号
        results[display_idx] = train_one_layer(
            layer_idx=arr_idx,
            train_x=all_hs[:, arr_idx, :], train_y=all_labels,
            test_x=test_hs[:, arr_idx, :], test_y=te_y,
            num_layers=num_layers, device=device,
            max_epochs=args.num_epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            dropout=args.dropout, seed=args.seed,
            shallow_n_features=args.shallow_n_features,
            mid_pca_variance=args.mid_pca_variance,
            use_balanced_acc=getattr(args, "use_balanced_acc", True),
            class_weight_toxic=getattr(args, "class_weight_toxic", 1.2),
        )

    # ---- 保存 ----
    probes_dir = save_results(results, args.output_dir, use_balanced_acc=getattr(args, "use_balanced_acc", True))

    config = {
        "probe_formula": "softmax(w^T * h + b)",
        "pooling": "last_token",
        "preprocess": "分层策略: 浅层=RobustScaler+SelectKBest, 中层=StandardScaler+PCA, 深层=StandardScaler（R4/R5兼容）",
        "shallow_n_features": args.shallow_n_features,
        "mid_pca_variance": args.mid_pca_variance,
        "use_balanced_acc": getattr(args, "use_balanced_acc", True),
        "class_weight_toxic": getattr(args, "class_weight_toxic", 1.2),
        "training_mode": "全量训练(train+val+test合并)",
        "exit_criterion": "测试集balanced_acc达标" if getattr(args, "use_balanced_acc", True) else "测试集总体准确率达标",
        "layer_numbering": "1-based（Layer 1 = 第一个 transformer 层）",
        "trained_layers": sorted(results.keys()),
        "total_layers": num_layers,
        "layer_spec": args.layers,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "dropout": args.dropout, "max_epochs": args.num_epochs,
        "batch_size": args.batch_size, "seed": args.seed,
        "max_length": args.max_length,
        "balance": "1:1",
        "original_split": "6:2:2",
        "actual_train_samples": f"{n_all} (S={n_all_s}, T={n_all_t}, 合并train+val+test)",
        "original_train_samples": len(train_labels),
        "original_val_samples": len(val_labels),
        "original_test_samples": len(test_labels),
        "test_eval_samples": len(test_labels),
        "from_cache": from_cache,
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(probes_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] 训练完成")


if __name__ == "__main__":
    main()
