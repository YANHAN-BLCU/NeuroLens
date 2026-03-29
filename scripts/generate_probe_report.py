#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务 2.2-ac：探针性能验证 & 输出验证报告

从 train_probes_balanced.py 的训练产物（outputs/probes/）中自动定位并完成：
  a. 验证探针在各层的准确率，确认 15 层后准确率>=90%、28 层达峰值(目标93%)
  c. 输出探针性能验证报告，明确各层有害特征判别能力

输入目录结构（来自 train_probes_balanced.py）：
  outputs/probes/
  ├── hidden_states_cache.npz
  ├── config.json
  ├── summary.json
  ├── training_log.json
  └── layer_{i}/
      ├── probe.pt
      ├── preprocessor.pkl
      ├── metrics.json
      └── training_history.json

输出目录（与训练产物隔离）：
  outputs/probes_reports/<timestamp>/
    ├── validation_report.json
    ├── probe_validation_report.txt
    ├── fig1_best_val_acc_per_layer.png
    ├── fig2_val_acc_curves.png
    ├── fig3_val_acc_heatmap.png
    ├── fig4_loss_curves.png              [NEW] train/val loss 对比
    ├── fig5_roc_auc_per_layer.png        [NEW] ROC-AUC 逐层柱状图
    └── fig6_val_vs_test.png              [NEW] val vs test 一致性

用法:
  python scripts/generate_probe_report.py
  python scripts/generate_probe_report.py --probes_dir outputs/probes
  python scripts/generate_probe_report.py --no_plot
  python scripts/generate_probe_report.py --plot_layers 0 15 28 31
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================================
# 各层目标配置
# ======================================================================
LAYER_TARGETS = {
    "shallow": 0.76,
    "mid":     0.85,
    "deep":    0.90,
    "peak":    0.93,
    "deepest": 0.90,
}


def get_layer_target(layer_idx: int, num_layers: int) -> Tuple[float, str]:
    if layer_idx < 6:
        return 0.76, "shallow"
    if layer_idx < 15:
        return 0.85, "mid"
    peak = 28 if num_layers >= 32 else num_layers - 4
    if layer_idx == peak:
        return 0.93, "peak"
    if layer_idx >= 29:
        return 0.90, "deepest"
    return 0.90, "deep"


# ======================================================================
# 自动搜索探针输出目录
# ======================================================================
SEARCH_PATHS = [
    "outputs/probes",
]


def find_probes_dir(hint: Optional[Path] = None) -> Path:
    candidates = []
    if hint is not None:
        candidates.append(Path(hint))
    for p in SEARCH_PATHS:
        candidates.append(PROJECT_ROOT / p)
    for d in candidates:
        if d.is_dir() and any(d.glob("layer_*")):
            return d.resolve()
    raise FileNotFoundError(
        "No layer_* dirs found.\n"
        f"Searched: {[str(c) for c in candidates]}\n"
        "Use --probes_dir to specify."
    )


# ======================================================================
# 数据加载
# ======================================================================
def discover_layers(probes_dir: Path) -> List[int]:
    layers = []
    for d in probes_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            try:
                layers.append(int(d.name.split("_")[1]))
            except ValueError:
                continue
    return sorted(layers)


def load_layer_metrics(layer_dir: Path) -> Optional[Dict]:
    p = layer_dir / "metrics.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_layer_history(layer_dir: Path) -> Optional[Dict]:
    """加载 training_history.json（train_probes_balanced.py 输出格式）"""
    p = layer_dir / "training_history.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    h: Dict[str, list] = {}
    h["epochs"]         = raw.get("epoch", [])
    h["lr"]             = raw.get("lr", [])
    h["train_loss"]     = raw.get("train_loss", [])
    h["val_loss"]       = raw.get("val_loss", [])
    h["train_acc"]      = raw.get("train_acc", [])
    h["val_acc"]        = raw.get("val_acc", [])
    h["test_acc"]       = raw.get("test_acc", [])
    h["train_safe_acc"] = raw.get("train_safe", [])
    h["val_safe_acc"]   = raw.get("val_safe", [])
    h["test_safe_acc"]  = raw.get("test_safe", [])
    h["train_toxic_acc"]= raw.get("train_toxic", [])
    h["val_toxic_acc"]  = raw.get("val_toxic", [])
    h["test_toxic_acc"] = raw.get("test_toxic", [])
    h["val_balanced_acc"] = raw.get("val_balanced_acc", [])
    h["val_roc_auc"]    = raw.get("val_roc_auc", [])
    h["val_pr_auc"]     = raw.get("val_pr_auc", [])
    return h


# ======================================================================
# 验证逻辑
# ======================================================================
def validate_layer(layer_idx: int, metrics: Dict, num_layers: int) -> Dict:
    target_acc, layer_type = get_layer_target(layer_idx, num_layers)
    val_acc = metrics.get("val_acc")
    primary_acc = val_acc if val_acc is not None else 0.0

    return {
        "layer_idx":       layer_idx,
        "layer_type":      layer_type,
        "target_acc":      target_acc,
        "meets_target":    primary_acc >= target_acc,
        "val_acc":          val_acc,
        "val_balanced_acc": metrics.get("val_balanced_acc"),
        "val_safe_acc":     metrics.get("val_safe_acc"),
        "val_toxic_acc":    metrics.get("val_toxic_acc"),
        "val_roc_auc":      metrics.get("val_roc_auc"),
        "val_pr_auc":       metrics.get("val_pr_auc"),
        "test_acc":         metrics.get("test_acc"),
        "test_safe_acc":    metrics.get("test_safe_acc"),
        "test_toxic_acc":   metrics.get("test_toxic_acc"),
        "train_acc":        metrics.get("train_acc"),
        "train_safe_acc":   metrics.get("train_safe_acc"),
        "train_toxic_acc":  metrics.get("train_toxic_acc"),
    }


# ======================================================================
# (f) 自动诊断建议
# ======================================================================
def generate_diagnostics(validations: List[Dict], histories: Dict[int, Optional[Dict]]) -> List[str]:
    """根据验证结果自动生成改进建议"""
    tips: List[str] = []

    # 1) 有害类准确率过低
    toxic_accs = [v["val_toxic_acc"] for v in validations if v["val_toxic_acc"] is not None]
    if toxic_accs and np.mean(toxic_accs) < 0.50:
        tips.append(
            f"[Imbalance] val_toxic_acc mean={np.mean(toxic_accs):.2%} (< 50%), "
            f"probe is biased toward safe class. "
            f"Recommend: use 1:1 balanced training (train_probes_balanced.py)."
        )

    # 2) 深层未达标
    deep = [v for v in validations if v["layer_idx"] >= 15]
    deep_met = sum(1 for v in deep if v["meets_target"])
    if deep and deep_met / len(deep) < 0.5:
        tips.append(
            f"[DeepLayers] Only {deep_met}/{len(deep)} deep layers (>=15) met target. "
            f"Consider: increase epochs, tune lr, or use larger dataset."
        )

    # 3) Peak layer 28 远未达标
    l28 = next((v for v in validations if v["layer_idx"] == 28), None)
    if l28 and l28["val_acc"] is not None and l28["val_acc"] < 0.90:
        gap = 0.93 - l28["val_acc"]
        tips.append(
            f"[Peak] Layer 28 val_acc={l28['val_acc']:.2%}, gap to 93% target is {gap:.2%}. "
            f"This layer should be the strongest; check data quality and class balance."
        )

    # 4) 过拟合检测: train_acc >> val_acc
    overfit_layers = []
    for v in validations:
        ta = v.get("train_acc")
        va = v.get("val_acc")
        if ta is not None and va is not None and ta - va > 0.15:
            overfit_layers.append(v["layer_idx"])
    if overfit_layers:
        tips.append(
            f"[Overfit] {len(overfit_layers)} layers show train_acc - val_acc > 15%: "
            f"{overfit_layers[:8]}{'...' if len(overfit_layers) > 8 else ''}. "
            f"Consider: increase dropout/weight_decay, reduce epochs, or add more data."
        )

    # 5) val vs test 不一致
    big_gap_layers = []
    for v in validations:
        va = v.get("val_acc")
        ta = v.get("test_acc")
        if va is not None and ta is not None and abs(va - ta) > 0.10:
            big_gap_layers.append((v["layer_idx"], va, ta))
    if big_gap_layers:
        tips.append(
            f"[ValTestGap] {len(big_gap_layers)} layers show |val_acc - test_acc| > 10%: "
            f"{[x[0] for x in big_gap_layers[:8]]}. "
            f"Possible data split issue or high variance."
        )

    # 6) ROC-AUC 低
    rocs = [v["val_roc_auc"] for v in validations if v["val_roc_auc"] is not None]
    if rocs and np.mean(rocs) < 0.65:
        tips.append(
            f"[ROC-AUC] Mean ROC-AUC={np.mean(rocs):.4f} (< 0.65), "
            f"probe discrimination is weak. "
            f"Balanced training + more epochs may help."
        )

    # 7) 训练 loss 未收敛
    non_converged = []
    for i, h in histories.items():
        if h is not None and h["train_loss"] and len(h["train_loss"]) >= 3:
            last3 = h["train_loss"][-3:]
            if last3[-1] > 0.5 and (last3[0] - last3[-1]) < 0.02:
                non_converged.append(i)
    if non_converged:
        tips.append(
            f"[Convergence] {len(non_converged)} layers show train_loss plateau > 0.5: "
            f"{non_converged[:8]}. "
            f"Consider: increase lr, train longer, or check data labels."
        )

    if not tips:
        tips.append("[OK] No major issues detected.")

    return tips


# ======================================================================
# 可视化
# ======================================================================
def _init_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        matplotlib.rcParams["font.sans-serif"] = [font] + matplotlib.rcParams.get("font.sans-serif", [])
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def plot_best_val_acc_per_layer(validations, output_path, num_layers):
    """Fig.1: 各层最佳验证集准确率柱状图"""
    plt = _init_mpl()
    from matplotlib.patches import Patch

    layers = [v["layer_idx"] for v in validations]
    accs   = [v["val_acc"] if v["val_acc"] is not None else 0 for v in validations]
    meets  = [v["meets_target"] for v in validations]
    colors = ["#4CAF50" if m else "#E53935" for m in meets]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(layers, accs, color=colors, width=0.7, edgecolor="white", linewidth=0.5)

    for lo, hi, thr, label, clr in [
        (0, 5, 0.76, "shallow >=76%", "#2196F3"),
        (6, 14, 0.85, "mid >=85%", "#FF9800"),
        (15, 27, 0.90, "deep >=90%", "#9C27B0"),
        (28, 28, 0.93, "peak >=93%", "#F44336"),
        (29, num_layers - 1, 0.90, "deepest >=90%", "#9C27B0"),
    ]:
        ax.hlines(thr, lo - 0.5, hi + 0.5, colors=clr, linestyles="--", linewidth=1.5, label=label)

    best_idx = int(np.argmax(accs))
    ax.annotate(f"best: L{layers[best_idx]}\n{accs[best_idx]:.2%}",
                xy=(layers[best_idx], accs[best_idx]),
                xytext=(layers[best_idx] + 2, accs[best_idx] + 0.02), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#333"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#999"))

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Best Val Accuracy", fontsize=11)
    ax.set_title("Fig.1  Best Validation Accuracy per Layer", fontsize=13, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax2 = ax.twinx(); ax2.set_yticks([])
    ax2.legend(handles=[Patch(fc="#4CAF50", label=f"Met({sum(meets)})"),
                        Patch(fc="#E53935", label=f"Unmet({len(meets)-sum(meets)})")],
               loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Fig.1: {output_path}")


def plot_val_acc_curves(histories, validations, plot_layers, output_path, num_layers):
    """Fig.2: 关键层验证集 acc 训练曲线 (val_acc / val_safe / val_toxic)"""
    plt = _init_mpl()
    available = [l for l in plot_layers if l in histories and histories[l] is not None]
    if not available:
        print("[Plot] Fig.2 skipped: no history"); return

    n = len(available); cols = min(n, 4); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

    for idx, li in enumerate(available):
        r, c = divmod(idx, cols); ax = axes[r][c]; h = histories[li]; ep = h["epochs"]
        if h["val_acc"]:
            ax.plot(ep[:len(h["val_acc"])], h["val_acc"], "o-", color="#1976D2", ms=3, lw=1.5, label="val_acc")
        if h["val_safe_acc"]:
            ax.plot(ep[:len(h["val_safe_acc"])], h["val_safe_acc"], "s--", color="#4CAF50", ms=2.5, lw=1.2, label="val_safe")
        if h["val_toxic_acc"]:
            ax.plot(ep[:len(h["val_toxic_acc"])], h["val_toxic_acc"], "^--", color="#E53935", ms=2.5, lw=1.2, label="val_toxic")
        ta, lt = get_layer_target(li, num_layers)
        ax.axhline(ta, color="#FF9800", ls=":", lw=1.5, label=f"target>={ta:.0%}")
        v = next((v for v in validations if v["layer_idx"] == li), None)
        mk = "[OK]" if v and v["meets_target"] else "[X]"
        ax.set_title(f"Layer {li} ({lt}) {mk}\nbest={v['val_acc']:.2%}" if v and v["val_acc"] else f"Layer {li}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols); axes[r][c].set_visible(False)
    fig.suptitle("Fig.2  Validation Accuracy Curves", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Fig.2: {output_path}")


def plot_val_acc_heatmap(histories, layer_indices, output_path, num_layers):
    """Fig.3: 全层 val_acc 热力图"""
    plt = _init_mpl()
    max_ep = 0; valid = []
    for i in layer_indices:
        if i in histories and histories[i] and histories[i]["val_acc"]:
            max_ep = max(max_ep, len(histories[i]["val_acc"])); valid.append(i)
    if not valid:
        print("[Plot] Fig.3 skipped"); return

    mat = np.full((len(valid), max_ep), np.nan)
    for row, i in enumerate(valid):
        v = histories[i]["val_acc"]; mat[row, :len(v)] = v

    fig, ax = plt.subplots(figsize=(max(10, max_ep*0.6), max(6, len(valid)*0.25)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Layer")
    ax.set_title("Fig.3  Validation Accuracy Heatmap", fontsize=13, fontweight="bold")
    ax.set_xticks(range(max_ep)); ax.set_xticklabels(range(1, max_ep+1), fontsize=8)
    ax.set_yticks(range(len(valid))); ax.set_yticklabels(valid, fontsize=8)
    for bl in [6, 15, 28, 29]:
        if bl in valid:
            ax.axhline(valid.index(bl)-0.5, color="white", lw=1.5, ls="--")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("val_acc")
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Fig.3: {output_path}")


def plot_loss_curves(histories, plot_layers, output_path, num_layers):
    """Fig.4 [NEW]: train/val loss 对比曲线，检测过拟合"""
    plt = _init_mpl()
    available = [l for l in plot_layers if l in histories and histories[l] is not None
                 and histories[l]["train_loss"] and histories[l]["val_loss"]]
    if not available:
        print("[Plot] Fig.4 skipped: no loss data"); return

    n = len(available); cols = min(n, 4); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

    for idx, li in enumerate(available):
        r, c = divmod(idx, cols); ax = axes[r][c]; h = histories[li]; ep = h["epochs"]
        tl = h["train_loss"]; vl = h["val_loss"]
        ax.plot(ep[:len(tl)], tl, "o-", color="#1976D2", ms=3, lw=1.5, label="train_loss")
        ax.plot(ep[:len(vl)], vl, "s-", color="#E53935", ms=3, lw=1.5, label="val_loss")

        # 过拟合 gap 标注
        if tl and vl:
            gap = vl[-1] - tl[-1]
            if gap > 0.1:
                ax.annotate(f"gap={gap:.3f}", xy=(ep[len(vl)-1], vl[-1]),
                            fontsize=8, color="#E53935",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#E53935"))

        _, lt = get_layer_target(li, num_layers)
        ax.set_title(f"Layer {li} ({lt})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols); axes[r][c].set_visible(False)
    fig.suptitle("Fig.4  Train vs Val Loss (Overfit Check)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Fig.4: {output_path}")


def plot_roc_auc_per_layer(validations, output_path, num_layers):
    """Fig.5 [NEW]: ROC-AUC 逐层柱状图"""
    plt = _init_mpl()
    layers = [v["layer_idx"] for v in validations if v["val_roc_auc"] is not None]
    rocs   = [v["val_roc_auc"] for v in validations if v["val_roc_auc"] is not None]
    if not layers:
        print("[Plot] Fig.5 skipped: no ROC-AUC data"); return

    # 颜色按 AUC 值渐变
    colors = ["#4CAF50" if r >= 0.7 else "#FF9800" if r >= 0.6 else "#E53935" for r in rocs]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(layers, rocs, color=colors, width=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#999", ls=":", lw=1, label="random (0.5)")
    ax.axhline(0.7, color="#4CAF50", ls="--", lw=1.2, label="good (0.7)")

    best_idx = int(np.argmax(rocs))
    ax.annotate(f"best: L{layers[best_idx]}\nAUC={rocs[best_idx]:.4f}",
                xy=(layers[best_idx], rocs[best_idx]),
                xytext=(layers[best_idx]+2, rocs[best_idx]+0.03), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#333"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#999"))

    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("Fig.5  ROC-AUC per Layer", fontsize=13, fontweight="bold")
    ax.set_xticks(layers); ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    ax2 = ax.twinx(); ax2.set_yticks([])
    ax2.legend(handles=[Patch(fc="#4CAF50", label=">=0.7"),
                        Patch(fc="#FF9800", label="0.6~0.7"),
                        Patch(fc="#E53935", label="<0.6")],
               loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Fig.5: {output_path}")


def plot_val_vs_test(validations, output_path):
    """Fig.6 [NEW]: val_acc vs test_acc 散点图，检测一致性"""
    plt = _init_mpl()
    pairs = [(v["val_acc"], v["test_acc"], v["layer_idx"])
             for v in validations
             if v["val_acc"] is not None and v["test_acc"] is not None]
    if not pairs:
        print("[Plot] Fig.6 skipped: no val/test pair data"); return

    val_a, test_a, idxs = zip(*pairs)
    val_a, test_a = np.array(val_a), np.array(test_a)
    gaps = np.abs(val_a - test_a)

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(val_a, test_a, c=gaps, cmap="RdYlGn_r", s=50, edgecolors="#333", linewidths=0.5,
                         vmin=0, vmax=0.2)

    # 对角线
    mn, mx = min(val_a.min(), test_a.min()) - 0.02, max(val_a.max(), test_a.max()) + 0.02
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="val=test")

    # 标注偏差大的点
    for va, ta, li in pairs:
        if abs(va - ta) > 0.08:
            ax.annotate(f"L{li}", (va, ta), fontsize=7, ha="center",
                        xytext=(0, 8), textcoords="offset points")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("|val_acc - test_acc|", fontsize=10)
    ax.set_xlabel("Val Accuracy", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Fig.6  Val vs Test Accuracy Consistency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Plot] Fig.6: {output_path}")


# ======================================================================
# 报告生成
# ======================================================================
def generate_validation_report(
    probes_dir, layer_indices, validations, output_dir,
    figure_paths=None, diagnostics=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_layers = len(layer_indices)
    met_layers   = [v["layer_idx"] for v in validations if v["meets_target"]]
    unmet_layers = [v["layer_idx"] for v in validations if not v["meets_target"]]

    # 按层类型汇总
    type_groups: Dict[str, Dict] = {}
    for v in validations:
        lt = v["layer_type"]
        if lt not in type_groups:
            type_groups[lt] = {"layers": [], "met": 0, "total": 0, "accs": []}
        type_groups[lt]["layers"].append(v["layer_idx"])
        type_groups[lt]["total"] += 1
        if v["meets_target"]:
            type_groups[lt]["met"] += 1
        if v["val_acc"] is not None:
            type_groups[lt]["accs"].append(v["val_acc"])

    type_summary = {}
    for lt, info in type_groups.items():
        accs = info["accs"]
        type_summary[lt] = {
            "layers": info["layers"],
            "target": LAYER_TARGETS.get(lt, "N/A"),
            "met":    f"{info['met']}/{info['total']}",
            "range":  f"{min(accs):.2%}~{max(accs):.2%}" if accs else "N/A",
            "mean":   f"{np.mean(accs):.2%}" if accs else "N/A",
        }

    all_val = [(v["layer_idx"], v["val_acc"]) for v in validations if v["val_acc"] is not None]
    best_layer, best_acc = max(all_val, key=lambda x: x[1]) if all_val else (-1, 0.0)
    l28 = next((v for v in validations if v["layer_idx"] == 28), None)
    deep = [v for v in validations if v["layer_idx"] >= 15]
    deep_accs = [v["val_acc"] for v in deep if v["val_acc"] is not None]
    deep_met = sum(1 for v in deep if v["meets_target"])

    # ---- JSON report ----
    report = {
        "report_type":   "Probe Validation Report",
        "task":          "Task 2.2a/c",
        "generated_at":  datetime.now().isoformat(),
        "probes_dir":    str(probes_dir),
        "output_dir":    str(output_dir),
        "formula":       "P(toxic|h) = softmax(w^T * h + b)",
        "num_layers":    num_layers,
        "overall": {
            "met":   len(met_layers),
            "unmet": len(unmet_layers),
            "rate":  f"{len(met_layers)/max(num_layers,1):.0%}",
            "met_layers":   met_layers,
            "unmet_layers": unmet_layers,
        },
        "peak": {
            "best_layer": best_layer,
            "best_acc":   f"{best_acc:.2%}",
            "layer28_acc":  f"{l28['val_acc']:.4f}" if l28 and l28["val_acc"] else None,
            "layer28_met":  l28["meets_target"] if l28 else None,
        },
        "deep_layers": {
            "total": len(deep), "met": deep_met,
            "range": f"{min(deep_accs):.2%}~{max(deep_accs):.2%}" if deep_accs else "N/A",
            "mean":  f"{np.mean(deep_accs):.2%}" if deep_accs else "N/A",
        },
        "by_type":  type_summary,
        "per_layer": {
            str(v["layer_idx"]): {
                "type": v["layer_type"], "target": v["target_acc"], "met": v["meets_target"],
                "val": {k: v[k] for k in ["val_acc","val_balanced_acc","val_safe_acc","val_toxic_acc","val_roc_auc","val_pr_auc"]},
                "test": {k: v[k] for k in ["test_acc","test_safe_acc","test_toxic_acc"]},
                "train": {k: v[k] for k in ["train_acc","train_safe_acc","train_toxic_acc"]},
            } for v in validations
        },
    }
    if diagnostics:
        report["diagnostics"] = diagnostics
    if figure_paths:
        report["figures"] = {k: str(v) for k, v in figure_paths.items()}

    rp = output_dir / "validation_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Save] {rp}")

    # ---- TXT report ----
    L = []
    L.append("=" * 80)
    L.append("  Task 2.2a/c: Probe Validation Report")
    L.append("=" * 80)
    L.append(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  Probes:     {probes_dir}")
    L.append(f"  Output:     {output_dir}")
    L.append(f"  Formula:    P(toxic|h) = softmax(w^T * h + b)")
    L.append(f"  Layers:     {num_layers}")
    L.append("")

    L.append("-" * 80)
    L.append("  [Overall]")
    L.append(f"    Met: {len(met_layers)}/{num_layers} ({len(met_layers)/max(num_layers,1):.0%})")
    L.append(f"    Unmet: {unmet_layers}")
    L.append("")

    L.append("-" * 80)
    L.append("  [By Type]")
    for lt, info in type_summary.items():
        tgt = info['target']
        tgt_str = f">={tgt:.0%}" if isinstance(tgt, (int, float)) else str(tgt)
        L.append(f"    {lt} (L{info['layers'][0]}~{info['layers'][-1]}): "
                 f"target{tgt_str}  met={info['met']}  "
                 f"range={info['range']}  mean={info['mean']}")
    L.append("")

    L.append("-" * 80)
    L.append("  [Peak]")
    L.append(f"    Best: Layer {best_layer} val_acc={best_acc:.2%}")
    if l28:
        mk = "[OK]" if l28["meets_target"] else "[X]"
        val_str = f"{l28['val_acc']:.2%}" if l28.get("val_acc") is not None else "N/A"
        L.append(f"    L28:  {mk} val_acc={val_str} (target>=93%)")
    L.append("")

    L.append("-" * 80)
    L.append("  [Deep (>=15)]")
    L.append(f"    Met: {deep_met}/{len(deep)} ({deep_met/max(len(deep),1):.0%})")
    if deep_accs:
        L.append(f"    Range: {min(deep_accs):.2%}~{max(deep_accs):.2%}  Mean: {np.mean(deep_accs):.2%}")
    L.append("")

    L.append("-" * 80)
    L.append("  [Per-Layer]")
    L.append(f"  {'L':>4} {'Type':>6} {'Tgt':>5} {'Met':>5} {'ValAcc':>8} {'TstAcc':>8} "
             f"{'ToxAcc':>8} {'ROC':>7} {'V-T gap':>8}")
    L.append("  " + "-" * 72)
    for v in validations:
        i = v["layer_idx"]
        mk = " [OK]" if v["meets_target"] else "  [X]"
        va = f"{v['val_acc']:.2%}" if v["val_acc"] is not None else "  N/A "
        ta = f"{v['test_acc']:.2%}" if v["test_acc"] is not None else "  N/A "
        tx = f"{v['val_toxic_acc']:.2%}" if v["val_toxic_acc"] is not None else "  N/A "
        rc = f"{v['val_roc_auc']:.4f}" if v["val_roc_auc"] is not None else " N/A  "
        # val-test gap
        if v["val_acc"] is not None and v["test_acc"] is not None:
            gap = v["val_acc"] - v["test_acc"]
            gs = f"{gap:+.2%}"
        else:
            gs = "   N/A  "
        L.append(f"  {i:4d} {v['layer_type']:>6} >={v['target_acc']:.0%} {mk:>5} "
                 f"{va:>8} {ta:>8} {tx:>8} {rc:>7} {gs:>8}")
    L.append("")

    # 诊断建议
    if diagnostics:
        L.append("-" * 80)
        L.append("  [Diagnostics & Recommendations]")
        for tip in diagnostics:
            L.append(f"    {tip}")
        L.append("")

    if figure_paths:
        L.append("-" * 80)
        L.append("  [Figures]")
        for name, path in figure_paths.items():
            L.append(f"    {name}: {path.name}")
        L.append("")

    L.append("=" * 80)
    L.append("  Report complete.")
    L.append("=" * 80)

    txt = "\n".join(L)
    tp = output_dir / "probe_validation_report.txt"
    with open(tp, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[Save] {tp}")
    print("\n" + txt)
    return rp, tp


# ======================================================================
# main
# ======================================================================
def _auto_select_layers(layer_indices: List[int]) -> List[int]:
    """自动选取代表性层（浅/中/深/峰值/最深各取典型）"""
    sel = []
    shallow = [i for i in layer_indices if i < 6]
    if shallow: sel.append(shallow[0])
    mid = [i for i in layer_indices if 6 <= i < 15]
    if mid: sel.append(mid[len(mid)//2])
    for dl in [15, 21, 27]:
        if dl in layer_indices: sel.append(dl)
    if 28 in layer_indices: sel.append(28)
    deep = [i for i in layer_indices if i >= 29]
    if deep: sel.append(deep[-1])
    return sorted(set(sel))


def main():
    parser = argparse.ArgumentParser(
        description="Task 2.2a/c: Probe Validation & Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_probe_report.py
  python scripts/generate_probe_report.py --probes_dir outputs/probes
  python scripts/generate_probe_report.py --no_plot
  python scripts/generate_probe_report.py --plot_layers 0 15 28 31
""",
    )
    parser.add_argument("--probes_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Override output dir (default: outputs/probes_reports/<timestamp>/)")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--plot_layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--use_all_layers",
        action="store_true",
        help="Treat all found layers as usable even if they do not meet the target thresholds. "
             "Useful when you want to evaluate or deploy probes from every layer regardless of val metrics.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Task 2.2a/c: Probe Validation & Report")
    print("=" * 60)

    # ---- Locate probes ----
    probes_dir = find_probes_dir(args.probes_dir)
    print(f"\n[Dir] Probes: {probes_dir}")

    layer_indices = discover_layers(probes_dir)
    if not layer_indices:
        raise FileNotFoundError(f"No layer_* in {probes_dir}")
    num_layers = max(layer_indices) + 1
    print(f"[Dir] Layers: {len(layer_indices)} ({layer_indices[0]}~{layer_indices[-1]})")

    # ---- (e) 带时间戳的输出目录，隔离到 outputs/probes_reports/ ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "probes_reports" / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Out] Report dir: {output_dir}")

    # ---- Config ----
    cfg_path = probes_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[Cfg] {cfg_path}  model={config.get('model_id', '?')}")

    # ---- Validate ----
    print(f"\n[Validate] Per-layer check...")
    validations = []
    for i in layer_indices:
        m = load_layer_metrics(probes_dir / f"layer_{i}")
        if m is None:
            print(f"  [Skip] L{i}: no metrics.json"); continue
        v = validate_layer(i, m, num_layers)

        # 如果用户指定 --use_all_layers，则强制将该层标记为可用（但保留原始达标信息）
        if args.use_all_layers:
            v["used_override"] = True
            v["_orig_meets_target"] = v["meets_target"]
            v["meets_target"] = True

        validations.append(v)
        mk = "[OK]" if v["meets_target"] else "[X]"
        if v.get("used_override"):
            mk = mk + " (override)"
        vs = f"{v['val_acc']:.2%}" if v["val_acc"] is not None else "N/A"
        print(f"  L{i:2d} [{v['layer_type']:>4}] {mk} val={vs} (>={v['target_acc']:.0%})")

    met_n = sum(1 for v in validations if v["meets_target"])
    print(f"\n[Result] {len(validations)} layers, met: {met_n}/{len(validations)}")

    # ---- Histories ----
    print(f"\n[History] Loading...")
    histories: Dict[int, Optional[Dict]] = {}
    hc = 0
    for i in layer_indices:
        h = load_layer_history(probes_dir / f"layer_{i}")
        histories[i] = h
        if h: hc += 1
    print(f"[History] {hc}/{len(layer_indices)} loaded")

    # ---- (f) Diagnostics ----
    diagnostics = generate_diagnostics(validations, histories)
    print(f"\n[Diag] {len(diagnostics)} findings:")
    for d in diagnostics:
        print(f"  {d}")

    # ---- Plots ----
    fig_paths: Dict[str, Path] = {}
    if not args.no_plot:
        try:
            import matplotlib; matplotlib.use("Agg")
            CAN = True
        except ImportError:
            print("[Warn] matplotlib missing"); CAN = False

        if CAN:
            selected = args.plot_layers if args.plot_layers else _auto_select_layers(layer_indices)
            print(f"\n[Plot] Curve layers: {selected}")

            # Fig.1
            p = output_dir / "fig1_best_val_acc_per_layer.png"
            plot_best_val_acc_per_layer(validations, p, num_layers); fig_paths["fig1"] = p
            # Fig.2
            if hc:
                p = output_dir / "fig2_val_acc_curves.png"
                plot_val_acc_curves(histories, validations, selected, p, num_layers); fig_paths["fig2"] = p
                # Fig.3
                p = output_dir / "fig3_val_acc_heatmap.png"
                plot_val_acc_heatmap(histories, layer_indices, p, num_layers); fig_paths["fig3"] = p
                # (b) Fig.4 Loss
                p = output_dir / "fig4_loss_curves.png"
                plot_loss_curves(histories, selected, p, num_layers); fig_paths["fig4"] = p
            # (c) Fig.5 ROC-AUC
            p = output_dir / "fig5_roc_auc_per_layer.png"
            plot_roc_auc_per_layer(validations, p, num_layers); fig_paths["fig5"] = p
            # (d) Fig.6 Val vs Test
            p = output_dir / "fig6_val_vs_test.png"
            plot_val_vs_test(validations, p); fig_paths["fig6"] = p
    else:
        print("[Plot] Disabled (--no_plot)")

    # ---- Report ----
    rp, tp = generate_validation_report(
        probes_dir, layer_indices, validations, output_dir, fig_paths, diagnostics)

    print(f"\n[Done] Task 2.2a/c complete!")
    print(f"  Output dir:  {output_dir}")
    print(f"  JSON report: {rp.name}")
    print(f"  Text report: {tp.name}")
    for k, p in fig_paths.items():
        print(f"  {k}: {p.name}")


if __name__ == "__main__":
    main()
