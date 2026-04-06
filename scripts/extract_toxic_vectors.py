#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务 2.2-b：提取各层毒性向量 w_toxic，整理成向量字典

从 train_probes_balanced.py 训练好的线性探针中提取每一层的毒性向量 w_toxic
（即 softmax(w^T * h + b) 中有害类对应的权重方向），
将所有层的向量合并为统一的 toxic_vectors.npz 字典。

输入来源：
  train_probes_balanced.py 训练产物
    └ layer_{i}/probe.pt   →  通过 LinearProbe.get_toxic_vector() 提取
  train_linear_probe* / 旧管线（与 extract_toxicity_vectors.py 一致）
    └ layerNN/layerNN.pt   →  从 checkpoint 的 weight / linear.weight 等提取 w[1]-w[0]

输出：
  toxic_vectors.npz
    ├── vectors        (num_layers, hidden_dim)  float32  所有层 w_toxic 矩阵
    ├── biases         (num_layers,)             float32  所有层 bias
    ├── layer_indices  (num_layers,)             int32    层索引
    ├── layer_{i}      (hidden_dim,)             float32  第 i 层的 w_toxic
    ├── layer_{i}_bias ()                        float32  第 i 层的 bias
    └── meta           json 字符串               元信息

用法:
  python scripts/extract_toxic_vectors.py
  python scripts/extract_toxic_vectors.py --probes_dir outputs/probes
  python scripts/extract_toxic_vectors.py --output outputs/toxic_vectors/toxic_vectors.npz
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 LinearProbe（仅在有 probe.pt 时需要）
from engine.probes.linear_probe_balanced import load_probe

HAS_PROBE_MODULE = True


# ======================================================================
# 自动搜索探针输出目录
# ======================================================================
SEARCH_PATHS = [
    "outputs/linear_probes/layers",
    "outputs/linear_probes",
    "outputs/probes",
]


def parse_layer_index_from_dirname(name: str) -> Optional[int]:
    """
    从子目录名解析层号。支持 train_probes_balanced 等脚本常见命名：
      layer_1, layer_12, layer01, layer32
    """
    if not name.lower().startswith("layer"):
        return None
    rest = name[5:].lstrip("_")
    if rest.isdigit():
        return int(rest)
    return None


def _dir_has_layer_children(d: Path) -> bool:
    if not d.is_dir():
        return False
    try:
        for child in d.iterdir():
            if child.is_dir() and parse_layer_index_from_dirname(child.name) is not None:
                return True
    except OSError:
        return False
    return False


def find_probes_dir(hint: Optional[Path] = None) -> Path:
    """
    自动搜索包含各层探针子目录的输出目录

    支持的子目录命名：layer_{i}、layer{i}、layer{i:02d}（如 layer_1、layer01）。

    搜索优先级：
      1. 用户指定的 --probes_dir（若其下无层目录，会尝试 .../layers）
      2. outputs/linear_probes/layers、outputs/linear_probes、outputs/probes
    """
    candidates: List[Path] = []
    if hint is not None:
        h = Path(hint).expanduser()
        if not h.is_absolute():
            h = (PROJECT_ROOT / h).resolve()
        else:
            h = h.resolve()
        candidates.append(h)
        layers_sub = h / "layers"
        if layers_sub.is_dir():
            candidates.append(layers_sub.resolve())

    for p in SEARCH_PATHS:
        candidates.append((PROJECT_ROOT / p).resolve())

    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for d in unique_candidates:
        if _dir_has_layer_children(d):
            return d

    raise FileNotFoundError(
        "未找到包含各层探针子目录的输出（期望子目录名为 layer_N、layer_NN 等）。\n"
        f"已搜索: {[str(c) for c in unique_candidates]}\n"
        "请使用 --probes_dir 指定目录（通常为 .../outputs/linear_probes/layers 或 .../outputs/linear_probes）。"
    )


# ======================================================================
# 辅助函数
# ======================================================================
def discover_layers(probes_dir: Path) -> List[int]:
    """发现所有 layer_* / layerNN 子目录并返回排序后的层索引"""
    layers = []
    for d in probes_dir.iterdir():
        if not d.is_dir():
            continue
        idx = parse_layer_index_from_dirname(d.name)
        if idx is not None:
            layers.append(idx)
    return sorted(set(layers))


def find_layer_subdir(probes_dir: Path, layer_idx: int) -> Optional[Path]:
    """按层号定位子目录（兼容 layer_12、layer12、layer01 等命名）"""
    candidates = [
        probes_dir / f"layer_{layer_idx}",
        probes_dir / f"layer_{layer_idx:02d}",
        probes_dir / f"layer{layer_idx:02d}",
        probes_dir / f"layer{layer_idx}",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _bias_toxic_from_checkpoint(checkpoint: dict) -> float:
    """二分类线性层：b_toxic ≈ b[1] - b[0]；若无则 0。"""
    for key in ("linear.bias", "fc.bias"):
        if key not in checkpoint:
            continue
        b = checkpoint[key]
        if hasattr(b, "detach"):
            arr = b.detach().cpu().numpy().reshape(-1)
        else:
            arr = np.asarray(b).reshape(-1)
        if arr.size >= 2:
            return float(arr[1] - arr[0])
        if arr.size == 1:
            return float(arr[0])
    return 0.0


def _toxic_from_legacy_checkpoint(checkpoint: dict) -> Optional[Tuple[np.ndarray, float]]:
    """
    与 scripts/extract_toxicity_vectors.py、engine/neurons/data_loaders 一致：
    支持 toxicity_vector 预计算字段，或 2×d 的 weight / linear.weight / fc.weight。
    """
    if "toxicity_vector" in checkpoint:
        tv = checkpoint["toxicity_vector"]
        if isinstance(tv, torch.Tensor):
            tv = tv.detach().cpu().numpy()
        w = np.asarray(tv, dtype=np.float64).reshape(-1)
        return w, _bias_toxic_from_checkpoint(checkpoint)

    for key in ("linear.weight", "fc.weight", "weight"):
        if key not in checkpoint:
            continue
        w = checkpoint[key]
        if hasattr(w, "detach"):
            w = w.detach().cpu().numpy()
        else:
            w = np.asarray(w)
        if w.ndim == 2 and w.shape[0] == 2:
            toxic = (w[1] - w[0]).astype(np.float64)
            return toxic, _bias_toxic_from_checkpoint(checkpoint)
    return None


def load_layer_toxic_vector(layer_dir: Path) -> Optional[Tuple[np.ndarray, float]]:
    """
    优先 layer_dir/probe.pt（LinearProbe / train_probes_balanced）；
    否则尝试 best.pt、{文件夹名}.pt（旧 linear_probes 管线，如 layer01/layer01.pt）。

    Returns:
        (w_toxic, bias) 或 None
    """
    probe_path = layer_dir / "probe.pt"
    if probe_path.exists():
        try:
            probe, _ = load_probe(layer_dir, dropout=0.0)
            w_toxic, b = probe.get_toxic_vector()
            return w_toxic, b
        except Exception as e:
            print(f"  [Warn] 从 probe.pt 加载失败 ({layer_dir.name}): {e}")

    for name in ("best.pt", f"{layer_dir.name}.pt"):
        p = layer_dir / name
        if not p.exists():
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                continue
            out = _toxic_from_legacy_checkpoint(ckpt)
            if out is not None:
                return out
        except Exception as e:
            print(f"  [Warn] 从 {name} 加载失败 ({layer_dir.name}): {e}")

    return None


# ======================================================================
# 保存毒性向量字典
# ======================================================================
def save_toxic_vectors(
    toxic_vectors: Dict[int, Tuple[np.ndarray, float]],
    output_path: Path,
    probes_dir: Path,
):
    """
    将所有层的 w_toxic 合并保存为 toxic_vectors.npz

    存储格式：
      vectors        (num_layers, hidden_dim)  — 所有层 w_toxic 按层索引排列
      biases         (num_layers,)             — 对应 bias
      layer_indices  (num_layers,)             — 层索引
      layer_{i}      (hidden_dim,)             — 第 i 层独立存储（兼容旧用法）
      layer_{i}_bias ()                        — 第 i 层 bias 独立存储
      meta           json 字符串               — 元信息
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_keys = sorted(toxic_vectors.keys())

    # 各层独立存储
    per_layer = {}
    per_layer_bias = {}
    for idx in sorted_keys:
        w, b = toxic_vectors[idx]
        per_layer[f"layer_{idx}"] = w.astype(np.float32)
        per_layer_bias[f"layer_{idx}_bias"] = np.float32(b)

    # 统一矩阵格式
    all_vectors = np.stack([toxic_vectors[i][0] for i in sorted_keys])  # (N, dim)
    all_biases  = np.array([toxic_vectors[i][1] for i in sorted_keys], dtype=np.float32)
    layer_indices_arr = np.array(sorted_keys, dtype=np.int32)

    hidden_dim = int(all_vectors.shape[1])

    meta = json.dumps({
        "description": "各层毒性向量 w_toxic（线性探针有害类权重方向）",
        "formula":     "P(toxic|h) = softmax(w_toxic^T * h + b)",
        "num_layers":  len(toxic_vectors),
        "hidden_dim":  hidden_dim,
        "layer_indices": sorted_keys,
        "source":      str(probes_dir),
        "generated_at": datetime.now().isoformat(),
    }, ensure_ascii=False)

    np.savez_compressed(
        output_path,
        vectors=all_vectors,
        biases=all_biases,
        layer_indices=layer_indices_arr,
        **per_layer,
        **per_layer_bias,
        meta=np.array([meta]),
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    return hidden_dim, size_mb


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="任务 2.2b：提取各层毒性向量 w_toxic，整理成向量字典",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/extract_toxic_vectors.py
  python scripts/extract_toxic_vectors.py --probes_dir outputs/probes
  python scripts/extract_toxic_vectors.py --output outputs/toxic_vectors/toxic_vectors.npz
""",
    )
    parser.add_argument(
        "--probes_dir", type=Path, default=None,
        help="探针输出目录（含 layer_N / layerNN 子目录，如 outputs/linear_probes/layers）。默认自动搜索。",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="输出 .npz 路径。默认保存在 outputs/toxic_vectors/toxic_vectors.npz。",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("任务 2.2b：提取各层毒性向量 w_toxic")
    print("=" * 60)

    # ---- 定位探针目录 ----
    probes_dir = find_probes_dir(args.probes_dir)
    print(f"\n[Dir] 探针目录: {probes_dir}")

    # ---- 发现所有层 ----
    layer_indices = discover_layers(probes_dir)
    if not layer_indices:
        raise FileNotFoundError(
            f"未在 {probes_dir} 中发现层目录（期望 layer_N、layer_NN 等命名）"
        )
    print(f"[Dir] 发现 {len(layer_indices)} 层: "
          f"{layer_indices[0]} ~ {layer_indices[-1]}")

    # ---- 检查模块可用性 ----
    if not HAS_PROBE_MODULE:
        print("\n[Error] 无法导入 engine.probes.linear_probe_balanced 模块！")
        print("        请确认 engine/probes/linear_probe_balanced.py 存在且可导入。")
        sys.exit(1)

    # ---- 逐层提取毒性向量 ----
    print(f"\n[Extract] 逐层提取 w_toxic...")
    toxic_vectors: Dict[int, Tuple[np.ndarray, float]] = {}

    for i in layer_indices:
        layer_dir = find_layer_subdir(probes_dir, i)
        if layer_dir is None:
            print(f"  Layer {i:2d}: [Skip] 未找到对应子目录")
            continue
        tv = load_layer_toxic_vector(layer_dir)
        if tv is not None:
            w, b = tv
            norm = float(np.linalg.norm(w))
            toxic_vectors[i] = (w, b)
            print(f"  Layer {i:2d}: dim={w.shape[0]}  "
                  f"|w|={norm:.4f}  bias={b:.4f}")
        else:
            print(f"  Layer {i:2d}: [Skip] 无可用权重 (probe.pt / {layer_dir.name}.pt 等)")

    if not toxic_vectors:
        print("\n[Error] 未提取到任何毒性向量!")
        sys.exit(1)

    # ---- 统计 ----
    norms = [float(np.linalg.norm(v[0])) for v in toxic_vectors.values()]
    print(f"\n[Stat] 已提取: {len(toxic_vectors)} 层")
    print(f"       向量维度:     {list(toxic_vectors.values())[0][0].shape[0]}")
    print(f"       L2 范数范围:  {min(norms):.4f} ~ {max(norms):.4f}")
    print(f"       L2 范数均值:  {np.mean(norms):.4f}")

    # ---- 保存 ----
    if args.output:
        output_path = args.output
    else:
        output_path = PROJECT_ROOT / "outputs" / "toxic_vectors" / "toxic_vectors.npz"
    hidden_dim, size_mb = save_toxic_vectors(toxic_vectors, output_path, probes_dir)

    print(f"\n[Save] 毒性向量字典: {output_path} ({size_mb:.1f} MB)")
    print(f"       shape: vectors=({len(toxic_vectors)}, {hidden_dim})")
    print(f"       层索引: {sorted(toxic_vectors.keys())}")

    print(f"\n[Done] 任务 2.2b 完成!")
    print(f"  输出文件: {output_path}")
    print(f"\n使用示例:")
    print(f"  import numpy as np")
    print(f"  d = np.load('{output_path.name}', allow_pickle=True)")
    print(f"  w_toxic_28 = d['layer_28']        # 第28层毒性向量")
    print(f"  all_vectors = d['vectors']         # ({len(toxic_vectors)}, {hidden_dim})")
    print(f"  layer_indices = d['layer_indices']  # 层索引数组")


if __name__ == "__main__":
    main()
