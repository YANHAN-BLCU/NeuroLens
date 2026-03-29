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
    "outputs/probes",
]


def find_probes_dir(hint: Optional[Path] = None) -> Path:
    """
    自动搜索包含 layer_* 子目录的探针输出目录

    搜索优先级：
      1. 用户指定的 --probes_dir
      2. 常见输出路径
    """
    candidates = []
    if hint is not None:
        candidates.append(Path(hint))
    for p in SEARCH_PATHS:
        candidates.append(PROJECT_ROOT / p)

    for d in candidates:
        if d.is_dir() and any(d.glob("layer_*")):
            return d.resolve()

    raise FileNotFoundError(
        "未找到包含 layer_* 子目录的探针输出。\n"
        f"已搜索: {[str(c) for c in candidates]}\n"
        "请使用 --probes_dir 指定探针目录。"
    )


# ======================================================================
# 辅助函数
# ======================================================================
def discover_layers(probes_dir: Path) -> List[int]:
    """发现所有 layer_{i} 子目录并返回排序后的层索引"""
    layers = []
    for d in probes_dir.iterdir():
        if d.is_dir() and d.name.startswith("layer_"):
            try:
                idx = int(d.name.split("_")[1])
                layers.append(idx)
            except ValueError:
                continue
    return sorted(layers)


def load_layer_toxic_vector(layer_dir: Path) -> Optional[Tuple[np.ndarray, float]]:
    """
    从 layer_dir/probe.pt 加载 LinearProbe 并提取毒性向量

    仅支持 train_probes_balanced.py 产出的 LinearProbe 格式。

    Returns:
        (w_toxic, bias) 或 None
    """
    probe_path = layer_dir / "probe.pt"
    if not probe_path.exists():
        return None

    try:
        probe, _ = load_probe(layer_dir, dropout=0.0)
        w_toxic, b = probe.get_toxic_vector()
        return w_toxic, b
    except Exception as e:
        print(f"  [Warn] 从 probe.pt 加载失败 ({layer_dir.name}): {e}")
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
        help="探针输出目录（含 layer_* 子目录）。默认自动搜索。",
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
        raise FileNotFoundError(f"未在 {probes_dir} 中发现 layer_* 目录")
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
        layer_dir = probes_dir / f"layer_{i}"
        tv = load_layer_toxic_vector(layer_dir)
        if tv is not None:
            w, b = tv
            norm = float(np.linalg.norm(w))
            toxic_vectors[i] = (w, b)
            print(f"  Layer {i:2d}: dim={w.shape[0]}  "
                  f"|w|={norm:.4f}  bias={b:.4f}")
        else:
            print(f"  Layer {i:2d}: [Skip] 无 probe.pt")

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
