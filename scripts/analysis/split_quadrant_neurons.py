#!/usr/bin/env python3
"""
从四象限分类结果自动拆分 TSFT 所需的神经元文件

根据论文定义，将四象限分类结果拆分为：
    - S-A- 象限神经元  →  专用安全神经元 D(p,q)（标准 TSFT / VA+TSFT 阶段一）
    - S+A- 象限神经元  →  脆弱神经元（VA+TSFT 阶段二，仅在启用了 --vulnerable-only 时使用）

四象限含义：
    - S+A+: 参数对齐为正，激活投影为正  →  毒性特征增强（最危险）
    - S-A+: 参数对齐为负，激活投影为正  →  良性特征抑制
    - S+A-: 参数对齐为正，激活投影为负  →  毒性特征抑制（伪安全，需功能反转）
    - S-A-: 参数对齐为负，激活投影为负  →  良性特征增强（最安全，专用安全神经元）

TSFT 所需格式：
    - dedicated_safety_neurons: 格式为 {f"layer_{layer}_neuron_{neuron}": {...}}
    - vulnerable_neurons: 同上

使用方法：
    # 拆分四象限结果为 S-A- 和 S+A- 两个文件
    python scripts/split_quadrant_neurons.py ^
        --quadrant-results outputs/quadrant_classification/quadrant_classification.json ^
        --output-dir outputs/tsft_neurons

    # 只输出专用安全神经元（标准 TSFT 使用）
    python scripts/split_quadrant_neurons.py ^
        --quadrant-results outputs/quadrant_classification/quadrant_classification.json ^
        --output-dir outputs/tsft_neurons ^
        --safety-only

    # 指定 S-A- 和 S+A- 的额外阈值过滤
    python scripts/split_quadrant_neurons.py ^
        --quadrant-results outputs/quadrant_classification/quadrant_classification.json ^
        --output-dir outputs/tsft_neurons ^
        --safety-alignment-threshold -0.05 ^
        --vulnerable-alignment-threshold 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- 路径处理 ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if _PROJECT_ROOT.exists() and (_PROJECT_ROOT / "engine").exists():
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---- 核心函数 ----

def load_quadrant_results(
    quadrant_file: str,
) -> Dict[Tuple[int, int], Dict]:
    """
    加载四象限分类结果，返回 Dict[(layer, neuron), data]

    支持两种输入格式：
        1. {"layer_5_neuron_1024": {"layer_idx": 5, "neuron_idx": 1024, "quadrant": "S-A-", ...}}
        2. {"layer_5_neuron_1024": {"layer": 5, "neuron": 1024, "quadrant": "S-A-", ...}}
    """
    path = Path(quadrant_file)
    if not path.exists():
        raise FileNotFoundError(f"四象限结果文件不存在: {quadrant_file}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败 ({path}): {e}")

    neurons: Dict[Tuple[int, int], Dict] = {}
    skipped = 0

    for key, val in raw.items():
        if key in ("_statistics", "_metadata", "metadata", "statistics"):
            continue

        if isinstance(val, dict):
            li = val.get("layer_idx") or val.get("layer")
            ni = val.get("neuron_idx") or val.get("neuron")
        else:
            li, ni = None, None

        if li is None or ni is None:
            # 尝试从 key 解析：layer_X_neuron_Y 或 X_Y
            parts = key.split("_")
            if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "neuron":
                try:
                    li = int(parts[1])
                    ni = int(parts[3])
                except (ValueError, IndexError):
                    skipped += 1
                    continue
            elif len(parts) == 2:
                try:
                    li = int(parts[0])
                    ni = int(parts[1])
                except ValueError:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue
        else:
            li = int(li)
            ni = int(ni)

        if li is not None and ni is not None:
            neurons[(li, ni)] = val

    if skipped > 0:
        print(f"[Split] 跳过了 {skipped} 个无法解析的条目")

    print(f"[Split] 加载了 {len(neurons)} 个四象限分类神经元")
    return neurons


def split_by_quadrant(
    neurons: Dict[Tuple[int, int], Dict],
    safety_quadrants: Optional[List[str]] = None,
    vulnerable_quadrants: Optional[List[str]] = None,
    safety_alignment_min: Optional[float] = None,
    safety_alignment_max: Optional[float] = None,
    vulnerable_alignment_min: Optional[float] = None,
    vulnerable_alignment_max: Optional[float] = None,
) -> Tuple[
    Dict[Tuple[int, int], Dict],
    Dict[Tuple[int, int], Dict],
]:
    """
    按象限和可选的 alignment 范围拆分神经元。

    Args:
        neurons: 四象限分类结果
        safety_quadrants: 属于安全神经元的象限列表（默认 ["S-A-"]）
        vulnerable_quadrants: 属于脆弱神经元的象限列表（默认 ["S+A-"]）
        safety_alignment_min/max: 对 S-A- 神经元额外按 alignment 过滤（可选）
        vulnerable_alignment_min/max: 对 S+A- 神经元额外按 alignment 过滤（可选）

    Returns:
        (safety_neurons, vulnerable_neurons)
    """
    if safety_quadrants is None:
        safety_quadrants = ["S-A-"]
    if vulnerable_quadrants is None:
        vulnerable_quadrants = ["S+A-"]

    safety_out: Dict[Tuple[int, int], Dict] = {}
    vulnerable_out: Dict[Tuple[int, int], Dict] = {}

    for (layer, neuron), data in neurons.items():
        quadrant = data.get("quadrant", "")
        alignment = data.get("alignment", data.get("cosine_similarity", 0.0))

        if quadrant in safety_quadrants:
            if safety_alignment_min is not None and alignment < safety_alignment_min:
                continue
            if safety_alignment_max is not None and alignment > safety_alignment_max:
                continue
            safety_out[(layer, neuron)] = data

        if quadrant in vulnerable_quadrants:
            if vulnerable_alignment_min is not None and alignment < vulnerable_alignment_min:
                continue
            if vulnerable_alignment_max is not None and alignment > vulnerable_alignment_max:
                continue
            vulnerable_out[(layer, neuron)] = data

    return safety_out, vulnerable_out


def neurons_to_serializable(
    neurons: Dict[Tuple[int, int], Dict],
    source_file: str,
    quadrant: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    将 (layer, neuron) 键的神经元字典转换为 JSON 可序列化的格式。

    输出格式（兼容 tsft.py 的 load_dedicated_safety_neurons）：
        {
            "layer_L_NEURON": {
                "layer_idx": L, "neuron_idx": NEURON,
                "quadrant": "...", "alignment": ...,
                "activation_projection": ...,
                ...（保留原始字段）
            }
        }
    """
    result: Dict[str, Dict] = {}
    for (layer, neuron), data in neurons.items():
        key = f"layer_{layer}_neuron_{neuron}"
        enriched = {
            "layer_idx": layer,
            "neuron_idx": neuron,
        }
        # 保留原始数据中的所有字段（quadrant, alignment, activation_projection 等）
        enriched.update({k: v for k, v in data.items() if k not in ("layer_idx", "neuron_idx")})
        result[key] = enriched
    return result


def save_neurons_json(
    neurons: Dict[str, Dict],
    output_path: Path,
    source_file: str,
    quadrant: Optional[str] = None,
    extra_meta: Optional[Dict] = None,
    neurons_key: str = "dedicated_safety_neurons",
) -> None:
    """保存神经元为 JSON 文件，格式符合 tsft.py 的 load_dedicated_safety_neurons 要求。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict = {
        "source_file": str(source_file),
        "num_neurons": len(neurons),
        "quadrant": quadrant,
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "metadata": meta,
        neurons_key: neurons,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[Split] 已保存 {len(neurons)} 个神经元 → {output_path}")


def print_summary(
    safety_neurons: Dict,
    vulnerable_neurons: Dict,
    total: int,
) -> None:
    """打印拆分结果摘要。"""
    print("\n" + "=" * 60)
    print("[Split] 拆分结果摘要")
    print("=" * 60)
    print(f"  四象限总神经元数 : {total}")
    print(f"  S-A- 安全神经元 : {len(safety_neurons)}")
    print(f"  S+A- 脆弱神经元 : {len(vulnerable_neurons)}")

    if total > 0:
        print(f"  其他象限神经元  : {total - len(safety_neurons) - len(vulnerable_neurons)}")

    # 按象限统计
    quadrant_counts: Dict[str, int] = {}
    for data in [safety_neurons, vulnerable_neurons]:
        for item in data.values():
            q = item.get("quadrant", "?")
            quadrant_counts[q] = quadrant_counts.get(q, 0) + 1

    if quadrant_counts:
        print("\n  象限分布：")
        for q, c in sorted(quadrant_counts.items()):
            pct = c / total * 100
            tag = ""
            if q == "S-A-":
                tag = " → 专用安全神经元"
            elif q == "S+A-":
                tag = " → 脆弱神经元"
            print(f"    {q}: {c} ({pct:.1f}%){tag}")

    print("=" * 60)


# ---- CLI ----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从四象限分类结果自动拆分 TSFT 所需的神经元文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--quadrant-results",
        type=str,
        required=True,
        help="四象限分类结果文件路径（quadrant_classification.json）",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录（将创建 dedicated_safety_neurons.json 和 vulnerable_neurons.json）",
    )

    parser.add_argument(
        "--safety-only",
        action="store_true",
        help="只输出专用安全神经元（S-A-），不输出脆弱神经元",
    )

    # 额外过滤：按 alignment 值范围精细控制
    parser.add_argument(
        "--safety-alignment-min",
        type=float,
        default=None,
        help="S-A- 神经元 alignment 下限（例如 -0.1，默认不限）",
    )
    parser.add_argument(
        "--safety-alignment-max",
        type=float,
        default=None,
        help="S-A- 神经元 alignment 上限（例如 -0.01，默认不限）",
    )
    parser.add_argument(
        "--vulnerable-alignment-min",
        type=float,
        default=None,
        help="S+A- 神经元 alignment 下限（例如 0.01，默认不限）",
    )
    parser.add_argument(
        "--vulnerable-alignment-max",
        type=float,
        default=None,
        help="S+A- 神经元 alignment 上限（例如 0.5，默认不限）",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[Split] 加载四象限结果: {args.quadrant_results}")
    neurons = load_quadrant_results(args.quadrant_results)

    if not neurons:
        print("[Split] 错误: 四象限结果为空")
        return 1

    # 拆分
    safety_raw, vulnerable_raw = split_by_quadrant(
        neurons,
        safety_quadrants=["S-A-"],
        vulnerable_quadrants=["S+A-"],
        safety_alignment_min=args.safety_alignment_min,
        safety_alignment_max=args.safety_alignment_max,
        vulnerable_alignment_min=args.vulnerable_alignment_min,
        vulnerable_alignment_max=args.vulnerable_alignment_max,
    )

    # 打印摘要
    print_summary(safety_raw, vulnerable_raw, len(neurons))

    output_dir = Path(args.output_dir)

    # 保存 S-A- 专用安全神经元
    if safety_raw:
        safety_json = neurons_to_serializable(safety_raw, args.quadrant_results, quadrant="S-A-")
        save_neurons_json(
            safety_json,
            output_dir / "dedicated_safety_neurons.json",
            args.quadrant_results,
            quadrant="S-A-",
            extra_meta={
                "source": "S-A- quadrant (Benign feature enhancement)",
                "formula": "D(p,q) = S(q) \\ U(p)",
                "description": "专用安全神经元，用于标准 TSFT 和 VA+TSFT 阶段一（正常梯度更新）",
            },
        )
    else:
        print("[Split] 警告: 没有找到 S-A- 安全神经元，跳过保存")

    # 保存 S+A- 脆弱神经元
    if not args.safety_only and vulnerable_raw:
        vulnerable_json = neurons_to_serializable(
            vulnerable_raw, args.quadrant_results, quadrant="S+A-"
        )
        save_neurons_json(
            vulnerable_json,
            output_dir / "vulnerable_neurons.json",
            args.quadrant_results,
            quadrant="S+A-",
            extra_meta={
                "source": "S+A- quadrant (Toxic feature suppression — pseudo-safety)",
                "description": (
                    "脆弱神经元（参数对齐为正但激活时抑制毒性），"
                    "用于 VA+TSFT 阶段二（负梯度反转以真正发挥抑制毒性作用）"
                ),
            },
            neurons_key="vulnerable_neurons",
        )
    elif not args.safety_only:
        print("[Split] 警告: 没有找到 S+A- 脆弱神经元，跳过保存")

    print(f"\n[Split] 完成！所有文件已保存到目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
