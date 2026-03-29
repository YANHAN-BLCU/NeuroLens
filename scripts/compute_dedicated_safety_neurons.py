"""
计算专用安全神经元集合：从安全神经元集合中移除效用神经元

根据论文定义，专用安全神经元集合 D(p,q) 的计算公式为：
    D(p,q) = S(q) \\ U(p)
    
其中：
    - S(q) = {i | I_i^s is top q% in I^s}
      安全神经元集合：基于安全参考集（benign-response prompts）计算的重要性分数 I_i^s，
      选择前 q% 最重要的神经元
      
    - U(p) = {i | I_i^u is top p% in I^u}
      效用神经元集合：基于通用任务参考集（Alpaca 数据集）计算的重要性分数 I_i^u，
      选择前 p% 最重要的神经元
      
    - D(p,q) = S(q) \\ U(p)
      专用安全神经元集合：从安全神经元集合中移除效用神经元后的结果，
      即专门负责安全功能而不参与通用语言生成的神经元

流程说明：
1. 加载安全神经元集合 S(q)（从安全参考集计算，包含 top q% 的神经元）
2. 加载效用神经元集合 U(p)（从通用任务参考集计算，包含 top p% 的神经元）
3. 计算差集：D(p,q) = S(q) \\ U(p)
4. 保存专用安全神经元集合 D(p,q)

示例用法：
    # 方式1: 从筛选后的神经元文件中计算
    python scripts/compute_dedicated_safety_neurons.py \
        --safety_neurons_path outputs/safety_neurons_salad.json \
        --utility_neurons_path outputs/utility_neurons.json \
        --output_path outputs/dedicated_safety_neurons.json
    
    # 方式2: 从所有神经元分数文件中，使用阈值计算
    python scripts/compute_dedicated_safety_neurons.py \
        --safety_all_neurons_path outputs/safety_all_neurons_scores.json \
        --utility_all_neurons_path outputs/utility_all_neurons_scores.json \
        --safety_threshold_q 0.005 \
        --utility_threshold_p 0.001 \
        --output_path outputs/dedicated_safety_neurons.json
    
    # 方式2b: 从所有神经元分数文件中，不指定阈值（使用所有神经元）
    python scripts/compute_dedicated_safety_neurons.py \
        --safety_all_neurons_path outputs/safety_all_neurons_scores.json \
        --utility_all_neurons_path outputs/utility_all_neurons_scores.json \
        --output_path outputs/all_dedicated_safety_neurons.json
    
    # 方式3: 混合方式（安全神经元从筛选文件，效用神经元从所有分数文件）
    python scripts/compute_dedicated_safety_neurons.py \
        --safety_neurons_path outputs/safety_neurons_salad.json \
        --utility_all_neurons_path outputs/utility_all_neurons_scores.json \
        --utility_threshold_p 0.001 \
        --output_path outputs/dedicated_safety_neurons.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set, Tuple, Union


def load_neurons_from_file(file_path: str, key: str = None) -> Dict[str, Dict]:
    """
    从 JSON 文件中加载神经元
    
    Args:
        file_path: JSON 文件路径
        key: 要加载的键名（"safety_neurons", "utility_neurons", "all_neurons"）
             如果为 None，则自动检测
    
    Returns:
        神经元字典，格式为 {f"{layer}_{neuron}": {layer, neuron, score, rank, percentile, ...}}
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败 ({file_path}): {e}")
    
    # 自动检测键名
    if key is None:
        if "safety_neurons" in data:
            key = "safety_neurons"
        elif "utility_neurons" in data:
            key = "utility_neurons"
        elif "all_neurons" in data:
            key = "all_neurons"
        else:
            raise ValueError(f"无法自动检测键名，文件应包含 'safety_neurons', 'utility_neurons' 或 'all_neurons' 字段")
    
    # 如果指定的 key 不存在，尝试自动检测
    if key not in data:
        print(f"警告: 文件中不包含 '{key}' 字段，尝试自动检测...")
        if "safety_neurons" in data:
            key = "safety_neurons"
            print(f"  自动检测到 'safety_neurons' 字段")
        elif "utility_neurons" in data:
            key = "utility_neurons"
            print(f"  自动检测到 'utility_neurons' 字段")
        elif "all_neurons" in data:
            key = "all_neurons"
            print(f"  自动检测到 'all_neurons' 字段")
        else:
            raise ValueError(f"文件中不包含 '{key}' 字段，且无法自动检测")
    
    return data[key]


def select_neurons_by_threshold(
    all_neurons: Dict[str, Dict],
    threshold: float,
) -> Dict[str, Dict]:
    """
    从所有神经元中选择前 threshold% 的神经元
    
    Args:
        all_neurons: 所有神经元的字典
        threshold: 阈值（例如 0.005 表示前 0.5%）
    
    Returns:
        筛选后的神经元字典
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold 应该在 0~1 之间，例如 0.005 表示 0.5%，当前为 {threshold}")
    
    # 按分数排序
    sorted_neurons = sorted(
        all_neurons.items(),
        key=lambda x: x[1].get("score", 0),
        reverse=True,
    )
    
    total_neurons = len(sorted_neurons)
    num_selected = max(1, int(total_neurons * threshold))
    
    # 选择前 threshold% 的神经元
    selected_neurons = dict(sorted_neurons[:num_selected])
    
    return selected_neurons


def get_neuron_key_set(neurons: Dict[str, Dict]) -> Set[Tuple[int, int]]:
    """
    从神经元字典中提取 (layer, neuron) 元组集合
    
    Args:
        neurons: 神经元字典，格式为 {f"{layer}_{neuron}": {...}} 或 {(layer, neuron): {...}}
    
    Returns:
        (layer, neuron) 元组集合
    """
    neuron_set = set()
    for key, info in neurons.items():
        if isinstance(key, tuple):
            # 已经是 (layer, neuron) 格式
            neuron_set.add(key)
        else:
            # 字符串格式 "layer_neuron"
            if "layer" in info and "neuron" in info:
                neuron_set.add((info["layer"], info["neuron"]))
            else:
                # 尝试从键名解析
                parts = key.split("_")
                if len(parts) >= 2:
                    try:
                        layer = int(parts[0])
                        neuron = int(parts[1])
                        neuron_set.add((layer, neuron))
                    except ValueError:
                        continue
    return neuron_set


def compute_dedicated_safety_neurons(
    safety_neurons: Dict[str, Dict],
    utility_neurons: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], Dict[str, Union[int, float]]]:
    """
    计算专用安全神经元集合 D(p,q) = S(q) \\ U(p)
    
    根据论文定义，从安全神经元集合 S(q) 中移除效用神经元集合 U(p)，
    得到专用安全神经元集合 D(p,q)。
    
    对应论文中的定义：
        - S(q) = {i | I_i^s is top q% in I^s}
          安全神经元集合，基于安全参考集计算的重要性分数
        - U(p) = {i | I_i^u is top p% in I^u}
          效用神经元集合，基于通用任务参考集计算的重要性分数
        - D(p,q) = S(q) \\ U(p)
          专用安全神经元集合，即安全神经元中不属于效用神经元的神经元
    
    Args:
        safety_neurons: 安全神经元集合 S(q)，格式为 {f"{layer}_{neuron}": {layer, neuron, score, rank, percentile, ...}}
        utility_neurons: 效用神经元集合 U(p)，格式为 {f"{layer}_{neuron}": {layer, neuron, score, rank, percentile, ...}}
    
    Returns:
        Tuple[dedicated_neurons, stats]:
            - dedicated_neurons: 专用安全神经元集合 D(p,q)，格式与输入相同
            - stats: 统计信息字典，包含 num_safety, num_utility, num_overlap, num_dedicated 等
    """
    # 获取神经元键集合
    safety_set = get_neuron_key_set(safety_neurons)
    utility_set = get_neuron_key_set(utility_neurons)
    
    # 计算差集：D(p,q) = S(q) \\ U(p)
    dedicated_set = safety_set - utility_set
    
    # 计算统计信息
    overlap_set = safety_set & utility_set
    overlap_count = len(overlap_set)
    overlap_ratio = overlap_count / len(safety_set) if len(safety_set) > 0 else 0.0
    dedicated_ratio = len(dedicated_set) / len(safety_set) if len(safety_set) > 0 else 0.0
    
    stats = {
        "num_safety_neurons": len(safety_set),
        "num_utility_neurons": len(utility_set),
        "num_overlap_neurons": overlap_count,
        "overlap_ratio": overlap_ratio,
        "num_dedicated_safety_neurons": len(dedicated_set),
        "dedicated_ratio": dedicated_ratio,
    }
    
    print(f"\n=== 专用安全神经元计算统计 ===")
    print(f"安全神经元集合 S(q) 大小: {stats['num_safety_neurons']}")
    print(f"效用神经元集合 U(p) 大小: {stats['num_utility_neurons']}")
    print(f"重叠神经元数量 |S(q) ∩ U(p)|: {stats['num_overlap_neurons']}")
    print(f"重叠比例: {stats['overlap_ratio']*100:.2f}%")
    print(f"专用安全神经元集合 D(p,q) = S(q) \\ U(p) 大小: {stats['num_dedicated_safety_neurons']}")
    print(f"专用安全神经元占比: {stats['dedicated_ratio']*100:.2f}%")
    
    # 构建专用安全神经元字典
    dedicated_neurons = {}
    for key, info in safety_neurons.items():
        if isinstance(key, tuple):
            neuron_key = key
        else:
            # 从 info 中获取 layer 和 neuron
            if "layer" in info and "neuron" in info:
                neuron_key = (info["layer"], info["neuron"])
            else:
                # 尝试从键名解析
                parts = key.split("_")
                if len(parts) >= 2:
                    try:
                        neuron_key = (int(parts[0]), int(parts[1]))
                    except ValueError:
                        continue
                else:
                    continue
        
        if neuron_key in dedicated_set:
            # 使用原始键名或生成新键名
            if isinstance(key, str):
                dedicated_neurons[key] = info
            else:
                dedicated_neurons[f"{neuron_key[0]}_{neuron_key[1]}"] = info
    
    return dedicated_neurons, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算专用安全神经元集合：从安全神经元集合中移除效用神经元"
    )
    
    # 安全神经元输入选项（二选一）
    safety_group = parser.add_mutually_exclusive_group(required=True)
    safety_group.add_argument(
        "--safety_neurons_path",
        type=str,
        help="安全神经元文件路径（已筛选的神经元，包含 'safety_neurons' 字段）",
    )
    safety_group.add_argument(
        "--safety_all_neurons_path",
        type=str,
        help="安全神经元所有分数文件路径（包含 'all_neurons' 字段）",
    )
    
    # 效用神经元输入选项（二选一）
    utility_group = parser.add_mutually_exclusive_group(required=True)
    utility_group.add_argument(
        "--utility_neurons_path",
        type=str,
        help="效用神经元文件路径（已筛选的神经元，包含 'utility_neurons' 字段）",
    )
    utility_group.add_argument(
        "--utility_all_neurons_path",
        type=str,
        help="效用神经元所有分数文件路径（包含 'all_neurons' 字段）",
    )
    
    # 阈值参数（当使用 all_neurons_path 时可选，如果不指定则使用所有神经元）
    parser.add_argument(
        "--safety_threshold_q",
        type=float,
        default=None,
        help="安全阈值 q（例如 0.5%% = 0.005），当使用 --safety_all_neurons_path 时可选。如果不指定，则使用所有神经元",
    )
    parser.add_argument(
        "--utility_threshold_p",
        type=float,
        default=None,
        help="效用阈值 p（例如 0.1%% = 0.001），当使用 --utility_all_neurons_path 时可选。如果不指定，则使用所有神经元",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出文件路径（JSON 格式）",
    )
    
    args = parser.parse_args()
    
    # 加载安全神经元
    print("加载安全神经元...")
    if args.safety_neurons_path:
        # 检查文件是否包含 all_neurons 而不是 safety_neurons
        try:
            with open(args.safety_neurons_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败 ({args.safety_neurons_path}): {e}")
        if "all_neurons" in data and "safety_neurons" not in data:
                raise ValueError(
                    f"错误: 文件 {args.safety_neurons_path} 包含 'all_neurons' 字段而不是 'safety_neurons' 字段。\n"
                    f"如果文件包含所有神经元，请使用 --safety_all_neurons_path 参数（可选的 --safety_threshold_q 参数用于筛选）。\n"
                    f"例如: --safety_all_neurons_path {args.safety_neurons_path} --safety_threshold_q 0.005\n"
                    f"或者: --safety_all_neurons_path {args.safety_neurons_path}（使用所有神经元）"
                )
        safety_neurons = load_neurons_from_file(args.safety_neurons_path, None)  # 自动检测
        print(f"从文件加载了 {len(safety_neurons)} 个安全神经元")
    else:
        all_safety_neurons = load_neurons_from_file(args.safety_all_neurons_path, None)  # 自动检测
        if args.safety_threshold_q is None:
            # 不指定阈值，使用所有神经元
            safety_neurons = all_safety_neurons
            print(f"使用所有 {len(safety_neurons)} 个安全神经元（未指定阈值）")
        else:
            safety_neurons = select_neurons_by_threshold(all_safety_neurons, args.safety_threshold_q)
            print(f"从所有神经元中选择了 {len(safety_neurons)} 个安全神经元 (top {args.safety_threshold_q*100:.2f}%)")
    
    # 加载效用神经元
    print("加载效用神经元...")
    if args.utility_neurons_path:
        # 检查文件是否包含 all_neurons 而不是 utility_neurons
        try:
            with open(args.utility_neurons_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败 ({args.utility_neurons_path}): {e}")
        if "all_neurons" in data and "utility_neurons" not in data:
                raise ValueError(
                    f"错误: 文件 {args.utility_neurons_path} 包含 'all_neurons' 字段而不是 'utility_neurons' 字段。\n"
                    f"如果文件包含所有神经元，请使用 --utility_all_neurons_path 参数（可选的 --utility_threshold_p 参数用于筛选）。\n"
                    f"例如: --utility_all_neurons_path {args.utility_neurons_path} --utility_threshold_p 0.001\n"
                    f"或者: --utility_all_neurons_path {args.utility_neurons_path}（使用所有神经元）"
                )
        utility_neurons = load_neurons_from_file(args.utility_neurons_path, None)  # 自动检测
        print(f"从文件加载了 {len(utility_neurons)} 个效用神经元")
    else:
        all_utility_neurons = load_neurons_from_file(args.utility_all_neurons_path, None)  # 自动检测
        if args.utility_threshold_p is None:
            # 不指定阈值，使用所有神经元
            utility_neurons = all_utility_neurons
            print(f"使用所有 {len(utility_neurons)} 个效用神经元（未指定阈值）")
        else:
            utility_neurons = select_neurons_by_threshold(all_utility_neurons, args.utility_threshold_p)
            print(f"从所有神经元中选择了 {len(utility_neurons)} 个效用神经元 (top {args.utility_threshold_p*100:.2f}%)")
    
    # 计算专用安全神经元
    print("\n计算专用安全神经元...")
    dedicated_neurons, stats = compute_dedicated_safety_neurons(safety_neurons, utility_neurons)
    
    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载元数据（如果可用）
    metadata = {}
    if args.safety_neurons_path:
        try:
            with open(args.safety_neurons_path, "r", encoding="utf-8") as f:
                safety_data = json.load(f)
        except json.JSONDecodeError:
            safety_data = {}
        if "metadata" in safety_data:
            metadata["safety_metadata"] = safety_data["metadata"]
    elif args.safety_all_neurons_path:
        try:
            with open(args.safety_all_neurons_path, "r", encoding="utf-8") as f:
                safety_data = json.load(f)
        except json.JSONDecodeError:
            safety_data = {}
        if "metadata" in safety_data:
            metadata["safety_metadata"] = safety_data["metadata"]
            if args.safety_threshold_q is not None:
                metadata["safety_metadata"]["safety_threshold_q"] = args.safety_threshold_q

    if args.utility_neurons_path:
        try:
            with open(args.utility_neurons_path, "r", encoding="utf-8") as f:
                utility_data = json.load(f)
        except json.JSONDecodeError:
            utility_data = {}
        if "metadata" in utility_data:
            metadata["utility_metadata"] = utility_data["metadata"]
    elif args.utility_all_neurons_path:
        try:
            with open(args.utility_all_neurons_path, "r", encoding="utf-8") as f:
                utility_data = json.load(f)
        except json.JSONDecodeError:
            utility_data = {}
        if "metadata" in utility_data:
            metadata["utility_metadata"] = utility_data["metadata"]
            if args.utility_threshold_p is not None:
                metadata["utility_metadata"]["utility_threshold_p"] = args.utility_threshold_p
    
    # 构建输出数据，包含统计信息和公式定义
    output_data = {
        "metadata": {
            **metadata,
            **stats,
            "formula": "D(p,q) = S(q) \\\\ U(p)",
            "definition": {
                "S(q)": "安全神经元集合，基于安全参考集计算的重要性分数 I_i^s，选择前 q% 的神经元",
                "U(p)": "效用神经元集合，基于通用任务参考集计算的重要性分数 I_i^u，选择前 p% 的神经元",
                "D(p,q)": "专用安全神经元集合，从安全神经元集合中移除效用神经元后的结果",
            },
            "note": "专用安全神经元集合 D(p,q) = S(q) \\\\ U(p)，即从安全神经元中移除效用神经元后的结果",
        },
        "dedicated_safety_neurons": dedicated_neurons,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n专用安全神经元已保存到: {output_path}")
    
    # 显示前10个专用安全神经元
    if len(dedicated_neurons) > 0:
        print("\n前10个专用安全神经元（按分数排序）:")
        sorted_neurons = sorted(
            dedicated_neurons.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )[:10]
        for key, info in sorted_neurons:
            print(
                f"  {key}: "
                f"score={info.get('score', 0):.6f}, "
                f"rank={info.get('rank', 0)}, "
                f"percentile={info.get('percentile', 0):.4f}%"
            )
    else:
        print("\n警告: 没有找到专用安全神经元（所有安全神经元都是效用神经元）")


if __name__ == "__main__":
    main()
