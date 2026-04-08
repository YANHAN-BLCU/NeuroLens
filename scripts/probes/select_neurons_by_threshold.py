"""
从保存的所有神经元分数中，根据阈值选择前 q% 的神经元

用法示例:
    python scripts/select_neurons_by_threshold.py \
        --input_path outputs/safety_neurons_salad.json \
        --output_path outputs/safety_neurons_salad_q0.005.json \
        --threshold_q 0.005
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def select_neurons_by_threshold(
    all_neurons: Dict[str, Dict],
    threshold_q: float,
) -> Dict[str, Dict]:
    """
    从所有神经元中选择前 q% 的神经元
    
    Args:
        all_neurons: 所有神经元的字典，格式为 {f"{layer}_{neuron}": {score, rank, percentile, ...}}
        threshold_q: 阈值 q（例如 0.005 表示前 0.5%）
    
    Returns:
        筛选后的神经元字典
    """
    if threshold_q <= 0 or threshold_q > 1:
        raise ValueError(f"threshold_q 应该在 0~1 之间，例如 0.005 表示 0.5%，当前为 {threshold_q}")
    
    # 按分数排序
    sorted_neurons = sorted(
        all_neurons.items(),
        key=lambda x: x[1].get("score", 0),
        reverse=True,
    )
    
    total_neurons = len(sorted_neurons)
    num_selected = max(1, int(total_neurons * threshold_q))
    
    # 选择前 q% 的神经元
    selected_neurons = dict(sorted_neurons[:num_selected])
    
    print(f"总神经元数: {total_neurons}")
    print(f"选择前 {num_selected} 个神经元 (top {threshold_q*100:.2f}%)")
    
    return selected_neurons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从保存的所有神经元分数中，根据阈值选择前 q% 的神经元"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="输入文件路径（包含所有神经元分数的 JSON 文件）",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出文件路径（JSON 格式）",
    )
    
    parser.add_argument(
        "--threshold_q",
        type=float,
        required=True,
        help="安全阈值 q（例如 0.5%% = 0.005）",
    )
    
    args = parser.parse_args()
    
    # 读取输入文件
    print(f"读取输入文件: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 获取所有神经元
    if "all_neurons" in data:
        all_neurons = data["all_neurons"]
    elif "safety_neurons" in data:
        print("警告: 输入文件已包含筛选后的神经元，无法重新筛选")
        print("请使用包含 'all_neurons' 字段的文件")
        return
    else:
        raise ValueError("输入文件必须包含 'all_neurons' 或 'safety_neurons' 字段")
    
    # 根据阈值选择神经元
    selected_neurons = select_neurons_by_threshold(all_neurons, args.threshold_q)
    
    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            **data.get("metadata", {}),
            "safety_threshold_q": args.threshold_q,
            "num_safety_neurons": len(selected_neurons),
        },
        "safety_neurons": selected_neurons,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_path}")
    
    # 显示前10个神经元
    if len(selected_neurons) > 0:
        print("\n前10个安全神经元:")
        sorted_neurons = sorted(
            selected_neurons.items(),
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


if __name__ == "__main__":
    main()
