"""
使用 Alpaca 数据集识别效用神经元

流程说明：
1. 计算所有神经元的重要性分数（SNIP 分数）
2. 对所有神经元进行全局排序
3. 如果指定了阈值 p，选择前 p% 的神经元作为效用神经元候选集 U(p)

输出说明：
- 总是会保存所有神经元分数到 all_neurons_scores.json
- 如果应用了阈值（utility_threshold_p >= 0），同时保存筛选后的效用神经元到 utility_neurons.json

示例用法：
    # 方式1: 应用阈值，同时保存所有神经元分数和筛选后的效用神经元
    python scripts/run_utility_identifier.py \
        --model_name_or_path /path/to/model \
        --alpaca_path data/alpaca/alpaca_data.jsonl \
        --output_path outputs/utility_neurons.json \
        --utility_threshold_p 0.001 \
        --batch_size 8 \
        --num_samples 1000
    # 输出: outputs/utility_all_neurons_scores.json 和 outputs/utility_neurons.json
    
    # 方式2: 不应用阈值，只保存所有神经元分数
    python scripts/run_utility_identifier.py \
        --model_name_or_path /path/to/model \
        --alpaca_path data/alpaca/alpaca_data.jsonl \
        --output_path outputs/all_neurons_scores.json \
        --utility_threshold_p -1 \
        --batch_size 8 \
        --num_samples 1000
    # 输出: outputs/utility_all_neurons_scores.json（自动添加 utility_ 前缀以区分）
    
    # 然后使用辅助脚本根据阈值选择神经元:
    python scripts/select_neurons_by_threshold.py \
        --input_path outputs/utility_all_neurons_scores.json \
        --output_path outputs/utility_neurons_p0.001.json \
        --threshold_q 0.001
"""

import argparse
import json
import sys
from pathlib import Path

# 以 `python scripts/xxx.py` 运行时 sys.path[0] 为 scripts/，需把项目根加入路径才能 import engine
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.neurons.utility_identifier import identify_utility_neurons


def _log_to_guard_label(
    script_name: str,
    status: str,
    message: str,
    details: dict = None,
) -> None:
    """向 logs/guard_label.log 追加一条结构化运行记录（JSONL 格式）。"""
    import datetime

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "guard_label.log"

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": script_name,
        "status": status,
        "message": message,
    }
    if details:
        entry["details"] = details

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Alpaca 数据集识别效用神经元"
    )
    parser.add_argument(
        "--model_name_or_path",
        "--model-path",
        type=str,
        required=True,
        dest="model_name_or_path",
        help="模型路径或 HuggingFace 模型名称",
    )
    parser.add_argument(
        "--alpaca_path",
        type=str,
        required=True,
        help="Alpaca 数据集文件路径（JSONL 格式）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出文件路径（JSON 格式）",
    )
    parser.add_argument(
        "--utility_threshold_p",
        type=float,
        default=0.001,
        help="效用阈值 p（例如 0.1%% = 0.001）。设置为 -1 表示不应用阈值，保存所有神经元分数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批大小",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="使用的样本数（None 表示全部）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（None 表示自动选择）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _log_to_guard_label(
        "run_utility_identifier",
        "START",
        f"效用神经元识别启动 — p={args.utility_threshold_p}",
        details={
            "model": args.model_name_or_path,
            "dataset_path": args.alpaca_path,
            "utility_threshold_p": args.utility_threshold_p,
            "num_samples": args.num_samples,
        },
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和分词器
    print(f"加载模型和分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # 加载数据集以获取大小信息
    from engine.neurons.utility_identifier import AlpacaJsonlDataset
    dataset = AlpacaJsonlDataset(
        file_path=args.alpaca_path,
        max_samples=args.num_samples,
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 总是计算所有神经元的分数
    print("开始计算所有神经元分数...")
    from engine.neurons.snip_scorer import (
        compute_snip_scores,
        rank_and_annotate_snip_scores,
    )
    from engine.neurons.utility_identifier import default_utility_loss_fn
    
    # 计算所有神经元的 SNIP 分数
    snip_scores = compute_snip_scores(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        loss_fn=default_utility_loss_fn,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    
    # 对所有神经元进行排序并标注排名和百分位
    all_neurons = rank_and_annotate_snip_scores(snip_scores)
    print(f"计算完成，共 {len(all_neurons)} 个神经元")
    
    # 判断是否应用阈值并筛选效用神经元
    apply_threshold = args.utility_threshold_p >= 0
    if apply_threshold:
        # 根据阈值筛选效用神经元
        print(f"根据阈值 p: {args.utility_threshold_p*100:.2f}% 筛选效用神经元...")
        total_neurons = len(all_neurons)
        num_to_select = max(1, int(total_neurons * args.utility_threshold_p))
        
        # 按分数排序，选择前 p% 的神经元
        sorted_neurons = sorted(
            all_neurons.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )[:num_to_select]
        
        utility_neurons = dict(sorted_neurons)
        print(f"识别到 {len(utility_neurons)} 个效用神经元")
    else:
        # 不应用阈值
        utility_neurons = {}
    
    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据 output_path 生成两个文件路径
    # 如果 output_path 包含 "all_neurons" 或 "scores"，则作为所有神经元文件
    # 否则作为效用神经元文件
    # 注意：效用神经元的所有分数文件会自动添加 "utility_" 前缀以区分
    if "all_neurons" in output_path.stem or "scores" in output_path.stem:
        # 确保文件名包含 "utility_" 前缀
        stem = output_path.stem
        if not stem.startswith("utility_"):
            # 如果文件名是 "all_neurons_scores"，改为 "utility_all_neurons_scores"
            if stem == "all_neurons_scores":
                stem = "utility_all_neurons_scores"
            elif "all_neurons" in stem and not stem.startswith("utility_"):
                stem = "utility_" + stem
            elif "scores" in stem and not stem.startswith("utility_"):
                stem = "utility_" + stem
        all_neurons_path = output_path.parent / f"{stem}.json"
        # 生成效用神经元文件路径
        if "all_neurons" in stem:
            utility_stem = stem.replace("all_neurons", "utility_neurons")
        elif "scores" in stem:
            utility_stem = stem.replace("scores", "utility_neurons")
        else:
            utility_stem = stem.replace("utility_", "utility_neurons_")
        utility_neurons_path = output_path.parent / f"{utility_stem}.json"
        if utility_neurons_path == all_neurons_path:
            utility_neurons_path = output_path.parent / "utility_neurons.json"
    else:
        utility_neurons_path = output_path
        # 自动生成效用神经元的所有分数文件路径
        all_neurons_path = output_path.parent / "utility_all_neurons_scores.json"
    
    # 转换格式以便 JSON 序列化
    def serialize_neurons(neurons_dict):
        return {
            f"{layer}_{neuron}": {
                "layer": layer,
                "neuron": neuron,
                "score": float(info.get("score", 0)),
                "rank": int(info.get("rank", 0)),
                "percentile": float(info.get("percentile", 0)),
            }
            for (layer, neuron), info in neurons_dict.items()
        }
    
    # 保存所有神经元分数文件
    all_neurons_data = {
        "metadata": {
            "model": args.model_name_or_path,
            "dataset_path": args.alpaca_path,
            "num_samples_used": len(dataset),
            "neuron_type": "utility",  # 标识这是效用神经元分数
            "note": "包含所有效用神经元的分数、排名和百分位信息（基于通用任务参考集计算），可根据需要调整阈值选择不同百分比的神经元",
        },
        "all_neurons": serialize_neurons(all_neurons),
    }
    all_neurons_data["metadata"]["num_total_neurons"] = len(all_neurons)
    
    with open(all_neurons_path, "w", encoding="utf-8") as f:
        json.dump(all_neurons_data, f, indent=2, ensure_ascii=False)
    print(f"所有神经元分数已保存到: {all_neurons_path}")
    
    # 如果应用了阈值，保存筛选后的效用神经元文件
    if apply_threshold and len(utility_neurons) > 0:
        utility_neurons_data = {
            "metadata": {
                "model": args.model_name_or_path,
                "dataset_path": args.alpaca_path,
                "utility_threshold_p": args.utility_threshold_p,
                "num_samples_used": len(dataset),
                "num_utility_neurons": len(utility_neurons),
            },
            "utility_neurons": serialize_neurons(utility_neurons),
        }
        
        with open(utility_neurons_path, "w", encoding="utf-8") as f:
            json.dump(utility_neurons_data, f, indent=2, ensure_ascii=False)
        print(f"效用神经元已保存到: {utility_neurons_path}")
    
    # 显示前10个神经元（从所有神经元中）
    if all_neurons and len(all_neurons) > 0:
        print("\n前10个神经元（按分数排序）:")
        sorted_neurons = sorted(
            all_neurons.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )[:10]
        for (layer, neuron), info in sorted_neurons:
            print(
                f"  Layer {layer}, Neuron {neuron}: "
                f"score={info.get('score', 0):.6f}, "
                f"rank={info.get('rank', 0)}, "
                f"percentile={info.get('percentile', 0):.4f}%"
            )
        
        # 如果应用了阈值，也显示筛选后的效用神经元数量
        if apply_threshold and len(utility_neurons) > 0:
            print(f"\n筛选后的效用神经元数量: {len(utility_neurons)} (前 {args.utility_threshold_p*100:.2f}%)")

    _log_to_guard_label(
        "run_utility_identifier",
        "DONE",
        f"效用神经元识别完成 — 神经元总数={len(all_neurons)}, 效用神经元={len(utility_neurons)}",
        details={
            "num_total_neurons": len(all_neurons),
            "num_utility_neurons": len(utility_neurons),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log_to_guard_label(
            "run_utility_identifier",
            "ERROR",
            f"运行异常: {e}",
            details={"exception": str(e)},
        )
        raise

