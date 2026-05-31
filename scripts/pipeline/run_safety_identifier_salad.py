"""
使用 SALAD 数据集中的安全样本识别安全神经元

流程说明：
1. 计算所有神经元的重要性分数（SNIP 分数）
2. 对所有神经元进行全局排序
3. 如果指定了阈值 q，选择前 q% 的神经元作为安全神经元候选集 S(q)

输出说明：
- 总是会保存所有神经元分数到 all_neurons_scores.json
- 如果应用了阈值（safety_threshold_q >= 0），同时保存筛选后的安全神经元到 safety_neurons.json

示例用法：
    # 方式1: 应用阈值，同时保存所有神经元分数和筛选后的安全神经元
    python scripts/run_safety_identifier_salad.py \
        --model-path /path/to/model \
        --dataset_path data/salad/raw/defense_enhanced_set_train.jsonl \
        --source_type defense \
        --output_path outputs/safety_neurons_salad.json \
        --safety_threshold_q 0.005 \
        --batch_size 8 \
        --num_samples 1000
    # 输出: outputs/safety_all_neurons_scores.json 和 outputs/safety_neurons_salad.json
    
    # 方式2: 不应用阈值，只保存所有神经元分数
    python scripts/run_safety_identifier_salad.py \
        --model-path /path/to/model \
        --dataset_path data/salad/raw/defense_enhanced_set_train.jsonl \
        --source_type defense \
        --output_path outputs/all_neurons_scores.json \
        --safety_threshold_q -1 \
        --batch_size 8 \
        --num_samples 1000
    # 输出: outputs/safety_all_neurons_scores.json（自动添加 safety_ 前缀以区分）
    
    # 然后使用辅助脚本根据阈值选择神经元:
    python scripts/select_neurons_by_threshold.py \
        --input_path outputs/safety_all_neurons_scores.json \
        --output_path outputs/safety_neurons_q0.005.json \
        --threshold_q 0.005
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
from transformers import AutoTokenizer, AutoModelForCausalLM

from engine.neurons.safety_identifier import identify_safety_neurons
from engine.neurons.salad_safety_dataset import (
    SaladSafetyDataset,
    CombinedSaladSafetyDataset,
)


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
        pass  # 日志写入失败不影响主流程


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 SALAD 数据集中的安全样本识别安全神经元"
    )
    
    parser.add_argument(
        "--model_name_or_path",
        "--model-path",
        "--model_path",
        type=str,
        required=True,
        dest="model_name_or_path",
        help="模型路径或 HuggingFace 模型名称",
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        required=True,
        help="数据集文件路径（可以指定多个文件）",
    )
    
    parser.add_argument(
        "--source_type",
        type=str,
        default="auto",
        choices=["auto", "defense", "mcq", "evaluation", "text"],
        help="数据源类型（auto 表示自动检测，text 表示直接提取 question 字段）",
    )

    parser.add_argument(
        "--label_paths",
        type=str,
        nargs="+",
        default=None,
        help="标签文件路径列表（用于过滤 Safe 样本），与 dataset_path 一一对应",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出文件路径（JSON 格式）",
    )
    
    parser.add_argument(
        "--safety_threshold_q",
        type=float,
        default=0.005,
        help="安全阈值 q（例如 0.5%% = 0.005）。设置为 -1 表示不应用阈值，保存所有神经元分数",
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
        "run_safety_identifier_salad",
        "START",
        f"安全神经元识别启动 — q={args.safety_threshold_q}",
        details={
            "model": args.model_name_or_path,
            "dataset_path": args.dataset_path,
            "safety_threshold_q": args.safety_threshold_q,
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
    # 统一设为评估模式，并默认关闭全模型梯度，避免无关步骤产生计算图
    model.eval()
    model.requires_grad_(False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # 加载数据集
    print(f"加载 SALAD 安全数据集...")
    if len(args.dataset_path) == 1:
        # 单个文件
        dataset = SaladSafetyDataset(
            file_path=args.dataset_path[0],
            source_type=args.source_type,
            max_samples=args.num_samples,
        )
    else:
        # 多个文件
        source_types = [args.source_type] * len(args.dataset_path)
        dataset = CombinedSaladSafetyDataset(
            file_paths=args.dataset_path,
            source_types=source_types,
            label_paths=args.label_paths,
            max_total_samples=args.num_samples,
        )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 总是计算所有神经元的分数
    print("开始计算所有神经元分数...")
    from engine.neurons.snip_scorer import (
        compute_snip_scores,
        rank_and_annotate_snip_scores,
    )
    from engine.neurons.safety_identifier import default_safety_loss_fn
    
    # 在正式计算 SNIP 之前先清理一次 GPU 缓存，避免前面加载阶段的无用缓存
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 计算所有神经元的 SNIP 分数
    snip_scores = compute_snip_scores(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        loss_fn=default_safety_loss_fn,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    
    # 对所有神经元进行排序并标注排名和百分位
    all_neurons = rank_and_annotate_snip_scores(snip_scores)
    print(f"计算完成，共 {len(all_neurons)} 个神经元")
    
    # 判断是否应用阈值并筛选安全神经元
    apply_threshold = args.safety_threshold_q >= 0
    if apply_threshold:
        # 根据阈值筛选安全神经元
        print(f"根据阈值 q: {args.safety_threshold_q*100:.2f}% 筛选安全神经元...")
        total_neurons = len(all_neurons)
        num_to_select = max(1, int(total_neurons * args.safety_threshold_q))
        
        # 按分数排序，选择前 q% 的神经元
        sorted_neurons = sorted(
            all_neurons.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )[:num_to_select]
        
        safety_neurons = dict(sorted_neurons)
        print(f"识别到 {len(safety_neurons)} 个安全神经元")
    else:
        # 不应用阈值
        safety_neurons = {}
    
    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据 output_path 生成两个文件路径
    # 如果 output_path 包含 "all_neurons" 或 "scores"，则作为所有神经元文件
    # 否则作为安全神经元文件
    # 注意：安全神经元的所有分数文件会自动添加 "safety_" 前缀以区分
    if "all_neurons" in output_path.stem or "scores" in output_path.stem:
        # 确保文件名包含 "safety_" 前缀
        stem = output_path.stem
        if not stem.startswith("safety_"):
            # 如果文件名是 "all_neurons_scores"，改为 "safety_all_neurons_scores"
            if stem == "all_neurons_scores":
                stem = "safety_all_neurons_scores"
            elif "all_neurons" in stem and not stem.startswith("safety_"):
                stem = "safety_" + stem
            elif "scores" in stem and not stem.startswith("safety_"):
                stem = "safety_" + stem
        all_neurons_path = output_path.parent / f"{stem}.json"
        # 生成安全神经元文件路径
        if "all_neurons" in stem:
            safety_stem = stem.replace("all_neurons", "safety_neurons")
        elif "scores" in stem:
            safety_stem = stem.replace("scores", "safety_neurons")
        else:
            safety_stem = stem.replace("safety_", "safety_neurons_")
        safety_neurons_path = output_path.parent / f"{safety_stem}.json"
        if safety_neurons_path == all_neurons_path:
            safety_neurons_path = output_path.parent / "safety_neurons.json"
    else:
        safety_neurons_path = output_path
        # 自动生成安全神经元的所有分数文件路径
        all_neurons_path = output_path.parent / "safety_all_neurons_scores.json"
    
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
            "dataset_paths": args.dataset_path,
            "source_type": args.source_type,
            "num_samples_used": len(dataset),
            "neuron_type": "safety",  # 标识这是安全神经元分数
            "note": "包含所有安全神经元的分数、排名和百分位信息（基于安全参考集计算），可根据需要调整阈值选择不同百分比的神经元",
        },
        "all_neurons": serialize_neurons(all_neurons),
    }
    all_neurons_data["metadata"]["num_total_neurons"] = len(all_neurons)
    
    with open(all_neurons_path, "w", encoding="utf-8") as f:
        json.dump(all_neurons_data, f, indent=2, ensure_ascii=False)
    print(f"所有神经元分数已保存到: {all_neurons_path}")
    
    # 如果应用了阈值，保存筛选后的安全神经元文件
    if apply_threshold and len(safety_neurons) > 0:
        safety_neurons_data = {
            "metadata": {
                "model": args.model_name_or_path,
                "dataset_paths": args.dataset_path,
                "source_type": args.source_type,
                "safety_threshold_q": args.safety_threshold_q,
                "num_samples_used": len(dataset),
                "num_safety_neurons": len(safety_neurons),
            },
            "safety_neurons": serialize_neurons(safety_neurons),
        }
        
        with open(safety_neurons_path, "w", encoding="utf-8") as f:
            json.dump(safety_neurons_data, f, indent=2, ensure_ascii=False)
        print(f"安全神经元已保存到: {safety_neurons_path}")
    
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
        
        # 如果应用了阈值，也显示筛选后的安全神经元数量
        if apply_threshold and len(safety_neurons) > 0:
            print(f"\n筛选后的安全神经元数量: {len(safety_neurons)} (前 {args.safety_threshold_q*100:.2f}%)")

    _log_to_guard_label(
        "run_safety_identifier_salad",
        "DONE",
        f"安全神经元识别完成 — 神经元总数={len(all_neurons)}, 安全神经元={len(safety_neurons)}",
        details={
            "num_total_neurons": len(all_neurons),
            "num_safety_neurons": len(safety_neurons),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log_to_guard_label(
            "run_safety_identifier_salad",
            "ERROR",
            f"运行异常: {e}",
            details={"exception": str(e)},
        )
        raise
