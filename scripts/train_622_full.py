#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6:2:2 数据划分方案下的探针训练脚本，支持详细的训练过程日志记录。

功能概述：
    本脚本是 `train_linear_probes.py` 的扩展版本，专门用于 6:2:2 数据划分方案。
    与主脚本相比，本脚本增加了详细的训练过程日志记录功能，可以追踪每一轮、
    每一层的训练和验证准确率变化。

主要特性：
    1. 数据划分策略：
       - 采用 6:2:2 划分方案：60% 训练集、20% 验证集、20% 测试集
       - 使用分层抽样，确保各子集的类别分布与总体一致
       - 训练集使用所有剩余数据（不进行额外的类别平衡）
       - 支持对 Probe-Train 进行过采样，将正负类比例从 1:4.65 调整为 1:1.5
    
    2. 训练日志记录：
       - 记录每一轮（epoch）、每一层的训练准确率（train_acc）和验证准确率（val_acc）
       - 输出两种格式的日志：
         * CSV 格式：便于在 Excel 等工具中分析
         * JSON 格式：便于程序化读取和处理
       - 生成汇总日志，包含数据划分信息、超参数配置和每层最终指标
    
    3. 探针训练与保存：
       - 与主脚本一致的探针训练流程
       - 自动过滤准确率 < 75% 的浅层探针
       - 保存训练好的探针模型和配置信息

使用方法：
    # 从原始数据开始训练（需要 GPU）
    python scripts/train_622_full.py \
        --data_file data/salad/raw/base_evaluation.jsonl \
        --output_dir outputs/probes_622_full \
        --oversample_probe_train  # 启用过采样，将正负类比例调整为 1:1.5 \
        --max_samples 8000 \
        --num_epochs 80 \
        --lr 3e-3
    
    # 使用预提取的隐藏态缓存（跳过模型加载和特征提取）
    python train_622_full.py \\
        --hidden_states_cache path/to/cache.npz \\
        --output_dir outputs/probes_622_full \\
        --oversample_probe_train

输出文件：
    - outputs/probes_622_full/probes/llama-3-8b/: 探针模型文件
    - outputs/probes_622_full/logs/train_log_622_full_epochs_*.csv: 每轮每层准确率（CSV）
    - outputs/probes_622_full/logs/train_log_622_full_epochs_*.json: 每轮每层准确率（JSON）
    - outputs/probes_622_full/logs/train_log_622_full_summary_*.json: 汇总日志
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 train_linear_probes 的加载、划分与工具（不执行其 main）
_spec = importlib.util.spec_from_file_location(
    "train_linear_probes",
    PROJECT_ROOT / "scripts" / "train_linear_probes.py",
)
_tlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tlp)
load_attack_enhanced = _tlp.load_attack_enhanced
split_data_optimized = _tlp.split_data_optimized
set_seed = _tlp.set_seed
oversample_probe_train = _tlp.oversample_probe_train
oversample_hidden_states_and_labels = _tlp.oversample_hidden_states_and_labels

from engine.models import ModelManager  # noqa: E402
from engine.probes.linear_probe import (  # noqa: E402
    extract_hidden_states,
    train_layer_probes,
    save_probes,
)


def write_epoch_log(results: dict, log_dir: Path, run_id: str) -> None:
    """
    将每一轮、每一层的训练和验证准确率写入 CSV 和 JSON 文件。
    
    本函数从训练结果中提取每层的训练历史（training_history），包括：
    - epochs: 轮次列表
    - train_acc: 每轮的训练准确率
    - val_acc: 每轮的验证准确率
    
    输出两种格式：
    1. CSV 格式：便于在 Excel、Python pandas 等工具中分析
    2. JSON 格式：便于程序化读取和处理
    
    Args:
        results: 训练结果字典，格式为 {layer_idx: {training_history: {...}, ...}, ...}
        log_dir: 日志文件保存目录
        run_id: 运行 ID，用于生成唯一的文件名（格式：YYYYMMDD_HHMMSS）
    
    Note:
        - 如果某层的某个轮次没有准确率数据，对应字段为空字符串（CSV）或 None（JSON）
        - 文件命名格式：train_log_622_full_epochs_{run_id}.{csv|json}
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer_idx, res in sorted(results.items()):
        th = res.get("training_history") or {}
        epochs = th.get("epochs") or []
        train_accs = th.get("train_acc") or []
        val_accs = th.get("val_acc") or []
        for i, e in enumerate(epochs):
            rows.append({
                "layer": layer_idx,
                "epoch": int(e),
                "train_acc": train_accs[i] if i < len(train_accs) else None,
                "val_acc": val_accs[i] if i < len(val_accs) else None,
            })
    # CSV
    csv_path = log_dir / f"train_log_622_full_epochs_{run_id}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("layer,epoch,train_acc,val_acc\n")
        for r in rows:
            ta = "" if r["train_acc"] is None else f"{r['train_acc']:.6f}"
            va = "" if r["val_acc"] is None else f"{r['val_acc']:.6f}"
            f.write(f"{r['layer']},{r['epoch']},{ta},{va}\n")
    print(f"[Log] 每轮每层准确率已写入: {csv_path}")
    # JSON（便于程序读取）
    json_path = log_dir / f"train_log_622_full_epochs_{run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[Log] 每轮每层准确率(JSON)已写入: {json_path}")


def write_summary_log(
    results: dict,
    split_info: dict,
    args_dict: dict,
    log_dir: Path,
    run_id: str,
) -> None:
    """
    写入汇总日志，包含数据划分信息、超参数配置和每层最终性能指标。
    
    汇总日志包含以下信息：
    1. run_id: 运行 ID
    2. created_at: 创建时间（UTC）
    3. split: 数据划分信息（训练集、验证集、测试集的数量和类别分布）
    4. args: 超参数配置（学习率、批大小、训练轮数等）
    5. per_layer: 每层的最终性能指标
       - train_acc: 训练准确率
       - val_acc: 验证准确率
       - val_roc_auc: 验证集 ROC-AUC
       - val_pr_auc: 验证集 PR-AUC
       - meets_requirement: 是否满足准确率要求（>= 75%）
    
    Args:
        results: 训练结果字典，格式为 {layer_idx: {metrics: {...}, ...}, ...}
        split_info: 数据划分信息字典，包含各子集的样本数和类别分布
        args_dict: 超参数配置字典
        log_dir: 日志文件保存目录
        run_id: 运行 ID，用于生成唯一的文件名
    
    Note:
        - 文件命名格式：train_log_622_full_summary_{run_id}.json
        - 所有时间使用 UTC 时区
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    per_layer = {}
    for layer_idx, res in sorted(results.items()):
        m = res.get("metrics") or {}
        per_layer[int(layer_idx)] = {
            "train_acc": m.get("train_acc"),
            "val_acc": m.get("val_acc"),
            "val_roc_auc": m.get("val_roc_auc"),
            "val_pr_auc": m.get("val_pr_auc"),
            "meets_requirement": m.get("meets_requirement"),
        }
    summary = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split": split_info,
        "args": args_dict,
        "per_layer": per_layer,
    }
    path = log_dir / f"train_log_622_full_summary_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Log] 汇总日志已写入: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="6:2:2 剩余全作训练集 + 每轮每层准确率日志")
    parser.add_argument("--data_file", type=Path, default=Path("data/salad/raw/base_evaluation.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/probes_622_full"))
    parser.add_argument("--log_dir", type=Path, default=None, help="日志目录，默认 output_dir/logs")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hidden_states_cache", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--oversample_probe_train",
        action="store_true",
        help="对 Probe-Train 进行过采样，将正负类比例调整为 1:1.5（正类 40%，负类 60%）",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or (args.output_dir / "logs")
    split_info = {}

    # ========== 模式1：从预提取的隐藏态缓存加载 ==========
    # 适用于已经提取好隐藏态的情况，可以跳过模型加载和特征提取步骤，加快训练速度
    if args.hidden_states_cache and args.hidden_states_cache.exists():
        # 加载缓存文件（.npz 格式）
        # 注意：缓存文件必须按照 6:2:2 全训练划分方案生成
        data = np.load(args.hidden_states_cache, allow_pickle=True)
        
        # 提取训练集和验证集的隐藏态和标签
        hidden_states = [data["train_hs"][i] for i in range(len(data["train_labels"]))]
        labels = list(data["train_labels"])
        val_hidden_states = [data["val_hs"][i] for i in range(len(data["val_labels"]))]
        val_labels = list(data["val_labels"])
        
        # 获取模型结构信息
        num_layers = int(data["num_layers"])
        hidden_dim = int(data["hidden_dim"])
        
        # 对 Probe-Train 进行过采样（如果启用）
        # 将正负类比例从约 1:4.65（正类 17.7%）调整为 1:1.5（正类 40%，负类 60%）
        if args.oversample_probe_train:
            hidden_states, labels = oversample_hidden_states_and_labels(
                hidden_states, labels,
                target_pos_ratio=0.4,  # 目标正类比例 40%（对应 1:1.5）
                random_state=args.seed,
            )
        
        # 记录数据划分信息
        split_info = {"from_cache": str(args.hidden_states_cache), "train": len(labels), "val": len(val_labels)}
        test_texts, test_labels = [], []  # 缓存模式不包含测试集
        print(f"[Cache] 训练: {len(hidden_states)}, 验证: {len(val_hidden_states)}, num_layers={num_layers}, hidden_dim={hidden_dim}")
    # ========== 模式2：从原始数据开始（需要 GPU 进行特征提取） ==========
    else:
        # 步骤1：加载原始数据
        print(f"[Data] Loading {args.data_file} ...")
        all_texts, all_labels = load_attack_enhanced(args.data_file, max_samples=args.max_samples)
        if not all_texts:
            raise SystemExit("未加载到有效样本。")
        
        # 步骤2：按 6:2:2 方案划分数据
        # 60% 训练集、20% 验证集、20% 测试集，使用分层抽样保持类别分布一致
        probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
            split_data_optimized(
                all_texts, all_labels,
                seed=args.seed,
                use_ratio_6_2_2_full_train=True,  # 使用 6:2:2 剩余全作训练集方案
            )
        
        # 步骤3：对 Probe-Train 进行过采样（如果启用）
        # 将正负类比例从约 1:4.65（正类 17.7%）调整为 1:1.5（正类 40%，负类 60%）
        # 这样可以提升正类样本数量，改善探针训练效果
        if args.oversample_probe_train:
            probe_train_texts, probe_train_labels = oversample_probe_train(
                probe_train_texts, probe_train_labels,
                target_pos_ratio=0.4,  # 目标正类比例 40%（对应 1:1.5）
                random_state=args.seed,
            )
        
        split_info = {
            "train": len(probe_train_texts),
            "train_safe": sum(1 for l in probe_train_labels if l == 0),
            "train_toxic": sum(1 for l in probe_train_labels if l == 1),
            "val": len(probe_val_texts),
            "val_safe": sum(1 for l in probe_val_labels if l == 0),
            "val_toxic": sum(1 for l in probe_val_labels if l == 1),
            "test": len(test_texts),
        }
        # 加载模型并提取隐藏态
        # 如果指定了自定义模型路径，设置环境变量
        if args.model_name_or_path:
            import os
            os.environ["LLM_CONTAINER_PATH"] = args.model_name_or_path
        model_manager = ModelManager()
        tokenizer, model = model_manager.load_llm()
        device = next(model.parameters()).device
        print("[Hidden] 提取训练集隐藏态...")
        hidden_states = extract_hidden_states(
            model=model, tokenizer=tokenizer, texts=probe_train_texts,
            device=device, max_length=args.max_length, batch_size=args.batch_size,
        )
        print("[Hidden] 提取验证集隐藏态...")
        val_hidden_states = extract_hidden_states(
            model=model, tokenizer=tokenizer, texts=probe_val_texts,
            device=device, max_length=args.max_length, batch_size=args.batch_size,
        )
        labels = probe_train_labels
        val_labels = probe_val_labels
        num_layers, hidden_dim = hidden_states[0].shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_indices = list(range(len(labels)))

    results = train_layer_probes(
        hidden_states=hidden_states,
        labels=labels,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        train_indices=train_indices,
        val_indices=[],
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.probe_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_hidden_states=val_hidden_states,
        val_labels=val_labels,
        ensure_accuracy_requirements=True,
    )

    # 每轮每层准确率日志
    write_epoch_log(results, log_dir, run_id)
    write_summary_log(
        results,
        split_info,
        {
            "num_epochs": args.num_epochs,
            "probe_batch_size": args.probe_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        log_dir,
        run_id,
    )

    # 保存探针
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_probes(
        results=results,
        output_dir=args.output_dir,
        model_id="llama-3-8b",
        filter_threshold=0.75,
    )
    print(f"[Done] 探针已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
