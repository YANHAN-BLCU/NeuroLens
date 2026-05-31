"""效用神经元识别模块

在 Stanford Alpaca 数据集上识别效用神经元集合 U(p)。

本模块使用 Stanford Alpaca 数据集作为通用任务参考集，计算每个神经元的效用 SNIP 分数 I_i^u，
并选择 top p% 的神经元作为效用神经元集合 U(p)。
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, Any
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from .snip_scorer import (
    compute_snip_scores,
    rank_and_annotate_snip_scores,
    select_top_percent_neurons,
)


def default_utility_loss_fn(outputs, batch, model, device):
    """
    默认效用损失函数

    对于通用任务，同样使用标准的语言模型交叉熵损失，对应论文中的 L(x)。

    Args:
        outputs: 模型输出
        batch: 批次数据
        model: 模型
        device: 设备

    Returns:
        损失张量
    """
    logits = outputs.logits

    # 获取标签
    input_ids = batch.get("input_ids")
    if input_ids is None:
        input_ids = outputs.input_ids if hasattr(outputs, "input_ids") else None
        if input_ids is None:
            raise ValueError("无法获取标签用于计算损失")

    labels = input_ids.to(device)

    # 计算交叉熵损失
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    return loss


class AlpacaJsonlDataset(Dataset):
    """
    读取 Stanford Alpaca JSONL（每行一个 JSON）并输出 compute_snip_scores 支持的样本格式。

    期望每行至少包含：
      - {"input": {"prompt": "..."} , "output": "..."}

    兼容：
      - {"prompt": "...", "response": "..."} / {"prompt": "...", "output": "..."}
      - {"input": "...", "output": "..."}  (input 为字符串 prompt)
    """

    def __init__(self, file_path: Union[str, Path], *, max_samples: Optional[int] = None):
        self.file_path = str(file_path)
        self.samples: list[dict[str, Any]] = []

        max_n = None if (max_samples is None or max_samples <= 0) else int(max_samples)

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_n is not None and len(self.samples) >= max_n:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # 跳过坏行（不让整个流程失败）
                    continue

                # 统一成 Alpaca-style: {"input":{"prompt":...},"output":...}
                prompt = None
                output = None

                if isinstance(obj, dict):
                    if "input" in obj and "output" in obj:
                        inp = obj["input"]
                        output = obj.get("output")
                        if isinstance(inp, dict):
                            prompt = inp.get("prompt")
                        else:
                            prompt = inp
                    elif "prompt" in obj and ("response" in obj or "output" in obj):
                        prompt = obj.get("prompt")
                        output = obj.get("response", obj.get("output"))

                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                if not isinstance(output, str):
                    # 允许 output 缺失时退化为 text（loss 作用于全序列）
                    self.samples.append({"text": prompt.strip()})
                    continue

                self.samples.append({"input": {"prompt": prompt.strip()}, "output": output})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def identify_utility_neurons(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    utility_dataset: Union[Dataset, str, Path],
    device: torch.device,
    utility_threshold_p: float = 0.001,  # 对应论文中的 p（例如 0.1% = 0.001）
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    loss_fn: Optional[callable] = None,
) -> Dict[Tuple[int, int], Dict]:
    """
    识别效用神经元集合 U(p)

    在 Stanford Alpaca 数据集上计算 SNIP 分数 I_i^u，选择 top p% 的神经元作为效用神经元 U(p)。

    对应论文中的定义：
        - I_i^u: 在 Stanford Alpaca 数据集上，对每个神经元的 SNIP 分数
        - U(p) = { i | I_i^u 为 I^u 中 top p% }

    Args:
        model: 语言模型
        tokenizer: 分词器
        utility_dataset: Stanford Alpaca 数据集路径（JSONL 格式，例如 data/alpaca/alpaca_data.jsonl）
                        或 PyTorch Dataset 对象。推荐使用 Stanford Alpaca 数据集。
                        支持样本格式：
                        - Alpaca 标准格式: {"input":{"prompt":"..."},"output":"..."}
                        - 兼容格式: {"prompt": "...", "response": "..."} / {"prompt": "...", "output": "..."}
                        - 兼容格式: {"input": "...", "output": "..."}  (input 为字符串 prompt)
                        - 文本格式: {"text": "..."}
        device: 设备
        utility_threshold_p: 效用阈值 p（例如 0.1% = 0.001，选择 top 0.1%）
        batch_size: 批大小
        num_samples: 使用的样本数（None 表示全部）
        loss_fn: 自定义损失函数（None 则使用默认）

    Returns:
        Dict[(layer_idx, neuron_idx), {score, rank, percentile, layer, neuron}]:
            效用神经元集合 U(p) 中每个神经元的分数、排名和百分位

    Note:
        推荐使用 Stanford Alpaca 数据集（data/alpaca/alpaca_data.jsonl），
        该数据集包含约 52,000 个指令-响应对，适合作为通用任务参考集。
        
    Warning:
        如果效用神经元和安全神经元使用相同的数据集和损失函数，它们的 SNIP 分数会完全相同，
        导致无法有效区分"安全专用"与"通用效用"神经元。
        
        建议：
        - 安全神经元：使用 Stanford Alpaca 数据集（benign 参考集）
        - 效用神经元：使用 CSQA 或其他通用任务数据集（如 PIQA、RACE、MMLU）
        
        这样可以确保 I_i^s ≠ I_i^u，从而有效区分安全神经元和效用神经元。
        详见: docs/安全神经元与效用神经元重叠问题分析.md
    """
    if loss_fn is None:
        loss_fn = default_utility_loss_fn

    # 允许直接传入 Stanford Alpaca JSONL 路径
    if isinstance(utility_dataset, (str, Path)):
        dataset_path = Path(utility_dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"utility_dataset 路径不存在: {dataset_path}")
        if dataset_path.suffix.lower() == ".jsonl":
            utility_dataset = AlpacaJsonlDataset(dataset_path, max_samples=num_samples)
        else:
            raise ValueError(
                f"不支持的数据集路径格式: {dataset_path.suffix}（目前仅支持 .jsonl；或直接传入 PyTorch Dataset）"
            )

    # 计算 SNIP 分数 I_i^u（所有神经元）
    print(
        f"[Utility Identifier] 计算效用 SNIP 分数 I^u（阈值 p: {utility_threshold_p*100:.2f}%）..."
    )
    snip_scores = compute_snip_scores(
        model=model,
        tokenizer=tokenizer,
        dataset=utility_dataset,
        device=device,
        loss_fn=loss_fn,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    if not snip_scores:
        print("[Utility Identifier] 警告: 未计算到任何 SNIP 分数 I^u")
        return {}

    # 第一步：对所有神经元做全局排序并标注 rank / percentile
    annotated = rank_and_annotate_snip_scores(snip_scores)
    total_neurons = len(annotated)

    # 第二步：在完整排序结果上按比例选择前 p% 神经元
    utility_neurons = select_top_percent_neurons(
        annotated,
        top_percent=utility_threshold_p,
    )

    num_selected = len(utility_neurons)
    print(
        f"[Utility Identifier] 总神经元数: {total_neurons}, "
        f"选择前 {num_selected} 个 (top {utility_threshold_p*100:.2f}%) 作为 U(p)"
    )

    print(f"[Utility Identifier] 识别到 {len(utility_neurons)} 个效用神经元 U(p)")
    return utility_neurons