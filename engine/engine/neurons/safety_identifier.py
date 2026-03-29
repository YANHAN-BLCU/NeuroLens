"""安全神经元识别模块

在 benign 参考集上识别安全神经元候选集 S(q)，并提供与效用神经元集合 U(p)
做差得到最终专用安全神经元 D(p, q) = S(q) \\ U(p) 的工具函数。

## 数据集选择原则（根据论文要求）

根据 NeuroBreak 论文，安全神经元识别需要在 benign（安全）参考数据集上计算 SNIP 分数。

### 推荐数据集

1. **Stanford Alpaca 数据集**（默认推荐）
   - 路径: `data/alpaca/alpaca_data.jsonl`
   - 特点: 包含约 52,000 个安全的指令-响应对
   - 优势: 数据量大、通用性强、适合作为 benign 参考集
   - 使用: `AlpacaJsonlDataset(alpaca_path)`

2. **SALAD 数据集的安全部分**（替代方案）
   - 防御增强集: `data/salad/raw/defense_enhanced_set_train.jsonl`
   - 多选题集: `data/salad/raw/mcq_set_train.jsonl` (仅 gt=="A" 的样本)
   - 评估日志: `logs/base_evaluation.jsonl` (仅 guard.verdict=="allow" 的样本)
   - 优势: 更贴近实际安全场景，包含真实的安全/有害样本对
   - 使用: `SaladSafetyDataset(file_path, source_type)`

### 关键原则

⚠️ **必须使用与效用神经元不同的数据集**

- 如果效用神经元使用 CSQA/PIQA/RACE/MMLU 等通用任务数据集，
  则安全神经元应使用 Alpaca 或 SALAD 数据集
- 避免使用相同数据集，否则会导致 I_i^s = I_i^u，无法有效区分安全神经元和效用神经元
- 详见: `docs/安全神经元与效用神经元重叠问题分析.md`

## 算法流程

1. 在 benign 数据集上计算每个神经元的安全 SNIP 分数 I_i^s
2. 对所有神经元按 I_i^s 进行全局排序
3. 选择 top q% 的神经元作为安全神经元候选集 S(q)
4. 通过 `get_dedicated_safety_neurons()` 与效用神经元 U(p) 做差，得到专属安全神经元 D(p,q)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from .snip_scorer import (
    compute_snip_scores,
    rank_and_annotate_snip_scores,
    select_top_percent_neurons,
)

def default_safety_loss_fn(outputs, batch, model, device):
    """
    默认安全损失函数

    对应论文中的 L(x)，这里使用标准的语言模型交叉熵损失。

    Args:
        outputs: 模型输出
        batch: 批次数据
        model: 模型
        device: 设备

    Returns:
        损失张量
    """
    logits = outputs.logits

    # 获取标签（从输入中提取）
    input_ids = batch.get("input_ids")
    if input_ids is None:
        # 如果没有提供标签，使用输入作为标签（自回归任务）
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


def identify_safety_neurons(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    benign_dataset: Dataset,
    device: torch.device,
    safety_threshold_q: float = 0.005,  # 对应论文中的 q（例如 0.5% = 0.005）
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    loss_fn: Optional[callable] = None,
) -> Dict[Tuple[int, int], Dict]:
    """
    识别安全神经元候选集 S(q)

    在 benign 数据集上计算 SNIP 分数 I_i^s，选择 top q% 的神经元作为安全候选集 S(q)。

    对应论文中的定义：
        - I_i(x) = | w_i * ΔL(x) |
        - S(q) = { i | I_i^s 为 I^s 中 top q% }

    Args:
        model: 语言模型
        tokenizer: 分词器
        benign_dataset: Benign 参考数据集（安全样本）
                       
                       根据论文要求，应使用 benign（安全）参考数据集来识别安全神经元。
                       推荐数据集选择：
                       
                       1. **Stanford Alpaca 数据集**（默认推荐）
                          - 路径: `data/alpaca/alpaca_data.jsonl`
                          - 特点: 包含约 52,000 个安全的指令-响应对
                          - 优势: 数据量大、通用性强、适合作为 benign 参考集
                          - 使用: `AlpacaJsonlDataset(alpaca_path)`
                       
                       2. **SALAD 数据集的安全部分**（替代方案）
                          - 防御增强集: `data/salad/raw/defense_enhanced_set_train.jsonl`
                          - 多选题集: `data/salad/raw/mcq_set_train.jsonl` (仅 gt=="A" 的样本)
                          - 评估日志: `logs/base_evaluation.jsonl` (仅 guard.verdict=="allow" 的样本)
                          - 优势: 更贴近实际安全场景，包含真实的安全/有害样本对
                          - 使用: `SaladSafetyDataset(file_path, source_type)`
                       
                       3. **数据集选择原则**（重要）:
                          - 必须使用与效用神经元**不同的数据集**，以确保 I_i^s ≠ I_i^u
                          - 如果效用神经元使用 CSQA/PIQA/RACE/MMLU 等通用任务数据集，
                            则安全神经元应使用 Alpaca 或 SALAD 数据集
                          - 避免使用相同数据集，否则会导致安全神经元和效用神经元重叠率过高
                          - 详见: `docs/安全神经元与效用神经元重叠问题分析.md`
                       
                       每个样本需要包含 "text" 字段，或使用 Alpaca/SALAD 专用 Dataset 类
        device: 设备
        safety_threshold_q: 安全阈值 q（例如 0.5% = 0.005，选择 top 0.5%）
        batch_size: 批大小
        num_samples: 使用的样本数（None 表示全部）
        loss_fn: 自定义损失函数（None 则使用默认）

    Returns:
        Dict[(layer_idx, neuron_idx), {score, rank, percentile, layer, neuron}]:
            安全神经元集合 S(q) 中每个神经元的分数、排名和百分位
    """
    if loss_fn is None:
        loss_fn = default_safety_loss_fn

        # 计算 SNIP 分数 I_i^s（所有神经元）
    print(
        f"[Safety Identifier] 计算安全 SNIP 分数 I^s（阈值 q: {safety_threshold_q*100:.2f}%）..."
    )
    snip_scores = compute_snip_scores(
        model=model,
        tokenizer=tokenizer,
        dataset=benign_dataset,
        device=device,
        loss_fn=loss_fn,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    if not snip_scores:
        print("[Safety Identifier] 警告: 未计算到任何 SNIP 分数 I^s")
        return {}

    # 第一步：对所有神经元做全局排序并标注 rank / percentile
    annotated = rank_and_annotate_snip_scores(snip_scores)
    total_neurons = len(annotated)

    # 第二步：在完整排序结果上按比例选择前 q% 神经元，作为安全候选集 S(q)
    safety_neurons = select_top_percent_neurons(
        annotated,
        top_percent=safety_threshold_q,
    )

    num_selected = len(safety_neurons)
    print(
        f"[Safety Identifier] 总神经元数: {total_neurons}, "
        f"选择前 {num_selected} 个 (top {safety_threshold_q*100:.2f}%) 作为 S(q)"
    )

    print(f"[Safety Identifier] 识别到 {len(safety_neurons)} 个安全神经元 S(q)")
    return safety_neurons


def get_dedicated_safety_neurons(
    safety_neurons: Dict[Tuple[int, int], Dict],
    utility_neurons: Dict[Tuple[int, int], Dict],
) -> Dict[Tuple[int, int], Dict]:
    """
    计算专属安全神经元 D(p,q) = S(q) \\ U(p)
    
    从安全神经元候选集 S(q) 中排除效用神经元 U(p)，得到专属安全神经元集合 D(p,q)。
    
    对应论文中的定义：
        - D(p,q) = S(q) \\ U(p)
        - 即：在安全候选集中，但不在效用神经元集合中的神经元
    
    Args:
        safety_neurons: 安全神经元集合 S(q)，格式为 Dict[(layer_idx, neuron_idx), {score, rank, ...}]
        utility_neurons: 效用神经元集合 U(p)，格式为 Dict[(layer_idx, neuron_idx), {score, rank, ...}]
    
    Returns:
        Dict[(layer_idx, neuron_idx), {score, rank, percentile, layer, neuron, ...}]:
            专属安全神经元集合 D(p,q) = S(q) \\ U(p)
    
    Warning:
        如果 S(q) 和 U(p) 使用相同数据集和损失函数计算，它们的 SNIP 分数会完全相同，
        导致 D(p,q) 只是"重要性中等"的神经元（排名在 p% ~ q% 之间），而非真正的"专属安全神经元"。
        
        建议：
        - 安全神经元：使用 Stanford Alpaca 数据集（benign 参考集）
        - 效用神经元：使用 CSQA 或其他通用任务数据集（如 PIQA、RACE、MMLU）
        
        这样可以确保 I_i^s ≠ I_i^u，从而有效区分安全神经元和效用神经元。
    """
    dedicated = {
        key: value
        for key, value in safety_neurons.items()
        if key not in utility_neurons
    }
    
    overlap_count = len(safety_neurons) - len(dedicated)
    overlap_ratio = overlap_count / len(safety_neurons) if safety_neurons else 0
    
    print(
        f"[Dedicated Safety Neurons] S(q) 大小: {len(safety_neurons)}, "
        f"U(p) 大小: {len(utility_neurons)}, "
        f"重叠: {overlap_count}, "
        f"D(p,q) 大小: {len(dedicated)}"
    )
    
    if overlap_ratio > 0.5:
        print(
            f"[Warning] 安全神经元和效用神经元重叠率过高 ({overlap_ratio*100:.1f}%)！"
            f"这可能表明它们使用了相同的数据集和损失函数。"
            f"建议使用不同的数据集（如 Alpaca vs CSQA）来区分安全神经元和效用神经元。"
            f"详见: docs/安全神经元与效用神经元重叠问题分析.md"
        )
    elif overlap_ratio > 0.2:
        print(
            f"[Info] 安全神经元和效用神经元重叠率: {overlap_ratio*100:.1f}%"
        )
    
    return dedicated