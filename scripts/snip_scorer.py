"""
SNIP 分数计算模块

基于 SNIP (Single-shot Network Pruning based on Connection Sensitivity) 方法
计算每个神经元对损失的贡献度
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset


def compute_snip_scores(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: torch.device,
    loss_fn: Callable,
    batch_size: int = 8,
    num_samples: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    计算 SNIP 分数
    
    SNIP 分数表示移除某个神经元对损失的影响程度。
    计算公式：SNIP = |gradient × weight|
    
    Args:
        model: 语言模型（需要支持梯度计算）
        tokenizer: 分词器
        dataset: 数据集，每个样本需要包含 "text" 字段
        device: 计算设备
        loss_fn: 损失函数，签名: loss_fn(outputs, batch, model, device) -> loss_tensor
        batch_size: 批大小
        num_samples: 使用的样本数（None 表示全部）
    
    Returns:
        Dict[(layer_idx, neuron_idx), snip_score]: 每个神经元的 SNIP 分数
        其中 layer_idx 是层索引，neuron_idx 是神经元索引（在 MLP down_proj 中）
    """
    model.eval()
    model.requires_grad_(True)  # 需要梯度来计算 SNIP
    
    # 存储每个神经元的 SNIP 分数（累加）
    snip_scores = defaultdict(float)
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_batches = len(dataloader)
    if num_samples:
        total_batches = min(total_batches, (num_samples + batch_size - 1) // batch_size)
    
    print(f"[SNIP Scorer] 开始计算 SNIP 分数，共 {total_batches} 个批次...")
    
    total_samples = 0
    successful_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if num_samples and total_samples >= num_samples:
            break
        
        # 准备输入
        # DataLoader 返回的 batch["text"] 通常是列表，但需要处理各种情况
        if isinstance(batch, dict) and "text" in batch:
            texts = batch["text"]
            if not isinstance(texts, list):
                texts = [texts]
        elif isinstance(batch, list):
            # 如果 batch 是列表，提取所有文本
            texts = [item["text"] if isinstance(item, dict) else item for item in batch]
        else:
            raise ValueError(f"无法处理 batch 格式: {type(batch)}")
        
        if not texts:
            continue
        
        try:
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            ).to(device)
        except Exception as e:
            print(f"[SNIP Scorer] 警告: 批处理 {batch_idx} 的 tokenization 失败: {e}")
            continue
        
        try:
            # 前向传播
            outputs = model(**inputs)
            
            # 计算损失
            loss = loss_fn(outputs, batch, model, device)
            
            # 反向传播
            loss.backward()
        except Exception as e:
            print(f"[SNIP Scorer] 警告: 批处理 {batch_idx} 的前向/反向传播失败: {e}")
            model.zero_grad()
            continue
        
        # 遍历所有 Transformer 层（Llama 架构）
        # 假设模型结构为: model.model.layers (LlamaForCausalLM)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            print("[SNIP Scorer] 警告: 无法找到模型的层结构，跳过该批处理")
            model.zero_grad()
            continue
        
        # 计算每层每个神经元的 SNIP 分数
        for layer_idx, layer in enumerate(layers):
            # 获取 MLP 模块（Llama 架构中的 feed-forward）
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp = layer.feed_forward
            else:
                continue
            
            # 获取 down_proj（输出层，每个神经元对应一个输出）
            if hasattr(mlp, 'down_proj'):
                down_proj = mlp.down_proj
            elif hasattr(mlp, 'output'):
                down_proj = mlp.output
            else:
                continue
            
            # 计算该层每个神经元的 SNIP 分数
            # 处理量化模型：量化模型的权重可能是特殊类型
            try:
                # 检查权重和梯度是否存在
                if not hasattr(down_proj, 'weight') or down_proj.weight is None:
                    continue
                
                weight = down_proj.weight
                # 对于量化模型，可能需要特殊处理
                if hasattr(weight, 'data'):
                    weight_data = weight.data
                else:
                    # 量化模型可能没有 .data 属性，直接使用权重
                    weight_data = weight
                
                # 检查梯度
                if not hasattr(weight, 'grad') or weight.grad is None:
                    continue
                
                grad = weight.grad
                if hasattr(grad, 'data'):
                    grad_data = grad.data
                else:
                    grad_data = grad
                
                # 确保权重和梯度是 torch.Tensor 类型
                if not isinstance(weight_data, torch.Tensor) or not isinstance(grad_data, torch.Tensor):
                    # 量化模型可能需要特殊处理，跳过
                    continue
                
                # 确保权重和梯度在同一设备上，并且可以计算
                if weight_data.device != grad_data.device:
                    continue
                
                # 确保维度匹配
                if weight_data.shape != grad_data.shape:
                    continue
                
                # 对每个神经元（每行）计算 SNIP 分数
                # SNIP = |gradient × weight| 的 L1 范数
                for neuron_idx in range(down_proj.out_features):
                    try:
                        # 计算该神经元权重的梯度贡献
                        neuron_grad = grad_data[neuron_idx, :]  # 该神经元的所有输入连接
                        neuron_weight = weight_data[neuron_idx, :]
                        
                        # SNIP 分数 = |gradient × weight| 的 L1 范数
                        snip_score = torch.abs(neuron_grad * neuron_weight).sum().item()
                        snip_scores[(layer_idx, neuron_idx)] += snip_score
                    except (IndexError, RuntimeError, AttributeError) as e:
                        # 跳过有问题的神经元（可能是量化模型的特殊结构）
                        continue
                        
            except Exception as e:
                # 跳过有问题的层（可能是量化模型或其他特殊情况）
                continue
        
        # 清零梯度（为下一批准备）
        model.zero_grad()
        total_samples += len(texts)
        successful_batches += 1
        
        # 每10个批次显示一次进度
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"[SNIP Scorer] 进度: {batch_idx + 1}/{total_batches} 批次, {total_samples} 样本, "
                  f"已识别 {len(snip_scores)} 个神经元")
    
    # 平均化（除以样本数）
    if total_samples > 0:
        for key in snip_scores:
            snip_scores[key] /= total_samples
        print(f"[SNIP Scorer] 完成: 成功处理 {successful_batches}/{total_batches} 批次, "
              f"共 {total_samples} 个样本, 识别到 {len(snip_scores)} 个神经元")
    else:
        print("[SNIP Scorer] 警告: 未处理任何样本")
    
    return dict(snip_scores)


def compute_snip_scores_batch(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: list,
    device: torch.device,
    loss_fn: Callable,
    batch_size: int = 8,
) -> Dict[Tuple[int, int], float]:
    """
    批量计算 SNIP 分数的便捷函数
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        texts: 文本列表
        device: 计算设备
        loss_fn: 损失函数
        batch_size: 批大小
    
    Returns:
        Dict[(layer_idx, neuron_idx), snip_score]: SNIP 分数
    """
    # 创建简单的数据集
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            return {"text": self.texts[idx]}
    
    dataset = TextDataset(texts)
    return compute_snip_scores(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        loss_fn=loss_fn,
        batch_size=batch_size,
    )
