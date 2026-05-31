"""
梯度依赖关系分析模块

根据论文5.4节要求，分析神经元之间的梯度依赖关系（G_{i,j}）。

核心功能：
- 使用W_down神经元作为锚点，追踪来自前一层模块的参数级影响
- 通过测量参数扰动如何传播到W_down激活来量化上游神经元与安全机制的因果关系
- 计算梯度关联：G_{i,j} = ∂a^k_down,i / ∂w^k_upstream,j
  - a^k_down,i: 第k层down_proj的第i个神经元的激活值
  - w^k_upstream,j: 上游（前一层，即k-1层）第j个神经元的权重参数（down_proj权重）

实现原理：
1. 对目标神经元（down_proj的第i个神经元）的激活进行反向传播
2. 计算前一层（layer k-1）down_proj权重的梯度
3. 梯度绝对值作为关联强度：G_{i,j} = |∂a^k_down,i / ∂w^{k-1}_down,j|
4. 选择top-k%的强关联神经元建立连接图
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def _get_transformer_layers(model: nn.Module):
    """获取Transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _get_down_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的down_proj层"""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "down_proj"):
        return layer.feed_forward.down_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "output"):
        return layer.mlp.output
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
        return layer.mlp.fc2
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "w2"):
        return layer.mlp.w2
    return None


def _extract_text_from_sample(sample) -> Optional[str]:
    """从样本中提取文本"""
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        if 'text' in sample:
            return sample['text']
        elif 'question' in sample:
            question = sample['question']
            if isinstance(question, str):
                return question
        elif 'prompt' in sample:
            prompt = sample['prompt']
            if isinstance(prompt, str):
                return prompt
            elif isinstance(prompt, dict) and 'prompt' in prompt:
                return prompt['prompt']
        elif 'input' in sample:
            input_data = sample['input']
            if isinstance(input_data, str):
                return input_data
            elif isinstance(input_data, dict) and 'prompt' in input_data:
                return input_data['prompt']
    return None


def compute_gradient_dependency(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    target_neurons: Dict[Tuple[int, int], Dict],
    device: torch.device,
    top_k: float = 0.1,
    batch_size: int = 4,
    max_length: int = 1024,
    num_samples: Optional[int] = None,
    use_last_token: bool = True,
) -> Dict[Tuple[int, int], Dict]:
    """
    计算梯度依赖关系（G_{i,j}）：分析神经元之间的梯度依赖关系
    
    根据论文5.4节，使用W_down神经元作为锚点，追踪来自前一层模块的参数级影响。
    通过测量参数扰动如何传播到W_down激活来量化上游神经元与安全机制的因果关系。
    
    梯度关联定义为：G_{i,j} = ∂a^k_down,i / ∂w^k_upstream,j
    - a^k_down,i: 第k层down_proj的第i个神经元的激活值
    - w^k_upstream,j: 上游（前一层，即k-1层）第j个神经元的权重参数（down_proj权重）
    
    实现方法：
    1. 对目标神经元（down_proj的第i个神经元）的激活进行反向传播
    2. 计算前一层（layer k-1）down_proj权重的梯度
    3. 梯度绝对值作为关联强度：G_{i,j} = |∂a^k_down,i / ∂w^{k-1}_down,j|
    4. 选择top-k%的强关联神经元建立连接图
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        dataset: 用于计算梯度的数据集
        target_neurons: 目标神经元集合，格式为 Dict[(layer_idx, neuron_idx), Dict]
        device: 计算设备
        top_k: 保留前k%的强关联（默认0.1，即10%）
        batch_size: 批大小（建议4-8）
        max_length: 最大序列长度（建议512-1024）
        num_samples: 使用的样本数（None表示全部）
        use_last_token: 是否使用最后一个token的激活值（默认True）
    
    Returns:
        Dict[(layer_idx, neuron_idx), {
            'upstream_neurons': List[Tuple[int, int]],  # 上游神经元列表（前一层神经元）
            'gradient_strengths': List[float],  # 对应的梯度强度（G_{i,j}）
            'mean_gradient_strength': float,  # 平均梯度强度
            'max_gradient_strength': float,  # 最大梯度强度
        }]
    """
    print("[Gradient Dependency] 开始计算梯度依赖关系...")
    
    # 确保模型处于训练模式以计算梯度
    model.train()
    
    # 启用所有参数的梯度
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad_(True)
    
    # 获取模型层结构
    layers = _get_transformer_layers(model)
    if layers is None:
        raise ValueError("无法获取模型的层结构，请确保模型是Llama架构")
    
    num_layers = len(layers)
    print(f"[Gradient Dependency] 模型结构: {num_layers} 层")
    print(f"[Gradient Dependency] 目标神经元: {len(target_neurons)} 个")
    
    # 存储每个目标神经元的梯度关联
    # 格式: {(layer_idx, neuron_idx): {(upstream_layer, upstream_neuron): gradient_strength}}
    gradient_dependencies = defaultdict(lambda: defaultdict(float))
    
    # 自定义 collate_fn
    def custom_collate_fn(batch):
        """自定义 collate 函数，直接返回样本列表"""
        return batch
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    total_batches = len(dataloader)
    if num_samples:
        total_batches = min(total_batches, (num_samples + batch_size - 1) // batch_size)
    
    print(f"[Gradient Dependency] 开始处理，共 {total_batches} 个批次...")
    
    total_samples = 0
    successful_batches = 0
    
    # 为每个目标神经元设置hook来捕获激活值
    activation_hooks = {}
    activation_storage = {}
    
    def create_activation_hook(layer_idx: int, neuron_idx: int):
        """创建捕获目标神经元激活的hook（保存带梯度的激活值）"""
        def hook(module, input, output):
            # output 是 down_proj 的输出，形状为 (batch_size, seq_len, hidden_dim)
            if output is not None and isinstance(output, torch.Tensor):
                if use_last_token:
                    # 使用最后一个token的激活值
                    last_token_activation = output[:, -1, neuron_idx]  # (batch_size,)
                else:
                    # 使用所有token的平均激活值
                    last_token_activation = output[:, :, neuron_idx].mean(dim=1)  # (batch_size,)
                activation_storage[(layer_idx, neuron_idx)] = last_token_activation
        return hook
    
    # 注册hooks
    for (layer_idx, neuron_idx) in target_neurons.keys():
        if layer_idx >= num_layers:
            continue
        layer = layers[layer_idx]
        down_proj = _get_down_proj(layer)
        if down_proj is not None:
            hook_handle = down_proj.register_forward_hook(create_activation_hook(layer_idx, neuron_idx))
            activation_hooks[(layer_idx, neuron_idx)] = hook_handle
    
    # 按层分组目标神经元
    neurons_by_layer = defaultdict(list)
    for (layer_idx, neuron_idx) in target_neurons.keys():
        if layer_idx < num_layers:
            neurons_by_layer[layer_idx].append(neuron_idx)
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="计算梯度依赖")):
            if num_samples and total_samples >= num_samples:
                break
            
            # 提取文本
            if isinstance(batch, dict):
                samples = batch.get('samples', [batch])
            elif isinstance(batch, list):
                samples = batch
            else:
                samples = [batch]
            
            texts = []
            for sample in samples:
                text = _extract_text_from_sample(sample)
                if text:
                    texts.append(text)
            
            if not texts:
                continue
            
            # Tokenize
            try:
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                print(f"[Gradient Dependency] 警告: 批处理 {batch_idx} tokenization 失败: {e}")
                continue
            
            # 按层分组处理
            for target_layer_idx in sorted(neurons_by_layer.keys()):
                if target_layer_idx >= num_layers:
                    continue
                
                # 获取该层的所有目标神经元
                layer_neuron_indices = neurons_by_layer[target_layer_idx]
                
                model.zero_grad(set_to_none=True)
                activation_storage.clear()
                
                try:
                    # 前向传播，捕获该层所有目标神经元的激活值
                    with torch.enable_grad():
                        outputs = model(**inputs)
                    
                    # 检查是否捕获到该层的激活值
                    layer_activations = {}
                    for target_neuron_idx in layer_neuron_indices:
                        if (target_layer_idx, target_neuron_idx) in activation_storage:
                            layer_activations[target_neuron_idx] = activation_storage[(target_layer_idx, target_neuron_idx)]
                    
                    if not layer_activations:
                        continue
                    
                    # 准备前一层down_proj权重（用于计算梯度）
                    prev_down_proj_weight = None
                    prev_layer_idx = None
                    if target_layer_idx > 0:
                        prev_layer_idx = target_layer_idx - 1
                        prev_layer = layers[prev_layer_idx]
                        prev_down_proj = _get_down_proj(prev_layer)
                        
                        if prev_down_proj is not None and hasattr(prev_down_proj, 'weight'):
                            weight = prev_down_proj.weight
                            if weight is not None and weight.requires_grad:
                                prev_down_proj_weight = weight
                    
                    # 对每个神经元分别计算梯度
                    for target_neuron_idx in layer_neuron_indices:
                        if target_neuron_idx not in layer_activations:
                            continue
                        
                        try:
                            # 获取目标神经元的激活值（带梯度）
                            target_activation = layer_activations[target_neuron_idx]  # (batch_size,)
                            
                            if not target_activation.requires_grad:
                                continue
                            
                            # 使用激活值的和作为loss（对batch求和）
                            loss = target_activation.sum()
                            
                            # 如果前一层存在，计算梯度
                            if prev_down_proj_weight is not None and prev_layer_idx is not None:
                                # 使用 torch.autograd.grad 直接计算梯度
                                is_last_neuron = (target_neuron_idx == layer_neuron_indices[-1])
                                
                                try:
                                    upstream_grad = torch.autograd.grad(
                                        outputs=loss,
                                        inputs=prev_down_proj_weight,
                                        retain_graph=not is_last_neuron,
                                        create_graph=False,
                                        only_inputs=True,
                                        allow_unused=True,
                                    )[0]
                                    
                                    if upstream_grad is not None:
                                        # upstream_grad 形状: (hidden_dim, intermediate_size)
                                        # 对于前一层down_proj的每个神经元（第j个神经元）
                                        # 计算其对目标神经元的梯度关联强度
                                        # G_{i,j} = |∂a^k_down,i / ∂w^{k-1}_down,j|
                                        for upstream_neuron_idx in range(upstream_grad.shape[0]):
                                            # 获取该上游神经元权重的梯度向量
                                            weight_grad_vector = upstream_grad[upstream_neuron_idx]  # (intermediate_size,)
                                            
                                            # 使用L2范数作为梯度关联强度
                                            gradient_strength = weight_grad_vector.norm().item()
                                            
                                            if gradient_strength > 0:
                                                gradient_dependencies[
                                                    (target_layer_idx, target_neuron_idx)
                                                ][(prev_layer_idx, upstream_neuron_idx)] += gradient_strength
                                    
                                    # 释放梯度张量
                                    if upstream_grad is not None:
                                        del upstream_grad
                                
                                except RuntimeError as e:
                                    if "graph" in str(e).lower() or "retain" in str(e).lower():
                                        continue
                                    print(f"[Gradient Dependency] 警告: 批处理 {batch_idx}, 神经元 ({target_layer_idx}, {target_neuron_idx}) 梯度计算失败: {e}")
                                    continue
                            
                            # 释放激活值内存
                            del target_activation, loss
                        
                        except Exception as e:
                            print(f"[Gradient Dependency] 警告: 批处理 {batch_idx}, 神经元 ({target_layer_idx}, {target_neuron_idx}) 梯度计算失败: {e}")
                            continue
                    
                    # 释放前向传播的输出和激活存储
                    del outputs, layer_activations
                    activation_storage.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"[Gradient Dependency] 警告: 批处理 {batch_idx}, 层 {target_layer_idx} 前向传播失败: {e}")
                    model.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            total_samples += len(texts)
            successful_batches += 1
            
            # 每10个批次显示一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                total_correlations = sum(len(correlations) for correlations in gradient_dependencies.values())
                neurons_with_correlations = sum(1 for correlations in gradient_dependencies.values() if len(correlations) > 0)
                print(f"[Gradient Dependency] 进度: 批次 {batch_idx + 1}/{total_batches}, "
                      f"样本 {total_samples}, 已收集关联: {total_correlations} (涉及 {neurons_with_correlations} 个神经元)")
    
    finally:
        # 移除所有hooks
        for hook_handle in activation_hooks.values():
            hook_handle.remove()
    
    print(f"[Gradient Dependency] 完成: 成功处理 {successful_batches}/{total_batches} 批次, 共 {total_samples} 个样本")
    
    # 处理结果：选择top-k%的强关联
    results = {}
    
    for (target_layer_idx, target_neuron_idx) in target_neurons.keys():
        dependencies = gradient_dependencies[(target_layer_idx, target_neuron_idx)]
        
        if not dependencies:
            results[(target_layer_idx, target_neuron_idx)] = {
                'upstream_neurons': [],
                'gradient_strengths': [],
                'mean_gradient_strength': 0.0,
                'max_gradient_strength': 0.0,
            }
            continue
        
        # 按梯度强度排序
        sorted_dependencies = sorted(dependencies.items(), key=lambda x: x[1], reverse=True)
        
        # 选择top-k%
        num_keep = max(1, int(len(sorted_dependencies) * top_k))
        top_dependencies = sorted_dependencies[:num_keep]
        
        upstream_neurons = [neuron for neuron, _ in top_dependencies]
        gradient_strengths = [strength for _, strength in top_dependencies]
        
        # 计算统计信息
        mean_strength = np.mean(gradient_strengths) if gradient_strengths else 0.0
        max_strength = max(gradient_strengths) if gradient_strengths else 0.0
        
        # 归一化梯度强度（相对于最大值）
        if gradient_strengths and max_strength > 0:
            gradient_strengths = [s / max_strength for s in gradient_strengths]
        
        results[(target_layer_idx, target_neuron_idx)] = {
            'upstream_neurons': upstream_neurons,
            'gradient_strengths': gradient_strengths,
            'mean_gradient_strength': float(mean_strength),
            'max_gradient_strength': float(max_strength),
        }
    
    print(f"[Gradient Dependency] 完成分析，共 {len(results)} 个目标神经元")
    
    return results


def visualize_gradient_dependency(
    gradient_dependency: Dict[Tuple[int, int], Dict],
    output_path: str,
    top_n: int = 20,
):
    """
    可视化梯度依赖关系
    
    Args:
        gradient_dependency: 梯度依赖关系结果
        output_path: 输出文件路径（保存为JSON格式）
        top_n: 每个目标神经元显示前N个强关联
    """
    import json
    
    # 准备可视化数据
    visualization_data = {}
    
    for (layer_idx, neuron_idx), deps in gradient_dependency.items():
        upstream_neurons = deps.get('upstream_neurons', [])
        gradient_strengths = deps.get('gradient_strengths', [])
        
        # 只保留前top_n个
        top_upstream = upstream_neurons[:top_n]
        top_strengths = gradient_strengths[:top_n]
        
        visualization_data[f"layer_{layer_idx}_neuron_{neuron_idx}"] = {
            'target_neuron': [int(layer_idx), int(neuron_idx)],
            'upstream_neurons': [[int(l), int(n)] for l, n in top_upstream],
            'gradient_strengths': [float(s) for s in top_strengths],
            'mean_gradient_strength': deps.get('mean_gradient_strength', 0.0),
            'max_gradient_strength': deps.get('max_gradient_strength', 0.0),
        }
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(visualization_data, f, indent=2, ensure_ascii=False)
    
    print(f"[Gradient Dependency] 可视化数据已保存到: {output_path}")
