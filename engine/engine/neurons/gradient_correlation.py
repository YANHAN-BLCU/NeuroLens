"""
梯度关联分析模块

根据论文5.4节要求，分析神经元之间的梯度依赖关系（G_{i,j}）。

功能：
- 使用Wdown神经元作为锚点，追踪来自前一层模块的参数级影响
- 通过测量参数扰动如何传播到Wdown激活来量化上游神经元与安全机制的因果关系
- 计算梯度关联：G_{i,j} = ∂a^k_down,i / ∂w^k_upstream,j
  - a^k_down,i: 第k层down_proj的第i个神经元的激活值
  - w^k_upstream,j: 上游（前一层）第j个神经元的权重参数
- 选择top-k%的强关联神经元建立连接图

性能优化：
- 批量处理：对同一层的所有神经元只做一次前向传播，然后分别计算梯度
- 使用 torch.autograd.grad 直接计算梯度，避免重复的前向传播
- 智能内存管理：及时释放计算图和中间变量，减少内存占用
- 预计加速比：~N倍（N为每层平均神经元数），大幅减少计算时间
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm


def _get_transformer_layers(model: nn.Module):
    """获取Transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _get_mlp_module(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP模块"""
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    return None


def _get_down_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的down_proj层"""
    mlp = _get_mlp_module(layer)
    if mlp is None:
        return None
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj
    if hasattr(mlp, "output"):
        return mlp.output
    if hasattr(mlp, "fc2"):
        return mlp.fc2
    if hasattr(mlp, "w2"):
        return mlp.w2
    return None


def _get_up_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的up_proj层"""
    mlp = _get_mlp_module(layer)
    if mlp is None:
        return None
    if hasattr(mlp, "up_proj"):
        return mlp.up_proj
    if hasattr(mlp, "w1"):
        return mlp.w1
    if hasattr(mlp, "fc1"):
        return mlp.fc1
    return None


def _get_gate_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的gate_proj层"""
    mlp = _get_mlp_module(layer)
    if mlp is None:
        return None
    if hasattr(mlp, "gate_proj"):
        return mlp.gate_proj
    if hasattr(mlp, "w3"):
        return mlp.w3
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


def _get_actual_device_from_model(model: nn.Module) -> Optional[torch.device]:
    """
    从模型中推断实际设备（用于处理 meta tensor）
    
    当某些层是 meta tensor 时，我们需要从其他已加载的层推断实际设备。
    
    Args:
        model: 模型
        
    Returns:
        实际设备，如果无法推断则返回 None
    """
    # 方法1: 检查所有参数的设备
    param_devices = set()
    for param in model.parameters():
        if param.device.type != 'meta':
            param_devices.add(param.device)
    
    if param_devices:
        # 优先使用 GPU
        gpu_devices = [d for d in param_devices if d.type == 'cuda']
        if gpu_devices:
            return gpu_devices[0]
        return next(iter(param_devices))
    
    # 方法2: 检查 hf_device_map（accelerate）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # 找到第一个非 meta 设备
        for module_name, device_name in model.hf_device_map.items():
            if device_name != 'meta' and device_name is not None:
                try:
                    return torch.device(device_name)
                except:
                    pass
    
    # 方法3: 默认设备
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def _infer_input_device(model: nn.Module, default_device: torch.device) -> torch.device:
    """推断模型输入应该使用的设备（兼容性函数）"""
    actual_device = _get_actual_device_from_model(model)
    if actual_device is not None:
        return actual_device
    return default_device


def _ensure_layer_loaded(layer: nn.Module, layer_idx: int, model: nn.Module) -> bool:
    """
    确保层已被正确加载到实际设备（处理 device_map='auto' 的延迟加载）
    
    当使用 device_map='auto' 时，某些层可能被标记为 meta device 或延迟加载。
    通过访问层的参数，可以触发实际的加载。
    
    Args:
        layer: 模型层
        layer_idx: 层索引（用于错误信息）
        model: 完整模型（用于推断设备）
        
    Returns:
        如果层已成功加载则返回 True，否则返回 False
    """
    try:
        down_proj = _get_down_proj(layer)
        if down_proj is None:
            return False
        
        if not hasattr(down_proj, 'weight') or down_proj.weight is None:
            return False
        
        # 如果权重是 meta tensor，尝试移动到实际设备
        if down_proj.weight.device.type == 'meta':
            # 推断实际设备
            actual_device = _get_actual_device_from_model(model)
            if actual_device is None:
                return False
            
            # 尝试访问层的其他属性来触发加载
            try:
                # 如果层使用了 accelerate 的延迟加载，访问参数会触发加载
                _ = list(down_proj.parameters())
                
                # 再次检查权重设备
                if down_proj.weight.device.type == 'meta':
                    # 如果仍然是 meta，说明无法自动加载
                    # 这通常意味着模型使用了特殊的加载方式（如分片）
                    return False
            except Exception:
                return False
        
        # 尝试访问权重的形状来确保它已加载
        _ = down_proj.weight.shape
        
        return True
    except Exception as e:
        # 静默失败，让调用者处理
        return False


def _get_weight_tensor(module: nn.Module, layer_idx: int) -> Optional[torch.Tensor]:
    """
    安全地获取模块的权重张量，处理量化权重和 meta tensor
    
    Args:
        module: 模块（如 down_proj）
        layer_idx: 层索引（用于错误信息）
        
    Returns:
        权重张量，如果无法获取则返回 None
    """
    weight_tensor = None
    
    # 尝试获取权重
    if hasattr(module, 'weight'):
        weight_tensor = module.weight
    elif hasattr(module, 'base_layer') and hasattr(module.base_layer, 'weight'):
        weight_tensor = module.base_layer.weight
    
    if weight_tensor is None:
        return None
    
    # 检查是否是 meta tensor
    if weight_tensor.device.type == 'meta':
        return None
    
    # 处理量化权重
    is_quantized = False
    try:
        # 检查是否是量化权重
        if hasattr(weight_tensor, 'quant_state') or hasattr(module, 'quantization_config'):
            is_quantized = True
            # 尝试反量化
            if hasattr(weight_tensor, 'dequantize'):
                weight_tensor = weight_tensor.dequantize()
            elif hasattr(weight_tensor, 'data') and hasattr(weight_tensor.data, 'dequantize'):
                weight_tensor = weight_tensor.data.dequantize()
            else:
                # 尝试使用 state_dict 获取权重
                try:
                    state_dict = module.state_dict()
                    if 'weight' in state_dict:
                        weight_tensor = state_dict['weight']
                        if hasattr(weight_tensor, 'dequantize'):
                            weight_tensor = weight_tensor.dequantize()
                except:
                    pass
                
                # 如果还是量化权重，尝试直接访问 base_layer（BitsAndBytes 包装）
                if hasattr(module, 'base_layer'):
                    try:
                        base_weight = module.base_layer.weight
                        if hasattr(base_weight, 'dequantize'):
                            weight_tensor = base_weight.dequantize()
                        else:
                            weight_tensor = base_weight
                    except:
                        pass
    except Exception as e:
        # 量化权重处理失败，但继续尝试使用原始权重
        pass
    
    # 检查权重是否为浮点类型（梯度计算需要）
    if hasattr(weight_tensor, 'dtype') and weight_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        # 如果不是浮点类型，无法计算梯度
        return None
    
    return weight_tensor


def _enable_gradients_for_float_params(model: nn.Module):
    """
    只对浮点类型的参数启用梯度计算，跳过量化参数
    
    当模型使用 bitsandbytes 量化时，量化参数不是浮点类型，
    不能直接设置 requires_grad。此函数只对浮点类型参数设置 requires_grad。
    """
    for param in model.parameters():
        # 只对浮点类型的参数设置 requires_grad
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad_(True)
        # 对于量化参数（如 int8, uint8, int4 等），跳过


def _enable_gradients_selectively(model: nn.Module, target_layers: set, layers: list):
    """
    选择性启用梯度：只对需要的层（目标层和前一层）启用梯度计算
    
    Args:
        model: 模型
        target_layers: 目标层索引集合
        layers: 层列表
    """
    # 需要启用梯度的层：目标层和前一层
    layers_to_enable = set(target_layers)
    for layer_idx in target_layers:
        if layer_idx > 0:
            layers_to_enable.add(layer_idx - 1)
    
    # 禁用所有参数的梯度
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad_(False)
    
    # 只对需要的层启用梯度
    for layer_idx in layers_to_enable:
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            # 启用该层所有参数的梯度
            for param in layer.parameters():
                if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    param.requires_grad_(True)


def compute_gradient_correlation(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    target_neurons: Dict[Tuple[int, int], Dict],
    device: torch.device,
    top_k: float = 0.1,
    batch_size: int = 4,
    max_length: int = 1024,
    num_samples: Optional[int] = None,
    use_gradient_checkpointing: bool = False,
    selective_gradients: bool = True,
    skip_zero_gradients: bool = True,
) -> Dict[Tuple[int, int], Dict]:
    """
    计算梯度关联（G_{i,j}）：分析目标神经元与上游神经元的梯度依赖关系
    
    根据论文5.4节，使用Wdown神经元作为锚点，追踪来自前一层模块的参数级影响。
    通过测量参数扰动如何传播到Wdown激活来量化上游神经元与安全机制的因果关系。
    
    梯度关联定义为：G_{i,j} = ∂a^k_down,i / ∂w^k_upstream,j
    - a^k_down,i: 第k层down_proj的第i个神经元的激活值
    - w^k_upstream,j: 上游（前一层，即k-1层）第j个神经元的权重参数（down_proj权重）
    
    实现方法：
    1. 对目标神经元（down_proj的第i个神经元）的激活进行反向传播
    2. 计算前一层（layer k-1）down_proj权重的梯度
    3. 梯度绝对值作为关联强度：G_{i,j} = |∂a^k_down,i / ∂w^{k-1}_down,j|
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        dataset: 用于计算梯度的数据集
        target_neurons: 目标神经元集合，格式为 Dict[(layer_idx, neuron_idx), Dict]
        device: 计算设备
        top_k: 保留前k%的强关联（默认0.1，即10%）
        batch_size: 批大小（建议4-8，内存允许时可增大）
        max_length: 最大序列长度（建议512-1024，减少序列长度可显著加速）
        num_samples: 使用的样本数（None表示全部，建议100-500以平衡速度和准确率）
        use_gradient_checkpointing: 是否使用梯度检查点（减少激活值内存，但增加计算时间，建议False）
        selective_gradients: 是否只对需要的层启用梯度（减少梯度内存，建议True）
        skip_zero_gradients: 是否跳过零梯度计算（加速，默认True）
    
    Returns:
        Dict[(layer_idx, neuron_idx), {
            'upstream_neurons': List[Tuple[int, int]],  # 上游神经元列表（前一层神经元）
            'gradient_strengths': List[float],  # 对应的梯度强度（G_{i,j}）
        }]
    """
    print("[Gradient Correlation] 开始计算梯度关联...")
    
    # 预检查：检测量化模型
    print("[Gradient Correlation] 预检查: 检测模型配置...")
    has_quantized_weights = False
    quantized_layer_count = 0
    total_layer_count = 0
    
    layers_precheck = _get_transformer_layers(model)
    if layers_precheck:
        for layer_idx, layer in enumerate(layers_precheck):
            down_proj = _get_down_proj(layer)
            if down_proj is not None:
                total_layer_count += 1
                weight = _get_weight_tensor(down_proj, layer_idx)
                if weight is None:
                    quantized_layer_count += 1
                    has_quantized_weights = True
                elif hasattr(weight, 'dtype') and weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    quantized_layer_count += 1
                    has_quantized_weights = True
    
    if has_quantized_weights:
        print(f"[Gradient Correlation] ⚠️  警告: 检测到量化权重（{quantized_layer_count}/{total_layer_count} 层）")
        print(f"[Gradient Correlation] ⚠️  量化权重无法计算梯度，梯度关联分析可能无法正常工作！")
        print(f"[Gradient Correlation] ⚠️  解决方案:")
        print(f"[Gradient Correlation]     1. 重新运行时不使用 --load-in-4bit 或 --load-in-8bit")
        print(f"[Gradient Correlation]     2. 如果必须使用量化，尝试 --no-selective-gradients")
        print(f"[Gradient Correlation]     3. 使用更小的模型或减少 batch_size 以使用完整精度模型")
        print(f"[Gradient Correlation] 继续运行，但可能无法收集到梯度关联...")
    else:
        print(f"[Gradient Correlation] ✓ 模型权重检查通过（{total_layer_count} 层，无量化）")
    
    # 确保模型处于训练模式以计算梯度
    model.train()
    
    # 检查模型是否被 torch.compile 编译过（可能与 gradient checkpointing 不兼容）
    is_compiled = False
    if hasattr(model, '_orig_mod'):
        is_compiled = True
        print("[Gradient Correlation] 警告: 检测到模型可能被 torch.compile 编译过")
        print("[Gradient Correlation] 警告: 编译后的模型与 gradient checkpointing 可能存在兼容性问题")
    
    # 启用梯度检查点（如果支持）
    gradient_checkpointing_active = False
    if use_gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            gradient_checkpointing_active = True
            print("[Gradient Correlation] 已启用梯度检查点（减少激活值内存）")
        elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()
            gradient_checkpointing_active = True
            print("[Gradient Correlation] 已启用梯度检查点（减少激活值内存）")
        else:
            print("[Gradient Correlation] 警告: 模型不支持梯度检查点")
    
    # 如果检测到编译模型且启用了 gradient checkpointing，发出警告
    if is_compiled and gradient_checkpointing_active:
        print("[Gradient Correlation] 警告: 编译模型 + gradient checkpointing 可能存在兼容性问题")
        print("[Gradient Correlation] 如果遇到错误，建议禁用 --use-gradient-checkpointing")
    
    # 获取模型层结构（需要在启用梯度前获取）
    layers = _get_transformer_layers(model)
    if layers is None:
        raise ValueError("无法获取模型的层结构，请确保模型是Llama架构")
    
    # 选择性启用梯度或全部启用
    if selective_gradients:
        target_layers = set(layer_idx for layer_idx, _ in target_neurons.keys())
        _enable_gradients_selectively(model, target_layers, layers)
        print(f"[Gradient Correlation] 已选择性启用梯度（仅对 {len(target_layers)} 个目标层及其前一层）")
    else:
        # 启用梯度计算（只对浮点类型参数，跳过量化参数）
        _enable_gradients_for_float_params(model)
        print("[Gradient Correlation] 已启用所有参数的梯度")
    
    # 获取第一层的隐藏维度
    first_layer = layers[0]
    down_proj = _get_down_proj(first_layer)
    if down_proj is None:
        raise ValueError("无法获取模型的 down_proj 层")
    
    hidden_dim = down_proj.weight.data.shape[0]  # out_features
    num_layers = len(layers)
    
    print(f"[Gradient Correlation] 模型结构: {num_layers} 层, 隐藏维度: {hidden_dim}")
    print(f"[Gradient Correlation] 目标神经元: {len(target_neurons)} 个")
    
    # 调试：检查前几层的权重是否启用了梯度
    debug_layers = sorted(set(layer_idx for layer_idx, _ in target_neurons.keys()))[:3]
    for layer_idx in debug_layers:
        if layer_idx > 0:
            prev_layer = layers[layer_idx - 1]
            
            # 确保层已加载
            if not _ensure_layer_loaded(prev_layer, layer_idx - 1, model):
                print(f"[Gradient Correlation] 调试: 层 {layer_idx-1} 无法加载（可能是 meta tensor）")
                continue
            
            prev_down_proj = _get_down_proj(prev_layer)
            if prev_down_proj is not None:
                weight = _get_weight_tensor(prev_down_proj, layer_idx - 1)
                
                if weight is not None:
                    requires_grad = hasattr(weight, 'requires_grad') and weight.requires_grad
                    dtype = getattr(weight, 'dtype', 'unknown')
                    device = getattr(weight, 'device', 'unknown')
                    print(f"[Gradient Correlation] 调试: 层 {layer_idx-1} down_proj权重 dtype={dtype}, device={device}, requires_grad={requires_grad}")
                else:
                    print(f"[Gradient Correlation] 调试: 层 {layer_idx-1} down_proj权重获取失败（可能是量化权重或 meta tensor）")
    
    # 存储每个目标神经元的梯度关联
    # 格式: {(layer_idx, neuron_idx): {(upstream_layer, upstream_neuron): gradient_strength}}
    gradient_correlations = defaultdict(lambda: defaultdict(float))
    
    # 自定义 collate_fn：直接返回样本列表，避免 PyTorch 默认 collate 的字典合并问题
    # 这样可以处理不同样本有不同键的情况（如某些样本有 'question'，某些没有）
    def custom_collate_fn(batch):
        """自定义 collate 函数，直接返回样本列表"""
        return batch
    
    # 创建 DataLoader，使用自定义 collate_fn
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    total_batches = len(dataloader)
    if num_samples:
        total_batches = min(total_batches, (num_samples + batch_size - 1) // batch_size)
    
    print(f"[Gradient Correlation] 开始处理，共 {total_batches} 个批次...")
    
    total_samples = 0
    successful_batches = 0
    input_device = _infer_input_device(model, device)
    
    # 为每个目标神经元设置hook来捕获激活值
    activation_hooks = {}
    activation_storage = {}
    
    def create_activation_hook(layer_idx: int, neuron_idx: int):
        """创建捕获目标神经元激活的hook（保存带梯度的激活值）"""
        def hook(module, input, output):
            # output 是 down_proj 的输出，形状为 (batch_size, seq_len, hidden_dim)
            # 我们取最后一个token的激活值（保持梯度，不detach）
            if output is not None and isinstance(output, torch.Tensor):
                last_token_activation = output[:, -1, neuron_idx]  # (batch_size,)
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
    
    # 按层分组目标神经元，以便批量处理
    neurons_by_layer = defaultdict(list)
    for (layer_idx, neuron_idx) in target_neurons.keys():
        if layer_idx < num_layers:
            neurons_by_layer[layer_idx].append(neuron_idx)
    
    # 计算总迭代数（优化后：每个批次对每个层做一次前向传播，然后对每个神经元计算梯度）
    total_neurons = sum(len(neurons_by_layer[layer_idx]) for layer_idx in neurons_by_layer.keys())
    total_layers = len(neurons_by_layer)
    # 优化后：前向传播次数 = 批次 × 层数，梯度计算次数 = 批次 × 神经元数
    # 但梯度计算比前向传播快得多，所以主要时间在前向传播
    total_forward_passes = total_batches * total_layers
    total_gradient_computations = total_batches * total_neurons
    total_iterations = total_gradient_computations  # 主要迭代是梯度计算
    
    print(f"[Gradient Correlation] 优化后计算策略:")
    print(f"[Gradient Correlation]   - 前向传播次数: {total_forward_passes} (批次: {total_batches} × 层数: {total_layers})")
    print(f"[Gradient Correlation]   - 梯度计算次数: {total_gradient_computations} (批次: {total_batches} × 神经元: {total_neurons})")
    print(f"[Gradient Correlation]   - 预计加速比: ~{total_neurons // max(total_layers, 1)}x (相比优化前)")
    if total_iterations > 10000:
        print(f"[Gradient Correlation] 警告: 总迭代数较大，预计耗时较长。")
        print(f"[Gradient Correlation] 建议: 使用 --num-samples 参数减少样本数量以加速计算。")
        print(f"[Gradient Correlation] 例如: --num-samples 100 或 --num-samples 500")
    
    try:
        global_iteration = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="计算梯度关联")):
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
                inputs = {k: v.to(input_device) for k, v in inputs.items()}
            except Exception as e:
                print(f"[Gradient Correlation] 警告: 批处理 {batch_idx} tokenization 失败: {e}")
                continue
            
            # 按层分组处理，优化：对同一层的所有神经元只做一次前向传播
            for target_layer_idx in sorted(neurons_by_layer.keys()):
                if target_layer_idx >= num_layers:
                    continue
                
                # 获取该层的所有目标神经元
                layer_neuron_indices = neurons_by_layer[target_layer_idx]
                
                # 优化：对同一层的所有神经元，只做一次前向传播
                # 然后使用 retain_graph 和 torch.autograd.grad 分别计算每个神经元的梯度
                model.zero_grad(set_to_none=True)
                activation_storage.clear()
                
                try:
                    # 一次前向传播，捕获该层所有目标神经元的激活值
                    # 注意：如果启用了 gradient checkpointing，前向传播会被分段重算
                    # 这可能导致 hook 被多次调用，但应该不影响最终结果
                    # 为了兼容性，我们确保在 enable_grad 上下文中进行前向传播
                    with torch.enable_grad():
                        # 清空该层的激活值存储，避免 checkpoint 重算时的旧值干扰
                        for target_neuron_idx in layer_neuron_indices:
                            if (target_layer_idx, target_neuron_idx) in activation_storage:
                                del activation_storage[(target_layer_idx, target_neuron_idx)]
                        
                        # 执行前向传播（如果启用了 gradient checkpointing，会在这里分段计算）
                        # 添加错误处理，捕获可能的兼容性问题
                        try:
                            outputs = model(**inputs)
                        except RuntimeError as e:
                            # 如果是因为 gradient checkpointing 导致的错误，提供更清晰的错误信息
                            if "checkpoint" in str(e).lower() or "gradient" in str(e).lower():
                                print(f"[Gradient Correlation] 警告: 批次 {batch_idx} 层 {target_layer_idx} 前向传播失败（可能与 gradient checkpointing 不兼容）")
                                print(f"[Gradient Correlation] 错误详情: {e}")
                                print(f"[Gradient Correlation] 建议: 尝试禁用 --use-gradient-checkpointing 或减小 --batch-size")
                                raise RuntimeError(
                                    f"Gradient checkpointing 兼容性问题: {e}\n"
                                    "建议: 禁用 --use-gradient-checkpointing 或减小 --batch-size"
                                ) from e
                            raise
                    
                    # 检查是否捕获到该层的激活值
                    layer_activations = {}
                    for target_neuron_idx in layer_neuron_indices:
                        if (target_layer_idx, target_neuron_idx) in activation_storage:
                            layer_activations[target_neuron_idx] = activation_storage[(target_layer_idx, target_neuron_idx)]
                    
                    if not layer_activations:
                        # 调试信息：如果第一个批次没有捕获到激活值，打印警告
                        if batch_idx == 0:
                            print(f"[Gradient Correlation] 警告: 层 {target_layer_idx} 未捕获到激活值（可能hook未正确注册）")
                        continue
                    
                    # 准备前一层down_proj权重（用于计算梯度）
                    prev_down_proj_weight = None
                    prev_layer_idx = None
                    if target_layer_idx > 0:
                        prev_layer_idx = target_layer_idx - 1
                        prev_layer = layers[prev_layer_idx]
                        
                        # 确保层已加载
                        if not _ensure_layer_loaded(prev_layer, prev_layer_idx, model):
                            if batch_idx == 0:
                                print(f"[Gradient Correlation] 警告: 层 {prev_layer_idx} 无法加载（可能是 meta tensor），跳过该层的梯度计算")
                        else:
                            prev_down_proj = _get_down_proj(prev_layer)
                            
                            if prev_down_proj is not None:
                                # 使用辅助函数获取权重（处理量化权重和 meta tensor）
                                weight = _get_weight_tensor(prev_down_proj, prev_layer_idx)
                                
                                # 检查权重是否有效且启用了梯度
                                if weight is not None:
                                    # 检查是否启用了梯度
                                    if hasattr(weight, 'requires_grad') and weight.requires_grad:
                                        prev_down_proj_weight = weight
                                    else:
                                        # 如果权重没有启用梯度，尝试启用它
                                        if hasattr(weight, 'requires_grad_'):
                                            try:
                                                weight.requires_grad_(True)
                                                prev_down_proj_weight = weight
                                            except Exception as e:
                                                if batch_idx == 0:
                                                    print(f"[Gradient Correlation] 警告: 层 {prev_layer_idx} 权重无法启用梯度: {e}")
                                else:
                                    if batch_idx == 0:
                                        print(f"[Gradient Correlation] 警告: 层 {prev_layer_idx} 权重获取失败（可能是量化权重或 meta tensor）")
                    
                    # 对每个神经元分别计算梯度（使用 torch.autograd.grad，更高效）
                    num_neurons_processed = 0
                    for target_neuron_idx in layer_neuron_indices:
                        if target_neuron_idx not in layer_activations:
                            continue
                        
                        global_iteration += 1
                        num_neurons_processed += 1
                        
                        try:
                            # 获取目标神经元的激活值（带梯度）
                            target_activation = layer_activations[target_neuron_idx]  # (batch_size,)
                            
                            # 检查激活值是否有梯度
                            if not target_activation.requires_grad:
                                if batch_idx == 0 and target_neuron_idx == layer_neuron_indices[0]:
                                    print(f"[Gradient Correlation] 警告: 层 {target_layer_idx} 神经元 {target_neuron_idx} 的激活值没有梯度")
                                continue
                            
                            # 使用激活值的和作为loss（对batch求和）
                            loss = target_activation.sum()
                            
                            # 如果前一层存在，计算梯度
                            if prev_down_proj_weight is None and prev_layer_idx is not None and batch_idx == 0:
                                # 第一个批次时，如果权重为None，提供诊断信息
                                if target_neuron_idx == layer_neuron_indices[0]:  # 只对第一个神经元打印
                                    print(f"[Gradient Correlation] 诊断: 层 {target_layer_idx} 神经元 {target_neuron_idx}")
                                    print(f"  - 前一层索引: {prev_layer_idx}")
                                    if prev_layer_idx == 0:
                                        print(f"  ⚠️  目标层是第一层（layer 0），没有前一层，无法计算梯度关联")
                                    else:
                                        print(f"  ❌ 前一层权重获取失败，无法计算梯度")
                            
                            if prev_down_proj_weight is not None and prev_layer_idx is not None:
                                # 确保权重启用了梯度
                                if not prev_down_proj_weight.requires_grad:
                                    prev_down_proj_weight.requires_grad_(True)
                                
                                # 使用 torch.autograd.grad 直接计算梯度（更高效）
                                # 只计算前一层down_proj权重的梯度
                                # 最后一个神经元不需要 retain_graph
                                is_last_neuron = (num_neurons_processed == len([n for n in layer_neuron_indices if n in layer_activations]))
                                
                                try:
                                    upstream_grad = torch.autograd.grad(
                                        outputs=loss,
                                        inputs=prev_down_proj_weight,
                                        retain_graph=not is_last_neuron,  # 最后一个神经元可以释放计算图
                                        create_graph=False,
                                        only_inputs=True,
                                        allow_unused=True,  # 允许未使用的输入
                                    )[0]
                                    
                                    if upstream_grad is not None:
                                        # upstream_grad 形状: (hidden_dim, intermediate_size)
                                        # 对于前一层down_proj的每个神经元（第j个神经元）
                                        # 计算其对目标神经元的梯度关联强度
                                        # G_{i,j} = |∂a^k_down,i / ∂w^{k-1}_down,j|
                                        # 使用权重梯度的L2范数作为关联强度
                                        # 优化：如果启用skip_zero_gradients，先检查梯度是否全为零
                                        if skip_zero_gradients:
                                            # 快速检查：计算所有梯度的L2范数
                                            grad_norm = upstream_grad.norm().item()
                                            if grad_norm == 0:
                                                # 梯度全为零，跳过后续计算
                                                del upstream_grad
                                                continue
                                        
                                        # 记录找到的非零梯度数量（用于调试）
                                        non_zero_count = 0
                                        
                                        for upstream_neuron_idx in range(upstream_grad.shape[0]):
                                            # 获取该上游神经元权重的梯度向量
                                            weight_grad_vector = upstream_grad[upstream_neuron_idx]  # (intermediate_size,)
                                            
                                            # 优化：如果启用skip_zero_gradients，跳过零梯度向量
                                            if skip_zero_gradients:
                                                # 快速检查：计算该向量是否为零
                                                if weight_grad_vector.norm().item() == 0:
                                                    continue
                                            
                                            # 使用L2范数作为梯度关联强度
                                            gradient_strength = weight_grad_vector.norm().item()
                                            
                                            if gradient_strength > 0:
                                                non_zero_count += 1
                                                gradient_correlations[
                                                    (target_layer_idx, target_neuron_idx)
                                                ][(prev_layer_idx, upstream_neuron_idx)] += gradient_strength
                                        
                                        # 调试信息：第一个批次第一个神经元打印详细信息
                                        if batch_idx == 0 and target_neuron_idx == layer_neuron_indices[0]:
                                            print(f"[Gradient Correlation] 调试: 层 {target_layer_idx} 神经元 {target_neuron_idx}")
                                            print(f"  - 梯度形状: {upstream_grad.shape}")
                                            print(f"  - 梯度范数: {upstream_grad.norm().item():.6f}")
                                            print(f"  - 非零梯度数: {non_zero_count}/{upstream_grad.shape[0]}")
                                            if non_zero_count > 0:
                                                # 显示前几个非零梯度的值
                                                sample_strengths = []
                                                for idx in range(min(5, upstream_grad.shape[0])):
                                                    vec = upstream_grad[idx]
                                                    strength = vec.norm().item()
                                                    if strength > 0:
                                                        sample_strengths.append(strength)
                                                if sample_strengths:
                                                    print(f"  - 示例梯度强度: {sample_strengths[:3]}")
                                    
                                    # 释放梯度张量
                                    if upstream_grad is not None:
                                        del upstream_grad
                                
                                except RuntimeError as e:
                                    # 如果是因为计算图被释放导致的错误，提供更清晰的错误信息
                                    if "graph" in str(e).lower() or "retain" in str(e).lower():
                                        print(f"[Gradient Correlation] 警告: 批处理 {batch_idx}, 神经元 ({target_layer_idx}, {target_neuron_idx}) 梯度计算失败（计算图问题）: {e}")
                                    else:
                                        print(f"[Gradient Correlation] 警告: 批处理 {batch_idx}, 神经元 ({target_layer_idx}, {target_neuron_idx}) 梯度计算失败: {e}")
                                    continue
                            elif target_layer_idx == 0:
                                # 第0层没有前一层，跳过
                                pass
                            else:
                                # 前一层权重获取失败
                                if batch_idx == 0 and target_neuron_idx == layer_neuron_indices[0]:
                                    print(f"[Gradient Correlation] 警告: 层 {target_layer_idx} 无法获取前一层权重")
                            
                            # 释放激活值内存
                            del target_activation, loss
                        
                        except Exception as e:
                            print(f"[Gradient Correlation] 警告: 批处理 {batch_idx}, 神经元 ({target_layer_idx}, {target_neuron_idx}) 梯度计算失败: {e}")
                            continue
                    
                    # 释放前向传播的输出和激活存储
                    del outputs, layer_activations
                    activation_storage.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"[Gradient Correlation] 警告: 批处理 {batch_idx}, 层 {target_layer_idx} 前向传播失败: {e}")
                    model.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            total_samples += len(texts)
            successful_batches += 1
            
            # 每10个批次或每1000次迭代显示一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches or global_iteration % 1000 == 0:
                progress_pct = (global_iteration / total_iterations * 100) if total_iterations > 0 else 0
                # 统计已收集的梯度关联数量
                total_correlations = sum(len(correlations) for correlations in gradient_correlations.values())
                neurons_with_correlations = sum(1 for correlations in gradient_correlations.values() if len(correlations) > 0)
                print(f"[Gradient Correlation] 进度: 批次 {batch_idx + 1}/{total_batches}, "
                      f"迭代 {global_iteration}/{total_iterations} ({progress_pct:.1f}%), "
                      f"样本 {total_samples}, 已收集关联: {total_correlations} (涉及 {neurons_with_correlations} 个神经元)")
            
            # 第一个批次后检查是否收集到数据
            if batch_idx == 0:
                total_correlations = sum(len(correlations) for correlations in gradient_correlations.values())
                if total_correlations == 0:
                    print(f"[Gradient Correlation] 警告: 第一个批次后未收集到任何梯度关联")
                    print(f"[Gradient Correlation] 开始诊断...")
                    
                    # 诊断1: 检查激活值是否被捕获
                    print(f"[Gradient Correlation] 诊断1: 检查激活值捕获...")
                    activation_count = len(activation_storage)
                    print(f"  - 已注册的hook数量: {len(activation_hooks)}")
                    print(f"  - 捕获到的激活值数量: {activation_count}")
                    if activation_count == 0:
                        print(f"  ❌ 未捕获到任何激活值！")
                        print(f"  - 可能原因: hook未正确注册或前向传播未触发hook")
                        print(f"  - 检查: 确认目标神经元所在的层索引是否正确")
                    else:
                        print(f"  ✓ 已捕获到激活值")
                        # 检查激活值是否有梯度
                        sample_activation = next(iter(activation_storage.values()))
                        if hasattr(sample_activation, 'requires_grad'):
                            print(f"  - 激活值requires_grad: {sample_activation.requires_grad}")
                            if not sample_activation.requires_grad:
                                print(f"  ❌ 激活值未启用梯度！")
                                print(f"  - 修复: 确保在torch.enable_grad()上下文内进行前向传播")
                    
                    # 诊断2: 检查权重是否启用梯度
                    print(f"[Gradient Correlation] 诊断2: 检查权重梯度启用状态...")
                    weight_grad_enabled_count = 0
                    weight_grad_disabled_count = 0
                    weight_quantized_count = 0
                    
                    for layer_idx in sorted(neurons_by_layer.keys()):
                        if layer_idx > 0:
                            prev_layer_idx = layer_idx - 1
                            prev_layer = layers[prev_layer_idx]
                            
                            if not _ensure_layer_loaded(prev_layer, prev_layer_idx, model):
                                weight_quantized_count += 1
                                continue
                            
                            prev_down_proj = _get_down_proj(prev_layer)
                            if prev_down_proj is not None:
                                weight = _get_weight_tensor(prev_down_proj, prev_layer_idx)
                                if weight is not None:
                                    if hasattr(weight, 'requires_grad'):
                                        if weight.requires_grad:
                                            weight_grad_enabled_count += 1
                                        else:
                                            weight_grad_disabled_count += 1
                                    else:
                                        weight_quantized_count += 1
                    
                    print(f"  - 权重启用梯度: {weight_grad_enabled_count} 层")
                    print(f"  - 权重未启用梯度: {weight_grad_disabled_count} 层")
                    print(f"  - 量化权重（无法启用梯度）: {weight_quantized_count} 层")
                    
                    if weight_grad_enabled_count == 0 and weight_grad_disabled_count > 0:
                        print(f"  ❌ 所有权重都未启用梯度！")
                        print(f"  - 修复建议:")
                        print(f"    1. 如果使用了量化（--load-in-4bit/--load-in-8bit），量化权重无法计算梯度")
                        print(f"    2. 尝试禁用量化或使用 --no-selective-gradients")
                        print(f"    3. 检查模型是否处于训练模式: model.train()")
                    elif weight_quantized_count > 0:
                        print(f"  ⚠️  检测到量化权重，量化权重无法计算梯度")
                        print(f"  - 如果使用量化模型，这是正常的，但梯度关联分析可能无法正常工作")
                    
                    # 诊断3: 检查模型模式
                    print(f"[Gradient Correlation] 诊断3: 检查模型状态...")
                    print(f"  - 模型训练模式: {model.training}")
                    if not model.training:
                        print(f"  ⚠️  模型未处于训练模式，可能影响梯度计算")
                    
                    # 诊断4: 检查skip_zero_gradients设置
                    print(f"[Gradient Correlation] 诊断4: 检查配置...")
                    print(f"  - selective_gradients: {selective_gradients}")
                    print(f"  - skip_zero_gradients: {skip_zero_gradients}")
                    if skip_zero_gradients:
                        print(f"  ⚠️  已启用skip_zero_gradients，如果所有梯度都为零会被跳过")
                        print(f"  - 尝试: 使用 --no-skip-zero-gradients 禁用此选项")
                    
                    # 提供修复建议
                    print(f"\n[Gradient Correlation] ========== 修复建议 ==========")
                    
                    # 根据诊断结果提供针对性建议
                    if weight_quantized_count > 0 or (weight_grad_enabled_count == 0 and weight_grad_disabled_count == 0):
                        print(f"[Gradient Correlation] 🔴 主要问题: 量化权重无法计算梯度")
                        print(f"[Gradient Correlation] 解决方案:")
                        print(f"[Gradient Correlation]   重新运行脚本，移除量化参数:")
                        print(f"[Gradient Correlation]   python scripts/run_gradient_correlation.py \\")
                        print(f"[Gradient Correlation]       --model-path <你的模型路径> \\")
                        print(f"[Gradient Correlation]       --dataset-path <你的数据集路径> \\")
                        print(f"[Gradient Correlation]       --output-path <输出路径> \\")
                        print(f"[Gradient Correlation]       --target-neurons-path <目标神经元路径>")
                        print(f"[Gradient Correlation]   (不要使用 --load-in-4bit 或 --load-in-8bit)")
                    elif weight_grad_enabled_count == 0 and weight_grad_disabled_count > 0:
                        print(f"[Gradient Correlation] 🔴 主要问题: 所有权重都未启用梯度")
                        print(f"[Gradient Correlation] 解决方案:")
                        print(f"[Gradient Correlation]   尝试禁用选择性梯度:")
                        print(f"[Gradient Correlation]   python scripts/run_gradient_correlation.py \\")
                        print(f"[Gradient Correlation]       --model-path <你的模型路径> \\")
                        print(f"[Gradient Correlation]       --dataset-path <你的数据集路径> \\")
                        print(f"[Gradient Correlation]       --output-path <输出路径> \\")
                        print(f"[Gradient Correlation]       --target-neurons-path <目标神经元路径> \\")
                        print(f"[Gradient Correlation]       --no-selective-gradients")
                    elif activation_count == 0:
                        print(f"[Gradient Correlation] 🔴 主要问题: 未捕获到激活值")
                        print(f"[Gradient Correlation] 解决方案:")
                        print(f"[Gradient Correlation]   1. 检查目标神经元文件格式是否正确")
                        print(f"[Gradient Correlation]   2. 确认层索引在有效范围内（0 到 {num_layers-1}）")
                        print(f"[Gradient Correlation]   3. 检查数据集是否能成功tokenize")
                    elif skip_zero_gradients and weight_grad_enabled_count > 0:
                        print(f"[Gradient Correlation] 🟡 可能问题: 所有梯度都为零并被跳过")
                        print(f"[Gradient Correlation] 解决方案:")
                        print(f"[Gradient Correlation]   尝试禁用零梯度跳过:")
                        print(f"[Gradient Correlation]   python scripts/run_gradient_correlation.py \\")
                        print(f"[Gradient Correlation]       --model-path <你的模型路径> \\")
                        print(f"[Gradient Correlation]       --dataset-path <你的数据集路径> \\")
                        print(f"[Gradient Correlation]       --output-path <输出路径> \\")
                        print(f"[Gradient Correlation]       --target-neurons-path <目标神经元路径> \\")
                        print(f"[Gradient Correlation]       --no-skip-zero-gradients")
                    else:
                        print(f"[Gradient Correlation] 其他可能原因:")
                        print(f"[Gradient Correlation]   1. 检查目标神经元文件是否正确，层索引是否在有效范围内")
                        print(f"[Gradient Correlation]   2. 确认数据集格式正确，能够成功tokenize")
                        print(f"[Gradient Correlation]   3. 检查模型结构是否兼容")
                    
                    print(f"[Gradient Correlation] ======================================\n")
                    
                    # 如果激活值被捕获但权重未启用梯度，提供更具体的建议
                    if activation_count > 0 and weight_grad_enabled_count == 0:
                        print(f"[Gradient Correlation] 💡 提示: 激活值已捕获，但权重未启用梯度")
                        print(f"[Gradient Correlation]   这通常是因为使用了量化模型或selective_gradients配置问题")
                        print(f"[Gradient Correlation]   建议: 重新运行时不使用量化，或使用 --no-selective-gradients")
    
    finally:
        # 移除所有hooks
        for hook_handle in activation_hooks.values():
            hook_handle.remove()
    
    print(f"[Gradient Correlation] 完成: 成功处理 {successful_batches}/{total_batches} 批次, 共 {total_samples} 个样本")
    
    # 处理结果：选择top-k%的强关联
    results = {}
    
    for (target_layer_idx, target_neuron_idx) in target_neurons.keys():
        correlations = gradient_correlations[(target_layer_idx, target_neuron_idx)]
        
        if not correlations:
            results[(target_layer_idx, target_neuron_idx)] = {
                'upstream_neurons': [],
                'gradient_strengths': [],
            }
            continue
        
        # 按梯度强度排序
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # 选择top-k%
        num_keep = max(1, int(len(sorted_correlations) * top_k))
        top_correlations = sorted_correlations[:num_keep]
        
        upstream_neurons = [neuron for neuron, _ in top_correlations]
        gradient_strengths = [strength for _, strength in top_correlations]
        
        # 归一化梯度强度（相对于最大值）
        if gradient_strengths:
            max_strength = max(gradient_strengths)
            if max_strength > 0:
                gradient_strengths = [s / max_strength for s in gradient_strengths]
        
        results[(target_layer_idx, target_neuron_idx)] = {
            'upstream_neurons': upstream_neurons,
            'gradient_strengths': gradient_strengths,
        }
    
    print(f"[Gradient Correlation] 完成分析，共 {len(results)} 个目标神经元")
    
    return results
