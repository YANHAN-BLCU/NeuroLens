"""
参数对齐分析模块

根据论文5.4节要求，计算神经元参数与毒性向量的余弦相似度（S_i^k）。

功能：
- 计算每个目标神经元的W_down行向量与毒性向量w_toxic的余弦相似度
- 判断参数对齐方向（S+为正对齐，S-为负对齐）
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from pathlib import Path


def _get_transformer_layers(model: nn.Module):
    """获取Transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _get_up_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的up_proj层"""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
        return layer.mlp.up_proj
    if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "up_proj"):
        return layer.feed_forward.up_proj
    return None


def _get_down_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的down_proj层"""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "down_proj"):
        return layer.feed_forward.down_proj
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
            
            # 尝试将层移动到实际设备
            # 注意：对于使用 device_map='auto' 的模型，直接移动可能不会工作
            # 但我们可以尝试访问层来触发加载
            try:
                # 尝试访问层的其他属性来触发加载
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


def _safe_get_weight_numpy(weight: torch.Tensor) -> Optional[np.ndarray]:
    """
    安全地获取权重的 numpy 数组
    
    处理以下情况：
    - meta tensor（占位符，无实际数据）
    - 量化权重（需要先反量化）
    - GPU 权重（安全移动到 CPU，确保同步）
    - 内存管理（使用 clone 避免共享内存）
    
    Args:
        weight: PyTorch 张量
        
    Returns:
        numpy 数组，如果无法获取则返回 None
    """
    # 检查是否是 meta tensor
    if weight.device.type == 'meta':
        return None
    
    # 安全地移动到 CPU 并转换为 numpy
    try:
        # 使用 no_grad 确保不追踪梯度，节省内存
        with torch.no_grad():
            # 分离计算图
            weight_detached = weight.detach()
            
            # 如果权重在 GPU 上，安全地移动到 CPU
            if weight_detached.device.type == 'cuda':
                # 确保 GPU 操作完成（同步）
                torch.cuda.synchronize(weight_detached.device)
                
                # 移动到 CPU（这会自动同步）
                weight_cpu = weight_detached.cpu()
                
                # 再次同步确保数据已传输完成
                # 注意：cpu() 操作本身是同步的，但显式同步更安全
            else:
                weight_cpu = weight_detached
            
            # 使用 clone() 确保不共享内存，避免后续修改影响原张量
            # 这对于大模型的内存管理很重要
            weight_cpu_cloned = weight_cpu.clone()
            
            # 转换为 numpy
            # 注意：如果权重是量化格式，这里可能会失败
            numpy_array = weight_cpu_cloned.numpy()
            
            # 确保返回的是连续数组（某些操作可能产生非连续数组）
            if not numpy_array.flags['C_CONTIGUOUS']:
                numpy_array = np.ascontiguousarray(numpy_array)
            
            return numpy_array
            
    except NotImplementedError as e:
        # meta tensor 或其他不支持的操作
        if 'meta' in str(e).lower() or 'no data' in str(e).lower():
            return None
        raise
    except RuntimeError as e:
        # CUDA 相关错误
        if 'cuda' in str(e).lower() or 'device' in str(e).lower():
            print(f"[Parameter Alignment] 警告: GPU 操作失败 (设备: {weight.device}): {e}")
            return None
        raise
    except Exception as e:
        print(f"[Parameter Alignment] 警告: 无法获取权重 (设备: {weight.device}, 类型: {type(weight)}): {e}")
        return None


def compute_parameter_alignment(
    model: nn.Module,
    toxic_vectors_path: str,
    target_neurons: Optional[Dict[Tuple[int, int], Dict]] = None,
    projection_method: str = "up_proj_transpose",
) -> Dict[Tuple[int, int], Dict]:
    """
    计算参数对齐（S_i^k）：计算每个目标神经元的W_down行向量与毒性向量w_toxic的余弦相似度
    
    根据论文5.4节，参数对齐定义为：
        S_i^k = (w_down,i^k · w_toxic^k) / (||w_down,i^k|| ||w_toxic^k||)
    其中：
        - w_down,i^k ∈ R^d 是第k层MLP down_proj的第i行（对应第i个神经元）
        - w_toxic^k 是第k层的毒性向量
        - · 表示点积，||·|| 表示L2范数
    
    解释：
        - S_i^k > 0 (S+): 参数对齐为正，表示神经元参数方向促进有害内容生成
        - S_i^k < 0 (S-): 参数对齐为负，表示神经元参数方向有助于防御性转向
    
    Args:
        model: 语言模型
        toxic_vectors_path: 毒性向量文件路径（.npz格式）
            文件应包含：
                - 'vectors': (num_layers, hidden_dim) 所有层的毒性向量
                - 'layer_indices': (num_layers,) 层索引数组
        target_neurons: 目标神经元集合，格式为 Dict[(layer_idx, neuron_idx), Dict]
            如果为None，则分析所有层的所有神经元
        projection_method: 投影方法，可选值：
            - "up_proj_transpose" (默认): 使用 up_proj 的转置进行投影，最准确
                - 利用 MLP 的语义结构：up_proj 将 hidden_dim -> intermediate_size
                - 其转置将 intermediate_size -> hidden_dim，保持语义一致性
            - "truncate": 简单截取前 hidden_dim 个维度，快速但不准确
                - 会丢失约 71% 的维度信息
    
    Returns:
        Dict[(layer_idx, neuron_idx), {
            'cosine_similarity': float,  # 余弦相似度 [-1, 1]
            'alignment_type': 'S+' | 'S-',  # 对齐类型：正对齐或负对齐
            'neuron_weight_norm': float,  # 神经元权重向量的L2范数
            'toxic_vector_norm': float,  # 毒性向量的L2范数
        }]
    """
    print("[Parameter Alignment] 加载毒性向量...")
    
    # 加载毒性向量
    toxic_data = np.load(toxic_vectors_path, allow_pickle=True)
    vectors = toxic_data['vectors']  # (num_layers, hidden_dim)
    toxic_layer_indices = toxic_data['layer_indices']  # (num_layers,)
    
    print(f"[Parameter Alignment] 加载了 {len(toxic_layer_indices)} 层的毒性向量")
    
    # 获取模型层结构
    layers = _get_transformer_layers(model)
    if layers is None:
        raise ValueError("无法获取模型的层结构，请确保模型是Llama架构")
    
    # 构建层索引到毒性向量的映射
    layer_to_toxic_idx = {}
    for idx, layer_idx in enumerate(toxic_layer_indices):
        layer_to_toxic_idx[int(layer_idx)] = idx
    
    print("[Parameter Alignment] 计算参数对齐...")
    parameter_alignment = {}
    
    # 如果指定了目标神经元，只分析这些神经元
    if target_neurons is not None:
        target_layers = set(layer_idx for layer_idx, _ in target_neurons.keys())
        total_neurons = len(target_neurons)
        print(f"[Parameter Alignment] 将分析 {total_neurons} 个目标神经元，分布在 {len(target_layers)} 层中")
    else:
        target_layers = None
        total_neurons = None
    
    # 遍历所有层
    for layer_idx, layer in enumerate(layers):
        # 如果指定了目标神经元，跳过不在目标中的层
        if target_layers is not None and layer_idx not in target_layers:
            continue
        
        # 检查该层是否有毒性向量
        if layer_idx not in layer_to_toxic_idx:
            continue
        
        # 获取该层的毒性向量
        toxic_idx = layer_to_toxic_idx[layer_idx]
        w_toxic = vectors[toxic_idx]  # (hidden_dim,)
        w_toxic_norm = np.linalg.norm(w_toxic)
        
        if w_toxic_norm < 1e-10:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的毒性向量范数过小，跳过")
            continue
        
        # 获取 MLP down_proj 权重
        down_proj = _get_down_proj(layer)
        if down_proj is None:
            continue
        
        if not hasattr(down_proj, 'weight') or down_proj.weight is None:
            continue
        
        # 确保层已被正确加载（处理 device_map='auto' 的延迟加载）
        if not _ensure_layer_loaded(layer, layer_idx, model):
            # 如果层无法加载，尝试直接访问权重看看是否是 meta tensor
            if down_proj.weight.device.type == 'meta':
                print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重是 meta tensor，"
                      f"可能是由于使用 device_map='auto' 导致的延迟加载。"
                      f"建议使用 device_map=None 或明确的设备映射来加载模型。")
            continue
        
        # 获取权重张量（处理量化权重）
        weight_tensor = down_proj.weight
        
        # 检查是否是量化权重（BitsAndBytes），需要反量化
        is_quantized = False
        try:
            # 检查是否是量化权重
            if hasattr(weight_tensor, 'quant_state') or hasattr(down_proj, 'quantization_config'):
                is_quantized = True
                # 尝试反量化
                if hasattr(weight_tensor, 'dequantize'):
                    weight_tensor = weight_tensor.dequantize()
                elif hasattr(weight_tensor, 'data') and hasattr(weight_tensor.data, 'dequantize'):
                    weight_tensor = weight_tensor.data.dequantize()
                else:
                    # 对于 BitsAndBytes 4-bit，可能需要使用特殊方法
                    # 尝试使用 state_dict 获取权重
                    try:
                        state_dict = down_proj.state_dict()
                        if 'weight' in state_dict:
                            weight_tensor = state_dict['weight']
                            if hasattr(weight_tensor, 'dequantize'):
                                weight_tensor = weight_tensor.dequantize()
                    except:
                        pass
                    
                    # 如果还是量化权重，尝试直接访问 base_layer（BitsAndBytes 包装）
                    if hasattr(down_proj, 'base_layer'):
                        try:
                            base_weight = down_proj.base_layer.weight
                            if hasattr(base_weight, 'dequantize'):
                                weight_tensor = base_weight.dequantize()
                            else:
                                weight_tensor = base_weight
                        except:
                            pass
        except Exception as e:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 处理量化权重时出错: {e}")
        
        # 检查权重形状（直接使用张量的 shape 属性，避免转换问题）
        if not hasattr(weight_tensor, 'shape'):
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重没有 shape 属性，跳过该层")
            continue
        
        weight_shape = weight_tensor.shape
        
        # 处理可能的展平情况（量化权重可能被展平）
        if len(weight_shape) == 1:
            # 如果是一维，尝试根据已知的模型架构重新reshape
            # 对于 Llama-3-8B: down_proj 应该是 (4096, 7168)
            # 29360128 = 4096 * 7168，所以如果总元素数是这个，可以reshape
            total_elements = weight_shape[0]
            if total_elements == 29360128:  # 4096 * 7168
                # 尝试reshape为 (4096, 7168)
                try:
                    weight_tensor = weight_tensor.reshape(4096, 7168)
                    weight_shape = weight_tensor.shape
                    print(f"[Parameter Alignment] 层 {layer_idx}: 检测到展平的量化权重，已重新reshape为 {weight_shape}")
                except:
                    print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重是一维 ({weight_shape})，无法自动reshape，跳过该层")
                    continue
            else:
                print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重是一维 ({weight_shape})，跳过该层")
                continue
        
        if len(weight_shape) != 2:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重形状异常: {weight_shape}，期望 2D 张量，跳过该层")
            continue
        
        # 处理错误的 reshape 情况：如果总元素数是 29360128，尝试 reshape 为 [4096, 7168]
        # 29360128 = 4096 * 7168 (Llama-3-8B 的 down_proj 形状)
        total_elements = weight_tensor.numel() if hasattr(weight_tensor, 'numel') else weight_shape[0] * weight_shape[1]
        if total_elements == 29360128 and weight_shape != (4096, 7168):
            try:
                weight_tensor = weight_tensor.reshape(4096, 7168)
                weight_shape = weight_tensor.shape
                print(f"[Parameter Alignment] 层 {layer_idx}: 检测到错误的权重形状，已重新reshape为 {weight_shape}")
            except Exception as e:
                print(f"[Parameter Alignment] 警告: 层 {layer_idx} 无法reshape权重从 {weight_shape} 到 (4096, 7168): {e}，跳过该层")
                continue
        
        # 获取维度信息（直接使用张量的形状）
        hidden_dim_from_weight = weight_shape[0]  # out_features
        intermediate_size = weight_shape[1]  # in_features
        hidden_dim = w_toxic.shape[0]  # 毒性向量的维度
        
        # 验证维度一致性
        if hidden_dim_from_weight != hidden_dim:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 输出维度 ({hidden_dim_from_weight}) "
                  f"与毒性向量维度 ({hidden_dim}) 不匹配，跳过该层")
            if is_quantized:
                print(f"[Parameter Alignment] 调试: 权重形状 = {weight_shape}, 权重类型 = {type(weight_tensor)}, "
                      f"总元素数 = {weight_tensor.numel() if hasattr(weight_tensor, 'numel') else 'unknown'}")
            continue
        
        # 安全地获取权重 numpy 数组（用于后续计算）
        weight = _safe_get_weight_numpy(weight_tensor)
        if weight is None:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重无法转换为 numpy，跳过该层")
            continue
        
        # 再次检查权重形状（防止转换过程中被展平）
        if len(weight.shape) != 2:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重在转换为 numpy 后形状异常: {weight.shape}，"
                  f"原始张量形状: {weight_tensor.shape}，跳过该层")
            continue
        
        # 确保维度匹配（再次验证）
        if weight.shape[0] != hidden_dim_from_weight or weight.shape[1] != intermediate_size:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重形状不匹配: "
                  f"numpy 数组形状 {weight.shape} vs 张量形状 {weight_tensor.shape}，跳过该层")
            continue
        
        # 根据投影方法获取投影矩阵
        layer_projection_method = projection_method  # 使用局部变量，避免修改函数参数
        projection_matrix = None
        if layer_projection_method == "up_proj_transpose":
            # 获取 up_proj 用于投影
            up_proj = _get_up_proj(layer)
            if up_proj is None or not hasattr(up_proj, 'weight') or up_proj.weight is None:
                print(f"[Parameter Alignment] 警告: 层 {layer_idx} 无法获取 up_proj，回退到截取方法")
                layer_projection_method = "truncate"
            else:
                # up_proj 形状: (intermediate_size, hidden_dim)
                # 转置后: (hidden_dim, intermediate_size)
                # 用于将 (intermediate_size,) 投影到 (hidden_dim,)
                up_proj_weight_tensor = up_proj.weight
                
                # 处理量化权重
                try:
                    if hasattr(up_proj_weight_tensor, 'quant_state') or hasattr(up_proj, 'quantization_config'):
                        if hasattr(up_proj_weight_tensor, 'dequantize'):
                            up_proj_weight_tensor = up_proj_weight_tensor.dequantize()
                        elif hasattr(up_proj_weight_tensor, 'data') and hasattr(up_proj_weight_tensor.data, 'dequantize'):
                            up_proj_weight_tensor = up_proj_weight_tensor.data.dequantize()
                except:
                    pass
                
                up_proj_weight = _safe_get_weight_numpy(up_proj_weight_tensor)
                if up_proj_weight is None:
                    print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 up_proj 权重是 meta tensor 或无法访问，回退到截取方法")
                    layer_projection_method = "truncate"
                else:
                    projection_matrix = up_proj_weight.T  # (hidden_dim, intermediate_size)
                    
                    # 验证维度
                    if projection_matrix.shape != (hidden_dim, intermediate_size):
                        print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 up_proj 转置形状 ({projection_matrix.shape}) "
                              f"与预期 ({hidden_dim}, {intermediate_size}) 不匹配，回退到截取方法")
                        layer_projection_method = "truncate"
                        projection_matrix = None
        
        # 打印维度信息和投影方法（仅第一层，用于调试）
        if layer_idx == 0:
            print(f"[Parameter Alignment] 层 {layer_idx} 维度信息:")
            print(f"  down_proj 权重形状: {weight.shape} = (hidden_dim={hidden_dim}, intermediate_size={intermediate_size})")
            print(f"  毒性向量形状: {w_toxic.shape} = (hidden_dim={hidden_dim},)")
            print(f"  投影方法: {layer_projection_method}")
            if projection_matrix is not None:
                print(f"  投影矩阵形状: {projection_matrix.shape} = (hidden_dim={hidden_dim}, intermediate_size={intermediate_size})")
        
        # 统计该层需要分析的神经元数量
        if target_neurons is not None:
            layer_neurons = [n_idx for (l_idx, n_idx) in target_neurons.keys() if l_idx == layer_idx]
            num_layer_neurons = len(layer_neurons)
            if num_layer_neurons > 0:
                print(f"[Parameter Alignment] 处理层 {layer_idx}: {num_layer_neurons} 个神经元")
        else:
            num_layer_neurons = weight.shape[0]
            if layer_idx % 5 == 0 or layer_idx == len(layers) - 1:
                print(f"[Parameter Alignment] 处理层 {layer_idx}/{len(layers)-1}: {num_layer_neurons} 个神经元")
        
        # 计算每个神经元（每行）与毒性向量的余弦相似度
        neurons_processed = 0
        for neuron_idx in range(weight.shape[0]):
            # 如果指定了目标神经元，只分析目标神经元
            if target_neurons is not None and (layer_idx, neuron_idx) not in target_neurons:
                continue
            
            # 获取完整的神经元权重向量（在 intermediate_size 空间中）
            neuron_weight_full = weight[neuron_idx, :]  # (intermediate_size,)
            
            # 投影到 hidden_dim 空间
            if layer_projection_method == "up_proj_transpose" and projection_matrix is not None:
                # 使用 up_proj 转置进行投影：neuron_weight = up_proj^T @ neuron_weight_full
                # 这保持了 MLP 的语义结构，是最准确的投影方法
                neuron_weight = projection_matrix @ neuron_weight_full  # (hidden_dim,)
            else:
                # 回退到截取方法：使用前 hidden_dim 个维度
                # 注意：这种方法会丢失后 (intermediate_size - hidden_dim) 个维度的信息
                if intermediate_size >= hidden_dim:
                    neuron_weight = neuron_weight_full[:hidden_dim]  # (hidden_dim,)
                else:
                    # 如果 intermediate_size < hidden_dim（不应该发生），进行零填充
                    neuron_weight = np.zeros(hidden_dim, dtype=neuron_weight_full.dtype)
                    neuron_weight[:intermediate_size] = neuron_weight_full
            
            neuron_weight_norm = np.linalg.norm(neuron_weight)
            
            if neuron_weight_norm < 1e-10:
                continue
            
            # 计算余弦相似度（现在两个向量都在 hidden_dim 空间中）
            dot_product = np.dot(neuron_weight, w_toxic)
            cosine_sim = dot_product / (neuron_weight_norm * w_toxic_norm)
            
            # 判断对齐方向
            alignment_type = 'S+' if cosine_sim > 0 else 'S-'
            
            parameter_alignment[(layer_idx, neuron_idx)] = {
                'cosine_similarity': float(cosine_sim),
                'alignment_type': alignment_type,
                'neuron_weight_norm': float(neuron_weight_norm),
                'toxic_vector_norm': float(w_toxic_norm),
            }
            neurons_processed += 1
            
            # 每处理100个神经元打印一次进度（仅在分析所有神经元时）
            if target_neurons is None and neurons_processed % 100 == 0:
                print(f"[Parameter Alignment] 层 {layer_idx}: 已处理 {neurons_processed}/{num_layer_neurons} 个神经元")
        
        if target_neurons is not None and num_layer_neurons > 0:
            print(f"[Parameter Alignment] 层 {layer_idx} 完成: 处理了 {neurons_processed} 个神经元")
    
    print(f"[Parameter Alignment] 完成: 分析了 {len(parameter_alignment)} 个神经元")
    
    # 统计对齐类型分布
    if parameter_alignment:
        s_plus_count = sum(1 for v in parameter_alignment.values() if v['alignment_type'] == 'S+')
        s_minus_count = len(parameter_alignment) - s_plus_count
        print(f"[Parameter Alignment] 对齐分布: S+={s_plus_count}, S-={s_minus_count}")
        
        # 统计余弦相似度范围
        cosine_sims = [v['cosine_similarity'] for v in parameter_alignment.values()]
        print(f"[Parameter Alignment] 余弦相似度范围: [{min(cosine_sims):.4f}, {max(cosine_sims):.4f}], "
              f"均值={np.mean(cosine_sims):.4f}, 标准差={np.std(cosine_sims):.4f}")
    
    return parameter_alignment


def save_parameter_alignment(
    parameter_alignment: Dict[Tuple[int, int], Dict],
    output_path: Path,
    filename: str = "parameter_alignment.json",
):
    """
    保存参数对齐结果到JSON文件
    
    Args:
        parameter_alignment: 参数对齐结果
        output_path: 输出目录
        filename: 输出文件名
    """
    import json
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化格式
    serializable = {}
    for (layer_idx, neuron_idx), data in parameter_alignment.items():
        key = f"layer_{layer_idx}_neuron_{neuron_idx}"
        serializable[key] = {
            'layer_idx': int(layer_idx),
            'neuron_idx': int(neuron_idx),
            **data
        }
    
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"[Parameter Alignment] 结果已保存到: {output_file}")
    return output_file
