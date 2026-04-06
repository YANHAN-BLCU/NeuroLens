"""
参数对齐分析模块

根据论文5.4节要求，计算神经元参数与毒性向量的余弦相似度（S_i^k）。

功能：
- 计算每个目标神经元的W_down行向量与毒性向量w_toxic的余弦相似度
- 判断参数对齐方向（S+为正对齐，S-为负对齐）
- 支持从探针输出加载毒性向量（enhanced 模式）
"""

import numpy as np
import torch
import torch.nn as nn
import json
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path


# ============================================================================
# 探针毒性向量加载器（与 activation_projection.py 保持一致）
# ============================================================================

def load_toxic_vectors_for_parameter_alignment(
    probe_output_dir: Union[str, Path] = "outputs",
    prefer_layer: Optional[int] = None,
    target_layers: Optional[List[int]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    """
    从探针输出目录加载毒性向量用于参数对齐分析。

    支持多种数据格式：
    新格式（engine/probes/linear_probe_balanced.py）：
        - outputs/probes/model_id/layer_XX/probe.pt
        - outputs/probes/model_id/layer_XX/toxic_vector.npz
        - outputs/probes/model_id/layer_XX/metrics.json
    旧格式（scripts/train_linear_probe_labels.py）：
        - outputs/linear_probes/layers/layerXX/toxic_vector.npz
        - outputs/linear_probes/layers/layerXX/metrics.json
    聚合格式：
        - outputs/toxicity_vectors/all_layers_toxicity_vectors.json

    Args:
        probe_output_dir: 探针输出目录
        prefer_layer: 优先使用的层（单层模式）
        target_layers: 目标层列表（多层模式，自动选择可用层）

    Returns:
        Tuple[Dict, Dict]: (vectors, metadata)
    """
    import re
    probe_output_dir = Path(probe_output_dir)
    vectors = {}
    metadata = {}

    def _get_layer_idx(name: str) -> Optional[int]:
        match = re.match(r'layer_(\d+)', name)
        if match:
            return int(match.group(1))
        match = re.match(r'layer(\d+)', name)
        if match:
            return int(match.group(1))
        return None

    # 策略1: 新格式 outputs/probes/model_id/layer_XX/
    probes_dir = probe_output_dir / "probes"
    if probes_dir.exists():
        for model_folder in sorted(probes_dir.iterdir()):
            if not model_folder.is_dir():
                continue
            for layer_folder in sorted(model_folder.iterdir()):
                if not layer_folder.is_dir():
                    continue
                layer_idx = _get_layer_idx(layer_folder.name)
                if layer_idx is None or layer_idx in vectors:
                    continue

                meta = {}

                # 加载 toxic_vector.npz（优先）
                toxic_npz = layer_folder / "toxic_vector.npz"
                if toxic_npz.exists():
                    try:
                        data = np.load(toxic_npz, allow_pickle=True)
                        if 'w_toxic' in data:
                            vectors[layer_idx] = data['w_toxic']
                        elif 'w_toxic_normalized' in data:
                            vectors[layer_idx] = data['w_toxic_normalized']
                        meta['b'] = float(data.get('b', 0.0))
                    except Exception:
                        pass

                # 从 probe.pt 提取
                if layer_idx not in vectors:
                    probe_pt = layer_folder / "probe.pt"
                    if probe_pt.exists():
                        try:
                            ckpt = torch.load(probe_pt, map_location='cpu', weights_only=False)
                            if 'linear.weight' in ckpt:
                                w = ckpt['linear.weight']
                                if w.shape[0] == 2:
                                    vectors[layer_idx] = (w[1] - w[0]).cpu().numpy()
                            elif 'fc.weight' in ckpt:
                                w = ckpt['fc.weight']
                                if w.shape[0] == 2:
                                    vectors[layer_idx] = (w[1] - w[0]).cpu().numpy()
                        except Exception:
                            pass

                # 加载 metrics.json
                metrics_json = layer_folder / "metrics.json"
                if metrics_json.exists():
                    try:
                        with open(metrics_json, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        if 'val_acc' in metrics:
                            meta['cv_accuracy'] = float(metrics['val_acc'])
                            meta['val_acc'] = float(metrics['val_acc'])
                        elif 'avg_val_acc' in metrics:
                            meta['cv_accuracy'] = float(metrics['avg_val_acc'])
                        meta['std'] = float(metrics.get('std', metrics.get('std_val_acc', 0)))
                        if 'val_roc_auc' in metrics:
                            meta['val_roc_auc'] = float(metrics['val_roc_auc'])
                        if 'val_pr_auc' in metrics:
                            meta['val_pr_auc'] = float(metrics['val_pr_auc'])
                    except Exception:
                        pass

                if layer_idx in vectors and meta:
                    metadata[layer_idx] = meta

    # 策略2: 旧格式 outputs/linear_probes/layers/
    if not vectors:
        layers_dir = probe_output_dir / "linear_probes" / "layers"
        if layers_dir.exists():
            for layer_folder in sorted(layers_dir.iterdir()):
                if layer_folder.is_dir() and layer_folder.name.startswith("layer"):
                    layer_idx = _get_layer_idx(layer_folder.name)
                    if layer_idx is None or layer_idx in vectors:
                        continue

                    meta = {}
                    toxic_npz = layer_folder / "toxic_vector.npz"
                    metrics_json = layer_folder / "metrics.json"

                    if toxic_npz.exists():
                        try:
                            data = np.load(toxic_npz, allow_pickle=True)
                            vectors[layer_idx] = data['w_toxic']
                            meta['b'] = float(data['b']) if 'b' in data else 0.0
                        except Exception:
                            pass

                    if metrics_json.exists():
                        try:
                            with open(metrics_json, 'r') as f:
                                metrics = json.load(f)
                            meta['cv_accuracy'] = metrics.get('avg_val_acc',
                                                             metrics.get('val_acc', 0.0))
                            meta['std'] = metrics.get('std_val_acc', 0)
                        except Exception:
                            pass

                    if layer_idx in vectors and meta:
                        metadata[layer_idx] = meta

    # 策略3: outputs/toxicity_vectors/
    if not vectors:
        toxicity_dir = probe_output_dir / "toxicity_vectors"
        if toxicity_dir.exists():
            all_vectors_path = toxicity_dir / "all_layers_toxicity_vectors.json"
            if all_vectors_path.exists():
                try:
                    with open(all_vectors_path, 'r') as f:
                        all_data = json.load(f)

                    if 'vectors' in all_data and 'layer_indices' in all_data:
                        for idx, layer_idx in enumerate(all_data['layer_indices']):
                            layer_idx = int(layer_idx)
                            vectors[layer_idx] = np.array(all_data['vectors'][idx])
                            metadata[layer_idx] = {
                                'cv_accuracy': all_data.get('cv_accuracies', {}).get(str(layer_idx), 0.0),
                                'std': all_data.get('stds', {}).get(str(layer_idx), 0.0),
                            }
                except Exception:
                    pass

    if not vectors:
        print(f"[Parameter Alignment] 警告: 未能从 {probe_output_dir} 加载毒性向量")
        return {}, {}

    print(f"[Parameter Alignment] 加载了 {len(vectors)} 层的毒性向量")

    # 打印层质量
    if metadata:
        sorted_meta = sorted(metadata.items(), key=lambda x: x[1].get('cv_accuracy', 0), reverse=True)
        print(f"[Parameter Alignment] 毒性向量质量（前5层）:")
        for layer_idx, meta in sorted_meta[:5]:
            acc = meta.get('cv_accuracy', 0) * 100
            std = meta.get('std', 0) * 100
            print(f"  Layer {layer_idx:2d}: CV Acc = {acc:.2f}%, Std = {std:.2f}%")

    return vectors, metadata


def select_optimal_layers_for_alignment(
    metadata: Dict[int, Dict],
    num_layers_to_select: int = 5,
    strategy: str = "accuracy_stability",
) -> List[int]:
    """
    根据探针元数据选择最佳的参数对齐分析层。

    Args:
        metadata: 各层的元数据（包含 cv_accuracy 和 std）
        num_layers_to_select: 选择的最优层数
        strategy: 选择策略
            - "accuracy": 只看准确率
            - "stability": 只看稳定性（标准差）
            - "accuracy_stability": 综合准确率和稳定性
            - "layer_coverage": 跨层分布选择（覆盖多个层级）

    Returns:
        List[int]: 选定的层索引列表
    """
    if not metadata:
        # 默认返回探针报告中推荐的层
        return [28, 31, 15, 14, 20][:num_layers_to_select]

    if strategy == "accuracy":
        # 只看准确率
        sorted_layers = sorted(metadata.items(),
                             key=lambda x: x[1].get('cv_accuracy', 0),
                             reverse=True)
        return [l for l, _ in sorted_layers[:num_layers_to_select]]

    elif strategy == "stability":
        # 只看稳定性
        sorted_layers = sorted(metadata.items(),
                              key=lambda x: x[1].get('std', float('inf')))
        return [l for l, _ in sorted_layers[:num_layers_to_select]]

    elif strategy == "accuracy_stability":
        # 综合评分：0.6 * accuracy_normalized + 0.4 * (1 - std_normalized)
        accs = [m.get('cv_accuracy', 0) for m in metadata.values()]
        stds = [m.get('std', 0) for m in metadata.values()]

        if not accs:
            return list(metadata.keys())[:num_layers_to_select]

        max_acc, min_acc = max(accs), min(accs)
        max_std, min_std = max(stds) if stds else 1, min(stds) if stds else 0

        scores = {}
        for layer_idx, meta in metadata.items():
            acc_norm = (meta.get('cv_accuracy', 0) - min_acc) / (max_acc - min_acc + 1e-10)
            std_norm = (meta.get('std', 0) - min_std) / (max_std - min_std + 1e-10)
            scores[layer_idx] = 0.6 * acc_norm + 0.4 * (1 - std_norm)

        sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [l for l, _ in sorted_layers[:num_layers_to_select]]

    elif strategy == "layer_coverage":
        # 跨层分布选择：S级(27-32)、A级(21-26)、B级(10-20) 各选一些
        groups = {
            'S': [l for l in metadata.keys() if l >= 27],
            'A': [l for l in metadata.keys() if 21 <= l <= 26],
            'B': [l for l in metadata.keys() if 10 <= l <= 20],
            'C': [l for l in metadata.keys() if l < 10],
        }

        selected = []
        per_group = max(1, num_layers_to_select // len([g for g, ls in groups.items() if ls]))

        for group_name in ['S', 'A', 'B', 'C']:
            group_layers = groups.get(group_name, [])
            if not group_layers:
                continue
            # 每组按准确率排序
            sorted_group = sorted(group_layers,
                                 key=lambda l: metadata[l].get('cv_accuracy', 0),
                                 reverse=True)
            selected.extend(sorted_group[:per_group])
            if len(selected) >= num_layers_to_select:
                break

        return selected[:num_layers_to_select]

    else:
        return list(metadata.keys())[:num_layers_to_select]


def _get_transformer_layers(model: nn.Module):
    """获取Transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _remap_toxic_vectors_to_hf_layer_index(
    layer_toxic_vectors: Dict[int, np.ndarray],
    num_model_layers: int,
) -> Dict[int, np.ndarray]:
    """
    extract_toxic_vectors.py 从子目录 layer01..layer32 解析出的层号为 1..N，
    而 HuggingFace CausalLM 的 decoder 层索引为 0..N-1。
    若检测到「连续 1..num_model_layers」，则转为 0..num_model_layers-1，避免只对齐到 31 层。
    """
    if num_model_layers <= 0 or not layer_toxic_vectors:
        return layer_toxic_vectors
    keys = sorted(layer_toxic_vectors.keys())
    k_min, k_max = keys[0], keys[-1]
    if (
        k_min == 1
        and k_max == num_model_layers
        and len(keys) == num_model_layers
        and keys == list(range(1, num_model_layers + 1))
    ):
        print(
            "[Parameter Alignment] 毒性向量层号为 1-based（与 layer01.. 文件夹一致），"
            "已映射为 HF 的 0-based 层索引"
        )
        return {k - 1: layer_toxic_vectors[k] for k in keys}
    return layer_toxic_vectors


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
    toxic_vectors: Union[str, Path, Dict[int, np.ndarray], None] = None,
    target_neurons: Optional[Dict[Tuple[int, int], Dict]] = None,
    projection_method: str = "up_proj_transpose",
    prefer_layer: Optional[int] = None,
    probe_output_dir: Optional[Union[str, Path]] = None,
    # 兼容旧调用方的别名参数名
    toxic_vectors_path: Union[str, Path, None] = None,
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
    
    增强功能（支持三种毒性向量输入方式）：
    1. 直接传入 Dict[int, np.ndarray]：预加载的毒性向量
    2. 传入探针输出目录路径：自动从 outputs/linear_probes/ 加载
    3. 传入 .npz 文件路径：传统方式，加载指定文件
    
    Args:
        model: 语言模型
        toxic_vectors: 毒性向量，支持三种输入：
            - Dict[int, np.ndarray]: 预加载的毒性向量 {layer_idx: w_toxic_array}
            - str/Path: 探针输出目录路径（如 "outputs"），自动从中加载
            - .npz 文件路径：传统方式，加载指定文件
        target_neurons: 目标神经元集合，格式为 Dict[(layer_idx, neuron_idx), Dict]
            如果为None，则分析所有层的所有神经元
        projection_method: 投影方法，可选值：
            - "up_proj_transpose" (默认): 使用 up_proj 的转置进行投影，最准确
                - 利用 MLP 的语义结构：up_proj 将 hidden_dim -> intermediate_size
                - 其转置将 intermediate_size -> hidden_dim，保持语义一致性
            - "truncate": 简单截取前 hidden_dim 个维度，快速但不准确
                - 会丢失约 71% 的维度信息
        prefer_layer: 优先使用的层（当 toxic_vectors 为目录路径时使用）
        probe_output_dir: 探针输出目录（当 toxic_vectors 为目录路径时使用）
    
    Returns:
        Dict[(layer_idx, neuron_idx), {
            'cosine_similarity': float,  # 余弦相似度 [-1, 1]
            'alignment_type': 'S+' | 'S-',  # 对齐类型：正对齐或负对齐
            'neuron_weight_norm': float,  # 神经元权重向量的L2范数
            'toxic_vector_norm': float,  # 毒性向量的L2范数
            'toxic_layer': int,  # 使用的毒性向量所属层
            'probe_cv_accuracy': float,  # 该层的探针 CV 准确率（如果有）
        }]
    """
    # ========================================================================
    # 毒性向量加载（支持三种方式）
    # ========================================================================
    
    vectors_metadata = {}  # 用于存储探针元数据

    # 兼容旧调用方使用的 toxic_vectors_path 别名
    if toxic_vectors is None and toxic_vectors_path is not None:
        toxic_vectors = toxic_vectors_path
    
    # 解析毒性向量输入
    if isinstance(toxic_vectors, dict):
        # 方式1：直接传入 Dict
        layer_toxic_vectors = toxic_vectors
        print(f"[Parameter Alignment] 使用预加载的 {len(layer_toxic_vectors)} 层毒性向量")
    elif isinstance(toxic_vectors, (str, Path)):
        vectors_path = Path(toxic_vectors)
        
        # 检查是否是文件还是目录
        if vectors_path.is_file() and vectors_path.suffix == '.npz':
            # 方式3：传统 .npz 文件
            toxic_data = np.load(str(vectors_path), allow_pickle=True)
            layer_toxic_vectors = {}
            if 'vectors' in toxic_data and 'layer_indices' in toxic_data:
                for idx, layer_idx in enumerate(toxic_data['layer_indices']):
                    layer_toxic_vectors[int(layer_idx)] = toxic_data['vectors'][idx]
            print(f"[Parameter Alignment] 从文件加载了 {len(layer_toxic_vectors)} 层毒性向量")
        else:
            # 方式2：探针输出目录
            layer_toxic_vectors, vectors_metadata = load_toxic_vectors_for_parameter_alignment(
                toxic_vectors, prefer_layer=prefer_layer
            )
    else:
        raise ValueError(f"toxic_vectors 类型错误: {type(toxic_vectors)}")
    
    if not layer_toxic_vectors:
        raise ValueError("未能加载任何毒性向量")

    # 获取模型层结构（须先于层号映射：与 HF 的 0-based 对齐）
    layers = _get_transformer_layers(model)
    if layers is None:
        raise ValueError("无法获取模型的层结构，请确保模型是Llama架构")

    layer_toxic_vectors = _remap_toxic_vectors_to_hf_layer_index(
        layer_toxic_vectors, len(layers)
    )

    # 构建层索引到毒性向量的映射
    layer_to_toxic_idx = {}
    vectors_array_list = []
    for idx, (layer_idx, vec) in enumerate(sorted(layer_toxic_vectors.items())):
        layer_to_toxic_idx[int(layer_idx)] = idx
        vectors_array_list.append(vec)

    vectors = np.stack(vectors_array_list) if len(vectors_array_list) > 1 else np.array([vectors_array_list[0]])

    print(f"[Parameter Alignment] 加载了 {len(layer_to_toxic_idx)} 层的毒性向量")

    # 遍历所有层
    parameter_alignment = {}
    for layer_idx, layer in enumerate(layers):
        # 如果指定了目标神经元，跳过不在目标神经元中的层
        if target_neurons is not None and not any(k[0] == layer_idx for k in target_neurons):
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
        
        # 获取 up_proj 用于动态推断 intermediate_size（处理展平权重需要）
        up_proj_for_dim = None
        _up_proj_intermediate_size = None
        _up_proj_hidden_dim = None
        try:
            up_proj_for_dim = _get_up_proj(layer)
            if up_proj_for_dim is not None and hasattr(up_proj_for_dim, 'weight') and up_proj_for_dim.weight is not None:
                up_proj_shape = up_proj_for_dim.weight.shape
                if len(up_proj_shape) == 2:
                    _up_proj_intermediate_size = up_proj_shape[0]
                    _up_proj_hidden_dim = up_proj_shape[1]
        except Exception:
            pass

        # 处理可能的展平情况（量化权重可能被展平）
        if len(weight_shape) == 1:
            # 如果是一维，尝试根据 up_proj 推断的维度重新 reshape
            total_elements = weight_shape[0]
            if _up_proj_intermediate_size is not None and _up_proj_hidden_dim is not None:
                expected_total = _up_proj_hidden_dim * _up_proj_intermediate_size
                expected_shape = (_up_proj_hidden_dim, _up_proj_intermediate_size)
                if total_elements == expected_total:
                    try:
                        weight_tensor = weight_tensor.reshape(_up_proj_hidden_dim, _up_proj_intermediate_size)
                        weight_shape = weight_tensor.shape
                        print(f"[Parameter Alignment] 层 {layer_idx}: 检测到展平的量化权重，已重新reshape为 {weight_shape}")
                    except Exception:
                        print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重是一维 ({weight_shape})，无法reshape为 {expected_shape}，跳过该层")
                        continue
                else:
                    print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重是一维 ({weight_shape}，元素数={total_elements})，"
                          f"与 up_proj 维度 ({expected_shape}，元素数={expected_total}) 不匹配，跳过该层")
                    continue
            else:
                print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重是一维 ({weight_shape})，且无法从 up_proj 推断维度，跳过该层")
                continue

        if len(weight_shape) != 2:
            print(f"[Parameter Alignment] 警告: 层 {layer_idx} 的 down_proj 权重形状异常: {weight_shape}，期望 2D 张量，跳过该层")
            continue

        # 处理错误的 reshape 情况：如果总元素数与 up_proj 维度不符，尝试自动修复
        total_elements = weight_tensor.numel() if hasattr(weight_tensor, 'numel') else weight_shape[0] * weight_shape[1]
        if _up_proj_hidden_dim is not None and _up_proj_intermediate_size is not None:
            expected_total = _up_proj_hidden_dim * _up_proj_intermediate_size
            expected_shape = (_up_proj_hidden_dim, _up_proj_intermediate_size)
            if total_elements == expected_total and weight_shape != expected_shape:
                try:
                    weight_tensor = weight_tensor.reshape(_up_proj_hidden_dim, _up_proj_intermediate_size)
                    weight_shape = weight_tensor.shape
                    print(f"[Parameter Alignment] 层 {layer_idx}: 检测到错误的权重形状，已重新reshape为 {weight_shape}")
                except Exception as e:
                    print(f"[Parameter Alignment] 警告: 层 {layer_idx} 无法reshape权重从 {weight_shape} 到 {expected_shape}: {e}，跳过该层")
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
            
            # 获取该层对应的探针元数据
            probe_cv_acc = vectors_metadata.get(layer_idx, {}).get('cv_accuracy', 0.0) if vectors_metadata else 0.0
            
            parameter_alignment[(layer_idx, neuron_idx)] = {
                'cosine_similarity': float(cosine_sim),
                'alignment_type': alignment_type,
                'neuron_weight_norm': float(neuron_weight_norm),
                'toxic_vector_norm': float(w_toxic_norm),
                'toxic_layer': layer_idx,  # 使用的毒性向量所属层
                'probe_cv_accuracy': probe_cv_acc,  # 该层的探针 CV 准确率（如果有）
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
