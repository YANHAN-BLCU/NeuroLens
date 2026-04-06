"""
激活投影分析模块

根据论文5.4节要求，分析神经元在jailbreak样本中的激活模式（A_i^k）。

功能：
- 收集jailbreak样本在目标神经元上的激活值
- 将激活值投影到毒性向量上
- 分别统计成功和失败jailbreak样本的激活分布
- 支持从训练好的探针加载毒性向量（enhanced 模式）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, Tuple, Optional, List, Callable, Union
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# 探针毒性向量加载器（支持多种输出格式）
# ============================================================================

def load_toxic_vectors_from_probes(
    probe_output_dir: Union[str, Path] = "outputs",
    prefer_layer: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    """
    从探针训练输出加载毒性向量。

    支持多种数据格式：
    1. outputs/linear_probes/layers/layerXX/layerXX.pt  +  metrics.json
       （scripts/train_linear_probe_labels.py 输出格式）
    2. outputs/toxicity_vectors/all_layers_toxicity_vectors.json
       （scripts/extract_toxicity_vectors.py 聚合格式）
    3. outputs/toxicity_vectors/layer_toxicity_summary.json
       （仅含统计，不含向量）

    Args:
        probe_output_dir: 探针输出目录
        prefer_layer: 优先使用的层（默认为 None，自动选择）

    Returns:
        Tuple[Dict, Dict]:
            - vectors: {layer_idx: w_toxic_array}
            - metadata: {layer_idx: {cv_accuracy, std, b, ...}}
    """
    import re as re_module
    probe_output_dir = Path(probe_output_dir)
    vectors = {}
    metadata = {}

    def _get_layer_idx(name: str) -> Optional[int]:
        match = re_module.match(r'layer_(\d+)', name)
        if match:
            return int(match.group(1))
        match = re_module.match(r'layer(\d+)', name)
        if match:
            return int(match.group(1))
        return None

    # 收集所有 layer 目录（包括 linear_probes/layers/ 子目录）
    layer_dirs = []

    # 策略1: linear_probes/layers/ 目录
    layers_dir = probe_output_dir / "linear_probes" / "layers"
    if layers_dir.exists():
        for layer_folder in sorted(layers_dir.iterdir()):
            if layer_folder.is_dir() and _get_layer_idx(layer_folder.name) is not None:
                layer_dirs.append(layer_folder)

    # 策略2: 直接格式（layerXX/ 或 layer_XX/）
    for item in sorted(probe_output_dir.iterdir()):
        if item.is_dir() and _get_layer_idx(item.name) is not None:
            if not any(d.name == item.name for d in layer_dirs):
                layer_dirs.append(item)

    # 策略3: 从 outputs/toxicity_vectors/ 加载
    toxicity_dir = probe_output_dir / "toxicity_vectors"
    if toxicity_dir.exists():
        all_vectors_path = toxicity_dir / "all_layers_toxicity_vectors.json"
        if all_vectors_path.exists():
            try:
                with open(all_vectors_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)

                if 'layers' in all_data:
                    for layer_key, layer_data in all_data['layers'].items():
                        layer_idx = int(layer_key)
                        if 'w_toxic' in layer_data:
                            vectors[layer_idx] = np.array(layer_data['w_toxic'])
                        elif 'vector' in layer_data:
                            vectors[layer_idx] = np.array(layer_data['vector'])
                        metadata[layer_idx] = {
                            'cv_accuracy': float(layer_data.get('cv_accuracy', 0)),
                            'l2_norm': float(layer_data.get('vector_l2_norm', 0)),
                            'safe_acc': float(layer_data.get('cv_safe_acc', 0)),
                            'harm_acc': float(layer_data.get('cv_harmful_acc', 0)),
                        }
            except Exception:
                pass

        # layer_toxicity_summary.json 仅含统计，不含向量
        if not vectors:
            summary_path = toxicity_dir / "layer_toxicity_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    if 'layers_summary' in summary:
                        for layer_info in summary['layers_summary']:
                            layer_idx = int(layer_info['layer'])
                            metadata[layer_idx] = {
                                'cv_accuracy': float(layer_info.get('cv_accuracy', 0)),
                                'l2_norm': float(layer_info.get('l2_norm', 0)),
                            }
                except Exception:
                    pass

    # 策略4: 从 linear_probes/layers/layerXX/ 加载（补充向量和 metrics）
    for layer_folder in layer_dirs:
        layer_idx = _get_layer_idx(layer_folder.name)
        if layer_idx is None or layer_idx in vectors:
            continue

        meta = {}

        # 从 layerXX.pt 加载毒性向量（主要来源）
        for pt_name in [f"{layer_folder.name}.pt", "probe.pt", "best.pt"]:
            pt_path = layer_folder / pt_name
            if not pt_path.exists():
                continue
            try:
                checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
                if 'toxicity_vector' in checkpoint:
                    tv = checkpoint['toxicity_vector']
                    vectors[layer_idx] = np.array(tv) if not isinstance(tv, np.ndarray) else tv
                elif 'linear.weight' in checkpoint:
                    w = checkpoint['linear.weight']
                    if w.shape[0] == 2:
                        w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                        vectors[layer_idx] = w_arr[1] - w_arr[0]
                elif 'fc.weight' in checkpoint:
                    w = checkpoint['fc.weight']
                    if w.shape[0] == 2:
                        w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                        vectors[layer_idx] = w_arr[1] - w_arr[0]
                elif 'weight' in checkpoint:
                    w = checkpoint['weight']
                    if hasattr(w, 'shape') and w.shape[0] == 2:
                        w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                        vectors[layer_idx] = w_arr[1] - w_arr[0]
                break
            except Exception:
                continue

        # 加载 metrics.json（支持嵌套 avg_metrics 格式）
        metrics_json = layer_folder / "metrics.json"
        if metrics_json.exists():
            try:
                with open(metrics_json, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

                # 新格式：val_acc（顶层字段）
                if 'val_acc' in metrics:
                    meta['cv_accuracy'] = float(metrics['val_acc'])
                # 旧格式：avg_metrics.avg_val_acc
                elif 'avg_metrics' in metrics and 'avg_val_acc' in metrics['avg_metrics']:
                    meta['cv_accuracy'] = float(metrics['avg_metrics']['avg_val_acc'])

                # 标准差
                if 'std_val_acc' in metrics:
                    meta['std'] = float(metrics['std_val_acc'])
                elif 'avg_metrics' in metrics and 'std_val_acc' in metrics['avg_metrics']:
                    meta['std'] = float(metrics['avg_metrics']['std_val_acc'])

                # 旧格式安全/有害准确率
                if 'avg_metrics' in metrics:
                    if 'avg_val_s_acc' in metrics['avg_metrics']:
                        meta['safe_acc'] = float(metrics['avg_metrics']['avg_val_s_acc'])
                    if 'avg_val_h_acc' in metrics['avg_metrics']:
                        meta['harm_acc'] = float(metrics['avg_metrics']['avg_val_h_acc'])

                # 新格式额外指标
                if 'val_roc_auc' in metrics:
                    meta['val_roc_auc'] = float(metrics['val_roc_auc'])
                if 'val_pr_auc' in metrics:
                    meta['val_pr_auc'] = float(metrics['val_pr_auc'])

            except Exception:
                pass

        if layer_idx in vectors or meta:
            metadata[layer_idx] = meta

    if not vectors:
        print(f"[Toxic Vectors] 警告: 未能从 {probe_output_dir} 加载毒性向量")

    print(f"[Toxic Vectors] 加载了 {len(vectors)} 层的毒性向量")

    # 自动选择最佳层
    if prefer_layer is None and metadata:
        valid_meta = {k: v for k, v in metadata.items() if k in vectors}
        if valid_meta:
            best_layer = max(valid_meta.items(),
                           key=lambda x: x[1].get('cv_accuracy', 0))[0]
            print(f"[Toxic Vectors] 自动选择最佳层: Layer {best_layer}")

    return vectors, metadata


def get_best_toxic_vector_for_activation(
    probe_output_dir: Union[str, Path] = "outputs",
    prefer_layer: Optional[int] = None,
    target_layers: Optional[List[int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[int], Dict]:
    """
    获取最佳毒性向量用于激活投影分析。
    
    Args:
        probe_output_dir: 探针输出目录
        prefer_layer: 优先使用的层
        target_layers: 目标层列表（如果提供，返回这些层的平均毒性向量）
    
    Returns:
        Tuple: (w_toxic, layer_idx, metadata)
    """
    vectors, metadata = load_toxic_vectors_from_probes(probe_output_dir, prefer_layer=prefer_layer)

    if not vectors:
        return None, None, {}

    # 如果指定了目标层，计算平均毒性向量
    if target_layers:
        valid_layers = [l for l in target_layers if l in vectors]
        if valid_layers:
            avg_vector = np.mean([vectors[l] for l in valid_layers], axis=0)
            avg_metadata = {
                'layer': f"{min(valid_layers)}-{max(valid_layers)}",
                'num_layers': len(valid_layers),
                'cv_accuracy': np.mean([metadata.get(l, {}).get('cv_accuracy', 0) for l in valid_layers]),
            }
            return avg_vector, valid_layers[0], avg_metadata

    # 使用指定层或最佳层
    if prefer_layer is not None and prefer_layer in vectors:
        return vectors[prefer_layer], prefer_layer, metadata.get(prefer_layer, {})

    # 选择 CV 准确率最高的层
    valid_meta = {k: v for k, v in metadata.items() if k in vectors}
    if valid_meta:
        best_layer = max(valid_meta.items(), key=lambda x: x[1].get('cv_accuracy', 0))[0]
        return vectors[best_layer], best_layer, metadata.get(best_layer, {})

    return None, None, {}


def _get_transformer_layers(model: nn.Module):
    """获取Transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _remap_toxic_vectors_to_hf_layer_index(
    layer_vectors: Dict[int, np.ndarray],
    num_model_layers: int,
) -> Dict[int, np.ndarray]:
    """
    extract_toxic_vectors.py 从子目录 layer01..layer32 解析出的层号为 1..N，
    而 HuggingFace CausalLM 的 decoder 层索引为 0..N-1。
    若检测到「连续 1..num_model_layers」，则转为 0..num_model_layers-1。
    """
    if num_model_layers <= 0 or not layer_vectors:
        return layer_vectors
    keys = sorted(layer_vectors.keys())
    k_min, k_max = keys[0], keys[-1]
    if (
        k_min == 1
        and k_max == num_model_layers
        and len(keys) == num_model_layers
        and keys == list(range(1, num_model_layers + 1))
    ):
        print(
            "[激活投影] 毒性向量层号为 1-based（与 layer01.. 文件夹一致），"
            "已映射为 HF 的 0-based 层索引"
        )
        return {k - 1: layer_vectors[k] for k in keys}
    return layer_vectors


def _get_mlp_module(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP模块"""
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    return None


def _extract_text_from_sample(sample: Dict) -> Optional[str]:
    """从样本中提取文本"""
    # 支持多种数据格式
    if 'text' in sample:
        return sample['text']
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


def _is_jailbreak_successful(sample: Dict) -> Optional[bool]:
    """判断jailbreak是否成功"""
    # 支持多种字段名
    # 首先检查顶层字段
    if 'jailbreak_success' in sample:
        return bool(sample['jailbreak_success'])
    elif 'asr_label' in sample:
        # asr_label: 1表示成功，0表示失败
        return bool(sample['asr_label'] == 1)
    elif 'success' in sample:
        return bool(sample['success'])
    
    # 检查 guard 字段（嵌套结构）
    if 'guard' in sample and isinstance(sample['guard'], dict):
        guard = sample['guard']
        if 'jailbreak_success' in guard:
            return bool(guard['jailbreak_success'])
        elif 'asr_label' in guard:
            # asr_label: 1表示成功，0表示失败
            return bool(guard['asr_label'] == 1)
    
    # 检查 inference.guard 字段（更深层的嵌套）
    if 'inference' in sample and isinstance(sample['inference'], dict):
        inference = sample['inference']
        if 'guard' in inference and isinstance(inference['guard'], dict):
            guard = inference['guard']
            if 'jailbreak_success' in guard:
                return bool(guard['jailbreak_success'])
            elif 'asr_label' in guard:
                return bool(guard['asr_label'] == 1)
    
    return None


def compute_activation_projection(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    toxic_vectors: Union[str, Path, Dict[int, np.ndarray]],
    target_neurons: Optional[Dict[Tuple[int, int], Dict]],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 2048,
    num_samples: Optional[int] = None,
    prefer_layer: Optional[int] = None,
    probe_output_dir: Optional[Union[str, Path]] = None,
) -> Dict[Tuple[int, int], Dict]:
    """
    计算激活投影（A_i^k）：分析神经元在jailbreak样本中的激活模式
    
    根据论文5.4节，激活投影定义为：
        A_i^k = a_down,i^k · (w_toxic^k / ||w_toxic^k||)
    其中：
        - a_down,i^k ∈ R^d 是第k层第i个神经元在最后一个token的激活向量
          （维度 d 是 hidden_dim）
        - intermediate 是 MLP 中间层的激活值（gate*up 的输出，维度为 intermediate_size）
        - 使用 up_proj 的转置将 intermediate 从 intermediate_size 空间投影到 hidden_dim 空间：
          a_down = intermediate @ up_proj^T
        - 这保持了 MLP 的语义结构，与参数对齐方法（parameter_alignment.py）一致
        - w_toxic^k 是第k层的毒性向量
        - w_toxic^k / ||w_toxic^k|| 是归一化的毒性向量
    
    本函数分别统计成功和失败jailbreak样本的激活投影分布。
    
    增强功能（支持三种毒性向量输入方式）：
    1. 直接传入 Dict[int, np.ndarray]：预加载的毒性向量
    2. 传入探针输出目录路径：自动从 outputs/linear_probes/ 加载
    3. 传入 .npz 文件路径：传统方式，加载指定文件
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        dataset: 包含jailbreak样本的数据集
            每个样本应包含：
                - 文本字段（'text'、'prompt'或'input'）
                - jailbreak成功标志（'jailbreak_success'、'asr_label'或'success'）
        toxic_vectors: 毒性向量，支持三种输入：
            - Dict[int, np.ndarray]: 预加载的毒性向量 {layer_idx: w_toxic_array}
            - str/Path: 探针输出目录路径（如 "outputs"），自动从中加载
            - .npz 文件路径：传统方式，加载指定文件
        target_neurons: 目标神经元集合，格式为 Dict[(layer_idx, neuron_idx), Dict]
        device: 计算设备
        batch_size: 批大小
        max_length: tokenization最大长度
        num_samples: 使用的样本数限制（None表示全部）
            注意：会分别限制成功和失败样本的数量，确保两种类型的样本都能被充分分析
        prefer_layer: 优先使用的层（默认为 None，自动选择最佳层）
        probe_output_dir: 探针输出目录（当 toxic_vectors 为目录路径时使用）
    
    Returns:
        Dict[(layer_idx, neuron_idx), {
            'successful_mean': float,  # 成功样本的平均激活投影 A_i^k
            'failed_mean': float,  # 失败样本的平均激活投影 A_i^k
            'successful_std': float,  # 成功样本的标准差
            'failed_std': float,  # 失败样本的标准差
            'successful_count': int,  # 成功样本数量
            'failed_count': int,  # 失败样本数量
            'activation_projection': float,  # 激活投影值（用于象限分类，通常使用成功样本的平均值）
            'activation_diff': float,  # 成功-失败的平均差异（辅助信息）
            'toxic_layer': int,  # 使用的毒性向量所属层
            'probe_cv_accuracy': float,  # 该层的探针 CV 准确率（如果有）
        }]
    """
    # ========================================================================
    # 毒性向量加载（支持三种方式）
    # ========================================================================
    
    # 解析毒性向量输入
    if isinstance(toxic_vectors, dict):
        # 方式1：直接传入 Dict
        layer_vectors, vectors_metadata = toxic_vectors, {}
        print(f"[激活投影] 使用预加载的 {len(layer_vectors)} 层毒性向量")
    elif isinstance(toxic_vectors, (str, Path)):
        vectors_path = Path(toxic_vectors)
        
        # 检查是否是文件还是目录
        if vectors_path.is_file() and vectors_path.suffix == '.npz':
            # 方式3：传统 .npz 文件
            toxic_data = np.load(str(vectors_path), allow_pickle=True)
            layer_vectors = {}
            if 'vectors' in toxic_data and 'layer_indices' in toxic_data:
                for idx, layer_idx in enumerate(toxic_data['layer_indices']):
                    layer_vectors[int(layer_idx)] = toxic_data['vectors'][idx]
            vectors_metadata = {}
            print(f"[激活投影] 从文件加载了 {len(layer_vectors)} 层毒性向量")
        else:
            # 方式2：探针输出目录
            layer_vectors, vectors_metadata = load_toxic_vectors_from_probes(
                vectors_path, prefer_layer=prefer_layer
            )
    else:
        raise ValueError(f"toxic_vectors 类型错误: {type(toxic_vectors)}")
    
    if not layer_vectors:
        raise ValueError("未能加载任何毒性向量")

    # 获取模型层数后，将 1-based 层号映射为 HF 的 0-based 索引
    layers = _get_transformer_layers(model)
    if layers is not None:
        layer_vectors = _remap_toxic_vectors_to_hf_layer_index(
            layer_vectors, len(layers)
        )

    # 构建层索引到毒性向量的映射
    layer_to_toxic_idx = {}
    vectors_array_list = []
    for idx, (layer_idx, vec) in enumerate(sorted(layer_vectors.items())):
        layer_to_toxic_idx[int(layer_idx)] = idx
        vectors_array_list.append(vec)
    
    vectors = np.stack(vectors_array_list) if len(vectors_array_list) > 1 else np.array([vectors_array_list[0]])
    
    # ========================================================================
    # 分离成功和失败的样本（根据论文5.4节要求）
    # ========================================================================
    # 
    # 为什么需要分离样本？
    # 
    # 根据论文5.4节"Jailbreak Neurons Analysis"，激活投影（A_i^k）分析需要
    # 分别统计成功和失败jailbreak样本的激活分布，原因如下：
    # 
    # 1. **理解防御机制**：
    #    - 论文指出，虽然attribution方法能识别拒绝jailbreak的关键神经元，
    #      但"provide limited insight into the underlying defense mechanisms 
    #      or reasons for failure"
    #    - 通过对比成功和失败样本中神经元的激活模式，可以深入理解：
    #      * 为什么某些jailbreak成功（防御失败）
    #      * 为什么某些jailbreak失败（防御成功）
    #      * 神经元在不同情况下的行为差异
    # 
    # 2. **神经元功能分类**：
    #    - 论文使用"dual perspective"方法：参数对齐（S）和激活投影（A）
    #    - 激活投影 A_i^k = a_down,i^k · (w_toxic^k / ||w_toxic^k||) 量化了
    #      神经元在推理过程中的实际影响
    #    - 通过分别计算成功和失败样本的激活投影，可以：
    #      * 识别哪些神经元在成功jailbreak时激活更强（促进毒性）
    #      * 识别哪些神经元在失败jailbreak时激活更强（防御机制）
    #      * 计算激活差异（activation_diff = successful_mean - failed_mean）
    # 
    # 3. **象限分类的基础**：
    #    - 论文将神经元分为四个象限：S+A+, S-A+, S+A-, S-A-
    #    - 激活投影值（A）用于判断神经元是促进毒性（A+）还是抑制毒性（A-）
    #    - 通过对比成功和失败样本的激活投影，可以更准确地判断：
    #      * A+：激活投影为正，在成功jailbreak时激活更强 → 促进毒性
    #      * A-：激活投影为负，在失败jailbreak时激活更强 → 抑制毒性
    # 
    # 4. **上下文敏感性分析**：
    #    - 论文强调激活投影考虑了"context-sensitive dynamics"
    #    - 同一个神经元在不同样本（成功vs失败）中可能有不同的激活模式
    #    - 分离分析可以揭示神经元行为的上下文依赖性
    # 
    # 因此，本函数分别处理成功和失败样本，并分别统计：
    #   - successful_mean: 成功样本的平均激活投影
    #   - failed_mean: 失败样本的平均激活投影
    #   - activation_diff: 两者的差异（用于分析防御机制）
    # ========================================================================
    
    successful_samples = []
    failed_samples = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        is_success = _is_jailbreak_successful(sample)
        if is_success is None:
            continue  # 跳过无法判断的样本
        
        text = _extract_text_from_sample(sample)
        if text is None:
            continue
        
        if is_success:
            successful_samples.append(text)
        else:
            failed_samples.append(text)
    
    print(f"[激活投影] 样本: 成功 {len(successful_samples)}, 失败 {len(failed_samples)}")
    
    if len(successful_samples) == 0 and len(failed_samples) == 0:
        raise ValueError("数据集中没有有效的jailbreak样本")
    
    # 限制样本数（分别限制成功和失败样本，保持比例平衡）
    # 注意：如果只限制总数，可能会导致成功/失败样本不平衡
    # 这里分别限制，确保两种类型的样本都能被充分分析
    if num_samples is not None:
        original_successful_count = len(successful_samples)
        original_failed_count = len(failed_samples)
        successful_samples = successful_samples[:num_samples]
        failed_samples = failed_samples[:num_samples]
        if len(successful_samples) < original_successful_count or len(failed_samples) < original_failed_count:
            print(f"[激活投影] 样本数限制: 成功 {len(successful_samples)}/{original_successful_count}, "
                  f"失败 {len(failed_samples)}/{original_failed_count}")
    
    model.eval()
    
    # 如果未指定目标神经元，自动生成所有神经元的列表
    if target_neurons is None:
        # 获取模型层结构以确定神经元数量
        layers = _get_transformer_layers(model)
        if layers is None:
            raise ValueError("无法获取模型的层结构")
        
        # 获取第一层的隐藏维度
        first_layer = layers[0]
        down_proj = _get_down_proj(first_layer)
        if down_proj is None:
            raise ValueError("无法获取模型的 down_proj 层")
        
        hidden_dim = down_proj.weight.data.shape[0]  # out_features
        num_layers = len(layers)
        
        # 为所有层和所有神经元创建目标神经元字典
        target_neurons = {}
        for layer_idx in range(num_layers):
            if layer_idx in layer_to_toxic_idx:  # 只包含有毒性向量的层
                for neuron_idx in range(hidden_dim):
                    target_neurons[(layer_idx, neuron_idx)] = {}
        
        print(f"[激活投影] 目标神经元: {len(target_neurons)} 个 ({num_layers} 层 × {hidden_dim} 神经元/层)")
    
    # 存储激活投影结果
    activation_projections = defaultdict(lambda: {
        'successful': [],
        'failed': []
    })
    
    # 处理成功样本
    if len(successful_samples) > 0:
        _process_samples(
            model, tokenizer, successful_samples, target_neurons,
            layer_to_toxic_idx, vectors, device, batch_size, max_length,
            activation_projections, 'successful'
        )
    
    # 处理失败样本
    if len(failed_samples) > 0:
        print("[Activation Projection] 处理失败样本...")
        _process_samples(
            model, tokenizer, failed_samples, target_neurons,
            layer_to_toxic_idx, vectors, device, batch_size, max_length,
            activation_projections, 'failed'
        )
    
    # 计算统计量
    results = {}

    # 确保 target_neurons 不为 None
    if target_neurons is None:
        raise ValueError("target_neurons 不能为 None，应该在函数开始时自动生成")

    # 预先检查：哪些层的目标神经元没有对应的毒性向量
    missing_layers = {
        layer_idx for (layer_idx, _) in target_neurons.keys()
        if layer_idx not in layer_to_toxic_idx
    }
    if missing_layers:
        print(f"[激活投影] 警告: 以下层的目标神经元没有对应的毒性向量，将被跳过: {sorted(missing_layers)}")

    for (layer_idx, neuron_idx) in target_neurons.keys():
        if layer_idx not in layer_to_toxic_idx:
            continue
        
        successful_projs = activation_projections[(layer_idx, neuron_idx)]['successful']
        failed_projs = activation_projections[(layer_idx, neuron_idx)]['failed']
        
        if len(successful_projs) == 0 and len(failed_projs) == 0:
            continue
        
        successful_mean = float(np.mean(successful_projs)) if successful_projs else 0.0
        failed_mean = float(np.mean(failed_projs)) if failed_projs else 0.0
        
        # 获取该层对应的探针元数据
        toxic_layer = layer_idx
        probe_cv_acc = vectors_metadata.get(layer_idx, {}).get('cv_accuracy', 0.0) if vectors_metadata else 0.0
        
        results[(layer_idx, neuron_idx)] = {
            'successful_mean': successful_mean,
            'failed_mean': failed_mean,
            'successful_std': float(np.std(successful_projs)) if successful_projs else 0.0,
            'failed_std': float(np.std(failed_projs)) if failed_projs else 0.0,
            'successful_count': len(successful_projs),
            'failed_count': len(failed_projs),
            'activation_projection': successful_mean,  # 用于象限分类的激活投影值（使用成功样本的平均值）
            'activation_diff': float(successful_mean - failed_mean) 
                if (successful_projs and failed_projs) else 0.0,
            'toxic_layer': toxic_layer,  # 使用的毒性向量所属层
            'probe_cv_accuracy': probe_cv_acc,  # 该层的探针 CV 准确率（如果有）
        }
    
    print(f"[激活投影] 完成: {len(results)} 个神经元")
    
    return results


def _get_gate_proj(layer: nn.Module) -> Optional[nn.Module]:
    """获取MLP的gate_proj层"""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
        return layer.mlp.gate_proj
    if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "gate_proj"):
        return layer.feed_forward.gate_proj
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


def _process_samples(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    target_neurons: Optional[Dict[Tuple[int, int], Dict]],
    layer_to_toxic_idx: Dict[int, int],
    vectors: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_length: int,
    activation_projections: Dict,
    sample_type: str,
):
    """
    处理样本并计算激活投影
    
    根据论文，激活投影 A_i^k = a_down,i^k · (w_toxic^k / ||w_toxic^k||)
    其中 a_down,i^k ∈ R^d 是激活向量，d 是 hidden_dim。
    
    对于MLP结构：
    - gate_proj(x) -> gate
    - up_proj(x) -> up  
    - intermediate = SiLU(gate) * up -> 维度为 intermediate_size
    - down_proj(intermediate) -> output
    
    投影方法（与 parameter_alignment.py 一致）：
    - 使用 up_proj 的转置将 intermediate 从 intermediate_size 空间投影到 hidden_dim 空间
    - up_proj: (intermediate_size, hidden_dim)
    - up_proj^T: (hidden_dim, intermediate_size)
    - a_down = intermediate @ up_proj^T -> (batch_size, hidden_dim)
    - 这保持了 MLP 的语义结构，确保激活投影与参数对齐使用相同的投影方法
    """
    # 检查 target_neurons 是否为 None
    if target_neurons is None:
        raise ValueError("target_neurons 不能为 None，应该在调用此函数前自动生成所有神经元列表")
    
    # 获取模型层结构
    layers = _get_transformer_layers(model)
    if layers is None:
        raise ValueError("无法获取模型的层结构")
    
    # 为每个目标层注册hook来捕获MLP中间激活
    activation_hooks = {}
    activation_storage = {}
    
    # 注册hooks来捕获gate和up的输出
    for layer_idx in layer_to_toxic_idx.keys():
        if layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        gate_proj = _get_gate_proj(layer)
        up_proj = _get_up_proj(layer)
        
        if gate_proj is not None and up_proj is not None:
            # 使用闭包捕获layer_idx
            def make_hook(idx):
                def gate_hook(module, input, output):
                    if idx not in activation_storage:
                        activation_storage[idx] = {}
                    activation_storage[idx]['gate'] = output
                def up_hook(module, input, output):
                    if idx not in activation_storage:
                        activation_storage[idx] = {}
                    activation_storage[idx]['up'] = output
                return gate_hook, up_hook
            
            gate_hook, up_hook = make_hook(layer_idx)
            gate_handle = gate_proj.register_forward_hook(gate_hook)
            up_handle = up_proj.register_forward_hook(up_hook)
            activation_hooks[layer_idx] = (gate_handle, up_handle)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    try:
        sample_type_cn = "成功" if sample_type == "successful" else "失败"
        for i in tqdm(range(0, len(texts), batch_size), desc=f"处理{sample_type_cn}样本", total=total_batches):
            batch_texts = texts[i:i + batch_size]
            
            # 清空存储
            activation_storage.clear()
            
            # 分词
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            # 前向传播
            with torch.no_grad():
                _ = model(**inputs)
            
            # 立即删除输入张量以释放内存
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 处理每一层的激活
            for layer_idx in layer_to_toxic_idx.keys():
                if layer_idx not in activation_storage:
                    continue
                
                layer_data = activation_storage[layer_idx]
                if 'gate' not in layer_data or 'up' not in layer_data:
                    continue
                
                # gate 和 up 的输出
                gate_output = layer_data['gate']  # (batch_size, seq_len, intermediate_size)
                up_output = layer_data['up']  # (batch_size, seq_len, intermediate_size)
                
                # 计算 intermediate = SiLU(gate) * up
                intermediate = F.silu(gate_output) * up_output  # (batch_size, seq_len, intermediate_size)
                
                # 立即删除不再需要的中间张量
                del gate_output, up_output
                
                # 获取最后一个token的激活值
                last_token_intermediate = intermediate[:, -1, :]  # (batch_size, intermediate_size)
                
                # 删除完整的 intermediate 张量（只保留最后一个token）
                del intermediate
                
                # 获取毒性向量并归一化
                toxic_idx = layer_to_toxic_idx[layer_idx]
                w_toxic = torch.from_numpy(vectors[toxic_idx]).to(device)  # (hidden_dim,)
                w_toxic_norm = torch.norm(w_toxic)
                
                if w_toxic_norm < 1e-10:
                    continue
                
                w_toxic_normalized = w_toxic / w_toxic_norm  # (hidden_dim,)
                
                # 获取 up_proj 和 down_proj 以了解维度和进行投影
                layer = layers[layer_idx]
                up_proj = _get_up_proj(layer)
                down_proj = _get_down_proj(layer)
                
                if up_proj is None or down_proj is None:
                    continue
                
                if not hasattr(up_proj, 'weight') or up_proj.weight is None:
                    continue
                
                if not hasattr(down_proj, 'weight') or down_proj.weight is None:
                    continue
                
                # 获取 up_proj 权重张量（用于投影，处理量化权重）
                up_proj_weight_tensor = up_proj.weight
                
                # 获取 down_proj 权重张量（用于了解维度，处理量化权重）
                weight_tensor = down_proj.weight
                
                # 处理 up_proj 量化权重（用于投影）
                try:
                    if hasattr(up_proj_weight_tensor, 'quant_state') or hasattr(up_proj, 'quantization_config'):
                        if hasattr(up_proj_weight_tensor, 'dequantize'):
                            up_proj_weight_tensor = up_proj_weight_tensor.dequantize()
                        elif hasattr(up_proj_weight_tensor, 'data') and hasattr(up_proj_weight_tensor.data, 'dequantize'):
                            up_proj_weight_tensor = up_proj_weight_tensor.data.dequantize()
                        else:
                            # 尝试使用 state_dict 获取权重
                            try:
                                state_dict = up_proj.state_dict()
                                if 'weight' in state_dict:
                                    up_proj_weight_tensor = state_dict['weight']
                                    if hasattr(up_proj_weight_tensor, 'dequantize'):
                                        up_proj_weight_tensor = up_proj_weight_tensor.dequantize()
                            except:
                                pass
                            
                            # 如果还是量化权重，尝试直接访问 base_layer（BitsAndBytes 包装）
                            if hasattr(up_proj, 'base_layer'):
                                try:
                                    base_weight = up_proj.base_layer.weight
                                    if hasattr(base_weight, 'dequantize'):
                                        up_proj_weight_tensor = base_weight.dequantize()
                                    else:
                                        up_proj_weight_tensor = base_weight
                                except:
                                    pass
                except Exception:
                    pass
                
                # 处理 down_proj 量化权重（用于了解维度）
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
                except Exception:
                    pass
                
                # 检查权重形状（直接使用张量的 shape 属性，避免转换问题）
                if not hasattr(weight_tensor, 'shape'):
                    continue
                
                weight_shape = weight_tensor.shape
                
                # 获取实际的维度信息（从激活值和毒性向量）
                actual_intermediate_size = last_token_intermediate.shape[1]  # 从激活值获取
                actual_hidden_dim = w_toxic.shape[0]  # 从毒性向量获取
                
                # 处理可能的展平情况（量化权重可能被展平）
                if len(weight_shape) == 1:
                    # 如果是一维，使用实际维度进行 reshape
                    total_elements = weight_shape[0]
                    expected_elements = actual_hidden_dim * actual_intermediate_size
                    
                    if total_elements == expected_elements:
                        # 尝试使用实际维度进行 reshape
                        try:
                            weight_tensor = weight_tensor.reshape(actual_hidden_dim, actual_intermediate_size)
                            weight_shape = weight_tensor.shape
                        except Exception:
                            continue
                    else:
                        continue
                
                # 处理 [N, 1] 或 [1, N] 这种展平的情况
                if len(weight_shape) == 2 and (weight_shape[0] == 1 or weight_shape[1] == 1):
                    # 展平为一维
                    weight_tensor = weight_tensor.flatten()
                    weight_shape = weight_tensor.shape
                
                if len(weight_shape) != 2:
                    # 如果还是一维，尝试从总元素数推断维度
                    if len(weight_shape) == 1:
                        total_elements = weight_shape[0]
                        expected_elements = actual_hidden_dim * actual_intermediate_size
                        
                        # 如果总元素数不匹配，尝试从权重推断 intermediate_size
                        if total_elements != expected_elements:
                            # 尝试从权重推断：如果 total_elements 能被 hidden_dim 整除
                            inferred_intermediate_size = total_elements // actual_hidden_dim
                            if total_elements % actual_hidden_dim == 0 and inferred_intermediate_size > 0:
                                # 使用推断的 intermediate_size
                                actual_intermediate_size = inferred_intermediate_size
                                expected_elements = actual_hidden_dim * actual_intermediate_size
                            else:
                                continue
                        
                        # 尝试 reshape
                        try:
                            weight_tensor = weight_tensor.reshape(actual_hidden_dim, actual_intermediate_size)
                            weight_shape = weight_tensor.shape
                        except Exception:
                            continue
                    else:
                        continue
                
                # 处理错误的 reshape 情况：如果总元素数匹配，尝试 reshape 为正确的形状
                total_elements = weight_tensor.numel() if hasattr(weight_tensor, 'numel') else weight_shape[0] * weight_shape[1]
                expected_elements = actual_hidden_dim * actual_intermediate_size
                
                # 优先使用实际维度进行 reshape
                if total_elements == expected_elements and weight_shape != (actual_hidden_dim, actual_intermediate_size):
                    try:
                        weight_tensor = weight_tensor.reshape(actual_hidden_dim, actual_intermediate_size)
                        weight_shape = weight_tensor.shape
                    except Exception:
                        continue
                elif total_elements != expected_elements:
                    # 如果总元素数不匹配，尝试从权重推断 intermediate_size
                    inferred_intermediate_size = total_elements // actual_hidden_dim
                    if total_elements % actual_hidden_dim == 0 and inferred_intermediate_size > 0:
                        # 使用推断的 intermediate_size
                        actual_intermediate_size = inferred_intermediate_size
                        expected_elements = actual_hidden_dim * actual_intermediate_size
                        try:
                            weight_tensor = weight_tensor.reshape(actual_hidden_dim, actual_intermediate_size)
                            weight_shape = weight_tensor.shape
                        except Exception:
                            continue
                    else:
                        continue
                
                # 获取维度信息（直接使用张量的形状）
                hidden_dim_from_weight = weight_shape[0]  # out_features
                intermediate_size = weight_shape[1]  # in_features
                hidden_dim = w_toxic.shape[0]  # 毒性向量的维度
                
                # 验证维度一致性
                if hidden_dim_from_weight != hidden_dim:
                    continue
                
                # 检查 intermediate 的实际维度
                # 如果从权重推断的 intermediate_size 与激活值的维度不匹配，使用权重维度
                # 这可能发生在量化或压缩的情况下
                if actual_intermediate_size != intermediate_size:
                    if intermediate_size < actual_intermediate_size:
                        # 如果权重维度小于激活值维度，可能是量化导致的
                        # 使用权重维度，并截取激活值的前 intermediate_size 个维度
                        actual_intermediate_size = intermediate_size
                    else:
                        # 如果权重维度大于激活值维度，这不应该发生，跳过该层
                        continue
                
                # 验证维度合理性
                if hidden_dim <= 0 or intermediate_size <= 0:
                    continue
                
                # 根据论文，激活投影 A_i^k = a_down,i^k · (w_toxic^k / ||w_toxic^k||)
                # 其中 a_down,i^k ∈ R^d 是激活向量，d 是 hidden_dim
                # 
                # 对于 MLP 结构：
                # - intermediate 的维度是 intermediate_size（gate*up 的输出）
                # - 需要将 intermediate 从 intermediate_size 空间投影到 hidden_dim 空间
                # 
                # 根据 parameter_alignment.py 的方法，使用 up_proj 的转置进行投影：
                # - up_proj 形状: (intermediate_size, hidden_dim)
                # - up_proj^T 形状: (hidden_dim, intermediate_size)
                # - 投影: a_down = up_proj^T @ intermediate
                # 这保持了 MLP 的语义结构，与参数对齐方法一致
                
                # 检查 up_proj 权重形状
                if not hasattr(up_proj_weight_tensor, 'shape'):
                    continue
                
                up_proj_shape = up_proj_weight_tensor.shape
                if len(up_proj_shape) != 2:
                    continue
                
                # up_proj 形状应该是 (intermediate_size, hidden_dim)
                up_proj_intermediate_size = up_proj_shape[0]
                up_proj_hidden_dim = up_proj_shape[1]
                
                # 验证维度一致性
                if up_proj_hidden_dim != hidden_dim:
                    continue
                
                # 确保 intermediate 激活值的维度与 up_proj 的输入维度匹配
                if last_token_intermediate.shape[1] != up_proj_intermediate_size:
                    # 如果维度不匹配，截取或填充
                    if last_token_intermediate.shape[1] > up_proj_intermediate_size:
                        last_token_intermediate = last_token_intermediate[:, :up_proj_intermediate_size]
                    else:
                        # 零填充
                        padding = torch.zeros(
                            last_token_intermediate.shape[0],
                            up_proj_intermediate_size - last_token_intermediate.shape[1],
                            device=last_token_intermediate.device,
                            dtype=last_token_intermediate.dtype
                        )
                        last_token_intermediate = torch.cat([last_token_intermediate, padding], dim=1)
                
                # 将 up_proj 权重移动到正确的设备
                up_proj_weight = up_proj_weight_tensor.to(device)
                
                # 使用 up_proj 进行投影：a_down = intermediate @ up_proj
                # up_proj: (intermediate_size, hidden_dim) = (14336, 4096)
                # intermediate: (batch_size, intermediate_size) = (batch_size, 14336)
                # a_down: (batch_size, hidden_dim) = (batch_size, 4096)
                a_down = torch.matmul(last_token_intermediate, up_proj_weight)  # (batch_size, hidden_dim)
                
                # 对于每个目标神经元，计算激活投影
                for (target_layer_idx, neuron_idx) in target_neurons.keys():
                    if target_layer_idx != layer_idx:
                        continue
                    
                    if neuron_idx >= hidden_dim:
                        continue
                    
                    # 根据论文公式：A_i^k = (a_down,i^k · w_toxic^k) / ||w_toxic^k||
                    # a_down: (batch_size, hidden_dim)，每行是一个样本的激活向量
                    # w_toxic_normalized: (hidden_dim,)，归一化的毒性向量
                    # 点积结果：(batch_size,)，每个样本有一个激活投影值
                    # 这个标量值代表了第i个神经元在最后一个token的激活与毒性方向的对齐程度
                    proj = torch.sum(a_down * w_toxic_normalized, dim=1)  # (batch_size,)
                    
                    # 存储每个样本的投影值
                    for b in range(proj.shape[0]):
                        activation_projections[(layer_idx, neuron_idx)][sample_type].append(proj[b].item())
                    
                    # 立即删除投影结果以释放内存
                    del proj
                
                # 处理完该层后，清理该层的激活数据
                if layer_idx in activation_storage:
                    if 'gate' in activation_storage[layer_idx]:
                        del activation_storage[layer_idx]['gate']
                    if 'up' in activation_storage[layer_idx]:
                        del activation_storage[layer_idx]['up']
                    activation_storage[layer_idx].clear()
                
                # 清理该层处理中的中间张量
                del last_token_intermediate, w_toxic, w_toxic_normalized, a_down
                
                # 每处理一层后清理缓存（如果内存紧张）
                if torch.cuda.is_available() and len(layer_to_toxic_idx) > 5:
                    torch.cuda.empty_cache()
            
            # 处理完整个 batch 后，清空激活存储并清理缓存
            activation_storage.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    finally:
        # 移除所有hooks
        for layer_idx, hooks in activation_hooks.items():
            gate_handle, up_handle = hooks
            gate_handle.remove()
            up_handle.remove()
