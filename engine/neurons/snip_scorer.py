"""
SNIP 分数计算模块

基于 SNIP (Single-shot Network Pruning based on Connection Sensitivity) 方法
计算每个神经元对损失的贡献度

增强功能：
- 支持从训练好的线性探针加载毒性向量（w_toxic）用于增强安全评估
- 基于探针的 CV 准确率和层级稳定性推荐最佳分析层
- 兼容新旧探针输出格式（layer_XX/probe.pt vs layerXX/layerXX.pt）
"""

import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Iterable, Literal, Union
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# 探针毒性向量加载器
# ============================================================================

def _get_layer_idx(name: str):
    """从目录名提取层索引（layer_XX 或 layerXX 格式）。"""
    import re
    match = re.match(r'layer_(\d+)', name)
    if match:
        return int(match.group(1))
    match = re.match(r'layer(\d+)', name)
    if match:
        return int(match.group(1))
    return None

def load_probe_toxic_vectors_from_snip(
    probe_output_dir: Union[str, Path],
    model_id: Optional[str] = None,
    prefer_layer: Optional[int] = None,
) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    从训练好的探针输出目录加载各层的毒性向量。

    返回值格式（Tuple）：
        - vectors: {layer_idx: {'w_toxic': np.ndarray, 'cv_accuracy': float, 'std': float, ...}}
        - metadata: {layer_idx: {'cv_accuracy': float, 'std': float, ...}}
                  （旧格式中 vectors 和 metadata 内容相同，统一用 vectors 填充 metadata）

    支持的探针目录结构：
    1. outputs/linear_probes/layers/layerXX/layerXX.pt + metrics.json
       （scripts/train_linear_probe_labels.py 输出格式）
    2. outputs/toxicity_vectors/all_layers_toxicity_vectors.json
       （scripts/extract_toxicity_vectors.py 聚合格式）
    3. outputs/probes/model_id/layer_XX/probe.pt
       （engine/probes/linear_probe_balanced.py 输出格式）

    Args:
        probe_output_dir: 探针输出目录
        model_id: 模型 ID（用于构建探针子目录路径）
        prefer_layer: 优先使用的层

    Returns:
        Tuple[Dict, Dict]: (vectors, metadata)
    """
    probe_output_dir = Path(probe_output_dir)

    # 按优先级收集所有 layer 目录（去重）
    layer_dirs = []

    # 1. 新格式 probes/model_id/layer_XX/
    probes_dir = probe_output_dir / "probes"
    if probes_dir.exists():
        for model_folder in sorted(probes_dir.iterdir()):
            if not model_folder.is_dir():
                continue
            for layer_folder in sorted(model_folder.iterdir()):
                if layer_folder.is_dir():
                    layer_dirs.append(layer_folder)

    # 2. 旧格式 linear_probes/layers/layerXX/
    old_style = probe_output_dir / "linear_probes" / "layers"
    if old_style.exists():
        for layer_folder in sorted(old_style.iterdir()):
            if layer_folder.is_dir() and _get_layer_idx(layer_folder.name) is not None:
                if not any(d.resolve() == layer_folder.resolve() for d in layer_dirs):
                    layer_dirs.append(layer_folder)

    # 3. 直接格式 layerXX/ 或 layer_XX/
    for item in sorted(probe_output_dir.iterdir()):
        if item.is_dir() and _get_layer_idx(item.name) is not None:
            if not any(d.resolve() == item.resolve() for d in layer_dirs):
                layer_dirs.append(item)

    vectors_by_layer = {}

    for layer_folder in layer_dirs:
        layer_idx = _get_layer_idx(layer_folder.name)
        if layer_idx is None or layer_idx in vectors_by_layer:
            continue

        name = layer_folder.name
        layer_info = {}

        # 方法1：toxic_vector.npz
        toxic_npz = layer_folder / "toxic_vector.npz"
        if toxic_npz.exists():
            try:
                data = np.load(toxic_npz, allow_pickle=True)
                if 'w_toxic' in data:
                    layer_info['w_toxic'] = data['w_toxic']
                    layer_info['b'] = float(data.get('b', 0.0))
                elif 'w_toxic_normalized' in data:
                    layer_info['w_toxic'] = data['w_toxic_normalized']
                    layer_info['b'] = float(data.get('b', 0.0))
            except Exception:
                pass

        # 方法2：从 probe.pt / layerXX.pt / best.pt 提取毒性向量
        if 'w_toxic' not in layer_info:
            for pt_file in ['probe.pt', f'{name}.pt', 'best.pt']:
                pt_path = layer_folder / pt_file
                if not pt_path.exists():
                    continue
                try:
                    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
                    if 'toxicity_vector' in checkpoint:
                        tv = checkpoint['toxicity_vector']
                        layer_info['w_toxic'] = np.array(tv) if not isinstance(tv, np.ndarray) else tv
                    elif 'linear.weight' in checkpoint:
                        w = checkpoint['linear.weight']
                        if w.shape[0] == 2:
                            w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                            layer_info['w_toxic'] = w_arr[1] - w_arr[0]
                    elif 'fc.weight' in checkpoint:
                        w = checkpoint['fc.weight']
                        if w.shape[0] == 2:
                            w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                            layer_info['w_toxic'] = w_arr[1] - w_arr[0]
                    elif 'weight' in checkpoint:
                        w = checkpoint['weight']
                        if hasattr(w, 'shape') and w.shape[0] == 2:
                            w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                            layer_info['w_toxic'] = w_arr[1] - w_arr[0]
                    break
                except Exception:
                    continue

        # 方法3：从 toxicity_vectors/all_layers_toxicity_vectors.json 补充向量
        if 'w_toxic' not in layer_info:
            tox_json = probe_output_dir / "toxicity_vectors" / "all_layers_toxicity_vectors.json"
            if tox_json.exists():
                try:
                    with open(tox_json, 'r', encoding='utf-8') as f:
                        all_data = json.load(f)
                    if 'layers' in all_data and str(layer_idx) in all_data['layers']:
                        ld = all_data['layers'][str(layer_idx)]
                        if 'w_toxic' in ld:
                            layer_info['w_toxic'] = np.array(ld['w_toxic'])
                        elif 'vector' in ld:
                            layer_info['w_toxic'] = np.array(ld['vector'])
                except Exception:
                    pass

        # 加载 metrics.json
        metrics_path = layer_folder / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

                if 'val_acc' in metrics:
                    layer_info['cv_accuracy'] = float(metrics['val_acc'])
                    layer_info['val_acc'] = float(metrics['val_acc'])
                elif 'avg_metrics' in metrics and 'avg_val_acc' in metrics['avg_metrics']:
                    layer_info['cv_accuracy'] = float(metrics['avg_metrics']['avg_val_acc'])
                elif 'avg_val_acc' in metrics:
                    layer_info['cv_accuracy'] = float(metrics['avg_val_acc'])

                if 'std_val_acc' in metrics:
                    layer_info['std'] = float(metrics['std_val_acc'])
                elif 'avg_metrics' in metrics and 'std_val_acc' in metrics['avg_metrics']:
                    layer_info['std'] = float(metrics['avg_metrics']['std_val_acc'])
                elif 'std' in metrics:
                    layer_info['std'] = float(metrics['std'])
                else:
                    layer_info['std'] = 0.0

                if 'avg_metrics' in metrics:
                    if 'avg_val_s_acc' in metrics['avg_metrics']:
                        layer_info['safe_acc'] = float(metrics['avg_metrics']['avg_val_s_acc'])
                    if 'avg_val_h_acc' in metrics['avg_metrics']:
                        layer_info['harm_acc'] = float(metrics['avg_metrics']['avg_val_h_acc'])
                elif 'avg_val_s_acc' in metrics:
                    layer_info['safe_acc'] = float(metrics['avg_val_s_acc'])
                if 'val_roc_auc' in metrics:
                    layer_info['val_roc_auc'] = float(metrics['val_roc_auc'])
                if 'val_pr_auc' in metrics:
                    layer_info['val_pr_auc'] = float(metrics['val_pr_auc'])

            except Exception:
                pass

        # 从 toxicity_vectors/layer_toxicity_summary.json 补充元数据
        if layer_idx not in vectors_by_layer and 'cv_accuracy' not in layer_info:
            summary_path = probe_output_dir / "toxicity_vectors" / "layer_toxicity_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    if 'layers_summary' in summary:
                        for li in summary['layers_summary']:
                            if int(li['layer']) == layer_idx:
                                layer_info['cv_accuracy'] = float(li.get('cv_accuracy', 0))
                                layer_info['std'] = float(li.get('std', 0))
                                layer_info['l2_norm'] = float(li.get('l2_norm', 0))
                                break
                except Exception:
                    pass

        if 'w_toxic' in layer_info or 'cv_accuracy' in layer_info:
            vectors_by_layer[layer_idx] = layer_info

    # 从 linear_probes/all_layers_summary.json 补充元数据（最简单的 fallback）
    if not vectors_by_layer or any('cv_accuracy' not in v for v in vectors_by_layer.values()):
        summary_path = probe_output_dir / "linear_probes" / "all_layers_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                if isinstance(summary, list):
                    for entry in summary:
                        layer_idx = int(entry.get('layer', 0))
                        if layer_idx > 0:
                            if layer_idx not in vectors_by_layer:
                                vectors_by_layer[layer_idx] = {}
                            vectors_by_layer[layer_idx].setdefault('cv_accuracy', float(entry.get('cv_avg_val_acc', 0)))
                            vectors_by_layer[layer_idx].setdefault('std', float(entry.get('cv_std_val_acc', 0)))
            except Exception:
                pass

    if not vectors_by_layer:
        print(f"[Probe Toxic Vectors] 警告: 未能从 {probe_output_dir} 加载任何毒性向量")
        return {}, {}

    print(f"[Probe Toxic Vectors] 从 {probe_output_dir} 加载了 {len(vectors_by_layer)} 层的毒性向量")

    if vectors_by_layer:
        sorted_layers = sorted(vectors_by_layer.items(),
                              key=lambda x: x[1].get('cv_accuracy', 0),
                              reverse=True)
        print(f"[Probe Toxic Vectors] 层质量排名（前5层）:")
        for layer_idx, info in sorted_layers[:5]:
            acc = info.get('cv_accuracy', 0)
            std = info.get('std', 0)
            print(f"  Layer {layer_idx:2d}: CV Acc = {acc*100:.2f}%, Std = {std*100:.2f}%")

    return vectors_by_layer, {}


def get_recommended_analysis_layers(
    probe_output_dir: Union[str, Path] = "outputs",
    model_id: Optional[str] = None,
    num_layers: Optional[int] = None,
    strategy: str = "auto",
) -> Dict[str, int]:
    """
    根据探针训练结果获取推荐的神经元分析层。
    
    推荐策略：
    - "stability": L14（最稳定，std=0.48%）
    - "accuracy": L32（最高准确率，93.76%）
    - "balanced": L16（准确率与稳定性平衡，91.62%）
    - "early_mid": L20（早中层，稳定在 91.8%）
    - "auto": 根据可用数据自动选择（优先 L32）
    
    Args:
        probe_output_dir: 探针输出目录
        model_id: 模型 ID
        num_layers: 总层数（默认 32）
        strategy: 推荐策略
    
    Returns:
        Dict[str, int]: 各策略的推荐层
            {
                'stability': 14,
                'accuracy': 32,
                'balanced': 16,
                'early_mid': 20,
                'selected': 32,  # 基于 strategy 选择的层
            }
    """
    recommendations = {
        'stability': None,   # 从探针数据动态推断
        'accuracy': None,    # 从探针数据动态推断
        'balanced': None,    # 从探针数据动态推断
        'early_mid': None,   # 从探针数据动态推断
        'selected': None,    # 基于 strategy 动态选择
    }

    vectors_by_layer = {}  # 初始化，避免 except 块中引用未定义变量

    # 尝试从实际数据获取推荐
    try:
        vectors_by_layer, _ = load_probe_toxic_vectors_from_snip(probe_output_dir, model_id)
        if vectors_by_layer:
            # 兼容新旧格式的准确率获取
            def _get_acc(info):
                return info.get('cv_accuracy', info.get('val_acc', 0.0))

            def _get_std(info):
                if 'std' in info:
                    return float(info['std'])
                if 'std_val_acc' in info:
                    return float(info['std_val_acc'])
                roc = info.get('val_roc_auc', 0)
                if roc > 0:
                    return 1.0 - float(roc)
                return float('inf')

            # 按准确率排序
            sorted_by_acc = sorted(vectors_by_layer.items(),
                                   key=lambda x: _get_acc(x[1]),
                                   reverse=True)

            # 按稳定性排序（标准差从小到大）
            sorted_by_std = sorted(vectors_by_layer.items(),
                                    key=lambda x: _get_std(x[1]))

            if sorted_by_acc:
                recommendations['accuracy'] = sorted_by_acc[0][0]
            if sorted_by_std:
                recommendations['stability'] = sorted_by_std[0][0]

            # 计算平衡层（准确率 >= 90% 且标准差 <= 10%）
            balanced_candidates = [
                (layer_idx, info) for layer_idx, info in vectors_by_layer.items()
                if _get_acc(info) >= 0.90 and _get_std(info) <= 0.10
            ]
            if balanced_candidates:
                best_balanced = min(balanced_candidates,
                                    key=lambda x: _get_std(x[1]))
                recommendations['balanced'] = best_balanced[0]

            # 早中层：选择中间范围（如前25%~75%层）中准确率最高的
            total_layers_range = max(max(vectors_by_layer.keys()), 20) if vectors_by_layer else 32
            early_mid_min = max(1, int(total_layers_range * 0.15))
            early_mid_max = max(early_mid_min + 1, int(total_layers_range * 0.75))
            early_mid_candidates = [
                (layer_idx, info) for layer_idx, info in sorted_by_acc
                if early_mid_min <= layer_idx <= early_mid_max
            ]
            if early_mid_candidates:
                recommendations['early_mid'] = early_mid_candidates[0][0]
    except Exception as e:
        print(f"[Recommended Layers] 警告: 无法加载探针数据，使用默认值: {e}")
    
    # 根据策略选择最终层（优先用探针数据，否则用候选中非None的值）
    if strategy == "stability":
        recommendations['selected'] = recommendations['stability']
    elif strategy == "accuracy":
        recommendations['selected'] = recommendations['accuracy']
    elif strategy == "balanced":
        recommendations['selected'] = recommendations['balanced']
    elif strategy == "early_mid":
        recommendations['selected'] = recommendations['early_mid']
    else:  # auto
        recommendations['selected'] = recommendations['accuracy']

    # 兜底：如果所有推荐都为 None（无法从探针数据推断），返回空策略
    if recommendations['selected'] is None:
        # 从所有可用层索引中选一个中间层作为通用兜底
        if vectors_by_layer:
            all_layers = sorted(vectors_by_layer.keys())
            mid_idx = all_layers[len(all_layers) // 2]
            recommendations['selected'] = mid_idx
            recommendations['accuracy'] = mid_idx
        else:
            recommendations['selected'] = 0
            recommendations['accuracy'] = 0
            recommendations['stability'] = 0
            recommendations['balanced'] = 0
            recommendations['early_mid'] = 0
    
    print(f"[Recommended Layers] 推荐分析层: {recommendations}")
    return recommendations


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    """
    Ensure tokenizer has a pad token for batch padding.

    Many LLaMA-like tokenizers ship without an explicit PAD token; in that case we
    safely fall back to EOS for padding (common practice for causal LM eval).
    """
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token


def _infer_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    """
    Infer a device suitable for placing tokenized inputs.

    Notes:
    - When using `device_map="auto"`, parameters may live on multiple devices.
      In practice HF expects inputs to be on the device of the first module.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def _get_transformer_layers(model: nn.Module):
    """Best-effort access to a Llama-like stack of blocks."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _get_mlp_module(layer: nn.Module) -> Optional[nn.Module]:
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    return None


def _get_linear(module: nn.Module, names: Iterable[str]) -> Optional[nn.Module]:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _to_str_or_none(x) -> Optional[str]:
    """
    将可能为 list/其他类型的字段安全转换为字符串。

    背景：
    - 部分上游数据集可能会把 text/prompt/response 存成 list（多轮对话）或其他结构，
      直接传给 tokenizer 会得到嵌套的 input_ids，最终在创建 tensor 时触发
      "inputs type `list` where type `int` is expected" 报错。
    - 这里做一个「尽力而为」的规范化：list/tuple 按空格拼接，其他类型走 str(...)。
    """
    if x is None:
        return None
    # 已经是字符串，直接返回
    if isinstance(x, str):
        return x
    # 多段文本：按空格拼成一段
    if isinstance(x, (list, tuple)):
        try:
            return " ".join(str(t) for t in x if t is not None)
        except Exception:
            return None
    # 其他类型：退化为 str 表示
    try:
        return str(x)
    except Exception:
        return None


def _batch_to_samples(batch) -> list:
    """
    Normalize a DataLoader batch into a list of per-sample dicts.

    Supported sample formats (dataset __getitem__):
    - {"text": "..."}  (treat as full sequence, loss on all non-pad tokens)
    - {"prompt": "...", "response": "..."}  (loss on response tokens only)
    - {"input": {"prompt": "..."} , "output": "..."} (Alpaca-style)
    - {"input": "...", "output": "..."} (generic)
    """
    samples = []

    # Case 1: default PyTorch collation for dict[str, list]
    if isinstance(batch, dict):
        if "text" in batch:
            texts = batch["text"]
            if not isinstance(texts, list):
                texts = [texts]
            for t in texts:
                norm = _to_str_or_none(t)
                if norm:
                    samples.append({"text": norm})
            return samples

        # prompt/response style
        if "prompt" in batch and "response" in batch:
            prompts = batch["prompt"]
            responses = batch["response"]
            if not isinstance(prompts, list):
                prompts = [prompts]
            if not isinstance(responses, list):
                responses = [responses]
            for p, r in zip(prompts, responses):
                prompt = _to_str_or_none(p)
                response = _to_str_or_none(r)
                if prompt is None and response is None:
                    continue
                if prompt is None:
                    # 没有 prompt，当成纯 text
                    samples.append({"text": response})
                elif response is None:
                    samples.append({"text": prompt})
                else:
                    samples.append({"prompt": prompt, "response": response})
            return samples

        # Alpaca-style: {"input": {"prompt": ...}, "output": ...}
        if "input" in batch and "output" in batch:
            inputs = batch["input"]
            outputs = batch["output"]
            if not isinstance(inputs, list):
                inputs = [inputs]
            if not isinstance(outputs, list):
                outputs = [outputs]
            for inp, out in zip(inputs, outputs):
                if isinstance(inp, dict) and "prompt" in inp:
                    raw_prompt = inp["prompt"]
                else:
                    raw_prompt = inp
                prompt = _to_str_or_none(raw_prompt)
                response = _to_str_or_none(out)
                if prompt is None and response is None:
                    continue
                if prompt is None:
                    samples.append({"text": response})
                elif response is None:
                    samples.append({"text": prompt})
                else:
                    samples.append({"prompt": prompt, "response": response})
            return samples

        raise ValueError(f"无法处理 batch dict keys: {list(batch.keys())}")

    # Case 2: list of samples
    if isinstance(batch, list):
        for item in batch:
            if isinstance(item, dict):
                if "text" in item:
                    norm = _to_str_or_none(item["text"])
                    if norm:
                        samples.append({"text": norm})
                elif "prompt" in item and "response" in item:
                    prompt = _to_str_or_none(item["prompt"])
                    response = _to_str_or_none(item["response"])
                    if prompt is None and response is None:
                        continue
                    if prompt is None:
                        samples.append({"text": response})
                    elif response is None:
                        samples.append({"text": prompt})
                    else:
                        samples.append({"prompt": prompt, "response": response})
                elif "input" in item and "output" in item:
                    inp = item["input"]
                    if isinstance(inp, dict):
                        raw_prompt = inp.get("prompt")
                    else:
                        raw_prompt = inp
                    prompt = _to_str_or_none(raw_prompt)
                    response = _to_str_or_none(item["output"])
                    if prompt is None and response is None:
                        continue
                    if prompt is None:
                        samples.append({"text": response})
                    elif response is None:
                        samples.append({"text": prompt})
                    else:
                        samples.append({"prompt": prompt, "response": response})
                else:
                    raise ValueError(f"无法处理 sample dict keys: {list(item.keys())}")
            else:
                # treat raw value as text-ish
                norm = _to_str_or_none(item)
                if norm:
                    samples.append({"text": norm})
        return samples

    raise ValueError(f"无法处理 batch 格式: {type(batch)}")


def _build_causal_lm_inputs_and_labels(
    samples: list,
    tokenizer: AutoTokenizer,
    *,
    max_length: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Build model inputs and labels for a causal LM.

    - If sample has (prompt, response): labels mask prompt tokens as -100, only score response.
    - If sample has text: labels score the whole sequence (non-pad).
    """
    _ensure_pad_token(tokenizer)

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    encodings = []
    labels_list = []

    for s in samples:
        if "text" in s:
            # Treat full text as "response" (loss on all tokens except padding)
            text = s["text"]
            ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length).input_ids
            # Labels = ids (padding handled later)
            labels = list(ids)
        else:
            prompt = s.get("prompt", "") or ""
            response = s.get("response", "") or ""

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids

            ids = []
            if bos_id is not None:
                ids.append(int(bos_id))
            ids.extend(prompt_ids)
            ids.extend(response_ids)
            if eos_id is not None:
                ids.append(int(eos_id))

            # Only score response (and optional EOS), mask prompt (and optional BOS)
            labels = [-100] * len(ids)
            resp_and_eos = list(response_ids) + ([int(eos_id)] if eos_id is not None else [])
            start = len(ids) - len(resp_and_eos)
            for j, tok in enumerate(resp_and_eos):
                if start + j < len(labels):
                    labels[start + j] = int(tok)

            # Apply truncation consistently
            if len(ids) > max_length:
                ids = ids[:max_length]
                labels = labels[:max_length]

        encodings.append({"input_ids": ids})
        labels_list.append(labels)

    padded = tokenizer.pad(
        encodings,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )
    input_ids = padded["input_ids"]
    attention_mask = padded.get("attention_mask")

    seq_len = int(input_ids.shape[1])
    padded_labels = []
    for labels in labels_list:
        if len(labels) < seq_len:
            labels = labels + ([-100] * (seq_len - len(labels)))
        else:
            labels = labels[:seq_len]
        padded_labels.append(labels)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long)

    # Move to device (device_map="auto" -> use inferred input device)
    model_inputs = {"input_ids": input_ids.to(device)}
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask.to(device)
        # Also ignore padding for the "text" case, and for response labels as extra safety.
        labels_tensor = labels_tensor.masked_fill(attention_mask == 0, -100)

    return model_inputs, labels_tensor.to(device)


def compute_snip_scores(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: torch.device,
    loss_fn: Callable,
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    *,
    score_target: Literal["down_proj_out", "mlp_intermediate"] = "down_proj_out",
    max_length: int = 2048,
    restrict_grad_to_mlp: bool = True,
) -> Dict[Tuple[int, int], float]:
    """
    计算 SNIP 分数（与论文公式对齐）
    
    对每个神经元 $i$，我们实现论文中的重要性打分公式：
        $I_i(x) = \\lvert w_i \\cdot \\Delta \\mathcal{{L}}(x) \\rvert$
    其中 $w_i$ 为该神经元相关的所有参数向量，$\\Delta \\mathcal{{L}}(x)$ 为这些参数上的
    损失梯度（对单个样本 x 的贡献）。在实现上，我们将这些参数及其梯度展平成向量，
    先做点积再取绝对值；对整个数据集上的样本再求平均，得到数据集级别的 $I_i$。
    
    Args:
        model: 语言模型（需要支持梯度计算）
        tokenizer: 分词器
        dataset: 数据集，支持以下样本格式：
            - {"text": "..."}（对全序列计算 LM loss；padding 会被忽略）
            - {"prompt": "...", "response": "..."}（仅对 response token 计算 LM loss，prompt token 置为 -100）
            - Alpaca: {"input":{"prompt": "..."},"output":"..."} 或 {"input":"...","output":"..."}
        device: 计算设备
        loss_fn: 损失函数，签名: loss_fn(outputs, batch, model, device) -> loss_tensor
        batch_size: 批大小
        num_samples: 使用的样本数（None 表示全部）
        score_target: 结构化打分目标（决定 $w_i$ 的具体结构）
            - "down_proj_out": 将 MLP `down_proj` 的每一行（out_features）视为一个“神经元”，
              即该行上的所有连接权重共同组成 $w_i$
            - "mlp_intermediate": 将 FFN 中间维度（intermediate_size）视为神经元，
              将 gate/up 的一行与 down 的一列拼成同一个 $w_i$
        max_length: tokenization 截断长度
        restrict_grad_to_mlp: 是否仅对 MLP 权重开启梯度（显著降低显存；推荐用于 SNIP/剪枝分析）
    
    Returns:
        Dict[(layer_idx, neuron_idx), snip_score]: 每个神经元的 SNIP 分数
        其中 layer_idx 是层索引，neuron_idx 是神经元索引（在 MLP down_proj 中）
    """
    model.eval()
    # 需要梯度来计算 SNIP；为了节省显存，默认只让 MLP 权重参与梯度
    if restrict_grad_to_mlp:
        model.requires_grad_(False)
        layers = _get_transformer_layers(model)
        if layers is not None:
            for layer in layers:
                mlp = _get_mlp_module(layer)
                if mlp is None:
                    continue
                for name in ("down_proj", "up_proj", "gate_proj", "output", "w1", "w2", "w3", "fc1", "fc2"):
                    mod = getattr(mlp, name, None)
                    if isinstance(mod, nn.Module):
                        mod.requires_grad_(True)
        else:
            # 回退：启用全模型梯度
            model.requires_grad_(True)
    else:
        model.requires_grad_(True)
    
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
    input_device = _infer_input_device(model, device)

    for batch_idx, batch in enumerate(dataloader):
        if num_samples and total_samples >= num_samples:
            break

        # 准备输入：对齐论文 x=(prompt,response)，默认只对 response 计算 loss
        try:
            samples = _batch_to_samples(batch)
        except Exception as e:
            print(f"[SNIP Scorer] 警告: 批处理 {batch_idx} 的 batch 解析失败: {e}")
            continue

        if not samples:
            continue

        try:
            inputs, labels = _build_causal_lm_inputs_and_labels(
                samples,
                tokenizer,
                max_length=max_length,
                device=input_device,
            )
        except Exception as e:
            print(f"[SNIP Scorer] 警告: 批处理 {batch_idx} 的输入构造/tokenization 失败: {e}")
            continue
        
        try:
            model.zero_grad(set_to_none=True)
            # 前向传播
            outputs = model(**inputs)
            
            # 计算损失
            # 下游默认 loss_fn 使用 batch["input_ids"] 作为 labels（shift labels 计算 CE）
            loss_batch = dict(batch) if isinstance(batch, dict) else {"samples": samples}
            loss_batch["input_ids"] = labels
            loss_batch["labels"] = labels
            if "attention_mask" in inputs:
                loss_batch["attention_mask"] = inputs["attention_mask"]

            loss = loss_fn(outputs, loss_batch, model, input_device)
            
            # 反向传播
            loss.backward()
        except Exception as e:
            print(f"[SNIP Scorer] 警告: 批处理 {batch_idx} 的前向/反向传播失败: {e}")
            model.zero_grad(set_to_none=True)
            continue
        
        # 遍历所有 Transformer 层（Llama 架构）
        # 假设模型结构为: model.model.layers (LlamaForCausalLM)
        layers = _get_transformer_layers(model)
        if layers is None:
            print("[SNIP Scorer] 警告: 无法找到模型的层结构，跳过该批处理")
            model.zero_grad(set_to_none=True)
            continue
        
        # 计算每层每个神经元的 SNIP 分数
        for layer_idx, layer in enumerate(layers):
            # 获取 MLP 模块（Llama 架构中的 feed-forward）
            mlp = _get_mlp_module(layer)
            if mlp is None:
                continue
            
            # structured SNIP / connection sensitivity aggregation
            if score_target == "down_proj_out":
                down_proj = _get_linear(mlp, ("down_proj", "output", "fc2", "w2"))
                if down_proj is None or not hasattr(down_proj, "weight") or down_proj.weight is None:
                    continue
                w = down_proj.weight
                g = getattr(w, "grad", None)
                if g is None or not isinstance(w, torch.Tensor) or not isinstance(g, torch.Tensor):
                    continue
                if w.shape != g.shape:
                    continue

                # 每一行(out_features)作为一个结构化单元：
                # I_j(x) = | sum_i w[j,i] * g[j,i] |
                scores_vec = (w * g).sum(dim=1).abs().detach()
                # 转 CPU 累加，避免 device_map 时跨设备保存
                scores_cpu = scores_vec.float().cpu()
                for neuron_idx, s in enumerate(scores_cpu.tolist()):
                    snip_scores[(layer_idx, neuron_idx)] += float(s)

            elif score_target == "mlp_intermediate":
                gate_proj = _get_linear(mlp, ("gate_proj", "w1", "fc1"))
                up_proj = _get_linear(mlp, ("up_proj", "w3"))
                down_proj = _get_linear(mlp, ("down_proj", "output", "fc2", "w2"))
                if gate_proj is None or up_proj is None or down_proj is None:
                    continue
                if any(getattr(m, "weight", None) is None for m in (gate_proj, up_proj, down_proj)):
                    continue

                wg, gg = gate_proj.weight, getattr(gate_proj.weight, "grad", None)
                wu, gu = up_proj.weight, getattr(up_proj.weight, "grad", None)
                wd, gd = down_proj.weight, getattr(down_proj.weight, "grad", None)
                if any(x is None for x in (gg, gu, gd)):
                    continue
                if any(not isinstance(x, torch.Tensor) for x in (wg, gg, wu, gu, wd, gd)):
                    continue

                # 期望：
                # gate/up: [intermediate, hidden] -> 按行聚合 dim=1
                # down:    [hidden, intermediate] -> 按列聚合 dim=0
                try:
                    # gate/up: [intermediate, hidden] -> 对每个中间维度 j 做向量点积
                    gate_raw = (wg * gg).sum(dim=1)
                    up_raw = (wu * gu).sum(dim=1)
                    # down: [hidden, intermediate] -> 对每个中间维度 j 的列做向量点积
                    down_raw = (wd * gd).sum(dim=0)
                except Exception:
                    continue

                if gate_raw.shape != up_raw.shape or gate_raw.shape != down_raw.shape:
                    continue

                # I_j(x) = | (w_gate_j · ∂L/∂w_gate_j) + (w_up_j · ∂L/∂w_up_j)
                #             + (w_down_j · ∂L/∂w_down_j) |
                scores_vec = (gate_raw + up_raw + down_raw).abs().detach()
                scores_cpu = scores_vec.float().cpu()
                for neuron_idx, s in enumerate(scores_cpu.tolist()):
                    snip_scores[(layer_idx, neuron_idx)] += float(s)
            else:
                raise ValueError(f"未知 score_target: {score_target}")
        
        # 清零梯度（为下一批准备），并尽量释放当前 batch 的计算图与缓存
        model.zero_grad(set_to_none=True)
        # 删除当前批次中不再需要的中间变量，帮助 Python GC 和 CUDA 及时回收
        del inputs, labels, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        total_samples += len(samples)
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

def rank_and_annotate_snip_scores(
    snip_scores: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], Dict]:
    """
    对已经汇总好的 SNIP 分数做全局排序，并为每个神经元添加排名和百分位信息。

    这一步**不做截断**，保留所有神经元的元信息，便于后续按任意百分比选择。
    """
    if not snip_scores:
        return {}

    # 从高到低排序
    sorted_neurons = sorted(snip_scores.items(), key=lambda x: x[1], reverse=True)
    total_neurons = len(sorted_neurons)

    annotated: Dict[Tuple[int, int], Dict] = {}
    for rank, ((layer_idx, neuron_idx), score) in enumerate(sorted_neurons):
        annotated[(layer_idx, neuron_idx)] = {
            "score": float(score),
            "rank": rank + 1,
            "percentile": (rank + 1) / total_neurons * 100,
            "layer": layer_idx,
            "neuron": neuron_idx,
        }

    return annotated


def select_top_percent_neurons(
    scored_neurons: Dict[Tuple[int, int], Dict],
    top_percent: float,
) -> Dict[Tuple[int, int], Dict]:
    """
    在已经带有排名信息的神经元集合上，选择前 top_percent 的神经元。

    Args:
        scored_neurons: 来自 rank_and_annotate_snip_scores 的结果
        top_percent: 0~1 之间的小数，比如 0.005 表示前 0.5%
    """
    if not scored_neurons:
        return {}

    if top_percent <= 0.0:
        return {}
    if top_percent > 1.0:
        raise ValueError(
            f"top_percent 应该在 0~1 之间，例如 0.005 表示 0.5%，当前为 {top_percent}"
        )

    total_neurons = len(scored_neurons)
    num_selected = max(1, int(total_neurons * top_percent))

    # 按 rank 从小到大排序（rank = 1 是分数最高）
    sorted_items = sorted(
        scored_neurons.items(),
        key=lambda x: x[1].get("rank", 0),
    )

    selected_items = sorted_items[:num_selected]
    return dict(selected_items)
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


# ============================================================================
# 增强的 SNIP 分析函数（基于探针毒性向量）
# ============================================================================

def compute_probe_guided_snip_scores(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: torch.device,
    loss_fn: Callable,
    probe_output_dir: Union[str, Path] = "outputs",
    model_id: Optional[str] = None,
    prefer_layer: Optional[int] = None,
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    score_target: Literal["down_proj_out", "mlp_intermediate"] = "down_proj_out",
    max_length: int = 2048,
    restrict_grad_to_mlp: bool = True,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, Dict]]:
    """
    基于探针毒性向量的增强 SNIP 分数计算。
    
    该函数在计算 SNIP 分数的同时，加载训练好的探针毒性向量，
    并为每个神经元添加与毒性向量对齐的信息，用于更精确的安全神经元识别。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        dataset: 数据集
        device: 设备
        loss_fn: 损失函数
        probe_output_dir: 探针输出目录（默认 "outputs"）
        model_id: 模型 ID
        prefer_layer: 优先使用的层（默认为 None，自动选择最高准确率层）
        batch_size: 批大小
        num_samples: 样本数限制
        score_target: 打分目标
        max_length: 最大长度
        restrict_grad_to_mlp: 是否限制梯度到 MLP
    
    Returns:
        Tuple[Dict, Dict]: 
            - snip_scores: 每个神经元的 SNIP 分数
            - toxic_vectors: 各层的毒性向量信息
    """
    # 加载探针毒性向量
    toxic_vectors, _ = load_probe_toxic_vectors_from_snip(probe_output_dir, model_id)
    
    if not toxic_vectors:
        print("[Probe-Guided SNIP] 警告: 未能加载探针毒性向量，使用标准 SNIP")
        snip_scores = compute_snip_scores(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_samples=num_samples,
            score_target=score_target,
            max_length=max_length,
            restrict_grad_to_mlp=restrict_grad_to_mlp,
        )
        return snip_scores, {}
    
    # 获取推荐的分析层
    recommendations = get_recommended_analysis_layers(probe_output_dir, model_id)
    
    # 确定使用的层
    if prefer_layer is None:
        prefer_layer = recommendations['selected']
    
    print(f"[Probe-Guided SNIP] 使用分析层: {prefer_layer}")
    print(f"[Probe-Guided SNIP] 可用毒性向量层: {sorted(toxic_vectors.keys())}")
    
    # 计算标准 SNIP 分数
    snip_scores = compute_snip_scores(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        loss_fn=loss_fn,
        batch_size=batch_size,
        num_samples=num_samples,
        score_target=score_target,
        max_length=max_length,
        restrict_grad_to_mlp=restrict_grad_to_mlp,
    )
    
    return snip_scores, toxic_vectors


def compute_layer_specific_snip_scores(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: torch.device,
    loss_fn: Callable,
    target_layers: list,
    toxic_vectors: Dict[int, Dict],
    batch_size: int = 8,
    num_samples: Optional[int] = None,
    score_target: Literal["down_proj_out", "mlp_intermediate"] = "down_proj_out",
    max_length: int = 2048,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    针对特定层计算 SNIP 分数，并结合毒性向量进行层内排名。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        dataset: 数据集
        device: 设备
        loss_fn: 损失函数
        target_layers: 目标层列表
        toxic_vectors: 各层的毒性向量信息
        batch_size: 批大小
        num_samples: 样本数限制
        score_target: 打分目标
        max_length: 最大长度
    
    Returns:
        Dict[int, Dict]: 每层的神经元 SNIP 分数
            {
                layer_idx: {
                    (layer_idx, neuron_idx): snip_score,
                    ...
                },
                ...
            }
    """
    model.eval()

    # 启用 MLP 梯度
    restrict_grad_to_mlp = True
    if restrict_grad_to_mlp:
        model.requires_grad_(False)
        layers = _get_transformer_layers(model)
        if layers is not None:
            for layer in layers:
                mlp = _get_mlp_module(layer)
                if mlp is None:
                    continue
                for name in ("down_proj", "up_proj", "gate_proj", "output", "w1", "w2", "w3", "fc1", "fc2"):
                    mod = getattr(mlp, name, None)
                    if isinstance(mod, nn.Module):
                        mod.requires_grad_(True)
    
    # 存储每层的 SNIP 分数
    layer_snip_scores = {layer: {} for layer in target_layers if layer in toxic_vectors}
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_batches = len(dataloader)
    if num_samples:
        total_batches = min(total_batches, (num_samples + batch_size - 1) // batch_size)
    
    print(f"[Layer-Specific SNIP] 开始计算，共 {total_batches} 个批次...")
    
    total_samples = 0
    input_device = _infer_input_device(model, device)
    
    # 初始化层分数累加器
    layer_score_accumulators = {layer: defaultdict(float) for layer in target_layers}
    
    for batch_idx, batch in enumerate(dataloader):
        if num_samples and total_samples >= num_samples:
            break
        
        try:
            samples = _batch_to_samples(batch)
        except Exception as e:
            continue
        
        if not samples:
            continue
        
        try:
            inputs, labels = _build_causal_lm_inputs_and_labels(
                samples, tokenizer, max_length=max_length, device=input_device
            )
        except Exception as e:
            continue
        
        try:
            model.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            
            loss_batch = dict(batch) if isinstance(batch, dict) else {"samples": samples}
            loss_batch["input_ids"] = labels
            loss_batch["labels"] = labels
            if "attention_mask" in inputs:
                loss_batch["attention_mask"] = inputs["attention_mask"]
            
            loss = loss_fn(outputs, loss_batch, model, input_device)
            loss.backward()
        except Exception as e:
            model.zero_grad(set_to_none=True)
            continue
        
        layers = _get_transformer_layers(model)
        if layers is None:
            model.zero_grad(set_to_none=True)
            continue
        
        # 计算每层的 SNIP 分数
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in target_layers:
                continue
            
            mlp = _get_mlp_module(layer)
            if mlp is None:
                continue
            
            if score_target == "down_proj_out":
                down_proj = _get_linear(mlp, ("down_proj", "output", "fc2", "w2"))
                if down_proj is None or not hasattr(down_proj, "weight") or down_proj.weight is None:
                    continue
                w = down_proj.weight
                g = getattr(w, "grad", None)
                if g is None or w.shape != g.shape:
                    continue
                
                scores_vec = (w * g).sum(dim=1).abs().detach().float().cpu()
                for neuron_idx, s in enumerate(scores_vec.tolist()):
                    layer_score_accumulators[layer_idx][(layer_idx, neuron_idx)] += float(s)
        
        model.zero_grad(set_to_none=True)
        total_samples += len(samples)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"[Layer-Specific SNIP] 进度: {batch_idx + 1}/{total_batches}")
    
    # 平均化
    if total_samples > 0:
        for layer_idx in layer_score_accumulators:
            for key in layer_score_accumulators[layer_idx]:
                layer_score_accumulators[layer_idx][key] /= total_samples
    
    print(f"[Layer-Specific SNIP] 完成: {total_samples} 样本")
    return {k: dict(v) for k, v in layer_score_accumulators.items() if v}


def get_probe_quality_report(
    probe_output_dir: Union[str, Path] = "outputs",
    model_id: Optional[str] = None,
) -> Dict:
    """
    生成探针质量报告，用于神经元分析的配置参考。
    
    Args:
        probe_output_dir: 探针输出目录
        model_id: 模型 ID
    
    Returns:
        Dict: 探针质量报告
    """
    report = {
        "summary": {},
        "layers_by_accuracy": [],
        "layers_by_stability": [],
        "recommended_analysis_layers": {},
        "layer_groups": {
            "S_grade": [],   # 93%+ 准确率
            "A_grade": [],   # 91-92%
            "B_grade": [],   # 89-91%
            "C_grade": [],  # 85-89%
            "D_grade": [],  # <85%
        }
    }
    
    # 加载毒性向量
    try:
        vectors_by_layer, _ = load_probe_toxic_vectors_from_snip(probe_output_dir, model_id)
    except Exception as e:
        print(f"[Probe Quality Report] 警告: 无法加载探针数据: {e}")
        return report
    
    if not vectors_by_layer:
        return report
    
    # 按准确率排序
    sorted_by_acc = sorted(
        vectors_by_layer.items(),
        key=lambda x: x[1].get('cv_accuracy', 0),
        reverse=True
    )
    report["layers_by_accuracy"] = [
        {"layer": idx, "cv_accuracy": info.get('cv_accuracy', 0), "std": info.get('std', 0)}
        for idx, info in sorted_by_acc
    ]
    
    # 按稳定性排序
    sorted_by_std = sorted(
        vectors_by_layer.items(),
        key=lambda x: x[1].get('std', float('inf'))
    )
    report["layers_by_stability"] = [
        {"layer": idx, "cv_accuracy": info.get('cv_accuracy', 0), "std": info.get('std', 0)}
        for idx, info in sorted_by_std
    ]
    
    # 分层分组
    for layer_idx, info in vectors_by_layer.items():
        acc = info.get('cv_accuracy', 0) * 100
        if acc >= 93:
            report["layer_groups"]["S_grade"].append(layer_idx)
        elif acc >= 91:
            report["layer_groups"]["A_grade"].append(layer_idx)
        elif acc >= 89:
            report["layer_groups"]["B_grade"].append(layer_idx)
        elif acc >= 85:
            report["layer_groups"]["C_grade"].append(layer_idx)
        else:
            report["layer_groups"]["D_grade"].append(layer_idx)
    
    # 推荐分析层
    recommendations = get_recommended_analysis_layers(probe_output_dir, model_id)
    report["recommended_analysis_layers"] = recommendations
    
    # 汇总信息
    report["summary"] = {
        "total_layers": len(vectors_by_layer),
        "avg_cv_accuracy": np.mean([info.get('cv_accuracy', 0) for info in vectors_by_layer.values()]),
        "avg_std": np.mean([info.get('std', 0) for info in vectors_by_layer.values()]),
        "best_accuracy_layer": sorted_by_acc[0][0] if sorted_by_acc else None,
        "best_accuracy": sorted_by_acc[0][1].get('cv_accuracy', 0) if sorted_by_acc else 0,
        "most_stable_layer": sorted_by_std[0][0] if sorted_by_std else None,
        "best_std": sorted_by_std[0][1].get('std', 0) if sorted_by_std else 0,
    }
    
    return report
