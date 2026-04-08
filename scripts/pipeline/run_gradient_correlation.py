#!/usr/bin/env python3
r"""
梯度关联分析运行脚本

根据论文5.4节要求，使用Wdown神经元作为锚点，追踪来自前一层模块的参数级影响。
通过测量参数扰动如何传播到Wdown激活来量化上游神经元与安全机制的因果关系。

梯度关联公式：G_{i,j} = ∂a^k_down,i / ∂w^k_upstream,j
- a^k_down,i: 第k层down_proj的第i个神经元的激活值
- w^k_upstream,j: 上游（前一层，即k-1层）第j个神经元的权重参数（down_proj权重）

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/run_gradient_correlation.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_correlation \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json

从宿主机运行（通过 docker exec）：
    docker exec -it neurobreak-container python /workspace/scripts/run_gradient_correlation.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_correlation \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json

Windows 环境使用（本地运行）：
    python scripts/run_gradient_correlation.py ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --dataset-path logs/base_evaluation.jsonl ^
        --output-path outputs/gradient_correlation ^
        --target-neurons-path outputs/snip_scores/safety_neurons.json

内存优化选项（如果遇到 GPU 内存不足）：
    python /workspace/scripts/run_gradient_correlation.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_correlation \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json \
        --load-in-4bit \
        --clear-cache \
        --batch-size 2 \
        --num-samples 100 \
        --use-gradient-checkpointing \
        --selective-gradients


速度优化选项（在保证准确率的前提下加速）：
    python /workspace/scripts/run_gradient_correlation.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_correlation \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json \
        --num-samples 200 \
        --max-length 512 \
        --batch-size 8 \
        --selective-gradients \
        --skip-zero-gradients

注意：
- 必须指定 --target-neurons-path，因为梯度关联分析需要明确的目标神经元
- 目标神经元文件格式应为 JSON，包含神经元信息（支持 safety_neurons、utility_neurons 或 all_neurons 字段）
- 数据集应为 JSONL 格式，每行包含文本字段：'text'、'prompt' 或 'input.prompt'
- 脚本会自动检测项目根目录，支持 Docker 环境和本地环境
- 如果遇到 GPU 内存不足，建议使用以下优化选项：
  * --load-in-4bit 或 --load-in-8bit：模型量化（需要安装 bitsandbytes）
  * --use-gradient-checkpointing：梯度检查点（减少激活值内存，但增加计算时间）
  * --selective-gradients：只对需要的层启用梯度（默认启用，减少梯度内存）
  * --batch-size 2 或 1：减小批次大小
  * --num-samples 100：限制样本数量
- --clear-cache 参数可以在加载模型前清理 GPU 缓存
- 梯度关联分析计算量较大，建议使用较小的 batch_size 和 num_samples 进行测试
- 速度优化建议（在保证准确率的前提下）：
  * --num-samples 200-500：减少样本数量（对准确率影响较小，但显著加速）
  * --max-length 512-1024：减少序列长度（对准确率影响较小，但显著加速）
  * --batch-size 8-16：如果内存允许，增大批次大小（可加速，但受限于内存）
  * --skip-zero-gradients：跳过零梯度计算（默认启用，可加速）
  * 禁用 --use-gradient-checkpointing：梯度检查点会增加计算时间（仅用于内存不足时）
  * 启用 --selective-gradients：只对需要的层启用梯度（默认启用，减少计算量）
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset

# 添加工作目录到路径
# 支持两种方式：1) 从脚本位置推断项目根目录 2) 使用 /workspace（Docker 环境）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.exists() and (PROJECT_ROOT / 'engine').exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    # Docker 环境或使用绝对路径
    workspace_path = os.getenv('WORKSPACE_PATH', '/workspace')
    if os.path.exists(workspace_path):
        sys.path.insert(0, workspace_path)
    else:
        # 尝试当前工作目录
        cwd = Path.cwd()
        if (cwd / 'engine').exists():
            sys.path.insert(0, str(cwd))
        else:
            # 最后尝试 /workspace
            sys.path.insert(0, '/workspace')

from engine.neurons.gradient_dependency import compute_gradient_dependency


def check_model_quantization(model) -> Tuple[bool, int, int]:
    """
    检查模型是否使用了量化
    
    Args:
        model: 模型对象
        
    Returns:
        (has_quantized_weights, quantized_layer_count, total_layer_count)
    """
    has_quantized_weights = False
    quantized_layer_count = 0
    total_layer_count = 0
    
    # 检查模型是否有量化配置
    if hasattr(model, 'quantization_config') or hasattr(model, 'hf_quantizer'):
        has_quantized_weights = True
    
    # 检查 Transformer 层
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    
    if layers:
        for layer_idx, layer in enumerate(layers):
            # 检查 MLP 模块
            mlp = None
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
            elif hasattr(layer, "feed_forward"):
                mlp = layer.feed_forward
            
            if mlp:
                # 检查 down_proj
                down_proj = None
                if hasattr(mlp, "down_proj"):
                    down_proj = mlp.down_proj
                elif hasattr(mlp, "output"):
                    down_proj = mlp.output
                elif hasattr(mlp, "fc2"):
                    down_proj = mlp.fc2
                elif hasattr(mlp, "w2"):
                    down_proj = mlp.w2
                
                if down_proj is not None:
                    total_layer_count += 1
                    # 检查权重
                    weight = None
                    if hasattr(down_proj, 'weight'):
                        weight = down_proj.weight
                    elif hasattr(down_proj, 'base_layer') and hasattr(down_proj.base_layer, 'weight'):
                        weight = down_proj.base_layer.weight
                    
                    if weight is not None:
                        # 检查是否是量化权重
                        if hasattr(weight, 'quant_state') or hasattr(down_proj, 'quantization_config'):
                            quantized_layer_count += 1
                            has_quantized_weights = True
                        elif weight.device.type == 'meta':
                            # meta tensor 也可能表示未加载或量化
                            quantized_layer_count += 1
                            has_quantized_weights = True
                        elif hasattr(weight, 'dtype') and weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                            quantized_layer_count += 1
                            has_quantized_weights = True
    
    return has_quantized_weights, quantized_layer_count, total_layer_count


class TextDataset(Dataset):
    """文本数据集，支持 JSONL 格式"""
    
    def __init__(self, file_path: str):
        self.samples = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        # 检查文件格式
        if file_path.endswith('.jsonl'):
            # JSONL 格式：从每行 JSON 中提取文本
            print(f'[Gradient Correlation] 检测到 JSONL 格式，加载样本...')
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line.strip())
                        self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f'[Gradient Correlation] 警告: 第 {line_num} 行 JSON 解析失败: {e}')
                        continue
                    except Exception as e:
                        print(f'[Gradient Correlation] 警告: 第 {line_num} 行处理失败: {e}')
                        continue
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，仅支持 .jsonl 格式")
        
        print(f'[Gradient Correlation] 成功加载 {len(self.samples)} 个样本')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_target_neurons(neurons_file: str) -> Optional[Dict[Tuple[int, int], Dict]]:
    """
    从JSON文件中加载目标神经元
    
    支持多种JSON格式：
    1. 嵌套格式：包含 'safety_neurons', 'utility_neurons', 'all_neurons', 'dedicated_safety_neurons' 键
       - 例如：dedicated_safety_neurons.json
       - 键格式：支持 "layer_X_neuron_Y" 或 "X_Y" (如 "31_4062")
       - 值格式：支持 {layer, neuron, ...} 或 {layer_idx, neuron_idx, ...}
    
    2. 直接格式：根级别就是神经元字典
       - 例如：activation_projection.json, quadrant_classification.json
       - 键格式：必须为 "layer_X_neuron_Y" (如 "layer_31_neuron_4062")
       - 值格式：支持 {layer_idx, neuron_idx, ...} 或 {layer, neuron, ...}
    
    Args:
        neurons_file: JSON文件路径
    
    Returns:
        目标神经元字典，格式为 Dict[(layer_idx, neuron_idx), Dict]
        如果文件不存在或格式不正确，返回 None
    """
    if not os.path.exists(neurons_file):
        print(f"[Gradient Correlation] 警告: 目标神经元文件不存在: {neurons_file}")
        return None
    
    print(f"[Gradient Correlation] 加载目标神经元: {neurons_file}")
    
    with open(neurons_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 自动检测键名（支持多种神经元集合格式）
    neurons_key = None
    for key in ['safety_neurons', 'utility_neurons', 'all_neurons', 'dedicated_safety_neurons']:
        if key in data:
            neurons_key = key
            print(f"[Gradient Correlation] 检测到神经元集合键: {key}")
            break
    
    if neurons_key is None:
        # 尝试直接解析为神经元字典格式（layer_X_neuron_Y）
        # 支持 activation_projection.json 和 quadrant_classification.json 等格式
        if any(k.startswith('layer_') and '_neuron_' in k for k in data.keys()):
            print("[Gradient Correlation] 检测到直接神经元格式（layer_X_neuron_Y，如 activation_projection.json）")
            target_neurons = {}
            for key, value in data.items():
                # 支持 layer_idx/neuron_idx 或 layer/neuron 字段
                if 'layer_idx' in value and 'neuron_idx' in value:
                    layer_idx = int(value['layer_idx'])
                    neuron_idx = int(value['neuron_idx'])
                    target_neurons[(layer_idx, neuron_idx)] = value
                elif 'layer' in value and 'neuron' in value:
                    layer_idx = int(value['layer'])
                    neuron_idx = int(value['neuron'])
                    target_neurons[(layer_idx, neuron_idx)] = value
                elif key.startswith('layer_') and '_neuron_' in key:
                    # 从键名解析：layer_X_neuron_Y
                    try:
                        parts = key.split('_')
                        if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'neuron':
                            layer_idx = int(parts[1])
                            neuron_idx = int(parts[3])
                            target_neurons[(layer_idx, neuron_idx)] = value
                    except (ValueError, IndexError):
                        continue
            print(f"[Gradient Correlation] 从直接格式加载了 {len(target_neurons)} 个神经元")
            return target_neurons
        else:
            print(f"[Gradient Correlation] 错误: 无法识别神经元格式")
            print(f"[Gradient Correlation] 支持的格式：")
            print(f"  - 嵌套格式：包含 'dedicated_safety_neurons' 等键（如 dedicated_safety_neurons.json）")
            print(f"  - 直接格式：根级别为神经元字典，键为 'layer_X_neuron_Y'（如 activation_projection.json）")
            return None
    
    neurons_data = data[neurons_key]
    
    # 解析神经元数据
    target_neurons = {}
    for key, value in neurons_data.items():
        # 支持多种格式：
        # 1. "layer_X_neuron_Y": {layer_idx, neuron_idx, ...} 或 {layer, neuron, ...}
        # 2. "X_Y": {layer, neuron, ...} (下划线分隔格式)
        # 3. 直接包含 layer_idx/neuron_idx 或 layer/neuron 的字典
        
        # 首先尝试从值中获取
        if 'layer_idx' in value and 'neuron_idx' in value:
            layer_idx = int(value['layer_idx'])
            neuron_idx = int(value['neuron_idx'])
            target_neurons[(layer_idx, neuron_idx)] = value
        elif 'layer' in value and 'neuron' in value:
            layer_idx = int(value['layer'])
            neuron_idx = int(value['neuron'])
            target_neurons[(layer_idx, neuron_idx)] = value
        elif '_' in key:
            # 尝试从键名解析
            try:
                parts = key.split('_')
                # 格式1: layer_X_neuron_Y
                if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'neuron':
                    layer_idx = int(parts[1])
                    neuron_idx = int(parts[3])
                    target_neurons[(layer_idx, neuron_idx)] = value
                # 格式2: X_Y (下划线分隔，如 "31_4062")
                elif len(parts) == 2:
                    layer_idx = int(parts[0])
                    neuron_idx = int(parts[1])
                    target_neurons[(layer_idx, neuron_idx)] = value
            except (ValueError, IndexError):
                continue
    
    print(f"[Gradient Correlation] 成功加载 {len(target_neurons)} 个目标神经元")
    return target_neurons


def save_gradient_correlation(
    gradient_correlation: Dict[Tuple[int, int], Dict],
    output_path: Path,
    filename: str = "gradient_correlation.json",
):
    """
    保存梯度关联结果到JSON文件
    
    Args:
        gradient_correlation: 梯度关联结果字典
        output_path: 输出目录路径
        filename: 输出文件名
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename
    
    # 转换元组键为字符串键（JSON不支持元组作为键）
    json_data = {}
    for (layer_idx, neuron_idx), data in gradient_correlation.items():
        key = f"layer_{layer_idx}_neuron_{neuron_idx}"
        json_data[key] = {
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "upstream_neurons": [
                {"layer_idx": l, "neuron_idx": n} 
                for l, n in data['upstream_neurons']
            ],
            "gradient_strengths": data['gradient_strengths'],
            "num_upstream_neurons": len(data['upstream_neurons']),
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"[Gradient Correlation] 结果已保存到: {output_file}")
    print(f"[Gradient Correlation] 共保存 {len(json_data)} 个目标神经元的梯度关联信息")


def main():
    parser = argparse.ArgumentParser(
        description="计算神经元之间的梯度关联（G_{i,j}）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径（本地路径或 HuggingFace 模型ID）"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="数据集路径（JSONL格式）"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--target-neurons-path",
        type=str,
        required=True,
        help="目标神经元文件路径（JSON格式）"
    )
    
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.1,
        help="保留前k%%的强关联（默认0.1，即10%%）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批大小（默认4，梯度计算需要更多内存）"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="最大序列长度（默认2048）"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="使用的样本数（None表示全部，默认None）"
    )
    
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="使用4-bit量化加载模型（节省内存）"
    )
    
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="使用8-bit量化加载模型（节省内存）"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="在加载模型前清理GPU缓存"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（'cuda' 或 'cpu'，默认自动检测）"
    )
    
    parser.add_argument(
        "--use-gradient-checkpointing",
        action="store_true",
        help="使用梯度检查点（减少激活值内存，但增加计算时间）"
    )
    
    parser.add_argument(
        "--selective-gradients",
        action="store_true",
        default=True,
        help="只对需要的层启用梯度（减少梯度内存，默认启用）"
    )
    
    parser.add_argument(
        "--no-selective-gradients",
        dest="selective_gradients",
        action="store_false",
        help="对所有层启用梯度（禁用选择性梯度）"
    )
    
    parser.add_argument(
        "--skip-zero-gradients",
        action="store_true",
        default=True,
        help="跳过零梯度计算以加速（默认启用）"
    )
    
    parser.add_argument(
        "--no-skip-zero-gradients",
        dest="skip_zero_gradients",
        action="store_false",
        help="不跳过零梯度计算（禁用零梯度跳过）"
    )
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(f"[Gradient Correlation] 使用设备: {device}")
    
    # 清理GPU缓存（如果需要）
    if args.clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[Gradient Correlation] 已清理GPU缓存")
    
    # 加载模型
    print(f"[Gradient Correlation] 加载模型: {args.model_path}")
    model_kwargs = {}
    
    using_quantization = False
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs['quantization_config'] = quantization_config
        print("[Gradient Correlation] ⚠️  使用4-bit量化")
        using_quantization = True
    elif args.load_in_8bit:
        model_kwargs['load_in_8bit'] = True
        print("[Gradient Correlation] ⚠️  使用8-bit量化")
        using_quantization = True
    
    if using_quantization:
        print("[Gradient Correlation] ⚠️  警告: 量化模型无法计算梯度！")
        print("[Gradient Correlation] ⚠️  梯度关联分析需要完整精度的权重来计算梯度。")
        print("[Gradient Correlation] ⚠️  建议: 移除 --load-in-4bit 或 --load-in-8bit 参数，使用完整精度模型。")
        print("[Gradient Correlation] ⚠️  如果内存不足，可以尝试:")
        print("[Gradient Correlation]      - 减小 --batch-size (如 --batch-size 2 或 1)")
        print("[Gradient Correlation]      - 减小 --num-samples (如 --num-samples 100)")
        print("[Gradient Correlation]      - 减小 --max-length (如 --max-length 512)")
        print("[Gradient Correlation] 继续运行，但可能无法收集到梯度关联...")
        print()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            **model_kwargs
        )
    except Exception as e:
        print(f"[Gradient Correlation] 模型加载失败: {e}")
        return
    
    # 加载后立即检查量化状态
    print("[Gradient Correlation] 检查模型量化状态...")
    has_quantized, quantized_count, total_count = check_model_quantization(model)
    if has_quantized:
        print(f"[Gradient Correlation] ⚠️  检测到量化权重: {quantized_count}/{total_count} 层")
        if not using_quantization:
            print("[Gradient Correlation] ⚠️  模型可能已预量化，或使用了其他量化方法")
        print("[Gradient Correlation] ⚠️  量化权重无法计算梯度，梯度关联分析可能无法正常工作！")
        print()
    else:
        print(f"[Gradient Correlation] ✓ 模型权重检查通过（{total_count} 层，无量化）")
        print()
    
    # 加载分词器
    print(f"[Gradient Correlation] 加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    except Exception as e:
        print(f"[Gradient Correlation] 分词器加载失败: {e}")
        return
    
    # 加载数据集
    print(f"[Gradient Correlation] 加载数据集: {args.dataset_path}")
    try:
        dataset = TextDataset(args.dataset_path)
    except Exception as e:
        print(f"[Gradient Correlation] 数据集加载失败: {e}")
        return
    
    # 加载目标神经元
    target_neurons = load_target_neurons(args.target_neurons_path)
    if target_neurons is None or len(target_neurons) == 0:
        print("[Gradient Correlation] 错误: 未找到有效的目标神经元")
        return
    
    # 计算梯度关联
    print(f"[Gradient Correlation] 开始计算梯度关联...")
    print(f"[Gradient Correlation] 参数: top_k={args.top_k}, batch_size={args.batch_size}, "
          f"max_length={args.max_length}, num_samples={args.num_samples}")
    print(f"[Gradient Correlation] 优化选项: selective_gradients={args.selective_gradients}, "
          f"skip_zero_gradients={args.skip_zero_gradients}, "
          f"gradient_checkpointing={args.use_gradient_checkpointing}")
    print()
    
    # 预检查：如果使用了量化，给出最终警告
    if has_quantized:
        print("[Gradient Correlation] ⚠️  最终警告: 检测到量化权重，梯度关联分析可能失败！")
        print("[Gradient Correlation] ⚠️  如果第一个批次后未收集到任何梯度关联，请:")
        print("[Gradient Correlation]     1. 重新运行时不使用量化参数")
        print("[Gradient Correlation]     2. 查看详细故障排除指南: docs/gradient_correlation_troubleshooting.md")
        print("[Gradient Correlation]     3. 查看快速修复指南: docs/gradient_correlation_quick_fix.md")
        print()
    
    try:
        gradient_correlation = compute_gradient_dependency(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            target_neurons=target_neurons,
            device=device,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_samples=args.num_samples,
            use_last_token=True,
        )
    except KeyboardInterrupt:
        print("\n[Gradient Correlation] 用户中断")
        return
    except Exception as e:
        print(f"\n[Gradient Correlation] ❌ 计算失败: {e}")
        print("[Gradient Correlation] 如果遇到梯度相关错误，请检查:")
        print("  1. 是否使用了量化模型（--load-in-4bit/--load-in-8bit）")
        print("  2. 模型权重是否正确加载")
        print("  3. 目标神经元索引是否在有效范围内")
        print("  4. 查看详细错误信息和堆栈跟踪:")
        import traceback
        traceback.print_exc()
        print("\n[Gradient Correlation] 故障排除指南:")
        print("  - 快速修复: docs/gradient_correlation_quick_fix.md")
        print("  - 详细指南: docs/gradient_correlation_troubleshooting.md")
        return
    
    # 检查结果
    if gradient_correlation is None or len(gradient_correlation) == 0:
        print("\n[Gradient Correlation] ⚠️  警告: 未收集到任何梯度关联结果")
        print("[Gradient Correlation] 可能的原因:")
        print("  1. 使用了量化模型（量化权重无法计算梯度）")
        print("  2. 所有权重都未启用梯度")
        print("  3. 未捕获到任何激活值")
        print("[Gradient Correlation] 请查看诊断输出和故障排除指南:")
        print("  - 快速修复: docs/gradient_correlation_quick_fix.md")
        print("  - 详细指南: docs/gradient_correlation_troubleshooting.md")
        return
    
    # 保存结果
    output_path = Path(args.output_path)
    save_gradient_correlation(gradient_correlation, output_path)
    
    # 统计结果
    total_upstream = sum(len(data.get('upstream_neurons', [])) for data in gradient_correlation.values())
    print(f"[Gradient Correlation] ✓ 成功计算 {len(gradient_correlation)} 个目标神经元的梯度关联")
    print(f"[Gradient Correlation] ✓ 共发现 {total_upstream} 个上游神经元关联")
    print("[Gradient Correlation] 完成！")


if __name__ == "__main__":
    main()
