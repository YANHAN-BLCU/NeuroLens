#!/usr/bin/env python3
r"""
激活投影分析运行脚本

根据论文5.4节要求，计算神经元在jailbreak样本中的激活投影（A_i^k）。

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/run_activation_projection.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection

    python /workspace/scripts/run_activation_projection.py \
        --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection

可选：指定目标神经元（只分析这些神经元）：
    python /workspace/scripts/run_activation_projection.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json

从宿主机运行（通过 docker exec）：
    docker exec -it neurobreak-container python /workspace/scripts/run_activation_projection.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection

Windows 环境使用（本地运行）：
    python scripts/run_activation_projection.py ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz ^
        --dataset-path logs/base_evaluation.jsonl ^
        --output-path outputs/activation_projection

内存优化选项（如果遇到 GPU 内存不足）：
    # 使用 4-bit 量化（最节省内存，推荐）
    python /workspace/scripts/run_activation_projection.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection \
        --load-in-4bit \
        --clear-cache

    # 使用 4-bit 量化 + 指定目标神经元
    python /workspace/scripts/run_activation_projection.py \
        --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection \
        --target-neurons-path /workspace/outputs/dedicated_safety_neurons.json \
        --load-in-4bit \
        --clear-cache

    # 使用 8-bit 量化（比 4-bit 占用更多内存，但精度更高）
    python /workspace/scripts/run_activation_projection.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/activation_projection \
        --load-in-8bit \
        --clear-cache

注意：
- 如果不指定 --target-neurons-path，将分析所有层的所有神经元
- 目标神经元文件格式应为 JSON，包含神经元信息（支持 safety_neurons、utility_neurons 或 all_neurons 字段）
- 数据集应为 JSONL 格式，每行包含：
    - 文本字段：'text'、'prompt' 或 'input.prompt'
    - jailbreak成功标志：'jailbreak_success'、'asr_label'、'success' 或 'guard.jailbreak_success'、'guard.asr_label'
- 脚本会自动检测项目根目录，支持 Docker 环境和本地环境
- 如果遇到 GPU 内存不足，建议使用 --load-in-4bit 或 --load-in-8bit 参数（需要安装 bitsandbytes）
- --clear-cache 参数可以在加载模型前清理 GPU 缓存
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

from engine.neurons.activation_projection import compute_activation_projection


class JailbreakDataset(Dataset):
    """Jailbreak数据集，支持 JSONL 格式，包含文本和jailbreak成功标志"""
    
    def __init__(self, file_path: str):
        self.samples = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        # 检查文件格式
        if file_path.endswith('.jsonl'):
            # JSONL 格式：从每行 JSON 中提取文本和jailbreak标志
            print(f'[Activation Projection] 检测到 JSONL 格式，加载jailbreak样本...')
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line.strip())
                        self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f'[Activation Projection] 警告: 第 {line_num} 行 JSON 解析失败: {e}')
                        continue
                    except Exception as e:
                        print(f'[Activation Projection] 警告: 第 {line_num} 行处理失败: {e}')
                        continue
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，仅支持 .jsonl 格式")
        
        print(f'[Activation Projection] 成功加载 {len(self.samples)} 个样本')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_target_neurons(neurons_file: str) -> Optional[Dict[Tuple[int, int], Dict]]:
    """
    从JSON文件中加载目标神经元
    
    Args:
        neurons_file: JSON文件路径，应包含 safety_neurons、utility_neurons 或 all_neurons 字段
    
    Returns:
        目标神经元字典，格式为 Dict[(layer_idx, neuron_idx), Dict]
        如果文件不存在或格式不正确，返回 None
    """
    if not os.path.exists(neurons_file):
        print(f"[Activation Projection] 警告: 目标神经元文件不存在: {neurons_file}")
        return None
    
    print(f"[Activation Projection] 加载目标神经元: {neurons_file}")
    
    with open(neurons_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 自动检测键名
    neurons_key = None
    for key in ['safety_neurons', 'utility_neurons', 'all_neurons', 'dedicated_safety_neurons']:
        if key in data:
            neurons_key = key
            break
    
    if neurons_key is None:
        # 尝试直接解析为神经元字典格式（layer_X_neuron_Y）
        if any(k.startswith('layer_') and '_neuron_' in k for k in data.keys()):
            print("[Activation Projection] 检测到直接神经元格式（layer_X_neuron_Y）")
            target_neurons = {}
            for key, value in data.items():
                # 支持 layer_idx/neuron_idx 或 layer/neuron 字段
                # 保留所有字段（包括 score、rank、percentile 等）
                if 'layer_idx' in value and 'neuron_idx' in value:
                    layer_idx = int(value['layer_idx'])
                    neuron_idx = int(value['neuron_idx'])
                    target_neurons[(layer_idx, neuron_idx)] = value
                elif 'layer' in value and 'neuron' in value:
                    layer_idx = int(value['layer'])
                    neuron_idx = int(value['neuron'])
                    target_neurons[(layer_idx, neuron_idx)] = value
            print(f"[Activation Projection] 从直接格式加载了 {len(target_neurons)} 个神经元")
            return target_neurons
        else:
            print(f"[Activation Projection] 警告: 无法识别神经元格式，将分析所有神经元")
            return None
    
    neurons_data = data[neurons_key]
    print(f"[Activation Projection] 从 '{neurons_key}' 字段加载神经元...")
    
    target_neurons = {}
    for key, value in neurons_data.items():
        # 支持多种格式：
        # 1. "layer_X_neuron_Y": {layer_idx, neuron_idx, ...} 或 {layer, neuron, ...}
        # 2. "X_Y": {layer, neuron, ...} (下划线分隔格式)
        # 3. 直接包含 layer_idx/neuron_idx 或 layer/neuron 的字典
        
        # 首先尝试从值中获取
        # 保留所有字段（包括 score、rank、percentile 等）
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
    
    print(f"[Activation Projection] 成功加载 {len(target_neurons)} 个目标神经元")
    return target_neurons


def save_activation_projection(
    activation_projection: Dict[Tuple[int, int], Dict],
    output_path: Path,
    filename: str = "activation_projection.json",
):
    """
    保存激活投影结果到JSON文件
    
    Args:
        activation_projection: 激活投影结果
        output_path: 输出目录
        filename: 输出文件名
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化格式（只保存激活投影相关数据，不包含 score、rank、percentile 等字段）
    serializable = {}
    for (layer_idx, neuron_idx), data in activation_projection.items():
        key = f"layer_{layer_idx}_neuron_{neuron_idx}"
        neuron_data = {
            'layer_idx': int(layer_idx),
            'neuron_idx': int(neuron_idx),
            **data
        }
        
        serializable[key] = neuron_data
    
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"[Activation Projection] 结果已保存到: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='激活投影分析 - 计算神经元在jailbreak样本中的激活投影',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--toxic-vectors-path',
        type=str,
        required=True,
        help='毒性向量文件路径（.npz格式）'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Jailbreak数据集路径（.jsonl格式）'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='输出路径'
    )
    parser.add_argument(
        '--target-neurons-path',
        type=str,
        default=None,
        help='目标神经元文件路径（可选，JSON格式）。如果提供，只分析这些神经元；否则分析所有神经元'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='activation_projection.json',
        help='输出文件名（默认: activation_projection.json）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批大小（默认: 8）'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='tokenization最大长度（默认: 2048）'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='使用的样本数限制（None表示全部，默认: None）。'
             '注意：会分别限制成功和失败样本的数量，确保两种类型的样本都能被充分分析'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='计算设备（默认: auto，自动检测）'
    )
    parser.add_argument(
        '--torch-dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'float32'],
        help='模型数据类型（默认: auto，GPU使用float16，CPU使用float32）'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='使用 8-bit 量化加载模型（节省显存，需要 bitsandbytes）'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='使用 4-bit 量化加载模型（节省显存，需要 bitsandbytes）'
    )
    parser.add_argument(
        '--low-cpu-mem-usage',
        action='store_true',
        default=True,
        help='使用低 CPU 内存模式加载模型（默认启用）'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='在加载模型前清理 GPU 缓存'
    )
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("========================================")
    print("激活投影分析 - Activation Projection")
    print("========================================")
    print()
    print("参数配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  毒性向量路径: {args.toxic_vectors_path}")
    print(f"  数据集路径: {args.dataset_path}")
    print(f"  输出路径: {args.output_path}")
    if args.target_neurons_path:
        print(f"  目标神经元路径: {args.target_neurons_path}")
    else:
        print("  目标神经元: 未指定（将分析所有神经元）")
    print(f"  输出文件名: {args.output_filename}")
    print(f"  批大小: {args.batch_size}")
    print(f"  最大长度: {args.max_length}")
    print(f"  样本数限制: {args.num_samples if args.num_samples else '全部'}")
    print(f"  设备: {args.device}")
    print(f"  数据类型: {args.torch_dtype}")
    if args.load_in_8bit:
        print(f"  量化: 8-bit")
    elif args.load_in_4bit:
        print(f"  量化: 4-bit")
    if args.clear_cache:
        print(f"  清理缓存: 是")
    print()
    
    # 检查文件是否存在
    if os.path.exists(args.model_path):
        print(f"[Activation Projection] 使用本地模型路径: {args.model_path}")
    else:
        print(f"[Activation Projection] 模型路径不存在，将尝试从 Hugging Face Hub 加载: {args.model_path}")
    
    if not os.path.exists(args.toxic_vectors_path):
        raise FileNotFoundError(f"毒性向量文件不存在: {args.toxic_vectors_path}")
    
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {args.dataset_path}")
    
    # 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[Activation Projection] 设备: {device}")
    
    # 清理 GPU 缓存（如果请求）
    if args.clear_cache and device.type == 'cuda':
        print("[Activation Projection] 清理 GPU 缓存...")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"[Activation Projection] GPU 内存使用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB / {torch.cuda.get_device_properties(device).total_memory/1024**3:.2f} GB")
    
    # 检查量化选项
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("不能同时使用 8-bit 和 4-bit 量化")
    
    use_quantization = args.load_in_8bit or args.load_in_4bit
    if use_quantization:
        try:
            import bitsandbytes  # noqa: F401
            print(f"[Activation Projection] 使用 {'8-bit' if args.load_in_8bit else '4-bit'} 量化")
        except ImportError:
            raise ImportError("使用量化需要安装 bitsandbytes: pip install bitsandbytes")
    
    # 确定数据类型
    if use_quantization:
        # 量化模式下，torch_dtype 会被忽略，但我们需要设置它
        torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        print(f"[Activation Projection] 量化模式，基础数据类型: {torch_dtype}")
    elif args.torch_dtype == 'auto':
        if device.type == 'cuda':
            torch_dtype = torch.float16
            print("[Activation Projection] 检测到 CUDA，使用 FP16 加载模型")
        else:
            torch_dtype = torch.float32
            print("[Activation Projection] 使用 CPU，使用 FP32 加载模型")
    elif args.torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # 加载模型
    print("[Activation Projection] 加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 准备模型加载参数
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'low_cpu_mem_usage': args.low_cpu_mem_usage,
    }
    
    # 添加量化参数（使用 BitsAndBytesConfig）
    if args.load_in_8bit:
        # 8-bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model_kwargs['quantization_config'] = quantization_config
        # 8-bit 量化时，device_map 会自动处理
        if device.type == 'cuda':
            model_kwargs['device_map'] = {"": device}
    elif args.load_in_4bit:
        # 4-bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs['quantization_config'] = quantization_config
        # 4-bit 量化时，device_map 会自动处理
        if device.type == 'cuda':
            model_kwargs['device_map'] = {"": device}
    else:
        # 非量化模式
        if device.type == 'cuda':
            # 使用明确的设备映射，确保所有层都在 GPU 上
            model_kwargs['device_map'] = {"": device}
        else:
            # CPU 模式：不使用 device_map，然后手动移动到 CPU
            model_kwargs['device_map'] = None
    
    print(f"[Activation Projection] 模型加载参数: {', '.join([f'{k}={v}' for k, v in model_kwargs.items() if k != 'device_map' or v is not None])}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs
        )
        
        # CPU 模式下，如果没有使用 device_map，需要手动移动
        if device.type == 'cpu' and not use_quantization and model_kwargs.get('device_map') is None:
            model = model.to(device)
    except torch.cuda.OutOfMemoryError as e:
        print("\n" + "="*60)
        print("错误: GPU 内存不足！")
        print("="*60)
        print("\n建议的解决方案：")
        print("1. 使用 8-bit 量化: 添加 --load-in-8bit 参数")
        print("2. 使用 4-bit 量化: 添加 --load-in-4bit 参数（更节省内存）")
        print("3. 清理 GPU 缓存: 添加 --clear-cache 参数")
        print("4. 减小批大小: 使用 --batch-size 4 或更小")
        print("5. 使用 CPU: 设置 --device cpu（会很慢）")
        print("\n示例命令：")
        print("  python scripts/run_activation_projection.py \\")
        print("    --model-path <path> \\")
        print("    --toxic-vectors-path <path> \\")
        print("    --dataset-path <path> \\")
        print("    --output-path <path> \\")
        print("    --load-in-4bit \\")
        print("    --clear-cache \\")
        print("    --batch-size 4")
        print("="*60)
        raise
    
    # 确保模型处于评估模式
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[Activation Projection] 模型加载完成")
    
    # 显示 GPU 内存使用情况
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - reserved
        print(f"[Activation Projection] GPU 内存使用: {allocated:.2f} GB 已分配 / {reserved:.2f} GB 已保留 / {total:.2f} GB 总计")
        print(f"[Activation Projection] GPU 可用内存: {free:.2f} GB")
        
        # 如果可用内存不足，给出警告和建议
        if free < 2.0:  # 小于 2GB
            print()
            print("⚠️  警告: GPU 可用内存不足！")
            print("建议:")
            if args.batch_size > 4:
                print(f"  - 减小批大小: 当前 {args.batch_size}，建议使用 --batch-size 4 或更小")
            if args.max_length > 1024:
                print(f"  - 减小序列长度: 当前 {args.max_length}，建议使用 --max-length 1024 或更小")
            if not args.load_in_4bit and not args.load_in_8bit:
                print("  - 使用量化: 添加 --load-in-4bit 或 --load-in-8bit 参数")
            print("  - 清理缓存: 添加 --clear-cache 参数")
            print()
    
    print()
    
    # 加载数据集
    print("[Activation Projection] 加载数据集...")
    dataset = JailbreakDataset(args.dataset_path)
    print(f"[Activation Projection] 数据集大小: {len(dataset)}")
    print()
    
    # 加载目标神经元（如果提供）
    target_neurons = None
    if args.target_neurons_path:
        target_neurons = load_target_neurons(args.target_neurons_path)
        if target_neurons is None:
            print("[Activation Projection] 警告: 无法加载目标神经元，将分析所有神经元")
            print("注意: 分析所有神经元可能需要很长时间和大量内存")
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                return
        print()
    
    # 计算激活投影
    print("=" * 50)
    activation_projection = compute_activation_projection(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        toxic_vectors_path=args.toxic_vectors_path,
        target_neurons=target_neurons,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_samples=args.num_samples,
    )
    print("=" * 50)
    print()
    
    # 保存结果
    output_path = Path(args.output_path)
    output_file = save_activation_projection(
        activation_projection=activation_projection,
        output_path=output_path,
        filename=args.output_filename,
    )
    
    # 显示统计信息
    if activation_projection:
        print()
        print("统计信息:")
        print(f"  分析的神经元数量: {len(activation_projection)}")
        
        # 统计成功和失败样本数量
        successful_counts = [v['successful_count'] for v in activation_projection.values() if v['successful_count'] > 0]
        failed_counts = [v['failed_count'] for v in activation_projection.values() if v['failed_count'] > 0]
        
        if successful_counts:
            print(f"  成功样本数量: {min(successful_counts)} - {max(successful_counts)} (范围)")
        if failed_counts:
            print(f"  失败样本数量: {min(failed_counts)} - {max(failed_counts)} (范围)")
        
        # 统计激活投影值
        activation_projs = [v['activation_projection'] for v in activation_projection.values()]
        if activation_projs:
            sorted_projs = sorted(activation_projs)
            print(f"  激活投影值统计:")
            print(f"    最小值: {min(activation_projs):.4f}")
            print(f"    最大值: {max(activation_projs):.4f}")
            print(f"    均值: {sum(activation_projs)/len(activation_projs):.4f}")
            # 计算中位数（正确处理奇数和偶数长度）
            if len(sorted_projs) % 2 == 0:
                median = (sorted_projs[len(sorted_projs)//2 - 1] + sorted_projs[len(sorted_projs)//2]) / 2
            else:
                median = sorted_projs[len(sorted_projs)//2]
            print(f"    中位数: {median:.4f}")
        else:
            print("  激活投影值统计: 无数据")
        
        # 统计成功和失败的平均差异
        activation_diffs = [v['activation_diff'] for v in activation_projection.values() if v['activation_diff'] != 0.0]
        if activation_diffs:
            print(f"  成功-失败差异统计:")
            print(f"    最小值: {min(activation_diffs):.4f}")
            print(f"    最大值: {max(activation_diffs):.4f}")
            print(f"    均值: {sum(activation_diffs)/len(activation_diffs):.4f}")
        
        # 显示前10个最高和最低的激活投影值
        sorted_by_proj = sorted(activation_projection.items(), key=lambda x: x[1]['activation_projection'], reverse=True)
        if sorted_by_proj:
            print()
            print("  前10个最高激活投影值:")
            for (layer_idx, neuron_idx), data in sorted_by_proj[:10]:
                print(f"    Layer {layer_idx}, Neuron {neuron_idx}: {data['activation_projection']:.4f} "
                      f"(成功: {data['successful_mean']:.4f}, 失败: {data['failed_mean']:.4f})")
            
            print()
            print("  前10个最低激活投影值:")
            for (layer_idx, neuron_idx), data in sorted_by_proj[-10:]:
                print(f"    Layer {layer_idx}, Neuron {neuron_idx}: {data['activation_projection']:.4f} "
                      f"(成功: {data['successful_mean']:.4f}, 失败: {data['failed_mean']:.4f})")
    
    print()
    print("=" * 50)
    print("激活投影分析完成！")
    print(f"结果已保存到: {output_file}")
    print("=" * 50)


if __name__ == '__main__':
    main()
