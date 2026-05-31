#!/usr/bin/env python3
r"""
参数对齐分析运行脚本

根据论文5.4节要求，计算神经元参数与毒性向量的余弦相似度（S_i^k）。

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/run_parameter_alignment.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment

    python /workspace/scripts/run_parameter_alignment.py \
        --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment

可选：指定目标神经元（只分析这些神经元）：
    python /workspace/scripts/run_parameter_alignment.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json

从宿主机运行（通过 docker exec）：
    docker exec -it neurobreak-container python /workspace/scripts/run_parameter_alignment.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment

Windows 环境使用（本地运行）：
    python scripts/run_parameter_alignment.py ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --toxic-vectors-path outputs/toxic_vectors/toxic_vectors.npz ^
        --output-path outputs/parameter_alignment

内存优化选项（如果遇到 GPU 内存不足）：
    # 使用 4-bit 量化（最节省内存，推荐）
    python /workspace/scripts/run_parameter_alignment.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment \
        --load-in-4bit \
        --clear-cache

    # 使用 4-bit 量化 + 指定目标神经元
    python /workspace/scripts/run_parameter_alignment.py \
        --model-path /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment \
        --target-neurons-path /workspace/outputs/dedicated_safety_neurons.json \
        --load-in-4bit \
        --clear-cache

    # 使用 8-bit 量化（比 4-bit 占用更多内存，但精度更高）
    python /workspace/scripts/run_parameter_alignment.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --output-path /workspace/outputs/parameter_alignment \
        --load-in-8bit \
        --clear-cache

注意：
- 如果不指定 --target-neurons-path，将分析所有层的所有神经元
- 目标神经元文件格式应为 JSON，包含神经元信息（支持 safety_neurons、utility_neurons 或 all_neurons 字段）
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

# 添加工作目录到路径
# 支持两种方式：1) 从脚本位置推断项目根目录 2) 使用 /workspace（Docker 环境）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

from engine.neurons.parameter_alignment import compute_parameter_alignment, save_parameter_alignment


def _log_to_guard_label(
    script_name: str,
    status: str,
    message: str,
    details: dict = None,
) -> None:
    """向 logs/guard_label.log 追加一条结构化运行记录（JSONL 格式）。"""
    import datetime
    import json as _json

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "guard_label.log"

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": script_name,
        "status": status,
        "message": message,
    }
    if details:
        entry["details"] = details

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
        print(f"[Parameter Alignment] 警告: 目标神经元文件不存在: {neurons_file}")
        return None
    
    print(f"[Parameter Alignment] 加载目标神经元: {neurons_file}")
    
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
            print("[Parameter Alignment] 检测到直接神经元格式（layer_X_neuron_Y）")
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
            print(f"[Parameter Alignment] 从直接格式加载了 {len(target_neurons)} 个神经元")
            return target_neurons
        else:
            print(f"[Parameter Alignment] 警告: 无法识别神经元格式，将分析所有神经元")
            return None
    
    neurons_data = data[neurons_key]
    print(f"[Parameter Alignment] 从 '{neurons_key}' 字段加载神经元...")
    
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
    
    print(f"[Parameter Alignment] 成功加载 {len(target_neurons)} 个目标神经元")
    return target_neurons


def main():
    parser = argparse.ArgumentParser(
        description='参数对齐分析 - 计算神经元参数与毒性向量的余弦相似度',
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
        default='parameter_alignment.json',
        help='输出文件名（默认: parameter_alignment.json）'
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

    _log_to_guard_label(
        "run_parameter_alignment",
        "START",
        f"参数对齐分析启动",
        details={
            "model_path": args.model_path,
            "toxic_vectors_path": args.toxic_vectors_path,
            "target_neurons_path": args.target_neurons_path,
        },
    )

    # 打印配置信息
    print("========================================")
    print("参数对齐分析 - Parameter Alignment")
    print("========================================")
    print()
    print("参数配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  毒性向量路径: {args.toxic_vectors_path}")
    print(f"  输出路径: {args.output_path}")
    if args.target_neurons_path:
        print(f"  目标神经元路径: {args.target_neurons_path}")
    else:
        print("  目标神经元: 未指定（将分析所有神经元）")
    print(f"  输出文件名: {args.output_filename}")
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
    # 注意：模型路径可以是本地路径或 Hugging Face Hub 模型名称
    # from_pretrained 会自动处理这两种情况
    if os.path.exists(args.model_path):
        print(f"[Parameter Alignment] 使用本地模型路径: {args.model_path}")
    else:
        print(f"[Parameter Alignment] 模型路径不存在，将尝试从 Hugging Face Hub 加载: {args.model_path}")
    
    if not os.path.exists(args.toxic_vectors_path):
        raise FileNotFoundError(f"毒性向量文件不存在: {args.toxic_vectors_path}")
    
    # 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[Parameter Alignment] 设备: {device}")
    
    # 清理 GPU 缓存（如果请求）
    if args.clear_cache and device.type == 'cuda':
        print("[Parameter Alignment] 清理 GPU 缓存...")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"[Parameter Alignment] GPU 内存使用: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB / {torch.cuda.get_device_properties(device).total_memory/1024**3:.2f} GB")
    
    # 检查量化选项
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("不能同时使用 8-bit 和 4-bit 量化")
    
    use_quantization = args.load_in_8bit or args.load_in_4bit
    if use_quantization:
        try:
            import bitsandbytes  # noqa: F401
            print(f"[Parameter Alignment] 使用 {'8-bit' if args.load_in_8bit else '4-bit'} 量化")
        except ImportError:
            raise ImportError("使用量化需要安装 bitsandbytes: pip install bitsandbytes")
    
    # 确定数据类型
    if use_quantization:
        # 量化模式下，torch_dtype 会被忽略，但我们需要设置它
        torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        print(f"[Parameter Alignment] 量化模式，基础数据类型: {torch_dtype}")
    elif args.torch_dtype == 'auto':
        if device.type == 'cuda':
            torch_dtype = torch.float16
            print("[Parameter Alignment] 检测到 CUDA，使用 FP16 加载模型")
        else:
            torch_dtype = torch.float32
            print("[Parameter Alignment] 使用 CPU，使用 FP32 加载模型")
    elif args.torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # 加载模型
    print("[Parameter Alignment] 加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # 准备模型加载参数
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'low_cpu_mem_usage': args.low_cpu_mem_usage,
    }
    
    # 量化：使用 quantization_config，避免 load_in_4bit 等 kwargs 泄漏到 LlamaForCausalLM.__init__
    # （部分 transformers / 环境下直接传 load_in_4bit 会触发 TypeError）
    if args.load_in_8bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        if device.type == 'cuda':
            model_kwargs['device_map'] = {"": device}
    elif args.load_in_4bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if device.type == 'cuda':
            model_kwargs['device_map'] = {"": device}
    else:
        # 非量化模式
        if device.type == 'cuda':
            # 使用明确的设备映射，确保所有层都在 GPU 上
            # 这避免了 device_map='auto' 可能导致的延迟加载问题
            model_kwargs['device_map'] = {"": device}
        else:
            # CPU 模式：不使用 device_map，然后手动移动到 CPU
            model_kwargs['device_map'] = None
    
    # 注意：使用 device_map='auto' 可能导致某些层被标记为 meta tensor 或延迟加载
    # 这对于需要访问所有层权重的参数对齐分析来说是有问题的
    # 因此，我们使用明确的设备映射来确保所有层都被正确加载
    
    print(f"[Parameter Alignment] 模型加载参数: {', '.join([f'{k}={v}' for k, v in model_kwargs.items() if k != 'device_map' or v is not None])}")
    
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
        print("4. 使用 CPU: 设置 --device cpu（会很慢）")
        print("\n示例命令：")
        print("  python scripts/run_parameter_alignment.py \\")
        print("    --model-path <path> \\")
        print("    --toxic-vectors-path <path> \\")
        print("    --output-path <path> \\")
        print("    --load-in-4bit \\")
        print("    --clear-cache")
        print("="*60)
        raise
    
    # 确保模型处于评估模式
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[Parameter Alignment] 模型加载完成")
    
    # 显示 GPU 内存使用情况
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"[Parameter Alignment] GPU 内存使用: {allocated:.2f} GB 已分配 / {reserved:.2f} GB 已保留 / {total:.2f} GB 总计")
    
    print()
    
    # 加载目标神经元（如果提供）
    target_neurons = None
    if args.target_neurons_path:
        target_neurons = load_target_neurons(args.target_neurons_path)
        if target_neurons is None:
            print("[Parameter Alignment] 将分析所有神经元")
        print()
    
    # 计算参数对齐
    print("=" * 50)
    parameter_alignment = compute_parameter_alignment(
        model=model,
        toxic_vectors=args.toxic_vectors_path,
        target_neurons=target_neurons,
    )
    print("=" * 50)
    print()
    
    # 保存结果
    output_path = Path(args.output_path)
    output_file = save_parameter_alignment(
        parameter_alignment=parameter_alignment,
        output_path=output_path,
        filename=args.output_filename,
    )
    
    # 显示统计信息
    if parameter_alignment:
        print()
        print("统计信息:")
        print(f"  分析的神经元数量: {len(parameter_alignment)}")
        
        s_plus = sum(1 for v in parameter_alignment.values() if v['alignment_type'] == 'S+')
        s_minus = len(parameter_alignment) - s_plus
        print(f"  S+ (正对齐): {s_plus} ({s_plus/len(parameter_alignment)*100:.2f}%)")
        print(f"  S- (负对齐): {s_minus} ({s_minus/len(parameter_alignment)*100:.2f}%)")
        
        cosine_sims = [v['cosine_similarity'] for v in parameter_alignment.values()]
        print(f"  余弦相似度统计:")
        print(f"    最小值: {min(cosine_sims):.4f}")
        print(f"    最大值: {max(cosine_sims):.4f}")
        print(f"    均值: {sum(cosine_sims)/len(cosine_sims):.4f}")
        print(f"    中位数: {sorted(cosine_sims)[len(cosine_sims)//2]:.4f}")
        
        # 显示前10个最高和最低的余弦相似度
        sorted_by_sim = sorted(parameter_alignment.items(), key=lambda x: x[1]['cosine_similarity'], reverse=True)
        print()
        print("  前10个最高余弦相似度 (S+):")
        for (layer_idx, neuron_idx), data in sorted_by_sim[:10]:
            print(f"    Layer {layer_idx}, Neuron {neuron_idx}: {data['cosine_similarity']:.4f} ({data['alignment_type']})")
        
        print()
        print("  前10个最低余弦相似度 (S-):")
        for (layer_idx, neuron_idx), data in sorted_by_sim[-10:]:
            print(f"    Layer {layer_idx}, Neuron {neuron_idx}: {data['cosine_similarity']:.4f} ({data['alignment_type']})")
    
    print()
    print("=" * 50)
    print("参数对齐分析完成！")
    print(f"结果已保存到: {output_file}")
    print("=" * 50)

    _log_to_guard_label(
        "run_parameter_alignment",
        "DONE",
        f"参数对齐分析完成 — 神经元数={len(parameter_alignment)}, 输出={output_file}",
        details={
            "num_neurons": len(parameter_alignment),
            "output_path": str(output_file),
        },
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        _log_to_guard_label(
            "run_parameter_alignment",
            "ERROR",
            f"运行异常: {e}",
            details={"exception": str(e)},
        )
        raise
