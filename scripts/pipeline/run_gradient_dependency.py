#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
梯度依赖关系分析运行脚本

根据论文5.4节要求，分析神经元之间的梯度依赖关系（G_{i,j}）。

核心功能：
- 使用W_down神经元作为锚点，追踪来自前一层模块的参数级影响
- 通过测量参数扰动如何传播到W_down激活来量化上游神经元与安全机制的因果关系
- 计算梯度关联：G_{i,j} = ∂a^k_down,i / ∂w^k_upstream,j
  - a^k_down,i: 第k层down_proj的第i个神经元的激活值
  - w^k_upstream,j: 上游（前一层，即k-1层）第j个神经元的权重参数（down_proj权重）

Docker 容器内 bash 运行示例
    python /workspace/scripts/run_gradient_dependency.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_dependency \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json

容器内使用 docker exec 运行示例
    docker exec -it neurobreak-container python /workspace/scripts/run_gradient_dependency.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_dependency \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json

Windows 命令行运行示例
    python scripts/run_gradient_dependency.py ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --dataset-path logs/base_evaluation.jsonl ^
        --output-path outputs/gradient_dependency ^
        --target-neurons-path outputs/snip_scores/safety_neurons.json

低显存 GPU 推荐配置
    python /workspace/scripts/run_gradient_dependency.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_dependency \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json \
        --batch-size 2 \
        --num-samples 100 \
        --max-length 512

标准配置示例
    python /workspace/scripts/run_gradient_dependency.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --output-path /workspace/outputs/gradient_dependency \
        --target-neurons-path /workspace/outputs/snip_scores/safety_neurons.json \
        --num-samples 200 \
        --max-length 512 \
        --batch-size 4

注意事项
- 目标神经元 JSON 文件应包含 safety_neurons、utility_neurons、all_neurons 或 dedicated_safety_neurons 字段
- 数据集必须是 JSONL 格式，支持 text、prompt 或 input.prompt 字段
- 低显存环境建议使用 Docker 容器
- 如遇 OOM 问题：减小 batch_size 或 num_samples
- 使用 --clear-cache 清理 GPU 缓存
- 批处理大小和样本数可根据 GPU 显存调整
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

# 项目根目录设置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.exists() and (PROJECT_ROOT / 'engine').exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    workspace_path = os.getenv('WORKSPACE_PATH', '/workspace')
    if os.path.exists(workspace_path):
        sys.path.insert(0, workspace_path)
    else:
        cwd = Path.cwd()
        if (cwd / 'engine').exists():
            sys.path.insert(0, str(cwd))
        else:
            sys.path.insert(0, '/workspace')

from engine.neurons.gradient_dependency import compute_gradient_dependency, visualize_gradient_dependency


def _log_to_guard_label(
    script_name: str,
    status: str,
    message: str,
    details: dict = None,
) -> None:
    """写入 logs/guard_label.log 日志文件（JSONL 格式）"""
    import datetime
    import json as _json

    log_dir = Path(__file__).resolve().parent.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'guard_label.log'

    entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'script': script_name,
        'status': status,
        'message': message,
    }
    if details:
        entry['details'] = details

    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        pass


class TextDataset(Dataset):
    """文本数据集加载器，支持 JSONL 格式"""

    def __init__(self, file_path: str):
        self.samples = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'文件不存在: {file_path}')

        if file_path.endswith('.jsonl'):
            print('[Gradient Dependency] 加载 JSONL 数据集...')
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line.strip())
                        self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f'[Gradient Dependency] 警告: 第{line_num}行JSON解析错误: {e}')
                        continue
                    except Exception as e:
                        print(f'[Gradient Dependency] 警告: 第{line_num}行读取错误: {e}')
                        continue
        else:
            raise ValueError(f'不支持的文件格式: {file_path}，仅支持 .jsonl 格式')

        print(f'[Gradient Dependency] 加载完成，共 {len(self.samples)} 条样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_target_neurons(neurons_file: str) -> Optional[Dict[Tuple[int, int], Dict]]:
    """
    从 JSON 文件加载目标神经元配置

    支持的 JSON 格式：
    1. 嵌套结构：包含 safety_neurons, utility_neurons, all_neurons, dedicated_safety_neurons 字段
       - 典型文件 dedicated_safety_neurons.json
       - 键名格式为 layer_X_neuron_Y 或 X_Y（如 31_4062）
       - 值格式为 {layer, neuron, ...} 或 {layer_idx, neuron_idx, ...}

    2. 扁平结构：键为 layer_X_neuron_Y 格式
       - 典型文件 activation_projection.json, quadrant_classification.json
       - 值格式为 {layer_idx, neuron_idx, ...} 或 {layer, neuron, ...}

    Args:
        neurons_file: JSON 文件路径

    Returns:
        目标神经元字典 Dict[(layer_idx, neuron_idx), Dict]
        加载失败时返回 None
    """
    if not os.path.exists(neurons_file):
        print(f'[Gradient Dependency] 错误: 目标神经元文件不存在: {neurons_file}')
        return None

    print(f'[Gradient Dependency] 加载目标神经元: {neurons_file}')

    with open(neurons_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 查找嵌套结构的神经元数据
    neurons_key = None
    for key in ['safety_neurons', 'utility_neurons', 'all_neurons', 'dedicated_safety_neurons']:
        if key in data:
            neurons_key = key
            print(f'[Gradient Dependency] 检测到神经元类别: {key}')
            break

    if neurons_key is None:
        # 尝试解析扁平结构（layer_X_neuron_Y 格式）
        if any(k.startswith('layer_') and '_neuron_' in k for k in data.keys()):
            print('[Gradient Dependency] 使用扁平结构解析 layer_X_neuron_Y 格式')
            target_neurons = {}
            for key, value in data.items():
                # 优先使用 layer_idx/neuron_idx，其次使用 layer/neuron
                if 'layer_idx' in value and 'neuron_idx' in value:
                    layer_idx = int(value['layer_idx'])
                    neuron_idx = int(value['neuron_idx'])
                    target_neurons[(layer_idx, neuron_idx)] = value
                elif 'layer' in value and 'neuron' in value:
                    layer_idx = int(value['layer'])
                    neuron_idx = int(value['neuron'])
                    target_neurons[(layer_idx, neuron_idx)] = value
                elif key.startswith('layer_') and '_neuron_' in key:
                    try:
                        parts = key.split('_')
                        if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'neuron':
                            layer_idx = int(parts[1])
                            neuron_idx = int(parts[3])
                            target_neurons[(layer_idx, neuron_idx)] = value
                    except (ValueError, IndexError):
                        continue
            print(f'[Gradient Dependency] 解析完成，共 {len(target_neurons)} 个目标神经元')
            return target_neurons
        else:
            print('[Gradient Dependency] 错误: 无法识别的神经元数据格式')
            print('[Gradient Dependency] 支持格式：')
            print('  - 嵌套结构：包含 dedicated_safety_neurons 等字段')
            print('  - 扁平结构：键为 layer_X_neuron_Y 格式')
            return None

    neurons_data = data[neurons_key]

    # 解析神经元位置信息
    target_neurons = {}
    for key, value in neurons_data.items():
        # 支持的格式：
        # 1. layer_X_neuron_Y: {layer_idx, neuron_idx, ...} 或 {layer, neuron, ...}
        # 2. X_Y: {layer, neuron, ...} （简化格式）
        # 3. 值的字段优先使用 layer_idx/neuron_idx

        if 'layer_idx' in value and 'neuron_idx' in value:
            layer_idx = int(value['layer_idx'])
            neuron_idx = int(value['neuron_idx'])
            target_neurons[(layer_idx, neuron_idx)] = value
        elif 'layer' in value and 'neuron' in value:
            layer_idx = int(value['layer'])
            neuron_idx = int(value['neuron'])
            target_neurons[(layer_idx, neuron_idx)] = value
        elif '_' in key:
            try:
                parts = key.split('_')
                # 格式1: layer_X_neuron_Y
                if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'neuron':
                    layer_idx = int(parts[1])
                    neuron_idx = int(parts[3])
                    target_neurons[(layer_idx, neuron_idx)] = value
                # 格式2: X_Y （简化格式）
                elif len(parts) == 2:
                    layer_idx = int(parts[0])
                    neuron_idx = int(parts[1])
                    target_neurons[(layer_idx, neuron_idx)] = value
            except (ValueError, IndexError):
                continue

    print(f'[Gradient Dependency] 加载完成，共 {len(target_neurons)} 个目标神经元')
    return target_neurons


def save_gradient_dependency(
    gradient_dependency: Dict[Tuple[int, int], Dict],
    output_path: Path,
    filename: str = 'gradient_dependency.json',
):
    """
    保存梯度依赖关系结果到 JSON 文件

    Args:
        gradient_dependency: 梯度依赖关系结果
        output_path: 输出目录路径
        filename: 输出文件名
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename

    json_data = {}
    for (layer_idx, neuron_idx), data in gradient_dependency.items():
        key = f'layer_{layer_idx}_neuron_{neuron_idx}'
        json_data[key] = {
            'layer_idx': layer_idx,
            'neuron_idx': neuron_idx,
            'upstream_neurons': [
                {'layer_idx': l, 'neuron_idx': n}
                for l, n in data['upstream_neurons']
            ],
            'gradient_strengths': data['gradient_strengths'],
            'mean_gradient_strength': data.get('mean_gradient_strength', 0.0),
            'max_gradient_strength': data.get('max_gradient_strength', 0.0),
            'num_upstream_neurons': len(data['upstream_neurons']),
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f'[Gradient Dependency] 结果已保存: {output_file}')
    print(f'[Gradient Dependency] 共 {len(json_data)} 个目标神经元及其梯度依赖信息')


def main():
    parser = argparse.ArgumentParser(
        description='梯度依赖关系分析：计算神经元间的梯度依赖 G_{i,j}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='模型路径，HuggingFace 模型 ID 或本地路径'
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='数据集路径，JSONL 格式'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='输出目录路径'
    )

    parser.add_argument(
        '--target-neurons-path',
        type=str,
        required=True,
        help='目标神经元配置文件路径，JSON 格式'
    )

    parser.add_argument(
        '--top-k',
        type=float,
        default=0.1,
        help='保留前 k%% 的强关联（默认 0.1，即 10%%）'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='批处理大小（默认 4，根据显存调整）'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=1024,
        help='最大序列长度（默认 1024）'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='使用的样本数量（None 表示全部）'
    )

    parser.add_argument(
        '--use-last-token',
        action='store_true',
        default=True,
        help='使用最后一个 token 的激活值（默认开启）'
    )

    parser.add_argument(
        '--no-use-last-token',
        dest='use_last_token',
        action='store_false',
        help='使用所有 token 的平均激活值'
    )

    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='计算前清理 GPU 缓存'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="计算设备：cuda 或 cpu，默认为自动检测"
    )

    args = parser.parse_args()

    _log_to_guard_label(
        'run_gradient_dependency',
        'START',
        '梯度依赖分析开始',
        details={
            'model_path': args.model_path,
            'dataset_path': args.dataset_path,
            'target_neurons_path': args.target_neurons_path,
            'batch_size': args.batch_size,
            'num_samples': args.num_samples,
        },
    )

    # 设备选择
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f'[Gradient Dependency] 计算设备: {device}')

    # 清理 GPU 缓存
    if args.clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('[Gradient Dependency] 已清理 GPU 缓存')

    # 加载模型
    print(f'[Gradient Dependency] 加载模型: {args.model_path}')
    print('[Gradient Dependency] 提示：模型将自动分配到可用 GPU')
    print('[Gradient Dependency] 提示：如遇显存不足，请减小 batch_size 或 num_samples')
    print()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )
    except Exception as e:
        print(f'[Gradient Dependency] 模型加载失败: {e}')
        _log_to_guard_label(
            'run_gradient_dependency',
            'ERROR',
            f'模型加载失败: {e}',
            details={'exception': str(e)},
        )
        return

    # 检测量化模型
    has_quantized = False
    if hasattr(model, 'quantization_config') or hasattr(model, 'hf_quantizer'):
        has_quantized = True
        print('[Gradient Dependency] 检测：量化模型已加载')
        print('[Gradient Dependency] 提示：量化模型可能影响梯度计算精度')
        print()

    # 加载分词器
    print('[Gradient Dependency] 加载分词器...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    except Exception as e:
        print(f'[Gradient Dependency] 分词器加载失败: {e}')
        return

    # 加载数据集
    print(f'[Gradient Dependency] 加载数据集: {args.dataset_path}')
    try:
        dataset = TextDataset(args.dataset_path)
    except Exception as e:
        print(f'[Gradient Dependency] 数据集加载失败: {e}')
        return

    # 加载目标神经元
    target_neurons = load_target_neurons(args.target_neurons_path)
    if target_neurons is None or len(target_neurons) == 0:
        print('[Gradient Dependency] 错误: 目标神经元为空或加载失败')
        return

    # 显示分析配置
    print('[Gradient Dependency] 分析配置:')
    print(f'[Gradient Dependency]   - top_k: {args.top_k}')
    print(f'[Gradient Dependency]   - batch_size: {args.batch_size}')
    print(f'[Gradient Dependency]   - max_length: {args.max_length}')
    print(f'[Gradient Dependency]   - num_samples: {args.num_samples}')
    print(f'[Gradient Dependency]   - use_last_token: {args.use_last_token}')
    print()

    # 量化模型提示
    if has_quantized:
        print('[Gradient Dependency] 注意：量化模型可能需要更大的 top_k 值以获得足够的梯度信号')
        print()

    # 执行梯度依赖分析
    try:
        gradient_dependency = compute_gradient_dependency(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            target_neurons=target_neurons,
            device=device,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_samples=args.num_samples,
            use_last_token=args.use_last_token,
        )
    except KeyboardInterrupt:
        print('\n[Gradient Dependency] 用户中断')
        return
    except Exception as e:
        print(f'\n[Gradient Dependency] 梯度依赖计算出错: {e}')
        print('[Gradient Dependency] 常见问题排查：')
        print('  1. 检查模型是否支持梯度计算')
        print('  2. 尝试减小 batch_size 或 num_samples')
        print('  3. 检查 GPU 显存是否充足')
        import traceback
        traceback.print_exc()
        return

    # 检查结果
    if gradient_dependency is None or len(gradient_dependency) == 0:
        print('\n[Gradient Dependency] 错误：梯度依赖结果为空')
        print('[Gradient Dependency] 可能原因：')
        print('  1. 目标神经元配置不正确')
        print('  2. 数据集为空或格式错误')
        print('  3. 模型前向传播失败')
        return

    # 保存结果
    output_path = Path(args.output_path)
    save_gradient_dependency(gradient_dependency, output_path)

    # 生成可视化数据
    visualization_path = output_path / 'gradient_dependency_visualization.json'
    visualize_gradient_dependency(gradient_dependency, str(visualization_path))

    # 统计结果
    total_upstream = sum(len(data.get('upstream_neurons', [])) for data in gradient_dependency.values())
    neurons_with_deps = sum(1 for data in gradient_dependency.values() if len(data.get('upstream_neurons', [])) > 0)
    print(f'[Gradient Dependency] 成功计算 {len(gradient_dependency)} 个目标神经元的梯度依赖关系')
    print(f'[Gradient Dependency] 共发现 {total_upstream} 个上游神经元关联')
    print(f'[Gradient Dependency] {neurons_with_deps} 个神经元有上游依赖关系')
    print('[Gradient Dependency] 完成！')

    _log_to_guard_label(
        'run_gradient_dependency',
        'DONE',
        f'梯度依赖分析完成 -- 神经元数={len(gradient_dependency)}, 上游关联={total_upstream}',
        details={
            'num_neurons': len(gradient_dependency),
            'num_upstream_associations': total_upstream,
        },
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        _log_to_guard_label(
            'run_gradient_dependency',
            'ERROR',
            f'运行错误: {e}',
            details={'exception': str(e)},
        )
        raise
