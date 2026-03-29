#!/usr/bin/env python3
"""
SNIP Scorer 脚本 - 在 Docker 容器中运行神经元分数计算

使用方法（在 Docker 容器内的 bash 中）：
    python /workspace/scripts/run_snip_scorer.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --mode safety \
        --filter-benign \
        --num-samples 100

或者使用简化的命令（对应原始 PowerShell 脚本的参数）：
    python /workspace/scripts/run_snip_scorer.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --mode safety \
        --filter-benign \
        --num-samples 100

从宿主机运行（通过 docker exec）：
    docker exec -it neurobreak-container python /workspace/scripts/run_snip_scorer.py \
        --model-path /cache/Meta-Llama-3-8B-Instruct \
        --dataset-path /workspace/logs/base_evaluation.jsonl \
        --toxic-vectors-path /workspace/outputs/toxic_vectors/toxic_vectors.npz \
        --mode safety \
        --filter-benign \
        --num-samples 100

注意：
- DatasetPath 支持两种格式：
  1. JSONL 文件：从每行的 JSON 对象中提取 "input.prompt" 字段作为文本
  2. 文本文件：每行一个文本样本
- 对于 JSONL 文件，可以使用 --filter-benign 参数只使用安全样本（asr_label=0）
- ToxicVectorsPath 是可选的，如果提供，将用于神经元功能分析（参数对齐和激活投影）
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

# 添加工作目录到路径
sys.path.insert(0, '/workspace')

from engine.neurons.safety_identifier import identify_safety_neurons
from engine.neurons.utility_identifier import identify_utility_neurons


class TextDataset(Dataset):
    """文本数据集，支持 JSONL 和纯文本格式"""
    
    def __init__(self, file_path, filter_benign=False):
        self.texts = []
        
        # 检查文件格式
        if file_path.endswith('.jsonl'):
            # JSONL 格式：从每行 JSON 中提取 prompt
            print(f'[SNIP] 检测到 JSONL 格式，从 input.prompt 提取文本...')
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line.strip())
                        
                        # 提取 prompt
                        if 'input' in sample and 'prompt' in sample['input']:
                            prompt = sample['input']['prompt']
                        elif 'prompt' in sample:
                            prompt = sample['prompt']
                        else:
                            continue
                        
                        if not prompt or not prompt.strip():
                            continue
                        
                        # 如果启用过滤，只使用安全样本
                        if filter_benign:
                            asr_label = None
                            if 'guard' in sample:
                                guard = sample['guard']
                                if 'asr_label' in guard:
                                    asr_label = guard['asr_label']
                                elif 'jailbreak_success' in guard:
                                    # jailbreak_success=True 表示有害，False 表示安全
                                    asr_label = 0 if not guard['jailbreak_success'] else 1
                            
                            # 只保留安全样本 (asr_label=0)
                            if asr_label is not None and asr_label != 0:
                                continue
                        
                        self.texts.append(prompt.strip())
                    except json.JSONDecodeError as e:
                        print(f'[SNIP] 警告: 第 {line_num} 行 JSON 解析失败: {e}')
                        continue
                    except Exception as e:
                        print(f'[SNIP] 警告: 第 {line_num} 行处理失败: {e}')
                        continue
        else:
            # 文本格式：每行一个文本样本
            print(f'[SNIP] 检测到文本格式，每行一个样本...')
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.texts.append(line.strip())
        
        print(f'[SNIP] 成功加载 {len(self.texts)} 个文本样本')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {'text': self.texts[idx]}


def compute_parameter_alignment(model, toxic_vectors_path, output_path, mode):
    """计算参数对齐（参数方向与毒性向量的相似度）"""
    print()
    print('[SNIP] 加载毒性向量进行神经元功能分析...')
    
    # 加载毒性向量
    toxic_data = np.load(toxic_vectors_path)
    vectors = toxic_data['vectors']  # (num_layers, hidden_dim)
    biases = toxic_data['biases']  # (num_layers,)
    toxic_layer_indices = toxic_data['layer_indices']  # (num_layers,)
    
    print(f'[SNIP] 加载了 {len(toxic_layer_indices)} 层的毒性向量')
    
    # 获取模型层结构
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        print('[SNIP] 警告: 无法获取模型层结构，跳过参数对齐分析')
        return None
    
    # 计算参数对齐（S_i^k）：每层 W_down 行向量与 w_toxic 的余弦相似度
    print('[SNIP] 计算参数对齐（参数方向与毒性向量的相似度）...')
    parameter_alignment = {}
    
    for layer_idx, layer in enumerate(layers):
        if layer_idx not in toxic_layer_indices:
            continue
        
        # 获取该层的毒性向量
        toxic_idx = np.where(toxic_layer_indices == layer_idx)[0]
        if len(toxic_idx) == 0:
            continue
        w_toxic = vectors[toxic_idx[0]]  # (hidden_dim,)
        
        # 获取 MLP down_proj 权重
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
            down_proj = layer.mlp.down_proj
        elif hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'down_proj'):
            down_proj = layer.feed_forward.down_proj
        else:
            continue
        
        if not hasattr(down_proj, 'weight') or down_proj.weight is None:
            continue
        
        weight = down_proj.weight.data.cpu().numpy()  # (out_features, in_features)
        
        # 计算每个神经元（每行）与毒性向量的余弦相似度
        neuron_alignments = []
        for neuron_idx in range(weight.shape[0]):
            neuron_weight = weight[neuron_idx, :]  # (in_features,)
            
            # 余弦相似度
            dot_product = np.dot(neuron_weight, w_toxic)
            norm_neuron = np.linalg.norm(neuron_weight)
            norm_toxic = np.linalg.norm(w_toxic)
            
            if norm_neuron > 0 and norm_toxic > 0:
                cosine_sim = dot_product / (norm_neuron * norm_toxic)
                neuron_alignments.append({
                    'neuron_idx': int(neuron_idx),
                    'cosine_similarity': float(cosine_sim),
                    'alignment_type': 'S+' if cosine_sim > 0 else 'S-'
                })
        
        if neuron_alignments:
            parameter_alignment[layer_idx] = neuron_alignments
    
    # 保存参数对齐结果
    alignment_file = output_path / f'{mode}_parameter_alignment.json'
    with open(alignment_file, 'w', encoding='utf-8') as f:
        json.dump(parameter_alignment, f, indent=2, ensure_ascii=False)
    
    print(f'[SNIP] 参数对齐结果保存到: {alignment_file}')
    print(f'[SNIP] 分析了 {len(parameter_alignment)} 层')
    
    return parameter_alignment


def main():
    parser = argparse.ArgumentParser(
        description='SNIP Scorer - 神经元分数计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.getenv('LLM_CONTAINER_PATH', '/workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct'),
        help='模型路径（容器内绝对路径，默认使用 LLM_CONTAINER_PATH 或 /workspace/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct）'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/workspace/logs/base_evaluation.jsonl',
        help='数据集路径（支持 JSONL 或文本文件）'
    )
    parser.add_argument(
        '--toxic-vectors-path',
        type=str,
        default=None,
        help='毒性向量路径（可选，用于神经元功能分析）'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/workspace/outputs/snip_scores',
        help='输出路径'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批大小'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=0,
        help='使用的样本数（0 表示使用全部样本）'
    )
    parser.add_argument(
        '--filter-benign',
        action='store_true',
        help='对于 JSONL 文件，只使用安全样本（asr_label=0）'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['safety', 'utility'],
        default='safety',
        help='模式：safety=安全神经元, utility=效用神经元'
    )
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("========================================")
    print("SNIP Scorer - 神经元分数计算")
    print("========================================")
    print()
    print("参数配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  数据集路径: {args.dataset_path}")
    if args.toxic_vectors_path:
        print(f"  毒性向量路径: {args.toxic_vectors_path}")
    else:
        print("  毒性向量路径: 未提供（跳过神经元功能分析）")
    print(f"  输出路径: {args.output_path}")
    print(f"  批大小: {args.batch_size}")
    if args.num_samples > 0:
        print(f"  样本数: {args.num_samples}")
    else:
        print("  样本数: 全部")
    print(f"  模式: {args.mode} (safety=安全神经元, utility=效用神经元)")
    if args.filter_benign:
        print("  过滤: 只使用安全样本 (asr_label=0)")
    print()
    
    # 配置
    model_path = args.model_path
    dataset_path = args.dataset_path
    toxic_vectors_path = args.toxic_vectors_path
    output_path = Path(args.output_path)
    batch_size = args.batch_size
    num_samples = args.num_samples if args.num_samples > 0 else None
    mode = args.mode
    
    print(f'[SNIP] 模型: {model_path}')
    print(f'[SNIP] 数据集: {dataset_path}')
    if toxic_vectors_path:
        print(f'[SNIP] 毒性向量: {toxic_vectors_path}')
    print(f'[SNIP] 模式: {mode}')
    print()
    
    # 加载模型（注意：SNIP 需要在浮点权重上计算梯度，这里强制使用 FP16/FP32，不使用 4bit 量化）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[SNIP] 设备: {device}')

    print('[SNIP] 加载模型和分词器（SNIP 阶段使用浮点模型，不走 4bit 量化）...')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if torch.cuda.is_available():
        # GPU 上使用 FP16，既能节省显存，又保持参数是可微的浮点类型
        print('[SNIP] 检测到 CUDA，使用 FP16 加载模型用于 SNIP 计算...')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
    else:
        # CPU 环境下使用 FP32，避免量化依赖
        print('[SNIP] 未检测到 CUDA，在 CPU 上以 FP32 加载模型用于 SNIP 计算...')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = TextDataset(dataset_path, filter_benign=args.filter_benign)
    print(f'[SNIP] 数据集大小: {len(dataset)}')
    
    # 计算 SNIP 分数
    if mode == 'safety':
        neurons = identify_safety_neurons(
            model=model,
            tokenizer=tokenizer,
            benign_dataset=dataset,
            device=device,
            safety_threshold_q=0.005,
            batch_size=batch_size,
            num_samples=num_samples,
        )
        output_file = output_path / 'safety_neurons.json'
    else:
        neurons = identify_utility_neurons(
            model=model,
            tokenizer=tokenizer,
            utility_dataset=dataset,
            device=device,
            utility_threshold_p=0.001,
            batch_size=batch_size,
            num_samples=num_samples,
        )
        output_file = output_path / 'utility_neurons.json'
    
    # 保存结果
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        # 将元组键转换为字符串以便 JSON 序列化
        neurons_str_keys = {
            f'{k[0]}_{k[1]}': v for k, v in neurons.items()
        }
        json.dump(neurons_str_keys, f, indent=2, ensure_ascii=False)
    
    print(f'[SNIP] 完成! 识别到 {len(neurons)} 个神经元')
    print(f'[SNIP] 结果保存到: {output_file}')
    
    # 显示前10个神经元的信息
    if len(neurons) > 0:
        print('\n[SNIP] Top 10 神经元:')
        sorted_neurons = sorted(neurons.items(), key=lambda x: x[1]['score'], reverse=True)[:10]
        for (layer, neuron), info in sorted_neurons:
            print(f'  Layer {layer}, Neuron {neuron}: score={info["score"]:.6f}, rank={info["rank"]}, percentile={info["percentile"]:.2f}%')
    
    # 如果提供了毒性向量，进行神经元功能分析
    if toxic_vectors_path and Path(toxic_vectors_path).exists():
        # 计算参数对齐
        parameter_alignment = compute_parameter_alignment(
            model=model,
            toxic_vectors_path=toxic_vectors_path,
            output_path=output_path,
            mode=mode
        )
        
        # 结合 SNIP 分数和参数对齐，生成综合分析
        if parameter_alignment:
            print('[SNIP] 生成综合分析（SNIP + 参数对齐）...')
            combined_analysis = {}
            for (layer_idx, neuron_idx), neuron_info in neurons.items():
                if layer_idx in parameter_alignment:
                    alignment_info = next(
                        (a for a in parameter_alignment[layer_idx] if a['neuron_idx'] == neuron_idx),
                        None
                    )
                    if alignment_info:
                        combined_analysis[(layer_idx, neuron_idx)] = {
                            **neuron_info,
                            'cosine_similarity': alignment_info['cosine_similarity'],
                            'alignment_type': alignment_info['alignment_type'],
                            'quadrant': f'{neuron_info.get("type", "SNIP")}{alignment_info["alignment_type"]}'
                        }
            
            if combined_analysis:
                combined_file = output_path / f'{mode}_combined_analysis.json'
                with open(combined_file, 'w', encoding='utf-8') as f:
                    # 将元组键转换为字符串以便 JSON 序列化
                    combined_analysis_str_keys = {
                        f'{k[0]}_{k[1]}': v for k, v in combined_analysis.items()
                    }
                    json.dump(combined_analysis_str_keys, f, indent=2, ensure_ascii=False)
                print(f'[SNIP] 综合分析结果保存到: {combined_file}')
                print(f'[SNIP] 共分析了 {len(combined_analysis)} 个神经元的综合特征')
    else:
        if toxic_vectors_path:
            print(f'[SNIP] 警告: 毒性向量文件不存在: {toxic_vectors_path}')
        else:
            print('[SNIP] 提示: 未提供毒性向量路径，跳过神经元功能分析')
    
    print()
    print("========================================")
    print("完成!")
    print(f"结果保存在: {output_path}")
    print("========================================")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[SNIP] 用户中断执行')
        sys.exit(1)
    except Exception as e:
        print(f'\n[SNIP] 错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)