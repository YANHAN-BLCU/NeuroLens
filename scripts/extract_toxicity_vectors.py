#!/usr/bin/env python3
"""
提取每层线性探针的毒性向量 (Toxicity Vector)

根据论文 5.2 节的方法：
  - 线性探针公式: P(y=toxic|h) = softmax(w_toxic · h + b)
  - w_toxic 即为毒性向量，代表引发有害内容的最佳方向
  - 沿此方向的攻击会以最小偏差将输出推入毒性区域
  - 每层的 w_toxic 定义为该层的毒性向量

用法:
    python extract_toxicity_vectors.py \
        --layers_dir outputs/linear_probes/layers \
        --output_dir outputs/toxicity_vectors \
        --num_layers 32

输出文件:
    outputs/toxicity_vectors/
    ├── all_layers_toxicity_vectors.json   # 所有层毒性向量 + 统计信息
    ├── toxicity_vectors.npy               # NumPy 格式，便于后续计算
    ├── layer_toxicity_summary.json         # 各层统计摘要
    └── analysis_report.txt                # 毒性向量分析报告
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm


# ========================= 配置 =========================

DEFAULT_LAYERS_DIR = r'outputs/linear_probes/layers'
DEFAULT_OUTPUT_DIR = r'outputs/toxicity_vectors'
HIDDEN_DIM = 4096


def parse_args():
    parser = argparse.ArgumentParser(description='提取每层线性探针的毒性向量')
    parser.add_argument('--layers_dir', type=str, default=DEFAULT_LAYERS_DIR,
                        help=f'线性探针模型目录 (默认: {DEFAULT_LAYERS_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--num_layers', type=int, default=32,
                        help='层数 (默认: 32)')
    parser.add_argument('--layer_prefix', type=str, default='layer',
                        help='层文件夹/文件名前缀 (默认: layer)')
    return parser.parse_args()


def load_toxicity_vector(model_path):
    """
    加载模型并提取毒性向量。
    优先使用预计算的 toxicity_vector，否则从权重计算。

    公式: w_toxic = w[1] - w[0]
        w[1]: 有害类 (toxic) 的权重
        w[0]: 安全类 (safe) 的权重
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'toxicity_vector' in checkpoint:
        tox_vec = checkpoint['toxicity_vector']
        if isinstance(tox_vec, torch.Tensor):
            tox_vec = tox_vec.detach().cpu().numpy()
        return tox_vec, checkpoint

    elif 'weight' in checkpoint:
        w = checkpoint['weight']
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        if len(w.shape) == 2 and w.shape[0] == 2:
            tox_vec = w[1] - w[0]
            return tox_vec, checkpoint
        else:
            raise ValueError(f"权重形状不符合预期: {w.shape}")

    else:
        raise KeyError(f"模型文件中缺少 toxicity_vector 或 weight 字段: {model_path}")


def compute_vector_stats(tox_vec):
    """计算毒性向量的统计信息"""
    return {
        'l2_norm': float(np.linalg.norm(tox_vec)),
        'l1_norm': float(np.linalg.norm(tox_vec, ord=1)),
        'max_abs': float(np.max(np.abs(tox_vec))),
        'mean': float(np.mean(tox_vec)),
        'std': float(np.std(tox_vec)),
        'positive_ratio': float(np.sum(tox_vec > 0) / len(tox_vec)),
        'min_value': float(np.min(tox_vec)),
        'max_value': float(np.max(tox_vec)),
    }


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    norm1 = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm1 == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm1)


def extract_all_layers(layers_dir, num_layers, layer_prefix='layer'):
    """遍历所有层文件夹，提取每个层的毒性向量"""
    results = {}
    missing_layers = []
    existing_layers = []

    print(f"\n{'='*60}")
    print(f"扫描目录: {layers_dir}")
    print(f"层数范围: 1 - {num_layers}")
    print(f"{'='*60}\n")

    for layer_idx in range(1, num_layers + 1):
        layer_name = f"{layer_prefix}{layer_idx:02d}"
        model_filename = f"{layer_name}.pt"
        model_path = os.path.join(layers_dir, layer_name, model_filename)

        if not os.path.exists(model_path):
            missing_layers.append(layer_idx)
            print(f"[{layer_name}] 模型文件不存在: {model_path}")
            continue

        try:
            tox_vec, checkpoint = load_toxicity_vector(model_path)
            stats = compute_vector_stats(tox_vec)

            results[layer_idx] = {
                'layer': layer_idx,
                'layer_name': layer_name,
                'model_path': model_path,
                'vector_shape': list(tox_vec.shape) if hasattr(tox_vec, 'shape') else [len(tox_vec)],
                'vector_l2_norm': stats['l2_norm'],
                'vector_statistics': stats,
            }

            if 'cv_avg_val_acc' in checkpoint:
                results[layer_idx]['cv_accuracy'] = checkpoint['cv_avg_val_acc']
            if 'cv_avg_val_s_acc' in checkpoint:
                results[layer_idx]['cv_safe_acc'] = checkpoint['cv_avg_val_s_acc']
            if 'cv_avg_val_h_acc' in checkpoint:
                results[layer_idx]['cv_harmful_acc'] = checkpoint['cv_avg_val_h_acc']

            existing_layers.append(layer_idx)
            print(f"[{layer_name}] ✓ 已提取 | L2范数: {stats['l2_norm']:.4f} | "
                  f"形状: {tox_vec.shape if hasattr(tox_vec, 'shape') else len(tox_vec)}")

        except Exception as e:
            print(f"[{layer_name}] ✗ 提取失败: {e}")
            missing_layers.append(layer_idx)

    return results, existing_layers, missing_layers


def compute_layer_relations(results, existing_layers):
    """计算各层毒性向量之间的关系"""
    relations = {
        'layer_similarities': {},
        'average_similarity': {},
        'most_similar_pairs': [],
    }

    if len(existing_layers) < 2:
        return relations

    tox_vectors = {l: results[l]['vector_statistics'] for l in existing_layers}

    print(f"\n计算层间相似度...")

    for i, layer_a in enumerate(existing_layers):
        layer_a_path = os.path.join(
            DEFAULT_LAYERS_DIR,
            f"{DEFAULT_LAYERS_DIR.split('/')[-1]}/{layer_a:02d}.pt" if DEFAULT_LAYERS_DIR else ""
        )

        for layer_b in existing_layers[i+1:]:
            model_a = os.path.join(
                DEFAULT_LAYERS_DIR,
                f"layer{layer_a:02d}/layer{layer_a:02d}.pt"
            )
            model_b = os.path.join(
                DEFAULT_LAYERS_DIR,
                f"layer{layer_b:02d}/layer{layer_b:02d}.pt"
            )

            if not os.path.exists(model_a) or not os.path.exists(model_b):
                continue

            try:
                tox_a, _ = load_toxicity_vector(model_a)
                tox_b, _ = load_toxicity_vector(model_b)
                sim = cosine_similarity(tox_a, tox_b)

                pair_key = f"L{layer_a:02d}-L{layer_b:02d}"
                relations['layer_similarities'][pair_key] = {
                    'layer_a': layer_a,
                    'layer_b': layer_b,
                    'cosine_similarity': sim,
                }

                relations['most_similar_pairs'].append({
                    'layer_a': layer_a,
                    'layer_b': layer_b,
                    'similarity': sim
                })

            except Exception as e:
                pass

    relations['most_similar_pairs'].sort(key=lambda x: x['similarity'], reverse=True)

    return relations


def save_results(results, existing_layers, missing_layers, output_dir, relations):
    """保存所有结果到文件"""

    os.makedirs(output_dir, exist_ok=True)

    tox_vectors_array = []
    layer_order = []

    for layer_idx in sorted(existing_layers):
        model_path = os.path.join(
            DEFAULT_LAYERS_DIR,
            f"layer{layer_idx:02d}/layer{layer_idx:02d}.pt"
        )
        if os.path.exists(model_path):
            tox_vec, _ = load_toxicity_vector(model_path)
            tox_vectors_array.append(tox_vec)
            layer_order.append(layer_idx)

    if tox_vectors_array:
        vectors_matrix = np.array(tox_vectors_array)
        np.save(os.path.join(output_dir, 'toxicity_vectors.npy'), vectors_matrix)
        print(f"\n[保存] toxicity_vectors.npy: 形状 {vectors_matrix.shape}")

    all_vectors_json = {
        'metadata': {
            'description': '每层线性探针的毒性向量 (Toxicity Vector)',
            'formula': 'w_toxic = w[1] - w[0]，其中 w[1] 为有害类权重，w[0] 为安全类权重',
            'paper_reference': 'Section 5.2 - 越狱探测',
            'total_layers': len(existing_layers),
            'missing_layers': missing_layers,
            'layer_order': layer_order,
        },
        'layers': results,
        'layer_relations': relations,
    }

    with open(os.path.join(output_dir, 'all_layers_toxicity_vectors.json'), 'w', encoding='utf-8') as f:
        json.dump(all_vectors_json, f, indent=2, ensure_ascii=False)
    print(f"[保存] all_layers_toxicity_vectors.json")

    summary = {
        'total_layers': len(existing_layers),
        'missing_layers': missing_layers,
        'vector_dimension': HIDDEN_DIM,
        'layers_summary': [],
    }

    for layer_idx in sorted(existing_layers):
        info = results[layer_idx]
        summary['layers_summary'].append({
            'layer': layer_idx,
            'l2_norm': info['vector_l2_norm'],
            'cv_accuracy': info.get('cv_accuracy', None),
        })

    with open(os.path.join(output_dir, 'layer_toxicity_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[保存] layer_toxicity_summary.json")

    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("              毒性向量分析报告 (Toxicity Vector Analysis)\n")
        f.write("=" * 70 + "\n\n")

        f.write("【论文背景】\n")
        f.write("根据论文 Section 5.2，毒性向量 w_toxic 代表了引发有害内容的最佳方向。\n")
        f.write("沿此方向的攻击会以最小偏差将输出推入毒性区域。\n\n")

        f.write("【公式定义】\n")
        f.write("P(y=toxic|h) = softmax(w_toxic · h + b)\n")
        f.write("其中 w_toxic = w[1] - w[0]，w[1] 为有害类权重，w[0] 为安全类权重。\n\n")

        f.write("【统计摘要】\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'层号':<8} {'L2范数':<12} {'CV准确率':<12} {'最大分量':<12} {'最小分量':<12}\n")
        f.write("-" * 70 + "\n")

        for layer_idx in sorted(existing_layers):
            info = results[layer_idx]
            stats = info['vector_statistics']
            cv_acc = info.get('cv_accuracy', 0)
            f.write(f"L{layer_idx:02d}     "
                    f"{info['vector_l2_norm']:<12.4f} "
                    f"{cv_acc:<12.4f} "
                    f"{stats['max_value']:<12.4f} "
                    f"{stats['min_value']:<12.4f}\n")

        f.write("-" * 70 + "\n\n")

        if relations.get('most_similar_pairs'):
            f.write("【最相似的层对 (Top 5)】\n")
            f.write("-" * 70 + "\n")
            for i, pair in enumerate(relations['most_similar_pairs'][:5]):
                f.write(f"{i+1}. L{pair['layer_a']:02d} <-> L{pair['layer_b']:02d}: "
                        f"余弦相似度 = {pair['similarity']:.4f}\n")
            f.write("\n")

        f.write("【使用建议】\n")
        f.write("1. 毒性向量可用于测量输入嵌入与毒性方向的相似度\n")
        f.write("2. 沿毒性向量方向移动嵌入可增加有害输出的概率\n")
        f.write("3. 各层毒性向量方向一致性可反映模型对有害内容的编码方式\n\n")

    print(f"[保存] analysis_report.txt")
    print(f"\n{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"成功提取: {len(existing_layers)} 层")
    if missing_layers:
        print(f"缺失层: {missing_layers}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    if not os.path.exists(args.layers_dir):
        print(f"[错误] 目录不存在: {args.layers_dir}")
        print("请确认 --layers_dir 参数指向包含 layer01/, layer02/, ... 文件夹的目录")
        sys.exit(1)

    results, existing_layers, missing_layers = extract_all_layers(
        args.layers_dir, args.num_layers, args.layer_prefix
    )

    if not existing_layers:
        print("\n[错误] 没有成功提取任何层的毒性向量！")
        sys.exit(1)

    relations = compute_layer_relations(results, existing_layers)

    save_results(results, existing_layers, missing_layers, args.output_dir, relations)

    print("\n✅ 毒性向量提取完成！")
    print(f"\n输出文件位于: {args.output_dir}/")
    print("  - all_layers_toxicity_vectors.json  (完整数据)")
    print("  - toxicity_vectors.npy              (NumPy 矩阵)")
    print("  - layer_toxicity_summary.json       (统计摘要)")
    print("  - analysis_report.txt               (分析报告)")


if __name__ == '__main__':
    main()
