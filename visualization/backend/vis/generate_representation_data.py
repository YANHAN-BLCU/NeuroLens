"""
生成表征视图（Representation View）数据
支持两种投影模式：标准 PCA 投影和决策边界对齐投影
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import os

def convert_to_native(obj):
    """将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj

def load_evaluation_data(jsonl_path):
    """加载评估数据，提取 ASR 标签和攻击方法"""
    asr_labels = []
    attack_types = []
    sample_ids = []
    prompts = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if 'guard' not in rec or 'input' not in rec:
                    continue
                asr_labels.append(rec['guard']['asr_label'])
                attack_types.append(rec['input']['original_sample'].get('method', 
                    rec['input']['original_sample'].get('source', 'Unknown')))
                sample_ids.append(rec['sample_id'])
                prompts.append(rec['input']['prompt'])
            except (json.JSONDecodeError, KeyError) as e:
                # 静默跳过无效行，只在调试时打印
                if line_num <= 10:  # 只打印前10个错误
                    print(f"Error parsing line {line_num}: {e}")
                continue
    
    return {
        'asr_labels': np.array(asr_labels),
        'attack_types': attack_types,
        'sample_ids': np.array(sample_ids),
        'prompts': prompts
    }

def load_hidden_states(cache_path, layer_idx):
    """加载指定层的隐藏状态"""
    data = np.load(cache_path, allow_pickle=True)
    
    # 合并 train, val, test 数据
    train_hs = data['train_hs']  # (N_train, 33, 4096)
    val_hs = data['val_hs']      # (N_val, 33, 4096)
    test_hs = data['test_hs']    # (N_test, 33, 4096)
    
    # 提取指定层
    all_hs = np.concatenate([train_hs[:, layer_idx, :], 
                             val_hs[:, layer_idx, :], 
                             test_hs[:, layer_idx, :]], axis=0)
    
    return all_hs

def standard_pca_projection(hidden_states, n_components=2):
    """标准 PCA 投影"""
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(hidden_states)
    return projected, pca

def decision_boundary_projection(hidden_states, w_toxic, bias):
    """决策边界对齐投影"""
    # 归一化 w_toxic 作为第一主成分方向
    w_toxic_norm = w_toxic / np.linalg.norm(w_toxic)
    
    # 中心化数据
    mean_h = np.mean(hidden_states, axis=0)
    h_centered = hidden_states - mean_h
    
    # 计算第一主成分（与 w_toxic 对齐）
    pc1 = w_toxic_norm
    
    # 使用 Gram-Schmidt 正交化得到垂直于 pc1 的子空间
    # 选择一个随机向量作为起始
    v = np.random.RandomState(42).randn(hidden_states.shape[1])
    v = v - np.dot(v, pc1) * pc1  # 减去在 pc1 上的投影
    v = v / np.linalg.norm(v)
    
    # 在垂直于 pc1 的子空间上做 PCA
    # 构建投影矩阵：将数据投影到垂直于 pc1 的子空间
    proj_matrix = np.eye(hidden_states.shape[1]) - np.outer(pc1, pc1)
    h_orthogonal = h_centered @ proj_matrix
    
    # 在正交子空间上做 PCA
    pca_ortho = PCA(n_components=1)
    pc2_vec = pca_ortho.fit_transform(h_orthogonal)
    # 获取主成分方向
    pc2 = pca_ortho.components_[0]
    pc2 = pc2 / np.linalg.norm(pc2)
    
    # 投影到 (pc1, pc2) 平面
    x_coords = h_centered @ pc1
    y_coords = h_centered @ pc2
    
    projected = np.column_stack([x_coords, y_coords])
    
    return projected, pc1, pc2, mean_h

def compute_decision_boundary_line(w_toxic, bias, pc1, pc2, mean_h, x_range, y_range):
    """计算决策边界线在投影平面上的表示
    决策边界：w_toxic^T * h + bias = 0
    在投影平面上：w_toxic^T * (mean_h + x*pc1 + y*pc2) + bias = 0
    即：w_toxic^T * mean_h + x*(w_toxic^T * pc1) + y*(w_toxic^T * pc2) + bias = 0
    """
    w_pc1 = np.dot(w_toxic, pc1)
    w_pc2 = np.dot(w_toxic, pc2)
    w_mean = np.dot(w_toxic, mean_h)
    
    # 如果 w_pc2 接近 0，边界线是垂直的
    if abs(w_pc2) < 1e-10:
        if abs(w_pc1) < 1e-10:
            return None  # 边界不在这个平面上
        x_boundary = -(w_mean + bias) / w_pc1
        return [[x_boundary, y_range[0]], [x_boundary, y_range[1]]]
    
    # 计算边界线上的点
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # y = -(w_mean + bias + x*w_pc1) / w_pc2
    points = []
    for x in np.linspace(x_min, x_max, 100):
        y = -(w_mean + bias + x * w_pc1) / w_pc2
        if y_min <= y <= y_max:
            points.append([x, y])
    
    if len(points) < 2:
        # 如果边界线不穿过这个区域，尝试从 y 计算 x
        for y in np.linspace(y_min, y_max, 100):
            x = -(w_mean + bias + y * w_pc2) / w_pc1
            if x_min <= x <= x_max:
                points.append([x, y])
    
    return points if len(points) >= 2 else None

def compute_density_contours(x, y, levels=5):
    """计算密度等高线"""
    try:
        # 准备数据
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        
        # 定义网格
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        margin_x = x_range * 0.1
        margin_y = y_range * 0.1
        
        x_grid = np.linspace(x_min - margin_x, x_max + margin_x, 100)
        y_grid = np.linspace(y_min - margin_y, y_max + margin_y, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # 计算密度
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = kde(positions).reshape(X.shape)
        
        # 计算等高线级别
        density_min = density.min()
        density_max = density.max()
        contour_levels = np.linspace(density_min, density_max * 0.8, levels)
        
        return {
            'x_grid': x_grid.tolist(),
            'y_grid': y_grid.tolist(),
            'density': density.tolist(),
            'levels': contour_levels.tolist()
        }
    except Exception as e:
        print(f"Error computing density contours: {e}")
        return None

def generate_representation_data(layer_idx, mode='standard', output_dir='.'):
    """生成表征视图数据"""
    
    # 加载数据
    print(f"Loading data for layer {layer_idx}, mode: {mode}...")
    
    eval_data = load_evaluation_data('../outputs/base_evaluation.jsonl')
    hidden_states = load_hidden_states('../outputs/probes/hidden_states_cache.npz', layer_idx)
    
    # 确保数据长度一致（取较小的）
    min_len = min(len(eval_data['asr_labels']), len(hidden_states))
    hidden_states = hidden_states[:min_len]
    asr_labels = eval_data['asr_labels'][:min_len]
    attack_types = eval_data['attack_types'][:min_len]
    sample_ids = eval_data['sample_ids'][:min_len]
    prompts = eval_data['prompts'][:min_len]
    
    print(f"Using {min_len} samples")
    
    # 执行投影
    if mode == 'standard':
        projected, pca = standard_pca_projection(hidden_states)
        decision_boundary = None
    else:  # decision_boundary
        toxic_vecs = np.load('../outputs/toxic_vectors/toxic_vectors.npz')
        w_toxic = toxic_vecs[f'layer_{layer_idx}']
        bias = toxic_vecs[f'layer_{layer_idx}_bias']
        
        projected, pc1, pc2, mean_h = decision_boundary_projection(hidden_states, w_toxic, bias)
        
        # 计算决策边界线
        x_range = [projected[:, 0].min(), projected[:, 0].max()]
        y_range = [projected[:, 1].min(), projected[:, 1].max()]
        boundary_points = compute_decision_boundary_line(
            w_toxic, bias, pc1, pc2, mean_h, x_range, y_range
        )
        
        decision_boundary = {
            'points': boundary_points,
            'w_toxic_direction': [float(pc1[0]), float(pc1[1])]  # 在投影平面上的方向
        } if boundary_points else None
    
    # 分离成功和失败的样本
    jailbroken_mask = asr_labels == 1
    benign_data = projected[~jailbroken_mask]
    harmful_data = projected[jailbroken_mask]
    
    # 计算等高线（仅标准模式）
    density_contours = None
    if mode == 'standard':
        # 为两类分别计算等高线
        if len(benign_data) > 10:
            benign_contours = compute_density_contours(
                benign_data[:, 0], benign_data[:, 1], levels=5
            )
        else:
            benign_contours = None
            
        if len(harmful_data) > 10:
            harmful_contours = compute_density_contours(
                harmful_data[:, 0], harmful_data[:, 1], levels=5
            )
        else:
            harmful_contours = None
        
        density_contours = {
            'benign': benign_contours,
            'harmful': harmful_contours
        }
    
    # 构建输出数据
    points = []
    for i in range(min_len):
        points.append({
            'id': str(sample_ids[i]),
            'x': float(projected[i, 0]),
            'y': float(projected[i, 1]),
            'jailbroken': bool(jailbroken_mask[i]),
            'method': attack_types[i],
            'instance_id': str(sample_ids[i]),
            'prompt': prompts[i][:200] + '...' if len(prompts[i]) > 200 else prompts[i]
        })
    
    result = {
        'mode': mode,
        'layer': layer_idx,
        'points': points,
        'density_contours': density_contours,
        'decision_boundary': decision_boundary
    }
    
    # 保存数据
    output_file = os.path.join(output_dir, f'representation_layer_{layer_idx}_{mode}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(result), f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")
    return result

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=32, help='Layer index (0-32)')
    parser.add_argument('--mode', type=str, default='standard', 
                       choices=['standard', 'decision_boundary'],
                       help='Projection mode')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--all-layers', action='store_true', 
                       help='Generate data for all layers')
    
    args = parser.parse_args()
    
    if args.all_layers:
        for layer in range(33):
            for mode in ['standard', 'decision_boundary']:
                generate_representation_data(layer, mode, args.output_dir)
    else:
        generate_representation_data(args.layer, args.mode, args.output_dir)
