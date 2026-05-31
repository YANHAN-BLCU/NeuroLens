"""
功能象限分类模块

根据论文5.4节要求，将神经元分为四个功能象限：S+A+, S-A+, S+A-, S-A-

功能：
- 根据参数对齐（S_i^k）和激活投影（A_i^k）结果，将神经元分类到四个象限
- 提供象限统计和可视化支持
- 支持从探针输出加载层质量数据用于可视化增强
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
from collections import Counter


# ============================================================================
# 探针层质量数据加载器（用于可视化增强）
# ============================================================================

def load_layer_quality_from_probes(
    probe_output_dir: Union[str, Path] = "outputs",
    model_id: Optional[str] = None,
) -> Dict[int, Dict]:
    """
    从探针输出加载各层的质量数据，用于象限可视化的增强。

    支持多种数据格式：
    新格式（engine/probes/linear_probe_balanced.py）：
        - outputs/probes/{model_id}/layer_XX/metrics.json
        （model_id 用于文档，实际路径解析不受其影响）
    旧格式（scripts/train_linear_probe_labels.py）：
        - outputs/linear_probes/layers/layerXX/metrics.json
    聚合格式：
        - outputs/toxicity_vectors/layer_toxicity_summary.json

    Args:
        probe_output_dir: 探针输出目录
        model_id: 模型标识符（仅用于文档；为 None 时从目录结构自动推断）

    Returns:
        Dict[int, Dict]: 各层的质量数据
            {
                layer_idx: {
                    'cv_accuracy': float,
                    'std': float,
                    'group': str,  # S/A/B/C/D
                    'safe_acc': float,
                    'harm_acc': float,
                    'val_roc_auc': float,    # 新格式
                    'val_pr_auc': float,     # 新格式
                    'is_invalid': bool,       # 新格式
                }
            }
    """
    import re
    probe_output_dir = Path(probe_output_dir)
    layer_quality = {}

    def _get_layer_idx(name: str) -> Optional[int]:
        match = re.match(r'layer_(\d+)', name)
        if match:
            return int(match.group(1))
        match = re.match(r'layer(\d+)', name)
        if match:
            return int(match.group(1))
        return None

    def _get_group(acc: float) -> str:
        """根据准确率确定分组"""
        acc_pct = acc * 100
        if acc_pct >= 93:
            return 'S'
        elif acc_pct >= 91:
            return 'A'
        elif acc_pct >= 89:
            return 'B'
        elif acc_pct >= 85:
            return 'C'
        else:
            return 'D'

    def _parse_metrics(metrics: Dict) -> Dict:
        """解析 metrics.json，兼容新旧格式"""
        result = {}

        # 新格式：val_acc
        if 'val_acc' in metrics:
            result['cv_accuracy'] = float(metrics['val_acc'])
            result['val_acc'] = float(metrics['val_acc'])
        # 旧格式：avg_val_acc
        elif 'avg_val_acc' in metrics:
            result['cv_accuracy'] = float(metrics['avg_val_acc'])

        # 标准差
        result['std'] = float(metrics.get('std', metrics.get('std_val_acc', 0)))

        # 旧格式指标
        if 'avg_val_s_acc' in metrics:
            result['safe_acc'] = float(metrics['avg_val_s_acc'])
        if 'avg_val_h_acc' in metrics:
            result['harm_acc'] = float(metrics['avg_val_h_acc'])

        # 新格式指标
        if 'val_roc_auc' in metrics:
            result['val_roc_auc'] = float(metrics['val_roc_auc'])
        if 'val_pr_auc' in metrics:
            result['val_pr_auc'] = float(metrics['val_pr_auc'])
        if 'is_invalid' in metrics:
            result['is_invalid'] = bool(metrics['is_invalid'])
        if 'filter_threshold' in metrics:
            result['filter_threshold'] = float(metrics['filter_threshold'])

        # 分组
        if 'cv_accuracy' in result:
            result['group'] = _get_group(result['cv_accuracy'])

        return result

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
                if layer_idx is None or layer_idx in layer_quality:
                    continue

                metrics_json = layer_folder / "metrics.json"
                if metrics_json.exists():
                    try:
                        with open(metrics_json, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        parsed = _parse_metrics(metrics)
                        if parsed:
                            layer_quality[layer_idx] = parsed
                    except Exception:
                        pass

    # 策略2: 旧格式 outputs/linear_probes/layers/
    if not layer_quality:
        layers_dir = probe_output_dir / "linear_probes" / "layers"
        if layers_dir.exists():
            for layer_folder in sorted(layers_dir.iterdir()):
                if layer_folder.is_dir() and layer_folder.name.startswith("layer"):
                    layer_idx = _get_layer_idx(layer_folder.name)
                    if layer_idx is None or layer_idx in layer_quality:
                        continue

                    metrics_json = layer_folder / "metrics.json"
                    if metrics_json.exists():
                        try:
                            with open(metrics_json, 'r') as f:
                                metrics = json.load(f)
                            parsed = _parse_metrics(metrics)
                            if parsed:
                                layer_quality[layer_idx] = parsed
                        except Exception:
                            pass

    # 策略3: outputs/toxicity_vectors/layer_toxicity_summary.json
    if not layer_quality:
        summary_path = probe_output_dir / "toxicity_vectors" / "layer_toxicity_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

                if 'layers_summary' in summary:
                    for layer_info in summary['layers_summary']:
                        layer_idx = int(layer_info['layer'])
                        layer_quality[layer_idx] = {
                            'cv_accuracy': layer_info.get('cv_accuracy', 0.0),
                            'std': 0.0,
                            'group': _get_group(layer_info.get('cv_accuracy', 0)),
                            'l2_norm': layer_info.get('l2_norm', 0.0),
                        }
            except Exception:
                pass

    return layer_quality


def _get_group_from_accuracy(accuracy: float) -> str:
    """根据准确率确定分组"""
    acc_pct = accuracy * 100
    if acc_pct >= 93:
        return 'S'
    elif acc_pct >= 91:
        return 'A'
    elif acc_pct >= 89:
        return 'B'
    elif acc_pct >= 85:
        return 'C'
    else:
        return 'D'


def classify_neuron_quadrants(
    parameter_alignment: Dict[Tuple[int, int], Dict],
    activation_projection: Dict[Tuple[int, int], Dict],
    threshold_s: float = 0.0,
    threshold_a: float = 0.0,
    layer_quality: Optional[Dict[int, Dict]] = None,
) -> Dict[Tuple[int, int], Dict]:
    """
    功能象限分类：根据参数对齐和激活投影结果，将神经元分为四个象限
    
    根据论文5.4节，神经元的功能象限分类基于：
        - S_i^k: 参数对齐（参数方向与毒性向量的余弦相似度）
            公式：S_i^k = (w_down,i^k · w_toxic^k) / (||w_down,i^k|| ||w_toxic^k||)
            - S+ (S_i^k > 0): 参数对齐为正，表示神经元参数方向促进有害内容生成
            - S- (S_i^k <= 0): 参数对齐为负，表示神经元参数方向有助于防御性转向
        - A_i^k: 激活投影（激活向量在归一化毒性向量上的投影）
            公式：A_i^k = a_down,i^k · (w_toxic^k / ||w_toxic^k||)
            - A+ (A_i^k > 0): 激活投影为正，在成功jailbreak时激活更强，促进毒性
            - A- (A_i^k <= 0): 激活投影为负，在失败jailbreak时激活更强，抑制毒性
    
    四个象限定义（基于论文）：
        - **S+A+**: 参数对齐为正（S+），激活投影为正（A+）
            → 毒性特征增强（Toxic feature enhancement）
            → 参数和激活都促进毒性，是最危险的神经元
        - **S-A+**: 参数对齐为负（S-），激活投影为正（A+）
            → 良性特征抑制（Benign feature suppression）
            → 参数方向抑制毒性，但激活时促进毒性（可能是防御机制失效）
        - **S+A-**: 参数对齐为正（S+），激活投影为负（A-）
            → 毒性特征抑制（Toxic feature suppression）
            → 参数方向促进毒性，但激活时抑制毒性（可能是防御机制生效）
        - **S-A-**: 参数对齐为负（S-），激活投影为负（A-）
            → 良性特征增强（Benign feature enhancement）
            → 参数和激活都抑制毒性，是最安全的神经元
    
    注意：
        - 激活投影（activation_projection）使用成功样本的平均值，用于象限分类
        - 激活差异（activation_diff = successful_mean - failed_mean）是辅助信息，
          用于分析防御机制，不直接用于象限分类
    
    Args:
        parameter_alignment: 参数对齐结果，来自 parameter_alignment.compute_parameter_alignment()
            格式为 Dict[(layer_idx, neuron_idx), {
                'cosine_similarity': float,  # 余弦相似度 [-1, 1]
                'alignment_type': 'S+' | 'S-',  # 对齐类型（基于阈值0.0）
                'neuron_weight_norm': float,
                'toxic_vector_norm': float,
                'toxic_layer': int,  # 使用的毒性向量所属层（增强字段）
                'probe_cv_accuracy': float,  # 该层的探针 CV 准确率（增强字段）
            }]
        activation_projection: 激活投影结果，来自 activation_projection.compute_activation_projection()
            格式为 Dict[(layer_idx, neuron_idx), {
                'activation_projection': float,  # 激活投影值 A_i^k（用于象限分类，= successful_mean）
                'successful_mean': float,  # 成功样本的平均激活投影
                'failed_mean': float,  # 失败样本的平均激活投影
                'activation_diff': float,  # successful_mean - failed_mean（辅助信息）
                'successful_std': float,
                'failed_std': float,
                'successful_count': int,
                'failed_count': int,
                'toxic_layer': int,  # 使用的毒性向量所属层（增强字段）
                'probe_cv_accuracy': float,  # 该层的探针 CV 准确率（增强字段）
            }]
        threshold_s: 参数对齐阈值（默认0.0）
            - 如果cosine_similarity > threshold_s，则为S+
            - 如果cosine_similarity <= threshold_s，则为S-
            注意：parameter_alignment中的alignment_type是基于阈值0.0计算的，
                  但这里允许使用自定义阈值进行重新分类
        threshold_a: 激活投影阈值（默认0.0）
            - 如果activation_projection > threshold_a，则为A+（激活投影为正，促进毒性）
            - 如果activation_projection <= threshold_a，则为A-（激活投影为负，抑制毒性）
        layer_quality: 层质量数据，来自 load_layer_quality_from_probes()（可选）
            用于添加探针质量信息到分类结果
    
    Returns:
        Dict[(layer_idx, neuron_idx), {
            'quadrant': 'S+A+' | 'S-A+' | 'S+A-' | 'S-A-',
            'alignment': float,  # 参数对齐值（cosine_similarity）
            'activation_projection': float,  # 激活投影值 A_i^k（用于象限分类）
            'activation_diff': float,  # 激活差异（successful_mean - failed_mean，辅助信息）
            'successful_mean': float,  # 成功样本的平均激活投影
            'failed_mean': float,  # 失败样本的平均激活投影
            'alignment_type': 'S+' | 'S-',  # 基于threshold_s判断的对齐类型
            'activation_type': 'A+' | 'A-',  # 基于threshold_a判断的激活类型
            'toxic_layer': int,  # 使用的毒性向量所属层
            'probe_cv_accuracy': float,  # 该层的探针 CV 准确率
            'layer_group': str,  # 层分组（S/A/B/C/D，来自 layer_quality）
        }]
    """
    print("[Quadrant Classification] 开始功能象限分类...")
    print(f"[Quadrant Classification] 参数对齐阈值: {threshold_s}, 激活投影阈值: {threshold_a}")
    
    # 找到同时存在于两个结果中的神经元
    common_neurons = set(parameter_alignment.keys()) & set(activation_projection.keys())
    
    if len(common_neurons) == 0:
        print("[Quadrant Classification] 警告: 没有找到同时存在于参数对齐和激活投影结果中的神经元")
        print(f"[Quadrant Classification] 参数对齐神经元数: {len(parameter_alignment)}")
        print(f"[Quadrant Classification] 激活投影神经元数: {len(activation_projection)}")
        return {}
    
    print(f"[Quadrant Classification] 找到 {len(common_neurons)} 个共同神经元")
    
    quadrant_results = {}
    quadrant_counts = Counter()
    missing_keys = []
    
    for (layer_idx, neuron_idx) in common_neurons:
        # 获取参数对齐信息
        align_data = parameter_alignment.get((layer_idx, neuron_idx))
        if align_data is None:
            missing_keys.append(f"参数对齐: ({layer_idx}, {neuron_idx})")
            continue
        
        if 'cosine_similarity' not in align_data:
            print(f"[Quadrant Classification] 警告: 神经元 ({layer_idx}, {neuron_idx}) 缺少 'cosine_similarity' 字段")
            continue
        
        cosine_sim = align_data['cosine_similarity']
        # 注意：parameter_alignment中的alignment_type是基于阈值0.0计算的，
        # 但这里使用自定义阈值threshold_s重新判断，以支持灵活的阈值设置
        
        # 获取激活投影信息
        proj_data = activation_projection.get((layer_idx, neuron_idx))
        if proj_data is None:
            missing_keys.append(f"激活投影: ({layer_idx}, {neuron_idx})")
            continue
        
        # 使用 activation_projection 字段（如果存在），否则使用 successful_mean
        # 根据 activation_projection.py 第317行，activation_projection = successful_mean
        activation_proj = proj_data.get('activation_projection')
        if activation_proj is None:
            activation_proj = proj_data.get('successful_mean', 0.0)
        
        # 获取其他辅助信息
        successful_mean = proj_data.get('successful_mean', activation_proj)
        failed_mean = proj_data.get('failed_mean', 0.0)
        activation_diff = proj_data.get('activation_diff', successful_mean - failed_mean)
        
        # 根据阈值判断对齐类型（使用自定义阈值，而不是原始的alignment_type）
        if cosine_sim > threshold_s:
            s_type = 'S+'
        else:
            s_type = 'S-'
        
        # 根据阈值判断激活类型（基于激活投影值本身）
        if activation_proj > threshold_a:
            a_type = 'A+'
        else:
            a_type = 'A-'
        
        # 组合象限
        quadrant = f"{s_type}{a_type}"
        quadrant_counts[quadrant] += 1
        
        # 获取探针元数据（从 parameter_alignment 和 activation_projection 中提取）
        toxic_layer = align_data.get('toxic_layer', layer_idx)
        probe_cv_acc = align_data.get('probe_cv_accuracy', 
                                proj_data.get('probe_cv_accuracy', 0.0))
        
        # 获取层分组（如果提供了 layer_quality）
        layer_group = None
        if layer_quality and toxic_layer in layer_quality:
            layer_group = layer_quality[toxic_layer].get('group')
        
        quadrant_results[(layer_idx, neuron_idx)] = {
            'quadrant': quadrant,
            'alignment': float(cosine_sim),
            'activation_projection': float(activation_proj),
            'activation_diff': float(activation_diff),
            'successful_mean': float(successful_mean),
            'failed_mean': float(failed_mean),
            'alignment_type': s_type,
            'activation_type': a_type,
            'toxic_layer': toxic_layer,  # 使用的毒性向量所属层
            'probe_cv_accuracy': probe_cv_acc,  # 探针 CV 准确率
            'layer_group': layer_group,  # 层分组（S/A/B/C/D）
        }
    
    if missing_keys:
        print(f"[Quadrant Classification] 警告: 发现 {len(missing_keys)} 个缺失键（前10个）: {missing_keys[:10]}")
    
    if len(quadrant_results) == 0:
        print("[Quadrant Classification] 警告: 没有成功分类任何神经元")
        return {}
    
    # 打印统计信息
    print(f"[Quadrant Classification] 象限分布:")
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        count = quadrant_counts.get(quadrant, 0)
        percentage = (count / len(quadrant_results) * 100) if quadrant_results else 0
        print(f"  {quadrant}: {count} ({percentage:.1f}%)")
    
    # 打印象限含义
    print(f"\n[Quadrant Classification] 象限含义:")
    print(f"  S+A+: 参数对齐为正，激活投影为正 → 毒性特征增强")
    print(f"  S-A+: 参数对齐为负，激活投影为正 → 良性特征抑制")
    print(f"  S+A-: 参数对齐为正，激活投影为负 → 毒性特征抑制")
    print(f"  S-A-: 参数对齐为负，激活投影为负 → 良性特征增强")
    
    return quadrant_results


def get_quadrant_statistics(
    quadrant_results: Dict[Tuple[int, int], Dict],
) -> Dict[str, Dict]:
    """
    获取象限统计信息
    
    Args:
        quadrant_results: 象限分类结果
    
    Returns:
        Dict[quadrant, {
            'count': int,
            'percentage': float,
            'avg_alignment': float,  # 平均参数对齐值（余弦相似度）
            'avg_activation_projection': float,  # 平均激活投影值（用于象限分类）
            'avg_activation_diff': float,  # 平均激活差异（successful_mean - failed_mean）
            'avg_successful_mean': float,  # 平均成功样本激活投影
            'avg_failed_mean': float,  # 平均失败样本激活投影
            'neurons': List[Tuple[int, int]],
        }]
    """
    from collections import defaultdict
    
    stats = defaultdict(lambda: {
        'count': 0,
        'alignments': [],
        'activation_projections': [],
        'activation_diffs': [],
        'successful_means': [],
        'failed_means': [],
        'neurons': [],
    })
    
    for (layer_idx, neuron_idx), data in quadrant_results.items():
        quadrant = data['quadrant']
        stats[quadrant]['count'] += 1
        stats[quadrant]['alignments'].append(data['alignment'])
        stats[quadrant]['activation_projections'].append(data['activation_projection'])
        stats[quadrant]['activation_diffs'].append(data['activation_diff'])
        stats[quadrant]['successful_means'].append(data.get('successful_mean', data['activation_projection']))
        stats[quadrant]['failed_means'].append(data.get('failed_mean', 0.0))
        stats[quadrant]['neurons'].append((layer_idx, neuron_idx))
    
    # 计算统计量
    result = {}
    total = len(quadrant_results)
    
    for quadrant in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        if quadrant not in stats:
            result[quadrant] = {
                'count': 0,
                'percentage': 0.0,
                'avg_alignment': 0.0,
                'avg_activation_projection': 0.0,
                'avg_activation_diff': 0.0,
                'avg_successful_mean': 0.0,
                'avg_failed_mean': 0.0,
                'neurons': [],
            }
            continue
        
        stat = stats[quadrant]
        result[quadrant] = {
            'count': stat['count'],
            'percentage': (stat['count'] / total * 100) if total > 0 else 0.0,
            'avg_alignment': float(np.mean(stat['alignments'])) if stat['alignments'] else 0.0,
            'avg_activation_projection': float(np.mean(stat['activation_projections'])) if stat['activation_projections'] else 0.0,
            'avg_activation_diff': float(np.mean(stat['activation_diffs'])) if stat['activation_diffs'] else 0.0,
            'avg_successful_mean': float(np.mean(stat['successful_means'])) if stat['successful_means'] else 0.0,
            'avg_failed_mean': float(np.mean(stat['failed_means'])) if stat['failed_means'] else 0.0,
            'neurons': stat['neurons'],
        }
    
    return result


def save_quadrant_classification(
    quadrant_results: Dict[Tuple[int, int], Dict],
    output_path: Path,
    filename: str = "quadrant_classification.json",
):
    """
    保存象限分类结果到JSON文件
    
    Args:
        quadrant_results: 象限分类结果
        output_path: 输出目录
        filename: 输出文件名
    """
    import json
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化格式
    serializable = {}
    for (layer_idx, neuron_idx), data in quadrant_results.items():
        key = f"layer_{layer_idx}_neuron_{neuron_idx}"
        serializable[key] = {
            'layer_idx': int(layer_idx),
            'neuron_idx': int(neuron_idx),
            **data
        }
    
    # 添加统计信息
    stats = get_quadrant_statistics(quadrant_results)
    serializable['_statistics'] = {
        quadrant: {
            'count': stat['count'],
            'percentage': stat['percentage'],
            'avg_alignment': stat['avg_alignment'],
            'avg_activation_projection': stat['avg_activation_projection'],
            'avg_activation_diff': stat['avg_activation_diff'],
            'avg_successful_mean': stat['avg_successful_mean'],
            'avg_failed_mean': stat['avg_failed_mean'],
        }
        for quadrant, stat in stats.items()
    }
    
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"[Quadrant Classification] 结果已保存到: {output_file}")
    return output_file


def filter_neurons_by_quadrant(
    quadrant_results: Dict[Tuple[int, int], Dict],
    quadrants: list,
) -> Dict[Tuple[int, int], Dict]:
    """
    根据象限过滤神经元
    
    Args:
        quadrant_results: 象限分类结果
        quadrants: 要保留的象限列表，如 ['S+A+', 'S-A+']
    
    Returns:
        过滤后的神经元字典
    """
    filtered = {
        key: value
        for key, value in quadrant_results.items()
        if value['quadrant'] in quadrants
    }
    
    print(f"[Quadrant Classification] 过滤后保留 {len(filtered)} 个神经元（象限: {quadrants}）")
    return filtered


# ============================================================================
# 增强的可视化函数（支持探针层质量数据）
# ============================================================================

def prepare_quadrant_visualization_data(
    quadrant_results: Dict[Tuple[int, int], Dict],
    layer_quality: Optional[Dict[int, Dict]] = None,
    include_neurons_by_layer: Optional[Dict[int, list]] = None,
) -> Dict:
    """
    准备用于可视化的象限分类数据。
    
    Args:
        quadrant_results: 象限分类结果
        layer_quality: 层质量数据（可选）
        include_neurons_by_layer: 每层的神经元列表（可选，用于生成详细可视化）
    
    Returns:
        Dict: 可视化数据
    """
    viz_data = {
        "summary": {},
        "layers": {},
        "quadrants": {},
        "neurons": [],
    }
    
    # 统计象限分布
    quadrant_counts = Counter(q['quadrant'] for q in quadrant_results.values())
    total = len(quadrant_results)
    
    viz_data["summary"] = {
        "total_neurons": total,
        "quadrant_distribution": {
            q: {"count": quadrant_counts.get(q, 0), "percentage": quadrant_counts.get(q, 0) / total * 100}
            for q in ['S+A+', 'S-A+', 'S+A-', 'S-A-']
        },
    }
    
    # 按层统计
    neurons_by_layer = {}
    for (layer_idx, neuron_idx), data in quadrant_results.items():
        if layer_idx not in neurons_by_layer:
            neurons_by_layer[layer_idx] = []
        neurons_by_layer[layer_idx].append((neuron_idx, data))
    
    for layer_idx, neurons in neurons_by_layer.items():
        layer_stats = Counter(n[1]['quadrant'] for n in neurons)
        layer_data = {
            "total": len(neurons),
            "quadrants": {q: layer_stats.get(q, 0) for q in ['S+A+', 'S-A+', 'S+A-', 'S-A-']},
        }
        
        # 添加层质量信息
        if layer_quality and layer_idx in layer_quality:
            layer_data["quality"] = layer_quality[layer_idx]
        
        viz_data["layers"][layer_idx] = layer_data
    
    # 按象限组织神经元
    for q in ['S+A+', 'S-A+', 'S+A-', 'S-A-']:
        viz_data["quadrants"][q] = {
            "count": quadrant_counts.get(q, 0),
            "percentage": quadrant_counts.get(q, 0) / total * 100,
            "neurons": [
                {
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "alignment": data['alignment'],
                    "activation_projection": data['activation_projection'],
                    "probe_cv_accuracy": data.get('probe_cv_accuracy'),
                    "layer_group": data.get('layer_group'),
                }
                for (layer_idx, neuron_idx), data in quadrant_results.items()
                if data['quadrant'] == q
            ],
        }
    
    # 扁平化神经元列表
    viz_data["neurons"] = [
        {
            "layer": layer_idx,
            "neuron": neuron_idx,
            **data,
        }
        for (layer_idx, neuron_idx), data in quadrant_results.items()
    ]
    
    return viz_data


def save_quadrant_visualization_data(
    quadrant_results: Dict[Tuple[int, int], Dict],
    output_path: Union[str, Path],
    layer_quality: Optional[Dict[int, Dict]] = None,
    filename: str = "quadrant_visualization.json",
) -> Path:
    """
    保存象限可视化的增强数据到 JSON 文件。
    
    Args:
        quadrant_results: 象限分类结果
        output_path: 输出目录
        layer_quality: 层质量数据（可选）
        filename: 输出文件名
    
    Returns:
        Path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备可视化数据
    viz_data = prepare_quadrant_visualization_data(quadrant_results, layer_quality)
    
    # 添加层质量信息（如果提供）
    if layer_quality:
        viz_data["layer_quality"] = layer_quality
    
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)
    
    print(f"[Quadrant Visualization] 可视化数据已保存到: {output_file}")
    return output_file
