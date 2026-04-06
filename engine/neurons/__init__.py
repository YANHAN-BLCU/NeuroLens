"""
安全神经元定位模块

本模块整合了基于探针输出的神经元安全分析功能，支持：
- SNIP 分数计算（增强版，支持探针毒性向量）
- 参数对齐分析（增强版，支持探针输出）
- 激活投影分析（增强版，支持探针输出）
- 功能象限分类（增强版，支持可视化集成）
- 安全神经元识别（增强版，支持探针指导）
- 数据加载器（支持从 outputs 目录加载隐藏态和标签）

兼容多种探针输出格式：
- 新格式（engine/probes/linear_probe_balanced.py）：
  outputs/probes/model_id/layer_XX/probe.pt
  outputs/probes/model_id/layer_XX/toxic_vector.npz
  outputs/probes/model_id/layer_XX/metrics.json
- 旧格式（scripts/train_linear_probe_labels.py）：
  outputs/linear_probes/layers/layerXX/layerXX.pt
  outputs/linear_probes/layers/layerXX/metrics.json
"""

from .data_loaders import (
    HiddenStateDataset,
    ProbeModel,
    load_all_probe_models,
    load_probe_toxic_vectors as load_probe_toxic_vectors_from_files,
    get_dataset_stats,
    get_recommended_probe_layers,
)
from .snip_scorer import (
    compute_snip_scores,
    compute_snip_scores_batch,
    rank_and_annotate_snip_scores,
    select_top_percent_neurons,
    load_probe_toxic_vectors_from_snip,
    get_recommended_analysis_layers,
    compute_probe_guided_snip_scores,
    compute_layer_specific_snip_scores,
    get_probe_quality_report,
)
from .safety_identifier import (
    identify_safety_neurons,
    default_safety_loss_fn,
    get_dedicated_safety_neurons,
    get_layer_weights_for_safety,
    select_safety_neurons_by_layer_quality,
    identify_safety_neurons_with_probe_guidance,
)
from .utility_identifier import (
    identify_utility_neurons,
    default_utility_loss_fn,
    AlpacaJsonlDataset,
)
from .salad_safety_dataset import (
    SaladSafetyDataset,
    CombinedSaladSafetyDataset,
)
from .parameter_alignment import (
    compute_parameter_alignment,
    save_parameter_alignment,
    load_toxic_vectors_for_parameter_alignment,
    select_optimal_layers_for_alignment,
)
from .activation_projection import (
    compute_activation_projection,
    load_toxic_vectors_from_probes,
    get_best_toxic_vector_for_activation,
)
from .quadrant_classification import (
    classify_neuron_quadrants,
    get_quadrant_statistics,
    save_quadrant_classification,
    filter_neurons_by_quadrant,
    load_layer_quality_from_probes,
    prepare_quadrant_visualization_data,
    save_quadrant_visualization_data,
)
from .gradient_dependency import (
    compute_gradient_dependency,
    visualize_gradient_dependency,
)


__all__ = [
    # 数据加载器
    "HiddenStateDataset",
    "ProbeModel",
    "load_all_probe_models",
    "load_probe_toxic_vectors_from_files",
    "get_dataset_stats",
    "get_recommended_probe_layers",
    # SNIP 相关
    "compute_snip_scores",
    "compute_snip_scores_batch",
    "rank_and_annotate_snip_scores",
    "select_top_percent_neurons",
    "load_probe_toxic_vectors_from_snip",
    "get_recommended_analysis_layers",
    "compute_probe_guided_snip_scores",
    "compute_layer_specific_snip_scores",
    "get_probe_quality_report",
    # 安全识别相关
    "identify_safety_neurons",
    "identify_safety_neurons_with_probe_guidance",
    "default_safety_loss_fn",
    "get_dedicated_safety_neurons",
    "get_layer_weights_for_safety",
    "select_safety_neurons_by_layer_quality",
    # 效用识别相关
    "identify_utility_neurons",
    "default_utility_loss_fn",
    "AlpacaJsonlDataset",
    # 数据集相关
    "SaladSafetyDataset",
    "CombinedSaladSafetyDataset",
    # 参数对齐相关
    "compute_parameter_alignment",
    "save_parameter_alignment",
    "load_toxic_vectors_for_parameter_alignment",
    "select_optimal_layers_for_alignment",
    # 激活投影相关
    "compute_activation_projection",
    "load_toxic_vectors_from_probes",
    "get_best_toxic_vector_for_activation",
    # 象限分类相关
    "classify_neuron_quadrants",
    "get_quadrant_statistics",
    "save_quadrant_classification",
    "filter_neurons_by_quadrant",
    "load_layer_quality_from_probes",
    "prepare_quadrant_visualization_data",
    "save_quadrant_visualization_data",
    # 梯度依赖相关
    "compute_gradient_dependency",
    "visualize_gradient_dependency",
]
