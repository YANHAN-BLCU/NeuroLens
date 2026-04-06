"""
数据加载器模块

专门处理从 outputs 目录加载隐藏态、标签和模型输出数据。
支持与 scripts/extract_hidden_states.py 和 scripts/train_linear_probe_labels.py 相同的数据格式。

数据目录结构：
    outputs/
    ├── data_set_output/           # 模型输出和标签（scripts/extract_hidden_states.py 输出）
    │   ├── base_set_outputs_*.jsonl      # 模型生成输出（无标签）
    │   ├── attack_enhanced_outputs.jsonl
    │   └── labels/                        # 标签文件（Qwen 标注）
    │       ├── base_set_outputs_*.jsonl
    │       └── attack_enhanced_outputs.jsonl
    ├── hidden_states/              # 隐藏态向量（scripts/extract_hidden_states.py 输出）
    │   ├── base_set_hidden_states_*.hs.npy   # 隐藏态
    │   ├── base_set_hidden_states_*.idx.npy  # 原始索引
    │   └── attack_enhanced_hidden_states.hs.npy
    └── linear_probes/             # 探针模型（scripts/train_linear_probe_labels.py 输出）
        └── layers/
            ├── layer01/layer01.pt
            └── layer32/layer32.pt
"""

import json
import re
import glob as glob_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================================
# 数据集类型定义
# ============================================================================

class HiddenStateDataset(Dataset):
    """
    隐藏态数据集，支持多文件分片加载。
    
    数据格式（来自 scripts/extract_hidden_states.py）：
    - 每个样本的隐藏态: shape (num_layers, hidden_dim) = (32, 4096)
    - 通过 idx.npy 记录每个隐藏态对应的 original_index
    
    用法:
        dataset = HiddenStateDataset("outputs/hidden_states")
        hs, label = dataset[0]  # hs: (32, 4096), label: 0/1
    """
    
    def __init__(
        self,
        hidden_states_dir: Union[str, Path],
        labels_dir: Optional[Union[str, Path]] = None,
        label_name_mapping: Optional[Dict[str, str]] = None,
        transform: Optional[callable] = None,
        target_layer: Optional[int] = None,
    ):
        """
        Args:
            hidden_states_dir: 隐藏态目录
            labels_dir: 标签目录（可选）
            label_name_mapping: 文件名映射，key是hs文件名，value是label文件名
            transform: 可选的变换函数
            target_layer: 如果指定，只返回该层的隐藏态 (hidden_dim,) 而非 (num_layers, hidden_dim)
        """
        self.hidden_states_dir = Path(hidden_states_dir) if hidden_states_dir else None
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.label_name_mapping = label_name_mapping or {}
        self.transform = transform
        self.target_layer = target_layer
        
        self.hs_chunks = []   # mmap 列表
        self.idx_chunks = []  # 索引列表
        self.all_labels = []   # 标签列表
        
        self._pos_to_chunk = []   # 全局索引 -> chunk索引
        self._pos_to_in_idx = [] # 全局索引 -> chunk内索引
        
        self._load_data()
    
    def _compute_label_name(self, hs_name: str) -> str:
        """从隐藏态文件名推导对应的标签文件名。"""
        name = re.sub(r'_hidden_states', '_outputs', hs_name)
        name = re.sub(r'\.hs\.npy$', '.jsonl', name)
        return name
    
    def _load_data(self):
        """加载所有隐藏态和标签文件。"""
        if not self.hidden_states_dir or not self.hidden_states_dir.exists():
            return
        
        hs_files = sorted(self.hidden_states_dir.glob('*.hs.npy'))
        
        for hs_path in hs_files:
            hs_basename = hs_path.name
            
            # 加载隐藏态（mmap只读）
            hs_mmap = np.load(hs_path, mmap_mode='r')
            self.hs_chunks.append(hs_mmap)
            
            # 加载索引
            idx_path = hs_path.with_suffix('.idx.npy')
            if idx_path.exists():
                idx_mmap = np.load(idx_path, mmap_mode='r')
                self.idx_chunks.append(idx_mmap)
            else:
                self.idx_chunks.append(np.arange(len(hs_mmap)))
            
            # 加载标签（如果提供）
            if self.labels_dir and self.labels_dir.exists():
                label_name = self.label_name_mapping.get(
                    hs_basename, 
                    self._compute_label_name(hs_basename)
                )
                label_path = self.labels_dir / label_name
                
                if label_path.exists():
                    labels = self._load_labels(label_path)
                    self.all_labels.extend(labels)
                else:
                    self.all_labels.extend([None] * len(hs_mmap))
            else:
                self.all_labels.extend([None] * len(hs_mmap))
            
            # 构建全局索引映射
            for chunk_idx in range(len(self.hs_chunks[-1])):
                self._pos_to_chunk.append(len(self.hs_chunks) - 1)
                self._pos_to_in_idx.append(chunk_idx)
    
    def _load_labels(self, label_path: Path) -> List[int]:
        """加载标签文件，建立 original_index -> label 映射。"""
        # 读取标签文件
        oi_to_label = {}
        with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    oi = data.get('original_index')
                    if oi is None:
                        continue
                    oi = int(oi)
                    
                    label_val = data.get('label', '')
                    if not label_val or label_val == 'Controversial':
                        continue
                    
                    if label_val == 'Unsafe':
                        oi_to_label[oi] = 1
                    elif label_val == 'Safe':
                        oi_to_label[oi] = 0
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        
        # 返回标签列表（按 idx.npy 的 original_index 顺序）
        if self.idx_chunks:
            idx_mmap = self.idx_chunks[-1]
            return [
                oi_to_label.get(int(oi), -1)
                for oi in idx_mmap
            ]
        return []
    
    def __len__(self):
        return sum(len(chunk) for chunk in self.hs_chunks)
    
    def __getitem__(self, idx: int) -> Tuple:
        """获取单个样本。"""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self)})")
        
        chunk_idx = self._pos_to_chunk[idx]
        in_idx = self._pos_to_in_idx[idx]
        
        # 获取隐藏态
        if self.target_layer is not None:
            # 返回单层 (hidden_dim,)
            hs = self.hs_chunks[chunk_idx][in_idx, self.target_layer - 1, :].astype(np.float32)
        else:
            # 返回所有层 (num_layers, hidden_dim)
            hs = self.hs_chunks[chunk_idx][in_idx].astype(np.float32)
        
        # 获取标签
        label = self.all_labels[idx]
        
        if self.transform:
            hs = self.transform(hs)
        
        return torch.from_numpy(hs) if isinstance(hs, np.ndarray) else hs, label


class ProbeModel(torch.nn.Module):
    """
    探针模型，与 scripts/train_linear_probe_labels.py 中的 Probe 类兼容。
    
    模型结构:
        Linear(hidden_dim, 2)
        其中 weight[0] = 安全类权重, weight[1] = 有害类权重
    
    毒性向量:
        tox_vec = weight[1] - weight[0]
    """
    
    def __init__(self, hidden_dim: Optional[int] = None):
        """
        Args:
            hidden_dim: 探针输入维度（必须与模型 hidden_dim 匹配）。
                        如果为 None，则在加载 checkpoint 时从实际权重中推断。
        """
        self._hidden_dim = hidden_dim
        self.fc: Optional[torch.nn.Linear] = None  # 延迟初始化
        super().__init__()

    def _ensure_fc(self, input_tensor: torch.Tensor):
        """确保 fc 层已初始化（延迟初始化）。"""
        if self.fc is None:
            if self._hidden_dim is None:
                # 从输入张量的最后一维推断
                actual_dim = input_tensor.shape[-1]
                self._hidden_dim = actual_dim
            self.fc = torch.nn.Linear(self._hidden_dim, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fc(x)
        return self.fc(x)
    
    def tox_vec(self) -> torch.Tensor:
        """获取毒性向量: weight[1] - weight[0]（需要先调用 forward 或手动设置 fc）"""
        if self.fc is None:
            raise RuntimeError("fc 层未初始化，请先调用 forward() 或通过 load_from_checkpoint 加载")
        return self.fc.weight[1] - self.fc.weight[0]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Union[str, Path]) -> 'ProbeModel':
        """从检查点加载模型，自动从权重形状推断 hidden_dim。"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 从权重形状推断 hidden_dim
        hidden_dim = None
        for weight_key in ('linear.weight', 'weight', 'fc.weight'):
            if weight_key in checkpoint:
                w = checkpoint[weight_key]
                if isinstance(w, np.ndarray):
                    if w.ndim == 2 and w.shape[0] == 2:
                        hidden_dim = w.shape[1]
                        break
                elif hasattr(w, 'shape'):
                    if w.shape[0] == 2:
                        hidden_dim = w.shape[1]
                        break

        model = cls(hidden_dim=hidden_dim)
        model.fc = torch.nn.Linear(hidden_dim, 2) if hidden_dim else None

        # 加载权重
        for weight_key in ('linear.weight', 'weight', 'fc.weight'):
            if weight_key in checkpoint and model.fc is not None:
                w = checkpoint[weight_key]
                if isinstance(w, np.ndarray):
                    model.fc.weight.data = torch.from_numpy(w)
                else:
                    model.fc.weight.data = w.cpu()
                break

        # 加载偏置
        model.fc.bias = torch.nn.Parameter(torch.zeros(2))
        for bias_key in ('linear.bias', 'bias', 'fc.bias'):
            if bias_key in checkpoint and model.fc is not None:
                b = checkpoint[bias_key]
                if isinstance(b, np.ndarray):
                    model.fc.bias.data = torch.from_numpy(b)
                else:
                    model.fc.bias.data = b.cpu()
                break

        return model


# ============================================================================
# 探针加载和管理
# ============================================================================

def _get_layer_index_from_name(folder_name: str) -> Optional[int]:
    """
    从目录名提取层索引。

    支持格式:
    - layer_0, layer_28 -> 0, 28
    - layer00, layer28 -> 0, 28
    - layer_00, layer_28 -> 0, 28
    """
    # 尝试 layer_XX 格式
    match = re.match(r'layer_(\d+)', folder_name)
    if match:
        return int(match.group(1))

    # 尝试 layerXX 格式（无下划线）
    match = re.match(r'layer(\d+)', folder_name)
    if match:
        return int(match.group(1))

    return None


def load_all_probe_models(
    probes_dir: Union[str, Path] = "outputs/probes",
    num_layers: Optional[int] = None,
) -> Dict[int, ProbeModel]:
    """
    加载所有层的探针模型。

    支持多种目录结构：
    - 新格式（engine/probes/linear_probe_balanced.py）：
      probes_dir/layer_0/probe.pt
      probes_dir/layer_28/probe.pt
    - 旧格式（scripts/train_linear_probe_labels.py）：
      probes_dir/linear_probes/layers/layer01/layer01.pt
      probes_dir/layers/layer32/layer32.pt
    - 直接格式：
      probes_dir/layerXX/layerXX.pt

    Args:
        probes_dir: 探针目录
        num_layers: 层数上限（为 None 时不过滤，允许任意索引）

    Returns:
        Dict[int, ProbeModel]: {layer_idx: model}
    """
    probes_dir = Path(probes_dir)
    models = {}

    if not probes_dir.exists():
        return models

    # 收集所有 layer 文件夹（包括 linear_probes/layers 子目录）
    layer_folders = []

    # 直接扫描根目录
    for item in probes_dir.iterdir():
        if item.is_dir():
            if _get_layer_index_from_name(item.name) is not None:
                layer_folders.append(item)

    # 如果存在 linear_probes/layers/ 子目录，也扫描它
    linear_probes_layers = probes_dir / "linear_probes" / "layers"
    if linear_probes_layers.exists():
        for item in linear_probes_layers.iterdir():
            if item.is_dir():
                if _get_layer_index_from_name(item.name) is not None:
                    layer_folders.append(item)

    # 去重（避免同一文件夹被扫描两次）
    seen_paths = set()
    unique_folders = []
    for folder in layer_folders:
        resolved = folder.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_folders.append(folder)

    for item in unique_folders:
        layer_idx = _get_layer_index_from_name(item.name)
        if layer_idx is None:
            continue
        if num_layers is not None and (layer_idx < 0 or layer_idx >= num_layers):
            continue

        # 尝试多种探针文件
        probe_path = None
        for candidate in ["probe.pt", "best.pt", f"{item.name}.pt"]:
            p = item / candidate
            if p.exists():
                probe_path = p
                break

        if probe_path is None:
            continue

        try:
            models[layer_idx] = ProbeModel.load_from_checkpoint(probe_path)
        except Exception:
            pass

    return models


def load_probe_toxic_vectors(
    probes_dir: Union[str, Path] = "outputs/probes",
    num_layers: Optional[int] = None,
    prefer_normalized: bool = True,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    """
    加载所有层的毒性向量和元数据。

    优先级：
    1. toxic_vector.npz（归一化向量）
    2. 从 probe.pt / layerXX.pt 提取（weight[1] - weight[0]）

    支持多种探针目录结构：
    - 新格式（engine/probes/linear_probe_balanced.py）：
      probes_dir/probes/model_id/layer_0/probe.pt
    - 旧格式（scripts/train_linear_probe_labels.py）：
      probes_dir/linear_probes/layers/layer01/layer01.pt
      probes_dir/linear_probes/layers/layer32/layer32.pt
    - 直接格式：
      probes_dir/toxicity_vectors/all_layers_toxicity_vectors.json

    metrics.json 支持的字段格式：
    - 新格式（engine/probes/）：val_acc, val_roc_auc, val_pr_auc
    - 旧格式（scripts/train_linear_probe_labels.py）：
      avg_metrics.avg_val_acc, avg_metrics.std_val_acc,
      avg_metrics.avg_val_s_acc, avg_metrics.avg_val_h_acc

    Args:
        probes_dir: 探针目录
        num_layers: 层数上限（为 None 时不过滤，允许任意索引）
        prefer_normalized: 是否优先使用归一化的毒性向量

    Returns:
        Tuple[Dict, Dict]:
            - vectors: {layer_idx: w_toxic_array}
            - metadata: {layer_idx: {cv_accuracy, std, ...}}
    """
    import re as re_module

    probes_dir = Path(probes_dir)
    vectors = {}
    metadata = {}

    if not probes_dir.exists():
        return vectors, metadata

    # 构建所有需要扫描的 layer 目录列表
    layer_dirs = []

    # 策略1: linear_probes/layers/
    old_style = probes_dir / "linear_probes" / "layers"
    if old_style.exists():
        for item in old_style.iterdir():
            if item.is_dir() and _get_layer_index_from_name(item.name) is not None:
                layer_dirs.append(item)

    # 策略2: 直接格式（layer_XX/ 或 layerXX/）
    for item in probes_dir.iterdir():
        if item.is_dir():
            idx = _get_layer_index_from_name(item.name)
            if idx is not None and (probes_dir / "linear_probes" / "layers" / item.name).exists() is False:
                # 避免重复添加 linear_probes/layers 中的目录
                if not any(d.name == item.name for d in layer_dirs):
                    layer_dirs.append(item)

    # 去重
    seen = set()
    unique_dirs = []
    for d in layer_dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            unique_dirs.append(d)

    for item in unique_dirs:
        layer_idx = _get_layer_index_from_name(item.name)
        if layer_idx is None:
            continue
        if num_layers is not None and (layer_idx < 0 or layer_idx >= num_layers):
            continue

        meta = {}

        # 方法1：加载 toxic_vector.npz（优先）
        tox_npz_path = item / "toxic_vector.npz"
        if tox_npz_path.exists():
            try:
                tox_data = np.load(tox_npz_path, allow_pickle=True)
                if prefer_normalized and 'w_toxic_normalized' in tox_data:
                    vectors[layer_idx] = tox_data['w_toxic_normalized']
                elif 'w_toxic' in tox_data:
                    vectors[layer_idx] = tox_data['w_toxic']
                elif 'w_toxic_normalized' in tox_data:
                    vectors[layer_idx] = tox_data['w_toxic_normalized']
                meta['b'] = float(tox_data.get('b', 0.0))
            except Exception:
                pass

        # 方法2：从 probe.pt / layerXX.pt 提取毒性向量
        if layer_idx not in vectors:
            for pt_name in ["probe.pt", "best.pt", f"{item.name}.pt"]:
                pt_path = item / pt_name
                if not pt_path.exists():
                    continue
                try:
                    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
                    # 新格式：linear.weight
                    if 'linear.weight' in checkpoint:
                        w = checkpoint['linear.weight']
                        if w.shape[0] == 2:
                            w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                            vectors[layer_idx] = w_arr[1] - w_arr[0]
                    # 旧格式：fc.weight
                    elif 'fc.weight' in checkpoint:
                        w = checkpoint['fc.weight']
                        if w.shape[0] == 2:
                            w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                            vectors[layer_idx] = w_arr[1] - w_arr[0]
                    # 最简格式：weight
                    elif 'weight' in checkpoint:
                        w = checkpoint['weight']
                        if hasattr(w, 'shape') and w.shape[0] == 2:
                            w_arr = w.cpu().numpy() if hasattr(w, 'cpu') else np.array(w)
                            vectors[layer_idx] = w_arr[1] - w_arr[0]
                    break
                except Exception:
                    continue

        # 加载 metrics.json
        metrics_path = item / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

                # 新格式：val_acc（顶层字段）
                if 'val_acc' in metrics:
                    meta['cv_accuracy'] = float(metrics['val_acc'])
                    meta['val_acc'] = float(metrics['val_acc'])
                # 旧格式：avg_metrics.avg_val_acc
                elif 'avg_metrics' in metrics and 'avg_val_acc' in metrics['avg_metrics']:
                    meta['cv_accuracy'] = float(metrics['avg_metrics']['avg_val_acc'])

                # 标准差：新格式 std_val_acc / 旧格式 avg_metrics.std_val_acc
                if 'std_val_acc' in metrics:
                    meta['std'] = float(metrics['std_val_acc'])
                elif 'avg_metrics' in metrics and 'std_val_acc' in metrics['avg_metrics']:
                    meta['std'] = float(metrics['avg_metrics']['std_val_acc'])

                # 新格式指标
                if 'val_roc_auc' in metrics:
                    meta['val_roc_auc'] = float(metrics['val_roc_auc'])
                if 'val_pr_auc' in metrics:
                    meta['val_pr_auc'] = float(metrics['val_pr_auc'])

                # 旧格式：avg_metrics.avg_val_s_acc / avg_metrics.avg_val_h_acc
                if 'avg_metrics' in metrics:
                    if 'avg_val_s_acc' in metrics['avg_metrics']:
                        meta['safe_acc'] = float(metrics['avg_metrics']['avg_val_s_acc'])
                    if 'avg_val_h_acc' in metrics['avg_metrics']:
                        meta['harm_acc'] = float(metrics['avg_metrics']['avg_val_h_acc'])

            except Exception:
                pass

        if layer_idx in vectors or meta:
            metadata[layer_idx] = meta

    return vectors, metadata


# ============================================================================
# 数据集信息工具
# ============================================================================

def get_dataset_stats(
    hidden_states_dir: Union[str, Path],
    labels_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    获取数据集统计信息。
    
    Returns:
        {
            'num_samples': int,
            'num_layers': int,
            'hidden_dim': int,
            'label_distribution': {'safe': int, 'unsafe': int, 'controversial': int},
            'files': [{'name': str, 'samples': int}, ...]
        }
    """
    hidden_states_dir = Path(hidden_states_dir)
    stats = {
        'num_samples': 0,
        'num_layers': 0,
        'hidden_dim': 0,
        'label_distribution': {'safe': 0, 'unsafe': 0, 'controversial': 0},
        'files': [],
    }
    
    if not hidden_states_dir.exists():
        return stats
    
    hs_files = sorted(hidden_states_dir.glob('*.hs.npy'))
    
    for hs_path in hs_files:
        hs = np.load(hs_path, mmap_mode='r')
        n_samples = hs.shape[0]
        
        stats['num_samples'] += n_samples
        if len(hs.shape) >= 2:
            stats['num_layers'] = max(stats['num_layers'], hs.shape[1])
        if len(hs.shape) >= 3:
            stats['hidden_dim'] = max(stats['hidden_dim'], hs.shape[2])
        
        stats['files'].append({
            'name': hs_path.name,
            'samples': n_samples,
        })
        
        # 加载标签（如果提供）
        if labels_dir:
            labels_path = Path(labels_dir) / re.sub(
                r'_hidden_states', '_outputs',
                hs_path.name.replace('.hs.npy', '.jsonl')
            )
            if labels_path.exists():
                with open(labels_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            label = data.get('label', '')
                            if label == 'Safe':
                                stats['label_distribution']['safe'] += 1
                            elif label == 'Unsafe':
                                stats['label_distribution']['unsafe'] += 1
                            elif label == 'Controversial':
                                stats['label_distribution']['controversial'] += 1
                        except:
                            pass
    
    return stats


# ============================================================================
# 推荐的探针分析层
# ============================================================================

def get_recommended_probe_layers(
    metadata: Optional[Dict[int, Dict]] = None,
    probes_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    """
    根据探针训练结果获取推荐的神经元分析层。

    Args:
        metadata: 来自 load_probe_toxic_vectors() 的元数据
        probes_dir: 探针目录（如果 metadata 为 None，则从目录加载）

    Returns:
        {
            'accuracy': int,     # 最高准确率层
            'stability': int,    # 最稳定层
            'balanced': int,     # 平衡层
            'early_mid': int,    # 早中层
        }
    """
    # defaults 为 None，使用动态 fallback（在函数末尾处理）
    defaults = {
        'accuracy': None,
        'stability': None,
        'balanced': None,
        'early_mid': None,
    }

    # 如果没有提供 metadata，尝试从 probes_dir 加载
    if metadata is None and probes_dir is not None:
        _, metadata = load_probe_toxic_vectors(probes_dir)

    if not metadata:
        # 无法从探针数据推断，返回 None（让调用方决定兜底策略）
        print("[Recommended Probe Layers] 警告: 无法获取探针元数据，返回 None")
        return {k: None for k in defaults}

    # 提取准确率（兼容新旧格式）
    def _get_cv_acc(info: Dict) -> float:
        # 新格式：val_acc
        if 'cv_accuracy' in info:
            return float(info['cv_accuracy'])
        if 'val_acc' in info:
            return float(info['val_acc'])
        return 0.0

    def _get_std(info: Dict) -> float:
        # 新格式：std_val_acc / 旧格式：avg_metrics.std_val_acc
        if 'std' in info:
            return float(info['std'])
        if 'std_val_acc' in info:
            return float(info['std_val_acc'])
        return float('inf')

    # 按准确率排序
    by_acc = sorted(
        metadata.items(),
        key=lambda x: _get_cv_acc(x[1]),
        reverse=True
    )

    # 按稳定性（标准差从小到大）
    by_std = sorted(
        metadata.items(),
        key=lambda x: _get_std(x[1])
    )

    recommendations = dict(defaults)

    if by_acc:
        recommendations['accuracy'] = by_acc[0][0]
    if by_std:
        recommendations['stability'] = by_std[0][0]

    # 找平衡层：准确率 >= 90% 且稳定性较好
    balanced_candidates = [
        (layer_idx, info) for layer_idx, info in metadata.items()
        if _get_cv_acc(info) >= 0.90
    ]
    if balanced_candidates:
        # 选择标准差最小的
        best_balanced = min(balanced_candidates, key=lambda x: _get_std(x[1]))
        recommendations['balanced'] = best_balanced[0]

    # 早中层：根据实际可用层范围动态计算中间区间
    all_layer_indices = sorted(metadata.keys())
    if all_layer_indices:
        total = len(all_layer_indices)
        min_layer = min(all_layer_indices)
        max_layer = max(all_layer_indices)
        range_size = max_layer - min_layer
        early_mid_min = min_layer + int(range_size * 0.15)
        early_mid_max = max_layer - int(range_size * 0.15)
    else:
        early_mid_min, early_mid_max = 15, 25  # 保底（理论上 metadata 非空时不会执行到这里
    early_mid_candidates = [
        (layer_idx, info) for layer_idx, info in by_acc
        if early_mid_min <= layer_idx <= early_mid_max
    ]
    if early_mid_candidates:
        recommendations['early_mid'] = early_mid_candidates[0][0]

    return recommendations


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'HiddenStateDataset',
    'ProbeModel',
    'load_all_probe_models',
    'load_probe_toxic_vectors',
    'get_dataset_stats',
    'get_recommended_probe_layers',
]
