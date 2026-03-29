"""
线性探针分类器

基于softmax的线性分类器，用于识别模型各层隐藏状态中的有害语义。
使用公式：softmax(w^T * h + b)

其中：
- w_toxic ∈ R^d 为权重向量
- h ∈ R^d 为隐藏状态
- d 为隐藏维度
- b 为偏置项

训练支持两种平衡方式（可单独或同时使用）：
- 过采样：use_oversample=True 时对训练集有害类有放回抽样，使安全:有害 ≈ oversample_target_ratio:1。
- 类别权重：use_class_weight=True 时按训练集逆频率构造 [w_safe, w_toxic]，传入 CrossEntropyLoss(weight=...)。
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class HiddenStateDataset(Dataset):
    """
    隐藏状态数据集
    
    用于包装隐藏状态和标签，供 PyTorch DataLoader 使用。
    优化：预转换为 Tensor，避免每次 __getitem__ 时重复转换。
    """
    
    def __init__(
        self, 
        hidden_states: List[np.ndarray], 
        labels: List[int],
        preload_to_tensor: bool = True,
    ):
        """
        Args:
            hidden_states: 隐藏状态列表，每个元素为 (hidden_dim,) 的数组（单层）
            labels: 标签列表，0=安全，1=有害
            preload_to_tensor: 是否预转换为 Tensor（推荐 True，提高效率）
        """
        if len(hidden_states) != len(labels):
            raise ValueError(f"隐藏状态数量 ({len(hidden_states)}) 与标签数量 ({len(labels)}) 不匹配")
        
        self.labels = labels
        
        if preload_to_tensor:
            # 预转换为 Tensor，避免每次 __getitem__ 时重复转换
            # 使用 torch.from_numpy 共享内存，更高效
            self.hidden_states = [
                torch.from_numpy(hs).float() if isinstance(hs, np.ndarray) else torch.FloatTensor(hs)
                for hs in hidden_states
            ]
            self.labels_tensor = torch.LongTensor(labels)
        else:
            # 保持原始格式，延迟转换（内存占用更小，但速度较慢）
            self.hidden_states = hidden_states
            self.labels_tensor = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.labels_tensor is not None:
            # 预加载模式：直接返回 Tensor
            return {
                "hidden_states": self.hidden_states[idx],
                "label": self.labels_tensor[idx]
            }
        else:
            # 延迟转换模式：每次创建新 Tensor
            return {
                "hidden_states": torch.FloatTensor(self.hidden_states[idx]),
                "label": torch.LongTensor([self.labels[idx]])[0]
            }


class LinearProbe(nn.Module):
    """
    线性探针分类器
    
    使用softmax激活的线性层进行分类：
    P(toxic | h) = softmax(w^T * h + b)
    
    优化点：
    1. 权重初始化：使用 Xavier/Kaiming 初始化提高训练稳定性
    2. 支持混合精度训练（FP16）以加速服务器训练
    3. 添加 dropout 防止过拟合（可选）
    4. 支持保存/加载模型状态
    """
    
    def __init__(
        self, 
        hidden_dim: int,
        dropout: float = 0.0,
        init_method: str = "xavier",
    ):
        """
        Args:
            hidden_dim: 隐藏状态维度
            dropout: Dropout 概率（0.0 表示不使用，推荐 0.1-0.2）
            init_method: 权重初始化方法 ("xavier", "kaiming", "normal")
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 线性层：hidden_dim -> 2 (安全/有害)
        self.linear = nn.Linear(hidden_dim, 2)
        
        # 可选的 dropout（用于正则化）
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # 权重初始化
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        """初始化权重，提高训练稳定性"""
        if method == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif method == "normal":
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        else:
            # 使用 PyTorch 默认初始化
            pass
        
        # 偏置初始化为 0
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_state: 隐藏状态，形状为 (batch_size, hidden_dim)
        
        Returns:
            分类logits，形状为 (batch_size, 2)
        """
        # 应用 dropout（如果启用）
        if self.dropout is not None and self.training:
            hidden_state = self.dropout(hidden_state)
        
        return self.linear(hidden_state)
    
    def predict_proba(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            hidden_state: 隐藏状态
        
        Returns:
            概率分布，形状为 (batch_size, 2)
        """
        logits = self.forward(hidden_state)
        return torch.softmax(logits, dim=-1)
    
    def predict(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        预测类别（返回类别索引）
        
        Args:
            hidden_state: 隐藏状态
        
        Returns:
            预测类别，形状为 (batch_size,)
        """
        logits = self.forward(hidden_state)
        return torch.argmax(logits, dim=-1)
    
    def get_toxic_vector(self) -> Tuple[np.ndarray, float]:
        """
        获取毒性向量（权重和偏置）
        
        Returns:
            (w_toxic, b): 权重向量和偏置项
        """
        w_toxic = self.linear.weight[1].detach().cpu().numpy()  # 有害类别的权重
        b = self.linear.bias[1].item()  # 有害类别的偏置
        return w_toxic, b
    
    def save_state_dict(self, path: Path):
        """保存模型状态（用于服务器训练时的检查点）"""
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_path(self, path: Path):
        """加载模型状态（从文件路径加载）"""
        super().load_state_dict(torch.load(path, map_location='cpu'))


def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 8,
    pooling_method: str = "mean",
) -> List[np.ndarray]:
    """
    提取模型各层的隐藏状态（对非 padding token 做平均池化）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        texts: 文本列表
        device: 设备
        max_length: 最大序列长度
        batch_size: 批次大小
    
    Args:
        pooling_method: 池化方式，支持:
            - "mean": 对非 padding token 做平均池化（默认）
            - "cls": 取 CLS token（序列首 token）
            - "last_token": 取最后一个非 padding token
    
    Returns:
        隐藏状态列表，每个元素为 (num_layers, hidden_dim) 的numpy数组
    """
    model.eval()
    all_hidden_states = []
    
    # 处理批次
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[提取隐藏状态] 共 {total_batches} 个批次，{len(texts)} 个样本")
    for i in tqdm(range(0, len(texts), batch_size), desc="提取隐藏状态", total=total_batches):
        batch_texts = texts[i:i + batch_size]
        
        # 分词
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # 前向传播，获取所有层的隐藏状态
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # 提取每层的池化隐藏状态
        # hidden_states 是一个元组，包含 embedding 层和所有 transformer 层的输出
        # 形状: (num_layers + 1, batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states
        
        batch_size_actual = inputs["input_ids"].shape[0]
        
        for b in range(batch_size_actual):
            # 获取该样本的实际序列长度（排除 padding）
            seq_len = (inputs["attention_mask"][b] == 1).sum().item()
            if seq_len <= 0:
                # 理论上不会发生，做个保护，直接跳过
                continue

            # 提取每层非 padding token 的平均池化隐藏状态
            layer_hidden_states = []
            for layer_idx, layer_hs in enumerate(hidden_states):
                # layer_hs 形状: (batch_size, seq_len, hidden_dim)
                # 根据 pooling_method 计算池化表示
                if pooling_method == "mean":
                    valid_tokens = layer_hs[b, :seq_len, :]  # (seq_len, hidden_dim)
                    pooled_hs = valid_tokens.mean(dim=0).cpu().numpy()
                elif pooling_method == "cls":
                    pooled_hs = layer_hs[b, 0, :].cpu().numpy()
                elif pooling_method == "last_token":
                    pooled_hs = layer_hs[b, seq_len - 1, :].cpu().numpy()
                else:
                    raise ValueError(
                        f"不支持的 pooling_method: {pooling_method}，可选值为 'mean'、'cls'、'last_token'"
                    )
                layer_hidden_states.append(pooled_hs)

            # 堆叠为 (num_layers, hidden_dim)
            all_hidden_states.append(np.stack(layer_hidden_states))
    
    return all_hidden_states


def get_layer_training_config(
    layer_idx: int,
    num_layers: int,
    base_epochs: int = 50,
    base_lr: float = 0.002,
    base_dropout: float = 0.1,
) -> Dict[str, any]:
    """
    根据层索引获取该层的训练配置
    
    策略：
    - 浅层（0-5）：需要更多训练，更高学习率，更少dropout（语义编码不完整）
    - 中层（6-14）：标准配置
    - 深层（15+）：可以适当减少dropout，提高学习率（语义编码更清晰）
    - 第28层：特殊优化，确保达到峰值93%
    
    Args:
        layer_idx: 层索引
        num_layers: 总层数
        base_epochs: 基础训练轮数
        base_lr: 基础学习率
        base_dropout: 基础dropout
    
    Returns:
        该层的训练配置字典
    """
    if layer_idx < 6:
        # 浅层：需要更多训练，更高学习率，更少dropout
        # 配合类别权重训练，适当降低学习率以提高稳定性
        return {
            "num_epochs": int(base_epochs * 1.5),  # 120轮（base_epochs=80时）
            "learning_rate": base_lr * 1.3,  # 从1.5改为1.3，降低学习率提高稳定性（约2.6e-3）
            "dropout": base_dropout * 0.6,  # 从0.5改为0.6，略微增加dropout提高稳定性（0.06）
            "min_required_acc": 0.76,  # 最低要求76%
        }
    elif layer_idx < 15:        # 中层：略增 dropout 减轻过拟合
        return {
            "num_epochs": base_epochs,
            "learning_rate": base_lr,
            "dropout": min(0.15, base_dropout * 1.2),  # 0.12，减轻过拟合
            "min_required_acc": 0.85,
        }
    elif layer_idx >= 15:
        peak_layer = 28 if num_layers >= 32 else num_layers - 4
        if layer_idx == peak_layer:
            return {
                "num_epochs": int(base_epochs * 1.2),
                "learning_rate": base_lr * 1.1,
                "dropout": min(0.12, base_dropout * 1.2),  # 0.12，减轻过拟合
                "min_required_acc": 0.93,
            }
        else:
            # 深层：略增 dropout 减轻过拟合
            return {
                "num_epochs": base_epochs,
                "learning_rate": base_lr,
                "dropout": min(0.15, base_dropout * 1.2),  # 0.12，减轻过拟合
                "min_required_acc": 0.90,
            }
    else:
        # 默认配置
        return {
            "num_epochs": base_epochs,
            "learning_rate": base_lr,
            "dropout": base_dropout,
            "min_required_acc": 0.75,
        }


def train_layer_probes(
    hidden_states: List[np.ndarray],
    labels: List[int],
    num_layers: int,
    hidden_dim: int,
    train_indices: List[int],
    val_indices: List[int],
    device: torch.device,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    val_hidden_states: Optional[List[np.ndarray]] = None,
    val_labels: Optional[List[int]] = None,
    test_hidden_states: Optional[List[np.ndarray]] = None,
    test_labels: Optional[List[int]] = None,
    ensure_accuracy_requirements: bool = True,  # 确保达到准确率要求
    use_class_weight: bool = False,  # 若 True，按训练集类别逆频率对损失加权，平衡少数类（有害）
    use_oversample: bool = False,  # 若 True，对训练集有害类过采样，使安全:有害 ≈ oversample_target_ratio:1
    oversample_target_ratio: float = 1.5,  # 过采样目标 安全:有害 比例
    oversample_seed: int = 42,
    train_seed: int = 42,  # 训练时的随机种子（用于固定模型初始化，确保重试可复现）
) -> Dict[int, Dict]:
    """
    训练各层的线性探针分类器
    
    Args:
        hidden_states: 训练集隐藏状态列表，每个元素为 (num_layers, hidden_dim)
        labels: 训练集标签列表，0=安全，1=有害
        num_layers: 层数
        hidden_dim: 隐藏维度
        train_indices: 训练集索引（相对于 hidden_states）
        val_indices: 验证集索引（如果 val_hidden_states 为 None，则相对于 hidden_states）
        device: 设备
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        val_hidden_states: 外部验证集隐藏状态（可选）
        val_labels: 外部验证集标签（可选，必须与 val_hidden_states 一起提供）
        test_hidden_states: 测试集隐藏状态（可选）；若提供则每轮记录 test_acc 到 training_history
        test_labels: 测试集标签（可选，必须与 test_hidden_states 一起提供）
        use_class_weight: 若 True，按训练集类别逆频率构造 [w_safe, w_toxic] 并传给 CrossEntropyLoss(weight=...)
        use_oversample: 若 True，对训练集有害类过采样至安全:有害 ≈ oversample_target_ratio:1
        oversample_target_ratio: 过采样目标比例（安全:有害）
        oversample_seed: 过采样随机种子
    
    Returns:
        每层的训练结果字典，包含：
        - model: 该层的 LinearProbe 模型
        - metrics:
            - train_acc: 最后一轮训练集准确率
            - val_acc: 该层「最佳验证集准确率」（用于 early stopping 与筛选）
            - test_acc: 若提供测试集，则对应于 val_acc 同一轮的测试集准确率；否则为 None
            - val_roc_auc / val_pr_auc: 与 val_acc 同一轮的 ROC-AUC / PR-AUC
            - min_required_acc / meets_requirement: 是否达到预设准确率要求
        - toxic_vector: 毒性向量 w_toxic 与偏置 b
        - training_history: 每轮的 train/val/test 曲线与学习率变化
    """
    # 准备数据
    train_hidden = [hidden_states[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    # 可选：过采样训练集有害类，使安全:有害 ≈ oversample_target_ratio:1
    if use_oversample and len(train_labels) > 0:
        safe_idx = [i for i, l in enumerate(train_labels) if l == 0]
        toxic_idx = [i for i, l in enumerate(train_labels) if l == 1]
        n_safe, n_toxic = len(safe_idx), len(toxic_idx)
        if n_toxic > 0:
            n_toxic_target = int(round(n_safe / oversample_target_ratio))
            if n_toxic < n_toxic_target:
                rng = np.random.RandomState(oversample_seed)
                extra = rng.choice(n_toxic, size=n_toxic_target - n_toxic, replace=True)
                extra_idx = [toxic_idx[i] for i in extra]
                indices = np.array(safe_idx + toxic_idx + extra_idx, dtype=np.intp)
                rng2 = np.random.RandomState(oversample_seed + 1)
                rng2.shuffle(indices)
                indices = indices.tolist()
                train_hidden = [train_hidden[i] for i in indices]
                train_labels = [train_labels[i] for i in indices]
                print(f"[Oversample] 训练集过采样: 安全={n_safe}, 有害 {n_toxic} -> {n_toxic_target}, 安全:有害={oversample_target_ratio}:1, 总样本={len(train_labels)}")

    # 类别权重：按训练集逆频率 [w_safe, w_toxic]，传入 CrossEntropyLoss(weight=...) 以平衡安全/有害
    class_weight_tensor = None
    if use_class_weight:
        n_total = len(train_labels)
        n_safe = sum(1 for l in train_labels if l == 0)
        n_toxic = sum(1 for l in train_labels if l == 1)
        if n_safe > 0 and n_toxic > 0:
            # 逆频率：少数类权重大，使损失中少数类贡献与多数类相当
            w_safe = n_total / (2.0 * n_safe)
            w_toxic = n_total / (2.0 * n_toxic)
            class_weight_tensor = torch.tensor([w_safe, w_toxic], dtype=torch.float32, device=device)
            print(f"[Class Weight] 安全(0) n={n_safe} weight={w_safe:.4f}, 有害(1) n={n_toxic} weight={w_toxic:.4f}")
        else:
            class_weight_tensor = None
    
    # 准备验证数据（优先使用外部验证集）
    if val_hidden_states is not None and val_labels is not None:
        # 使用外部验证集（按照文档划分的探针验证集）
        val_hidden = val_hidden_states
        val_labels_list = val_labels
    else:
        # 从训练集中划分验证集（旧方式）
        val_hidden = [hidden_states[i] for i in val_indices]
        val_labels_list = [labels[i] for i in val_indices]
    
    results = {}
    
    # 对每一层训练探针
    print(f"\n[训练层探针] 共 {num_layers} 层")
    print(f"[训练配置] 基础轮数={num_epochs}, 基础学习率={learning_rate:.4f}, "
          f"批大小={batch_size}, 权重衰减={weight_decay}")
    print(f"[训练策略] 浅层(0-5): 轮数×1.5, 学习率×1.3, dropout×0.6, 要求≥76%")
    print(f"[训练策略] 中层(6-14): 标准配置, 要求≥85%")
    print(f"[训练策略] 深层(15+): 标准配置, 要求≥90%")
    print(f"[训练策略] 峰值层(28): 轮数×1.2, 学习率×1.1, 要求≥93%")
    print(f"[训练策略] 未达标时最多重试2次，每次调整超参数后重新训练\n")
    
    for layer_idx in tqdm(range(num_layers), desc="训练层探针", total=num_layers):
        # 获取该层的训练配置（分层策略）
        layer_config = get_layer_training_config(
            layer_idx=layer_idx,
            num_layers=num_layers,
            base_epochs=num_epochs,
            base_lr=learning_rate,
            base_dropout=0.1,
        )
        
        layer_epochs = layer_config["num_epochs"]
        layer_lr = layer_config["learning_rate"]
        layer_dropout = layer_config["dropout"]
        min_required_acc = layer_config["min_required_acc"]
        
        # 输出该层的训练配置信息（在训练开始前输出一次）
        layer_type = (
            "浅层" if layer_idx < 6 else
            "峰值层" if layer_idx == (28 if num_layers >= 32 else num_layers - 4) else
            "中层" if layer_idx < 15 else "深层"
        )
        print(f"\n[Layer {layer_idx:2d}] 配置: {layer_type} | 轮数={layer_epochs} | "
              f"学习率={layer_lr:.4f} | dropout={layer_dropout:.3f} | 要求准确率>={min_required_acc:.0%}")
        
        # 提取该层的隐藏状态
        train_layer_hs = [hs[layer_idx] for hs in train_hidden]
        val_layer_hs = [hs[layer_idx] for hs in val_hidden]
        test_layer_hs = None
        test_loader = None
        if test_hidden_states is not None and test_labels is not None:
            test_layer_hs = [hs[layer_idx] for hs in test_hidden_states]
            test_dataset = HiddenStateDataset(test_layer_hs, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 创建数据集
        train_dataset = HiddenStateDataset(train_layer_hs, train_labels)
        val_dataset = HiddenStateDataset(val_layer_hs, val_labels_list)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # 各集安全/有害样本数（每层固定，用于每轮日志）
        n_train_safe = sum(1 for l in train_labels if l == 0)
        n_train_toxic = sum(1 for l in train_labels if l == 1)
        n_val_safe = sum(1 for l in val_labels_list if l == 0)
        n_val_toxic = sum(1 for l in val_labels_list if l == 1)
        n_test_safe = sum(1 for l in test_labels if l == 0) if (test_labels is not None and test_loader is not None) else None
        n_test_toxic = sum(1 for l in test_labels if l == 1) if (test_labels is not None and test_loader is not None) else None
        
        # 训练循环（如果未达到要求，会重试）
        # 重试机制：最多重试2次，每次重试时会调整超参数（学习率+10%, 轮数+15%, dropout-10%）
        # 重试时会在日志中显示调整后的超参数
        max_retries = 2 if ensure_accuracy_requirements else 0
        retry_count = 0
        best_val_acc = 0.0          # 该层全局最佳验证准确率（跨重试）
        best_val_epoch = 0          # 达到最佳验证准确率的轮数（跨重试）
        best_probe_state = None     # 与 best_val_acc 对应的模型参数
        best_val_roc_auc = 0.0      # 与 best_val_acc 对应的 ROC-AUC
        best_val_pr_auc = 0.0       # 与 best_val_acc 对应的 PR-AUC
        best_test_acc = None        # 若提供测试集：与 best_val_acc 同一轮的测试集准确率
        best_test_epoch = None      # 若提供测试集：达到最佳测试集准确率的轮数（与 best_val_acc 对应）
        best_train_acc = 0.0        # 该层全局最佳训练准确率（跨重试）
        best_train_epoch = 0        # 达到最佳训练准确率的轮数（跨重试）

        while retry_count <= max_retries:
            # 固定随机种子，确保每次重试的初始化可复现（基于层索引和重试次数）
            retry_seed = train_seed + layer_idx * 1000 + retry_count * 100
            random.seed(retry_seed)
            np.random.seed(retry_seed)
            torch.manual_seed(retry_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(retry_seed)
            
            # 创建模型（使用该层的配置）
            probe = LinearProbe(
                hidden_dim=hidden_dim,
                dropout=layer_dropout,
                init_method="xavier",
            ).to(device)
            
            criterion = nn.CrossEntropyLoss(weight=class_weight_tensor) if class_weight_tensor is not None else nn.CrossEntropyLoss()
            # 服务器训练优化：使用AdamW优化器（权重衰减更稳定）
            optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=layer_lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            
            # 服务器训练优化：使用学习率调度器（ReduceLROnPlateau）
            # 调整 patience 为 8，给模型更多时间提升（配合类别权重训练，需要更多耐心）
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=8,  # 从5增加到8，给模型更多时间提升
                verbose=False,
                min_lr=1e-5,
            )
            
            # 服务器训练优化：使用混合精度训练（如果支持）
            use_amp = device.type == 'cuda' and hasattr(torch.amp, 'GradScaler')
            scaler = torch.amp.GradScaler('cuda') if use_amp else None
            
            # 训练
            best_val_acc_this_try = 0.0
            best_probe_state_this_try = None
            best_val_roc_auc_this_try = 0.0
            best_val_pr_auc_this_try = 0.0
            best_test_acc_this_try = None  # 本次重试中，与 best_val_acc_this_try 对应的测试集准确率
            best_val_epoch_this_try = 0  # 本次重试中，达到最佳验证集准确率的轮数
            best_test_epoch_this_try = None  # 本次重试中，达到最佳测试集准确率的轮数（与 best_val_acc_this_try 对应）
            best_train_acc_this_try = 0.0  # 本次重试中，训练集的最高准确率
            best_train_epoch_this_try = 0  # 本次重试中，达到最佳训练集准确率的轮数
            no_improve_count = 0

            # 记录训练历史（用于生成曲线）
            training_history = {
                "epochs": [],
                "train_acc": [],
                "val_acc": [],
                "test_acc": [],
                "train_loss": [],
                "val_loss": [],
                "learning_rate": [],
            }
            
            # 训练循环：每轮计算训练集、验证集和测试集（如果提供）的准确率
            # 基于验证集准确率选择最佳模型、调整学习率、判断早停
            for epoch in tqdm(range(layer_epochs), desc=f"  Layer {layer_idx} Epochs", leave=False):
                # 训练阶段：前向传播、计算损失、反向传播
                probe.train()
                train_loss = 0.0
                train_preds = []
                train_targets = []
                
                for batch in train_loader:
                    hs = batch["hidden_states"].to(device, non_blocking=True)  # 异步传输
                    label = batch["label"].to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    # 混合精度训练（服务器优化）
                    if use_amp and scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits = probe(hs)
                            loss = criterion(logits, label)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = probe(hs)
                        loss = criterion(logits, label)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    train_targets.extend(label.cpu().numpy())
                
                # 验证阶段
                probe.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                val_probs = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        hs = batch["hidden_states"].to(device)
                        label = batch["label"].to(device)
                        
                        logits = probe(hs)
                        loss = criterion(logits, label)
                        probs = probe.predict_proba(hs)
                        
                        val_loss += loss.item()
                        val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                        val_targets.extend(label.cpu().numpy())
                        val_probs.extend(probs[:, 1].cpu().numpy())  # 有害类别的概率
                
                # 计算指标
                train_acc = accuracy_score(train_targets, train_preds)
                val_acc = accuracy_score(val_targets, val_preds)

                # 测试集准确率（若提供测试集则每轮计算并记录）
                test_acc = None
                if test_loader is not None:
                    test_preds = []
                    test_targets = []
                    with torch.no_grad():
                        for batch in test_loader:
                            hs = batch["hidden_states"].to(device)
                            label = batch["label"].to(device)
                            logits = probe(hs)
                            test_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                            test_targets.extend(label.cpu().numpy())
                    test_acc = float(accuracy_score(test_targets, test_preds))

                # 计算ROC-AUC和PR-AUC（验证集需同时含安全/有害）
                if len(set(val_targets)) > 1:
                    try:
                        val_roc_auc = roc_auc_score(val_targets, val_probs)
                        precision, recall, _ = precision_recall_curve(val_targets, val_probs)
                        val_pr_auc = auc(recall, precision)
                    except ValueError:
                        val_roc_auc = 0.0
                        val_pr_auc = 0.0
                else:
                    val_roc_auc = 0.0
                    val_pr_auc = 0.0

                # 学习率调度（基于验证准确率）
                scheduler.step(val_acc)

                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 记录训练历史
                training_history["epochs"].append(epoch + 1)
                training_history["train_acc"].append(float(train_acc))
                training_history["val_acc"].append(float(val_acc))
                training_history["test_acc"].append(test_acc if test_acc is not None else None)
                training_history["train_loss"].append(float(train_loss / len(train_loader)))
                training_history["val_loss"].append(float(val_loss / len(val_loader)))
                training_history["learning_rate"].append(float(current_lr))

                # 更新训练集最高准确率
                if train_acc > best_train_acc_this_try:
                    best_train_acc_this_try = train_acc
                    best_train_epoch_this_try = epoch + 1
                
                # 训练日志：每轮输出本轮准确率、以及之前最高准确率及其轮数
                log_parts = [
                    f"[Layer {layer_idx:2d}] Epoch {epoch + 1:3d}/{layer_epochs}",
                    f"训练集 本轮={train_acc:.2%} 最高={best_train_acc_this_try:.2%}@E{best_train_epoch_this_try}",
                    f"验证集 本轮={val_acc:.2%} 最高={best_val_acc_this_try:.2%}@E{best_val_epoch_this_try}",
                ]
                if test_acc is not None:
                    # 显示本轮测试集准确率和最高测试集准确率（与最高验证集准确率对应的那一轮）及其轮数
                    if best_test_acc_this_try is not None and best_test_epoch_this_try is not None:
                        log_parts.append(f"测试集 本轮={test_acc:.2%} 最高={best_test_acc_this_try:.2%}@E{best_test_epoch_this_try}")
                    else:
                        log_parts.append(f"测试集 本轮={test_acc:.2%} 最高=--")
                print(" | ".join(log_parts))

                # 保存最佳模型（同时记录该轮 roc/pr auc，便于结果与最佳模型一致）
                if val_acc > best_val_acc_this_try:
                    # 记录「本次重试」中验证集表现最好的那一轮，
                    # 并同步记录该轮对应的 ROC-AUC / PR-AUC / 测试集准确率（若存在）及其轮数
                    best_val_acc_this_try = val_acc
                    best_val_epoch_this_try = epoch + 1  # 记录达到最佳验证集准确率的轮数
                    best_probe_state_this_try = probe.state_dict().copy()
                    best_val_roc_auc_this_try = val_roc_auc
                    best_val_pr_auc_this_try = val_pr_auc
                    # 当验证集准确率提升时，同步更新测试集准确率及其轮数（如果提供测试集）
                    if test_acc is not None:
                        best_test_acc_this_try = test_acc
                        best_test_epoch_this_try = epoch + 1  # 记录达到最佳测试集准确率的轮数
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # 更新全局最佳（含该轮的 roc/pr auc 和轮数）
                if best_val_acc_this_try > best_val_acc:
                    # 更新「全局最佳」状态（跨所有重试）
                    best_val_acc = best_val_acc_this_try
                    best_val_epoch = best_val_epoch_this_try
                    best_probe_state = best_probe_state_this_try
                    best_val_roc_auc = best_val_roc_auc_this_try
                    best_val_pr_auc = best_val_pr_auc_this_try
                    best_test_acc = best_test_acc_this_try
                    best_test_epoch = best_test_epoch_this_try
                
                # 更新全局最佳训练准确率（独立于验证集最佳）
                if best_train_acc_this_try > best_train_acc:
                    best_train_acc = best_train_acc_this_try
                    best_train_epoch = best_train_epoch_this_try

                # 早停机制（减轻过拟合）：
                # 1. 达标后连续10轮验证集无提升 → 提前停止（已达标，避免过训练）
                # 2. 连续20轮验证集无提升 → 提前停止（防止过拟合）
                # 注意：早停基于验证集准确率，不基于训练集准确率
                if ensure_accuracy_requirements and best_val_acc_this_try >= min_required_acc and no_improve_count >= 10:
                    if retry_count == 0:
                        print(f"[Layer {layer_idx}] 已达到要求 ({best_val_acc_this_try:.2%} >= {min_required_acc:.0%})，提前停止训练")
                    # 确保全局最佳状态已更新
                    if best_val_acc_this_try > best_val_acc:
                        best_val_acc = best_val_acc_this_try
                        best_val_epoch = best_val_epoch_this_try
                        best_probe_state = best_probe_state_this_try
                        best_val_roc_auc = best_val_roc_auc_this_try
                        best_val_pr_auc = best_val_pr_auc_this_try
                        best_test_acc = best_test_acc_this_try
                        best_test_epoch = best_test_epoch_this_try
                    # 更新训练集最佳
                    if best_train_acc_this_try > best_train_acc:
                        best_train_acc = best_train_acc_this_try
                        best_train_epoch = best_train_epoch_this_try
                    break
                if no_improve_count >= 20:
                    if retry_count == 0:
                        print(f"[Layer {layer_idx}] 连续 20 轮验证无提升，提前停止（减轻过拟合）")
                    # 确保全局最佳状态已更新
                    if best_val_acc_this_try > best_val_acc:
                        best_val_acc = best_val_acc_this_try
                        best_val_epoch = best_val_epoch_this_try
                        best_probe_state = best_probe_state_this_try
                        best_val_roc_auc = best_val_roc_auc_this_try
                        best_val_pr_auc = best_val_pr_auc_this_try
                        best_test_acc = best_test_acc_this_try
                        best_test_epoch = best_test_epoch_this_try
                    # 更新训练集最佳
                    if best_train_acc_this_try > best_train_acc:
                        best_train_acc = best_train_acc_this_try
                        best_train_epoch = best_train_epoch_this_try
                    break

                # 检查是否达到准确率要求（至少训练10轮后再判断，避免前几轮随机初始化影响）
                # 如果未达标且未达到最大重试次数，会调整超参数后重试
                min_epochs_before_check = 10  # 至少训练10轮后再判断是否达到要求（提高稳定性）
                if ensure_accuracy_requirements and epoch + 1 >= min_epochs_before_check and best_val_acc_this_try < min_required_acc:
                    # 确保全局最佳状态已更新（即使未达到要求，也要保存当前最佳）
                    if best_val_acc_this_try > best_val_acc:
                        best_val_acc = best_val_acc_this_try
                        best_val_epoch = best_val_epoch_this_try
                        best_probe_state = best_probe_state_this_try
                        best_val_roc_auc = best_val_roc_auc_this_try
                        best_val_pr_auc = best_val_pr_auc_this_try
                        best_test_acc = best_test_acc_this_try
                        best_test_epoch = best_test_epoch_this_try
                    # 更新训练集最佳
                    if best_train_acc_this_try > best_train_acc:
                        best_train_acc = best_train_acc_this_try
                        best_train_epoch = best_train_epoch_this_try
                    
                    if retry_count < max_retries:
                        retry_count += 1
                        # 超参数调整策略（更温和，避免过度调整）：
                        # - 学习率增加10%（帮助模型跳出局部最优）
                        # - 训练轮数增加15%（给模型更多训练时间）
                        # - dropout减少10%（降低正则化强度，允许模型学习更复杂的模式）
                        layer_lr *= 1.1  # 学习率增加10%
                        layer_epochs = int(layer_epochs * 1.15)  # 训练轮数增加15%
                        layer_dropout = max(0.01, layer_dropout * 0.9)  # dropout减少10%，但不低于0.01
                        print(f"[Layer {layer_idx}] ⚠ 训练 {epoch + 1} 轮后验证集准确率 {best_val_acc_this_try:.2%} < 要求 {min_required_acc:.0%}，"
                              f"重试 {retry_count}/{max_retries}（调整后: lr={layer_lr:.4f}, epochs={layer_epochs}, dropout={layer_dropout:.3f}）")
                        break  # 退出 for epoch，进入下一轮 while 重试
                    else:
                        print(f"[Layer {layer_idx}] ⚠ 警告: 训练 {epoch + 1} 轮后准确率 {best_val_acc_this_try:.2%} < 要求 {min_required_acc:.0%}，"
                              "已达到最大重试次数，使用当前最佳模型")
                        break
                elif ensure_accuracy_requirements and epoch + 1 >= min_epochs_before_check and best_val_acc_this_try >= min_required_acc:
                    # 达到要求，确保全局最佳状态已更新
                    if best_val_acc_this_try > best_val_acc:
                        best_val_acc = best_val_acc_this_try
                        best_val_epoch = best_val_epoch_this_try
                        best_probe_state = best_probe_state_this_try
                        best_val_roc_auc = best_val_roc_auc_this_try
                        best_val_pr_auc = best_val_pr_auc_this_try
                        best_test_acc = best_test_acc_this_try
                        best_test_epoch = best_test_epoch_this_try
                    # 更新训练集最佳
                    if best_train_acc_this_try > best_train_acc:
                        best_train_acc = best_train_acc_this_try
                        best_train_epoch = best_train_epoch_this_try
                    
                    if retry_count > 0:
                        print(f"[Layer {layer_idx}] ✓ 重试成功: 训练 {epoch + 1} 轮后准确率 {best_val_acc_this_try:.2%} >= 要求 {min_required_acc:.0%}")
                    break
        
        # 加载最佳模型（基于验证集准确率选择的最佳模型）
        # 注意：这里加载的是验证集表现最好的模型，而不是训练集表现最好的模型
        if best_probe_state is not None:
            probe.load_state_dict(best_probe_state)
        else:
            print(f"[Layer {layer_idx}] ⚠ 警告: 使用最后一次训练的模型（未找到最佳模型）")

        w_toxic, b = probe.get_toxic_vector()

        # 检查是否达到要求
        layer_config = get_layer_training_config(
            layer_idx=layer_idx,
            num_layers=num_layers,
            base_epochs=num_epochs,
            base_lr=learning_rate,
            base_dropout=0.1,
        )
        min_required_acc = layer_config["min_required_acc"]

        # 是否分别在训练集 / 验证集 / 测试集上达到要求
        # - 训练集：如果 best_train_epoch 为空（极端情况），默认视为达标，避免因为记录问题导致误判
        # - 验证集：必须达到 min_required_acc（核心判据）
        # - 测试集：如果存在测试集指标，则同样要求达到 min_required_acc；若不存在测试集，则不约束
        train_ok = (best_train_epoch is None) or (best_train_acc >= min_required_acc)
        val_ok = best_val_acc >= min_required_acc
        if best_test_acc is None or best_test_epoch is None:
            test_ok = True
        else:
            test_ok = best_test_acc >= min_required_acc
        meets_requirement = train_ok and val_ok and test_ok

        # 计算最后一轮训练准确率（用于记录）
        # 注意：这里使用最后一轮的训练准确率，但最佳训练准确率已保存在 best_train_acc 中
        final_train_acc = train_acc  # 最后一轮的训练准确率
        
        # 保存结果（val_roc_auc/val_pr_auc/test_acc 为达到 best_val_acc 那一轮的值）
        results[layer_idx] = {
            "model": probe,
            "metrics": {
                "train_acc": final_train_acc,  # 最后一轮训练准确率
                "train_acc_best": best_train_acc,  # 最佳训练准确率
                "train_epoch_best": best_train_epoch,  # 达到最佳训练准确率的轮数
                "val_acc": best_val_acc,
                "val_epoch": best_val_epoch,  # 达到最佳验证集准确率的轮数
                "val_roc_auc": best_val_roc_auc,
                "val_pr_auc": best_val_pr_auc,
                "test_acc": best_test_acc,
                "test_epoch": best_test_epoch,  # 达到最佳测试集准确率的轮数（与 best_val_acc 对应）
                "min_required_acc": min_required_acc,
                "meets_requirement": meets_requirement,
            },
            "toxic_vector": {
                "w_toxic": w_toxic,
                "b": b,
            },
            "training_history": training_history,  # 添加训练历史
        }
        
        # 打印该层的训练结果（包含训练集、验证集与测试集的最终指标及达到最佳准确率的轮数）
        status = "✓" if meets_requirement else "✗"
        log_parts = [
            f"[Layer {layer_idx:2d}] {status}",
            f"训练集最佳准确率: {best_train_acc:.2%}@E{best_train_epoch}",
            f"验证集最佳准确率: {best_val_acc:.2%}@E{best_val_epoch} (要求: {min_required_acc:.0%}, {'达标' if meets_requirement else '未达标'})",
        ]
        if best_test_acc is not None and best_test_epoch is not None:
            log_parts.append(f"测试集最佳准确率: {best_test_acc:.2%}@E{best_test_epoch}")
        print(" | ".join(log_parts))
    
    return results


def save_probes(
    results: Dict[int, Dict],
    output_dir: Path,
    model_id: str = "llama-3-8b",
    filter_threshold: float = 0.75,
):
    """
    保存探针模型和毒性向量
    
    Args:
        results: 训练结果字典
        output_dir: 输出目录
        model_id: 模型ID
        filter_threshold: 准确率阈值，低于此值的浅层探针将被标记为无效（默认0.75，即75%）
    """
    output_dir = Path(output_dir) / "probes" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计无效层
    invalid_layers = []
    valid_layers = []
    
    # 保存每层的探针
    for layer_idx, result in results.items():
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        
        val_acc = result["metrics"]["val_acc"]
        
        # 自动过滤准确率 < 75% 的浅层探针（按照论文要求）
        is_invalid = val_acc < filter_threshold
        
        if is_invalid:
            invalid_layers.append(layer_idx)
            # 仍然保存，但标记为无效层
            print(f"[Filter] Layer {layer_idx}: 准确率 {val_acc:.2%} < {filter_threshold:.0%}，标记为无效层")
        else:
            valid_layers.append(layer_idx)
        
        # 保存模型
        torch.save(result["model"].state_dict(), layer_dir / "probe.pt")
        
        # 保存毒性向量（归一化）
        w_toxic = result["toxic_vector"]["w_toxic"]
        b = result["toxic_vector"]["b"]
        
        # 归一化毒性向量（L2归一化）
        w_norm = np.linalg.norm(w_toxic)
        if w_norm > 0:
            w_toxic_normalized = w_toxic / w_norm
        else:
            w_toxic_normalized = w_toxic
        
        np.savez(
            layer_dir / "toxic_vector.npz",
            w_toxic=w_toxic_normalized,  # 归一化后的毒性向量
            b=b,
            w_toxic_original=w_toxic,  # 保留原始向量
        )
        
        # 保存指标（包含是否无效的标记）
        metrics = result["metrics"].copy()
        metrics["is_invalid"] = is_invalid
        metrics["filter_threshold"] = filter_threshold
        
        with open(layer_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史（训练日志）
        if "training_history" in result:
            training_history = result["training_history"]
            with open(layer_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # 保存汇总信息（包含无效层标记）
    summary = {
        "model_id": model_id,
        "num_layers": len(results),
        "filter_threshold": filter_threshold,
        "invalid_layers": invalid_layers,
        "valid_layers": valid_layers,
        "layers": {
            str(layer_idx): {
                "val_acc": result["metrics"]["val_acc"],
                "val_roc_auc": result["metrics"]["val_roc_auc"],
                "val_pr_auc": result["metrics"]["val_pr_auc"],
                "is_invalid": result["metrics"]["val_acc"] < filter_threshold,
            }
            for layer_idx, result in results.items()
        },
    }
    
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 统计准确率要求达成情况
    requirement_stats = {
        "meets_requirement": [],
        "fails_requirement": [],
    }
    for layer_idx, result in results.items():
        metrics = result["metrics"]
        if metrics.get("meets_requirement", True):
            requirement_stats["meets_requirement"].append(layer_idx)
        else:
            requirement_stats["fails_requirement"].append((layer_idx, metrics["val_acc"], metrics.get("min_required_acc", 0.75)))
    
    # 打印统计信息
    print(f"\n[Summary] 探针已保存到: {output_dir}")
    print(f"[Summary] 总层数: {len(results)}")
    print(f"[Summary] 有效层: {len(valid_layers)} (准确率 >= {filter_threshold:.0%})")
    print(f"[Summary] 无效层: {len(invalid_layers)} (准确率 < {filter_threshold:.0%})")
    if invalid_layers:
        print(f"[Summary] 无效层列表: {invalid_layers}")
    
    # 打印准确率要求达成情况
    print(f"\n[Accuracy Requirements]")
    print(f"  达标层数: {len(requirement_stats['meets_requirement'])}/{len(results)}")
    if requirement_stats["fails_requirement"]:
        print(f"  ⚠ 未达标层数: {len(requirement_stats['fails_requirement'])}")
        for layer_idx, acc, req in requirement_stats["fails_requirement"]:
            print(f"    Layer {layer_idx}: {acc:.2%} < {req:.0%} (差距: {req - acc:.2%})")
    else:
        print(f"  ✓ 所有层均达到准确率要求！")
    
    # 打印准确率曲线关键点和训练统计
    accuracies = [(layer_idx, result["metrics"]["val_acc"]) for layer_idx, result in sorted(results.items())]
    train_accuracies = [(layer_idx, result["metrics"].get("train_acc_best", result["metrics"]["train_acc"])) 
                        for layer_idx, result in sorted(results.items())]
    
    if accuracies:
        min_acc_layer, min_acc = min(accuracies, key=lambda x: x[1])
        max_acc_layer, max_acc = max(accuracies, key=lambda x: x[1])
        layer_15_acc = next((acc for layer, acc in accuracies if layer == 15), None)
        layer_28_acc = next((acc for layer, acc in accuracies if layer == 28), None)
        
        # 计算平均准确率
        avg_val_acc = sum(acc for _, acc in accuracies) / len(accuracies)
        avg_train_acc = sum(acc for _, acc in train_accuracies) / len(train_accuracies) if train_accuracies else None
        
        print(f"\n[Accuracy Curve]")
        print(f"  最低验证准确率: Layer {min_acc_layer} = {min_acc:.2%}")
        print(f"  平均验证准确率: {avg_val_acc:.2%}")
        if avg_train_acc is not None:
            print(f"  平均训练准确率: {avg_train_acc:.2%} (过拟合程度: {avg_train_acc - avg_val_acc:.2%})")
        if layer_15_acc:
            status_15 = "✓" if layer_15_acc >= 0.90 else "✗"
            print(f"  Layer 15验证准确率: {layer_15_acc:.2%} {status_15} (要求: >=90%)")
        if layer_28_acc:
            status_28 = "✓" if layer_28_acc >= 0.93 else "✗"
            print(f"  Layer 28验证准确率: {layer_28_acc:.2%} {status_28} (要求: >=93%, 峰值层)")
        print(f"  最高验证准确率: Layer {max_acc_layer} = {max_acc:.2%}")
    
    # 保存完整的训练日志（所有层）
    all_training_log = {
        "model_id": model_id,
        "num_layers": len(results),
        "layers": {
            str(layer_idx): {
                "metrics": result["metrics"],
                "training_history": result.get("training_history", {}),
            }
            for layer_idx, result in results.items()
        },
    }
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(all_training_log, f, indent=2, ensure_ascii=False)
    
    print(f"[Logs] 训练日志已保存到: {output_dir / 'training_log.json'}")

