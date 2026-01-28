"""
线性探针分类器

基于softmax的线性分类器，用于识别模型各层隐藏状态中的有害语义。
使用公式：softmax(w^T * h + b)

其中：
- w_toxic ∈ R^d 为权重向量
- h ∈ R^d 为隐藏状态
- d 为隐藏维度
- b 为偏置项
"""

import json
import pickle
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
                # 取该样本前 seq_len 个 token，然后在 token 维度做平均池化
                valid_tokens = layer_hs[b, :seq_len, :]  # (seq_len, hidden_dim)
                pooled_hs = valid_tokens.mean(dim=0).cpu().numpy()
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
        return {
            "num_epochs": int(base_epochs * 1.5),  # 75轮
            "learning_rate": base_lr * 1.5,  # 3e-3
            "dropout": base_dropout * 0.5,  # 0.05
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
    use_class_weight: bool = True,  # 对有害类（少数类）提高权重，逆频率
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
        use_class_weight: 是否对有害类（少数类）逆频率加权，默认 True
    
    Returns:
        每层的训练结果字典，包含模型、指标和毒性向量
    """
    # 准备数据
    train_hidden = [hidden_states[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    # 类别权重：对有害类（少数类）提高权重，逆频率 w_i = n_total / (2 * n_i)，上限 5.0
    class_weight_tensor = None
    if use_class_weight:
        n_safe = sum(1 for l in train_labels if l == 0)
        n_toxic = sum(1 for l in train_labels if l == 1)
        n_total = len(train_labels)
        w_safe = n_total / (2.0 * max(1, n_safe))
        w_toxic = n_total / (2.0 * max(1, n_toxic))
        w_safe = min(5.0, w_safe)
        w_toxic = min(5.0, w_toxic)
        class_weight_tensor = torch.tensor([w_safe, w_toxic], dtype=torch.float32)
        print(f"[Class Weight] 安全(0)={n_safe} weight={w_safe:.3f}, 有害(1)={n_toxic} weight={w_toxic:.3f}")
    
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
    print(f"[训练层探针] 共 {num_layers} 层")
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
        
        # 训练循环（如果未达到要求，会重试）
        max_retries = 2 if ensure_accuracy_requirements else 0
        retry_count = 0
        best_val_acc = 0.0
        best_probe_state = None
        best_val_roc_auc = 0.0
        best_val_pr_auc = 0.0

        while retry_count <= max_retries:
            # 创建模型（使用该层的配置）
            probe = LinearProbe(
                hidden_dim=hidden_dim,
                dropout=layer_dropout,
                init_method="xavier",
            ).to(device)
            
            criterion = (
                nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
                if class_weight_tensor is not None
                else nn.CrossEntropyLoss()
            )
            # 服务器训练优化：使用AdamW优化器（权重衰减更稳定）
            optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=layer_lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            
            # 服务器训练优化：使用学习率调度器（ReduceLROnPlateau）
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
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
            
            for epoch in tqdm(range(layer_epochs), desc=f"  Layer {layer_idx} Epochs", leave=False):
                # 训练阶段
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

                # 保存最佳模型（同时记录该轮 roc/pr auc，便于结果与最佳模型一致）
                if val_acc > best_val_acc_this_try:
                    best_val_acc_this_try = val_acc
                    best_probe_state_this_try = probe.state_dict().copy()
                    best_val_roc_auc_this_try = val_roc_auc
                    best_val_pr_auc_this_try = val_pr_auc
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # 早停（减轻过拟合）：达标后 6 轮无提升即停；或任意连续 15 轮无提升即停
                if ensure_accuracy_requirements and best_val_acc_this_try >= min_required_acc and no_improve_count >= 6:
                    if retry_count == 0:
                        print(f"[Layer {layer_idx}] 已达到要求 ({best_val_acc_this_try:.2%} >= {min_required_acc:.0%})，提前停止训练")
                    break
                if no_improve_count >= 15:
                    if retry_count == 0:
                        print(f"[Layer {layer_idx}] 连续 15 轮验证无提升，提前停止（减轻过拟合）")
                    break

                # 更新全局最佳（含该轮的 roc/pr auc）
                if best_val_acc_this_try > best_val_acc:
                    best_val_acc = best_val_acc_this_try
                    best_probe_state = best_probe_state_this_try
                    best_val_roc_auc = best_val_roc_auc_this_try
                    best_val_pr_auc = best_val_pr_auc_this_try

                # 检查是否达到要求
                if ensure_accuracy_requirements and best_val_acc_this_try < min_required_acc:
                    if retry_count < max_retries:
                        retry_count += 1
                        layer_lr *= 1.2
                        layer_epochs = int(layer_epochs * 1.2)
                        layer_dropout *= 0.8
                        print(f"[Layer {layer_idx}] 准确率 {best_val_acc_this_try:.2%} < 要求 {min_required_acc:.0%}，"
                              f"重试 {retry_count}/{max_retries}（lr={layer_lr:.4f}, epochs={layer_epochs}, dropout={layer_dropout:.3f}）")
                        break  # 退出 for epoch，进入下一轮 while 重试
                    else:
                        print(f"[Layer {layer_idx}] ⚠ 警告: 准确率 {best_val_acc_this_try:.2%} < 要求 {min_required_acc:.0%}，"
                              "已达到最大重试次数，使用当前最佳模型")
                        break
                else:
                    if retry_count > 0:
                        print(f"[Layer {layer_idx}] ✓ 重试成功: 准确率 {best_val_acc_this_try:.2%} >= 要求 {min_required_acc:.0%}")
                    break
        
        # 加载最佳模型
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
        meets_requirement = best_val_acc >= min_required_acc

        # 保存结果（val_roc_auc/val_pr_auc 为达到 best_val_acc 那一轮的值）
        results[layer_idx] = {
            "model": probe,
            "metrics": {
                "train_acc": train_acc,
                "val_acc": best_val_acc,
                "val_roc_auc": best_val_roc_auc,
                "val_pr_auc": best_val_pr_auc,
                "min_required_acc": min_required_acc,
                "meets_requirement": meets_requirement,
            },
            "toxic_vector": {
                "w_toxic": w_toxic,
                "b": b,
            },
            "training_history": training_history,  # 添加训练历史
        }
        
        # 打印该层的训练结果
        status = "✓" if meets_requirement else "✗"
        print(f"[Layer {layer_idx:2d}] {status} 准确率: {best_val_acc:.2%} "
              f"(要求: {min_required_acc:.0%}, {'达标' if meets_requirement else '未达标'})")
    
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
    
    # 打印准确率曲线关键点
    accuracies = [(layer_idx, result["metrics"]["val_acc"]) for layer_idx, result in sorted(results.items())]
    if accuracies:
        min_acc_layer, min_acc = min(accuracies, key=lambda x: x[1])
        max_acc_layer, max_acc = max(accuracies, key=lambda x: x[1])
        layer_15_acc = next((acc for layer, acc in accuracies if layer == 15), None)
        layer_28_acc = next((acc for layer, acc in accuracies if layer == 28), None)
        
        print(f"\n[Accuracy Curve]")
        print(f"  最低准确率: Layer {min_acc_layer} = {min_acc:.2%}")
        if layer_15_acc:
            status_15 = "✓" if layer_15_acc >= 0.90 else "✗"
            print(f"  Layer 15准确率: {layer_15_acc:.2%} {status_15} (要求: >=90%)")
        if layer_28_acc:
            status_28 = "✓" if layer_28_acc >= 0.93 else "✗"
            print(f"  Layer 28准确率: {layer_28_acc:.2%} {status_28} (要求: >=93%, 峰值)")
        print(f"  最高准确率: Layer {max_acc_layer} = {max_acc:.2%}")
    
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

