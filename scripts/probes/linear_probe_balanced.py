"""
线性探针分类器 & 隐藏态提取 —— prompt 意图分类（安全 vs 有害）

任务：根据 LLM 各层隐藏态，判断输入 prompt 的语义意图
标签：guard.asr_label  0 = 安全意图（prompt 本身无害）  1 = 有害意图（prompt 含攻击/有毒内容）
数据：logs/clean_intent_dataset_10k.jsonl

公式：P(harmful_intent | h) = softmax(w^T * h + b)

提供：
  - LinearProbe        线性探针模型（与 train_probes_balanced.py 接口一致）
  - extract_hidden_states  从 LLM 提取各层隐藏态
  - load_probe         从 probe.pt + preprocessor.pkl 加载已训练的探针
  - get_layer_target   各层目标准确率
  - LAYER_TARGETS      目标配置

接口约定（与 train_probes_balanced.py 一致）：
  - LinearProbe(input_dim, dropout=0.1)
  - probe.pt 保存格式: torch.save(probe.state_dict(), path)
  - preprocessor.pkl 保存格式: pickle.dump({"scaler": StandardScaler}, f)
  - 隐藏态池化: last_token（最后一个非 padding token）
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ======================================================================
# 各层目标（验证集总体准确率）
# ======================================================================
LAYER_TARGETS = {
    "浅层":  0.76,   # 0-5
    "中层":  0.85,   # 6-14
    "深层":  0.90,   # 15-27
    "峰值层": 0.93,  # 28
    "最深层": 0.90,  # 29+
}


def get_layer_target(layer_idx: int, num_layers: int) -> Tuple[float, str]:
    """
    返回 (目标总体准确率, 层类型)

    Args:
        layer_idx: 层索引 (0-based)
        num_layers: 模型总层数

    Returns:
        (target_acc, layer_type)
    """
    if layer_idx < 6:
        return 0.76, "浅层"
    if layer_idx < 15:
        return 0.85, "中层"
    peak = 28 if num_layers >= 32 else num_layers - 4
    if layer_idx == peak:
        return 0.93, "峰值层"
    if layer_idx >= 29:
        return 0.90, "最深层"
    return 0.90, "深层"


# ======================================================================
# 线性探针：softmax(w^T * h + b)
# ======================================================================
class LinearProbe(nn.Module):
    """
    线性探针分类器 —— prompt 意图分类

    P(harmful_intent | h) = softmax(w^T * h + b)

    输出: [P(safe_intent), P(harmful_intent)]
    标签: 0=安全意图, 1=有害意图（对应 guard.asr_label）
    结构: Dropout(可选) -> Linear(input_dim, 2)
    初始化: Xavier uniform + bias=0
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: 隐藏状态维度
            dropout: Dropout 概率（0 表示不使用）
        """
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear = nn.Linear(input_dim, 2)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 logits (batch_size, 2)"""
        if self.dropout is not None and self.training:
            x = self.dropout(x)
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回概率分布 (batch_size, 2)"""
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """返回预测类别 (batch_size,)"""
        return torch.argmax(self.forward(x), dim=-1)

    def get_harmful_vector(self) -> Tuple[np.ndarray, float]:
        """
        获取有害意图向量（类别 1 的权重方向）

        Returns:
            (w_harmful, b): 有害意图类别的权重向量 (input_dim,) 和偏置标量
        """
        w_harmful = self.linear.weight[1].detach().cpu().numpy()
        b = self.linear.bias[1].item()
        return w_harmful, b

    # 保留旧名兼容
    get_toxic_vector = get_harmful_vector

    def get_safe_vector(self) -> Tuple[np.ndarray, float]:
        """
        获取安全意图向量（类别 0 的权重方向）

        Returns:
            (w_safe, b): 安全意图类别的权重向量 (input_dim,) 和偏置标量
        """
        w_safe = self.linear.weight[0].detach().cpu().numpy()
        b = self.linear.bias[0].item()
        return w_safe, b


# ======================================================================
# 加载已训练的探针
# ======================================================================
def load_probe(
    layer_dir: Path,
    device: torch.device = None,
    dropout: float = 0.0,
) -> Tuple["LinearProbe", Optional[object]]:
    """
    从 layer_{i}/ 目录加载已训练的探针和预处理器

    Args:
        layer_dir: 层目录路径（包含 probe.pt 和 preprocessor.pkl）
        device: 目标设备（默认 CPU）
        dropout: Dropout（推理时建议设为 0）

    Returns:
        (probe, scaler): 探针模型（eval 模式）和 StandardScaler

    示例::

        probe, scaler = load_probe(Path("outputs/probes/layer_28"))
        h = scaler.transform(hidden_state.reshape(1, -1))
        logits = probe(torch.tensor(h, dtype=torch.float32))
    """
    if device is None:
        device = torch.device("cpu")

    layer_dir = Path(layer_dir)

    # 加载模型权重 → 推断 input_dim
    state = torch.load(layer_dir / "probe.pt", map_location="cpu")
    input_dim = state["linear.weight"].shape[1]

    probe = LinearProbe(input_dim, dropout=dropout)
    probe.load_state_dict(state)
    probe.to(device).eval()

    # 加载预处理器
    scaler = None
    pkl_path = layer_dir / "preprocessor.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        scaler = data.get("scaler", data) if isinstance(data, dict) else data

    return probe, scaler


def load_all_probes(
    output_dir: Path,
    device: torch.device = None,
) -> Dict[int, Tuple["LinearProbe", Optional[object]]]:
    """
    加载所有层的探针

    Args:
        output_dir: 输出根目录（包含 layer_0/, layer_1/, ...）
        device: 目标设备

    Returns:
        {layer_idx: (probe, scaler)} 字典
    """
    output_dir = Path(output_dir)
    probes = {}
    for d in sorted(output_dir.glob("layer_*")):
        if d.is_dir() and (d / "probe.pt").exists():
            try:
                idx = int(d.name.split("_")[1])
                probes[idx] = load_probe(d, device=device)
            except (ValueError, RuntimeError):
                continue
    return probes


# ======================================================================
# 提取隐藏态
# ======================================================================
def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 32,
    pooling_method: str = "last_token",
    desc: str = "提取隐藏状态",
) -> List[np.ndarray]:
    """
    提取模型各层的隐藏状态

    Args:
        model: 语言模型
        tokenizer: 分词器
        texts: 文本列表
        device: 设备
        max_length: 最大序列长度
        batch_size: 批次大小
        pooling_method: 池化方式
            - "last_token": 最后一个非 padding token（默认，推荐）
            - "mean": 对非 padding token 做平均池化
            - "cls": 取首 token
        desc: 进度条描述

    Returns:
        隐藏状态列表，每个元素为 (num_layers, hidden_dim) 的 numpy 数组
    """
    model.eval()
    all_hidden_states = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"{desc} ({len(texts)}样本)", total=total_batches):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: tuple of (batch_size, seq_len, hidden_dim)
        # 包含 embedding 层 + 所有 transformer 层
        hidden_states = outputs.hidden_states
        batch_size_actual = inputs["input_ids"].shape[0]

        for b in range(batch_size_actual):
            seq_len = (inputs["attention_mask"][b] == 1).sum().item()
            if seq_len <= 0:
                continue

            layer_hidden_states = []
            for layer_hs in hidden_states:
                if pooling_method == "last_token":
                    pooled = layer_hs[b, seq_len - 1, :].cpu().numpy()
                elif pooling_method == "mean":
                    pooled = layer_hs[b, :seq_len, :].mean(dim=0).cpu().numpy()
                elif pooling_method == "cls":
                    pooled = layer_hs[b, 0, :].cpu().numpy()
                else:
                    raise ValueError(
                        f"不支持的 pooling_method: {pooling_method}，"
                        f"可选: 'last_token', 'mean', 'cls'"
                    )
                layer_hidden_states.append(pooled)

            all_hidden_states.append(np.stack(layer_hidden_states))

        # 释放 GPU 显存（hidden_states 包含所有层的大张量）
        del outputs, hidden_states, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_hidden_states
