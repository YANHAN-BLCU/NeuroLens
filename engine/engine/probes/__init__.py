"""
线性探针模块

提供分层线性探针分类器，用于识别模型表征中嵌入的有害语义。
"""

from .linear_probe import LinearProbe, extract_hidden_states, train_layer_probes

__all__ = ["LinearProbe", "extract_hidden_states", "train_layer_probes"]

