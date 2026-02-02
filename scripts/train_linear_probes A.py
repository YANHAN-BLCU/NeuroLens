#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练分层线性探针分类器，用于识别模型表征中的有害语义。

当前脚本仅使用 `base_evaluation.jsonl` 作为数据集来源，并面向服务器环境（GPU）训练：
- **自动使用4-bit量化**：通过 ModelManager 加载模型，自动启用4-bit量化以节省显存
- 从 `input.prompt` 读取文本
- 使用 Guard 的 `asr_label` / `verdict` 自动标记安全(0) 与有害(1)
- 数据划分：使用 --use_optimized_split（含 6:2:2 等）或默认优化划分（6:2:2 比例）
- 6:2:2 划分时对探针训练集过采样有害样本，使安全:有害 = 1.5:1；验证集保持与总体相近的分布；训练不使用类别权重
- 每层使用token 的平均池化状态，训练 softmax 线性探针
- 自动过滤准确率 < 75% 的浅层探针
- 输出每层指标与毒性向量 w_toxic, b（已归一化）

4-bit量化优势：
- 显存占用降低约75%（8B模型从16GB降至4-5GB）
- 支持在8GB GPU上训练
- 可以增大批大小，加快训练速度
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

# 将工程根目录加入路径
import sys

sys.path.append(str(Path(__file__).parent.parent))
from engine.models import ModelManager  # noqa: E402
from engine.probes.linear_probe import (  # noqa: E402
    extract_hidden_states,
    save_probes,
    train_layer_probes,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_attack_enhanced(
    file_path: Path,
    max_samples: int | None = None,
) -> Tuple[List[str], List[int]]:
    """
    从 base_evaluation.jsonl 加载数据集（input.prompt + guard labels）
    """
    texts: List[str] = []
    labels: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and len(texts) >= max_samples:
                break
            if not line.strip():
                continue
                
            sample = json.loads(line)

            # 提取文本：input.prompt（input 非 dict 时视为缺）
            inp = sample.get("input")
            prompt = (inp.get("prompt", "") or "") if isinstance(inp, dict) else ""
            if not prompt:
                continue
            
            # 提取标签（优先使用asr_label，其次使用verdict）
            guard_info = sample.get("guard", {}) if isinstance(sample.get("guard"), dict) else {}
            asr_label = guard_info.get("asr_label")
            verdict = (guard_info.get("verdict") or "").lower()
            
            if asr_label is not None:
                label = int(asr_label)  # 0=安全，1=有害
            elif verdict == "allow":
                label = 0  # 安全
            elif verdict in ("flag", "block"):
                label = 1  # 有害
            else:
                continue  # 跳过无法判断的样本
            
            texts.append(prompt)
            labels.append(label)

    return texts, labels


def oversample_probe_train(
    texts: List[str],
    labels: List[int],
    target_safe_toxic_ratio: float = 1.5,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    对探针训练集过采样有害样本，使安全:有害 = target_safe_toxic_ratio:1。
    保持安全样本不变，通过有放回抽样反复采用有害样本直至达到目标比例。
    """
    safe_texts = [t for t, l in zip(texts, labels) if l == 0]
    safe_labels = [0] * len(safe_texts)
    toxic_texts = [t for t, l in zip(texts, labels) if l == 1]
    toxic_labels = [1] * len(toxic_texts)
    n_safe, n_toxic = len(safe_texts), len(toxic_texts)
    if n_toxic == 0:
        return texts, labels
    # 目标有害数 = 安全数 / 目标比例，使 安全:有害 = target_safe_toxic_ratio:1
    n_toxic_target = int(round(n_safe / target_safe_toxic_ratio))
    if n_toxic >= n_toxic_target:
        # 已满足或有害更多，不做欠采样，直接返回
        return texts, labels
    rng = np.random.RandomState(seed)
    extra_idx = rng.choice(n_toxic, size=n_toxic_target - n_toxic, replace=True)
    extra_texts = [toxic_texts[i] for i in extra_idx]
    extra_labels = [1] * (n_toxic_target - n_toxic)
    new_texts = safe_texts + toxic_texts + extra_texts
    new_labels = safe_labels + toxic_labels + extra_labels
    # 打乱顺序
    combined = list(zip(new_texts, new_labels))
    random.Random(seed + 10).shuffle(combined)
    new_texts, new_labels = zip(*combined)
    n_safe_final = n_safe
    n_toxic_final = n_toxic_target
    print(f"[Oversample] 探针训练集过采样: 安全={n_safe_final}, 有害={n_toxic} -> {n_toxic_final}, 比例 安全:有害={target_safe_toxic_ratio}:1, 总样本={len(new_texts)}")
    return list(new_texts), list(new_labels)


def split_data_optimized(
    texts: List[str],
    labels: List[int],
    test_ratio: float = 0.20,  # 6:2:2 测试集 20%
    val_ratio: float = 0.20,   # 6:2:2 验证集 20%
    min_test_toxic: int = 100,  # 测试集最少有害样本数
    min_val_toxic: int = 100,   # 验证集最少有害样本数
    seed: int = 42,
    use_ratio_6_2_2: bool = False,  # 6:2:2 分层划分：Task-Train 60%→Probe-Train 42%+Probe-Val 18%，Task-Test 20%
    probe_val_ratio_in_train: float = 0.3,  # 仅 use_ratio_6_2_2 时有效：Task-Train 内探针验证集比例，0.3→18% 总数
    use_ratio_6_2_2_full_train: bool = False,  # 6:2:2 剩余全作训练集：60% 训练、20% 验证、20% 测试
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    优化的数据划分函数：优先分离测试集和验证集，训练集使用全部剩余数据，6:2:2 时后续过采样至安全:有害=1.5:1。
    
    策略：
    1. 先分析数据集的实际分布（安全/有害比例）
    2. 优先为测试集和验证集预留足够的有害样本（确保评估有效）
    3. 测试集和验证集保持与全集一致的有害/安全比例
    4. 训练集使用所有剩余数据（安全+有害），6:2:2 划分时在外部对探针训练集过采样至 1.5:1
    
    Args:
        texts: 文本列表
        labels: 标签列表，0=安全，1=有害
        test_ratio: 测试集占总数据的比例（默认0.20，即 6:2:2）
        val_ratio: 验证集占总数据的比例（默认0.20，即 6:2:2）
        min_test_toxic: 测试集最少有害样本数（默认100）
        min_val_toxic: 验证集最少有害样本数（默认100）
        seed: 随机种子
        use_ratio_6_2_2: 是否按 6:2:2 分层划分（60% 训练、20% 验证、20% 测试，每份有害/安全比例与总体相同）
    
    Returns:
        (probe_train_texts, probe_train_labels, 
         probe_val_texts, probe_val_labels,
         test_texts, test_labels)
    """
    # 数据验证
    if len(texts) == 0 or len(labels) == 0:
        raise ValueError("数据列表不能为空")
    if len(texts) != len(labels):
        raise ValueError(f"文本数量({len(texts)})与标签数量({len(labels)})不匹配")
    if not use_ratio_6_2_2 and not use_ratio_6_2_2_full_train:
        if not (0 < test_ratio < 1):
            raise ValueError(f"test_ratio 必须在 (0, 1) 之间，当前值: {test_ratio}")
        if not (0 < val_ratio < 1):
            raise ValueError(f"val_ratio 必须在 (0, 1) 之间，当前值: {val_ratio}")
        if test_ratio + val_ratio >= 1.0:
            raise ValueError(f"test_ratio({test_ratio:.1%}) + val_ratio({val_ratio:.1%}) >= 100%，没有空间用于训练集")
    
    # 分析数据集分布
    total_samples = len(texts)
    num_safe_total = sum(1 for l in labels if l == 0)
    num_toxic_total = sum(1 for l in labels if l == 1)
    safe_ratio_total = num_safe_total / total_samples
    toxic_ratio_total = num_toxic_total / total_samples
    
    print(f"[Data Analysis] 总数据: {total_samples} 个样本")
    print(f"[Data Analysis] 安全: {num_safe_total} ({safe_ratio_total:.1%}), 有害: {num_toxic_total} ({toxic_ratio_total:.1%})")
    
    # 6:2:2 剩余全作训练集：60% 训练、20% 验证、20% 测试，分层；训练集用满 60% 剩余数据，后续过采样至 1.5:1
    if use_ratio_6_2_2_full_train:
        rng = random.Random(seed)
        safe_indices = [i for i, label in enumerate(labels) if label == 0]
        toxic_indices = [i for i, label in enumerate(labels) if label == 1]
        rng.shuffle(safe_indices)
        random.Random(seed + 1).shuffle(toxic_indices)
        n_safe, n_toxic = len(safe_indices), len(toxic_indices)
        t_safe = int(n_safe * 0.6)
        v_safe = int(n_safe * 0.2)
        t_toxic = int(n_toxic * 0.6)
        v_toxic = int(n_toxic * 0.2)
        train_safe_idx = safe_indices[:t_safe]
        val_safe_idx = safe_indices[t_safe : t_safe + v_safe]
        test_safe_idx = safe_indices[t_safe + v_safe :]
        train_toxic_idx = toxic_indices[:t_toxic]
        val_toxic_idx = toxic_indices[t_toxic : t_toxic + v_toxic]
        test_toxic_idx = toxic_indices[t_toxic + v_toxic :]
        probe_train_indices = train_safe_idx + train_toxic_idx
        probe_val_indices = val_safe_idx + val_toxic_idx
        test_indices = test_safe_idx + test_toxic_idx
        rng.shuffle(probe_train_indices)
        random.Random(seed + 2).shuffle(probe_val_indices)
        random.Random(seed + 3).shuffle(test_indices)
        probe_train_texts = [texts[i] for i in probe_train_indices]
        probe_train_labels = [labels[i] for i in probe_train_indices]
        probe_val_texts = [texts[i] for i in probe_val_indices]
        probe_val_labels = [labels[i] for i in probe_val_indices]
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        pct_train = 100.0 * len(probe_train_texts) / total_samples
        pct_val = 100.0 * len(probe_val_texts) / total_samples
        pct_test = 100.0 * len(test_texts) / total_samples
        print(f"[Split Strategy] 6:2:2 剩余全作训练集：训练 {pct_train:.1f}%、验证 {pct_val:.1f}%、测试 {pct_test:.1f}%；训练集用满剩余数据，后续过采样至 1.5:1")
        print(f"  训练集 ({pct_train:.1f}%): {len(probe_train_texts)} - 安全={len(train_safe_idx)}, 有害={len(train_toxic_idx)}")
        print(f"  验证集 ({pct_val:.1f}%): {len(probe_val_texts)} - 安全={len(val_safe_idx)}, 有害={len(val_toxic_idx)}")
        print(f"  测试集 ({pct_test:.1f}%): {len(test_texts)} - 安全={len(test_safe_idx)}, 有害={len(test_toxic_idx)}")
        return probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels
    
    # 6:2:2 分层划分（按表）：Task-Train 60% → Probe-Train 42% + Probe-Val 18%；Task-Val 20% 留作它用；Task-Test 20% 最终评估
    # 每份有害/安全比例与总体相同。探针验证集比例 = probe_val_ratio_in_train（默认 0.3 → 18% 总数）
    if use_ratio_6_2_2:
        rng = random.Random(seed)
        safe_indices = [i for i, label in enumerate(labels) if label == 0]
        toxic_indices = [i for i, label in enumerate(labels) if label == 1]
        rng.shuffle(safe_indices)
        random.Random(seed + 1).shuffle(toxic_indices)
        n_safe, n_toxic = len(safe_indices), len(toxic_indices)
        # 顶层 60% Task-Train / 20% Task-Val / 20% Task-Test
        t_safe = int(n_safe * 0.6)
        task_val_safe = int(n_safe * 0.2)
        t_toxic = int(n_toxic * 0.6)
        task_val_toxic = int(n_toxic * 0.2)
        train_pool_safe = safe_indices[:t_safe]
        train_pool_toxic = toxic_indices[:t_toxic]
        test_safe_idx = safe_indices[t_safe + task_val_safe :]
        test_toxic_idx = toxic_indices[t_toxic + task_val_toxic :]
        # Task-Train 内 70% Probe-Train、30% Probe-Val（probe_val_ratio_in_train=0.3）
        pv_safe = int(len(train_pool_safe) * probe_val_ratio_in_train)
        pv_toxic = int(len(train_pool_toxic) * probe_val_ratio_in_train)
        pt_safe = len(train_pool_safe) - pv_safe
        pt_toxic = len(train_pool_toxic) - pv_toxic
        probe_train_safe_idx = train_pool_safe[:pt_safe]
        probe_val_safe_idx = train_pool_safe[pt_safe:]
        probe_train_toxic_idx = train_pool_toxic[:pt_toxic]
        probe_val_toxic_idx = train_pool_toxic[pt_toxic:]
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
        probe_val_indices = probe_val_safe_idx + probe_val_toxic_idx
        test_indices = test_safe_idx + test_toxic_idx
        rng.shuffle(probe_train_indices)
        random.Random(seed + 2).shuffle(probe_val_indices)
        random.Random(seed + 3).shuffle(test_indices)
        probe_train_texts = [texts[i] for i in probe_train_indices]
        probe_train_labels = [labels[i] for i in probe_train_indices]
        probe_val_texts = [texts[i] for i in probe_val_indices]
        probe_val_labels = [labels[i] for i in probe_val_indices]
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        pct_train = 100.0 * len(probe_train_texts) / total_samples
        pct_val = 100.0 * len(probe_val_texts) / total_samples
        pct_test = 100.0 * len(test_texts) / total_samples
        print(f"[Split Strategy] 6:2:2 分层 (use_ratio_6_2_2=True)：Task-Train 60% → Probe-Train {pct_train:.1f}% + Probe-Val {pct_val:.1f}%；Task-Test {pct_test:.1f}%；每份有害/安全比例与总体相同")
        print(f"  Probe-Train ({pct_train:.1f}%): {len(probe_train_texts)} - 安全={len(probe_train_safe_idx)}, 有害={len(probe_train_toxic_idx)}")
        print(f"  Probe-Val ({pct_val:.1f}%, Task-Train 内 {probe_val_ratio_in_train:.0%}): {len(probe_val_texts)} - 安全={len(probe_val_safe_idx)}, 有害={len(probe_val_toxic_idx)}")
        print(f"  Task-Test ({pct_test:.1f}%): {len(test_texts)} - 安全={len(test_safe_idx)}, 有害={len(test_toxic_idx)}")
        return probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels
    
    # 根据数据量自动调整最小值要求（避免训练集为空）
    # 策略：确保训练集至少有 min(50, num_toxic_total * 0.2) 个有害样本
    min_train_toxic_required = min(50, max(20, int(num_toxic_total * 0.2)))  # 训练集至少需要20%或50个（取较小值）
    max_available_for_test_val = num_toxic_total - min_train_toxic_required
    
    # 如果最小值要求过高，自动降低
    if min_test_toxic + min_val_toxic > max_available_for_test_val:
        # 按比例降低，但确保训练集有足够的样本
        scale_factor = max_available_for_test_val / (min_test_toxic + min_val_toxic)
        original_min_test_toxic = min_test_toxic
        original_min_val_toxic = min_val_toxic
        min_test_toxic = max(10, int(min_test_toxic * scale_factor))  # 至少保留10个
        min_val_toxic = max(10, int(min_val_toxic * scale_factor))  # 至少保留10个
        
        # 再次检查，确保训练集有足够的样本
        if min_test_toxic + min_val_toxic > max_available_for_test_val:
            # 如果还是太高，进一步降低（优先保证训练集）
            remaining_for_train = num_toxic_total - min_test_toxic - min_val_toxic
            if remaining_for_train < min_train_toxic_required:
                # 按比例进一步降低
                total_needed = min_test_toxic + min_val_toxic + min_train_toxic_required
                if total_needed > num_toxic_total:
                    scale_factor2 = num_toxic_total / total_needed
                    min_test_toxic = max(5, int(min_test_toxic * scale_factor2))
                    min_val_toxic = max(5, int(min_val_toxic * scale_factor2))
        
        print(f"[Info] 数据量较小，自动调整最小值要求:")
        print(f"  测试集最小值: {original_min_test_toxic} → {min_test_toxic}")
        print(f"  验证集最小值: {original_min_val_toxic} → {min_val_toxic}")
        print(f"  确保训练集至少有 {min_train_toxic_required} 个有害样本")
    
    # 分离安全样本和有害样本的索引
    safe_indices = [i for i, label in enumerate(labels) if label == 0]
    toxic_indices = [i for i, label in enumerate(labels) if label == 1]
    
    # 打乱
    rng = random.Random(seed)
    rng.shuffle(safe_indices)
    random.Random(seed + 1).shuffle(toxic_indices)
    
    # 第一步：先按 20%:80% 划分测试集和剩余数据（6:2:2 时 test_ratio=0.20）
    test_size = int(total_samples * test_ratio)
    
    # 测试集中的有害样本数 = 测试集总样本数 * (总有害样本数 / 总样本数)
    # 即按总数据的有害比例来计算，保证比例一致
    test_toxic_size = int(test_size * toxic_ratio_total)
    test_safe_size = test_size - test_toxic_size
    
    # 确保有害样本数满足最小值要求
    if test_toxic_size < min_test_toxic:
        print(f"[Info] 按比例计算的测试集有害样本数({test_toxic_size}) < 最小值({min_test_toxic})，调整为{min_test_toxic}")
        test_toxic_size = min(min_test_toxic, num_toxic_total)
        # 调整测试集大小以保持比例，或调整安全样本数
        if test_toxic_size > test_size:
            # 如果最小值超过测试集大小，则扩大测试集
            test_size = int(test_toxic_size / toxic_ratio_total)
            test_safe_size = test_size - test_toxic_size
        else:
            test_safe_size = int(test_toxic_size * (safe_ratio_total / toxic_ratio_total))
            # 如果超过目标大小，按目标大小调整
            if test_safe_size + test_toxic_size > test_size:
                test_safe_size = test_size - test_toxic_size
    
    # 确保不超过可用数量
    test_toxic_size = min(test_toxic_size, num_toxic_total)
    test_safe_size = min(test_safe_size, num_safe_total)
    
    # 如果计算出的测试集大小超过可用数量，调整（优先保证有害样本数）
    if test_toxic_size + test_safe_size > test_size:
        if test_toxic_size > num_toxic_total:
            test_toxic_size = num_toxic_total
            test_safe_size = min(test_size - test_toxic_size, num_safe_total)
        elif test_safe_size > num_safe_total:
            test_safe_size = num_safe_total
            test_toxic_size = min(test_size - test_safe_size, num_toxic_total)
    
    # 最终调整测试集大小（可能因为最小值要求而略有变化）
    actual_test_size = test_safe_size + test_toxic_size
    
    test_safe_idx = safe_indices[:test_safe_size]
    test_toxic_idx = toxic_indices[:test_toxic_size]
    test_indices = test_safe_idx + test_toxic_idx
    
    # 剩余数据（80%，用于验证+训练）
    remaining_safe_indices = safe_indices[test_safe_size:]
    remaining_toxic_indices = toxic_indices[test_toxic_size:]
    remaining_samples = len(remaining_safe_indices) + len(remaining_toxic_indices)
    
    test_safe = len(test_safe_idx)
    test_toxic = len(test_toxic_idx)
    test_actual_ratio = test_toxic / actual_test_size if actual_test_size > 0 else 0
    ratio_diff = abs(test_actual_ratio - toxic_ratio_total)
    
    print(f"[Split Strategy] 第一步：按{test_ratio:.0%}:{1-test_ratio:.0%}划分")
    print(f"  测试集 ({test_ratio:.0%}): {actual_test_size}个 - 安全={test_safe}, 有害={test_toxic}")
    print(f"    有害比例: {test_actual_ratio:.1%} (总数据: {toxic_ratio_total:.1%}, 差异: {ratio_diff:.2%})")
    print(f"    有害样本数: {test_toxic}个 {'✓ 满足最小值' if test_toxic >= min_test_toxic else f'⚠ 低于最小值({min_test_toxic})'}")
    print(f"  剩余数据 ({1-test_ratio:.0%}): {remaining_samples}个 - 安全={len(remaining_safe_indices)}, 有害={len(remaining_toxic_indices)}")
    
    # 第二步：从剩余数据中划分验证集（验证集占总数据 val_ratio，默认 20%）
    # 验证集大小基于总数据的比例
    val_size = int(total_samples * val_ratio)
    
    # 验证集中的有害样本数 = 验证集总样本数 * (总有害样本数 / 总样本数)
    # 即按总数据的有害比例来计算，保证比例一致
    val_toxic_needed = int(val_size * toxic_ratio_total)
    val_safe_needed = val_size - val_toxic_needed
    
    # 确保有害样本数满足最小值要求
    if val_toxic_needed < min_val_toxic:
        print(f"[Info] 按比例计算的验证集有害样本数({val_toxic_needed}) < 最小值({min_val_toxic})，调整为{min_val_toxic}")
        val_toxic_needed = min(min_val_toxic, len(remaining_toxic_indices))
        # 调整验证集大小以保持比例，或调整安全样本数
        if val_toxic_needed > val_size:
            # 如果最小值超过验证集大小，则扩大验证集
            val_size = int(val_toxic_needed / toxic_ratio_total)
            val_safe_needed = val_size - val_toxic_needed
        else:
            val_safe_needed = int(val_toxic_needed * (safe_ratio_total / toxic_ratio_total))
            # 如果超过目标大小，按目标大小调整
            if val_safe_needed + val_toxic_needed > val_size:
                val_safe_needed = val_size - val_toxic_needed
    
    # 确保不超过剩余可用数量
    val_toxic_needed = min(val_toxic_needed, len(remaining_toxic_indices))
    val_safe_needed = min(val_safe_needed, len(remaining_safe_indices))
    
    # 如果计算出的验证集大小超过可用数量，调整（优先保证有害样本数）
    if val_toxic_needed + val_safe_needed > val_size:
        if val_toxic_needed > len(remaining_toxic_indices):
            val_toxic_needed = len(remaining_toxic_indices)
            val_safe_needed = min(val_size - val_toxic_needed, len(remaining_safe_indices))
        elif val_safe_needed > len(remaining_safe_indices):
            val_safe_needed = len(remaining_safe_indices)
            val_toxic_needed = min(val_size - val_safe_needed, len(remaining_toxic_indices))
    
    # 最终调整验证集大小（可能因为最小值要求而略有变化）
    actual_val_size = val_safe_needed + val_toxic_needed
    val_actual_ratio = val_toxic_needed / actual_val_size if actual_val_size > 0 else 0
    val_ratio_diff = abs(val_actual_ratio - toxic_ratio_total)
    
    print(f"[Split Strategy] 第二步：从剩余数据中划分验证集")
    print(f"  验证集 ({val_ratio:.0%}): {actual_val_size}个 - 安全={val_safe_needed}, 有害={val_toxic_needed}")
    print(f"    有害比例: {val_actual_ratio:.1%} (总数据: {toxic_ratio_total:.1%}, 差异: {val_ratio_diff:.2%})")
    print(f"    有害样本数: {val_toxic_needed}个 {'✓ 满足最小值' if val_toxic_needed >= min_val_toxic else f'⚠ 低于最小值({min_val_toxic})'}")
    
    # 提取验证集
    probe_val_safe_idx = remaining_safe_indices[:val_safe_needed]
    probe_val_toxic_idx = remaining_toxic_indices[:val_toxic_needed]
    probe_val_indices = probe_val_safe_idx + probe_val_toxic_idx
    
    # 剩余数据用于训练集
    train_safe_indices = remaining_safe_indices[val_safe_needed:]
    train_toxic_indices = remaining_toxic_indices[val_toxic_needed:]
    
    # 第三步：划分训练集（使用所有剩余数据）
    remaining_toxic_for_train = len(train_toxic_indices)
    remaining_safe_for_train = len(train_safe_indices)
    
    print(f"\n[Split Strategy] 第三步：划分训练集（使用全部剩余数据，6:2:2 时后续过采样至 1.5:1）")
    print(f"  剩余数据: 安全={remaining_safe_for_train}, 有害={remaining_toxic_for_train}")
    
    if remaining_toxic_for_train == 0:
        raise ValueError(
            f"训练集无有害样本。测试集有害={test_toxic}、验证集有害={val_toxic_needed}、总有害={num_toxic_total}。"
            f"可增加数据量或降低 --min_test_toxic / --min_val_toxic。"
        )
    
    probe_train_indices = train_safe_indices + train_toxic_indices
    probe_train_safe_idx = train_safe_indices
    probe_train_toxic_idx = train_toxic_indices
    unused_safe = 0
    unused_toxic = 0
    print(f"  训练集: 安全={remaining_safe_for_train}, 有害={remaining_toxic_for_train}, 总计={len(probe_train_indices)}")
    
    # 提取数据
    probe_train_texts = [texts[i] for i in probe_train_indices]
    probe_train_labels = [labels[i] for i in probe_train_indices]
    probe_val_texts = [texts[i] for i in probe_val_indices]
    probe_val_labels = [labels[i] for i in probe_val_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # 统计信息
    probe_train_safe = sum(1 for l in probe_train_labels if l == 0)
    probe_train_toxic = sum(1 for l in probe_train_labels if l == 1)
    probe_val_safe = sum(1 for l in probe_val_labels if l == 0)
    probe_val_toxic = sum(1 for l in probe_val_labels if l == 1)
    
    # 计算实际比例
    train_ratio_actual = len(probe_train_texts) / total_samples
    val_ratio_actual = len(probe_val_texts) / total_samples
    test_ratio_actual = len(test_texts) / total_samples
    
    # 计算实际比例
    test_actual_ratio_final = test_toxic / len(test_texts) if len(test_texts) > 0 else 0
    val_actual_ratio_final = probe_val_toxic / len(probe_val_texts) if len(probe_val_texts) > 0 else 0
    
    print(f"\n[Split Result] 数据划分完成:")
    print(f"  测试集: {len(test_texts)} ({test_ratio_actual:.1%}) - 安全={test_safe}, 有害={test_toxic}")
    print(f"    有害比例: {test_actual_ratio_final:.1%} (总数据: {toxic_ratio_total:.1%}, 差异: {abs(test_actual_ratio_final - toxic_ratio_total):.2%})")
    print(f"  验证集: {len(probe_val_texts)} ({val_ratio_actual:.1%}) - 安全={probe_val_safe}, 有害={probe_val_toxic}")
    print(f"    有害比例: {val_actual_ratio_final:.1%} (总数据: {toxic_ratio_total:.1%}, 差异: {abs(val_actual_ratio_final - toxic_ratio_total):.2%})")
    train_ratio_actual_val = probe_train_safe / probe_train_toxic if probe_train_toxic > 0 else 0
    print(f"  训练集: {len(probe_train_texts)} ({train_ratio_actual:.1%}) - 安全={probe_train_safe}, 有害={probe_train_toxic}")
    
    # 数据使用情况
    total_used = len(probe_train_texts) + len(probe_val_texts) + len(test_texts)
    unused = total_samples - total_used
    usage_rate = total_used / total_samples * 100
    
    print(f"\n[Usage] 数据使用情况:")
    print(f"  已使用: {total_used}/{total_samples} ({usage_rate:.1f}%)")
    if unused > 0:
        print(f"  未使用: {unused}个样本 (安全={unused_safe}, 有害={unused_toxic})")
        if unused_toxic > 0:
            print(f"  [Warning] 有{unused_toxic}个有害样本未使用！")
    else:
        print(f"  ✓ 数据利用率100%，所有数据都被使用")
    
    # 验证关键指标
    print(f"\n[Validation] 关键指标检查:")
    test_toxic_ratio = test_toxic / len(test_texts) if len(test_texts) > 0 else 0
    val_toxic_ratio = probe_val_toxic / len(probe_val_texts) if len(probe_val_texts) > 0 else 0
    val_has_toxic = probe_val_toxic > 0

    # 检查比例一致性
    test_ratio_diff = abs(test_toxic_ratio - toxic_ratio_total)
    val_ratio_diff = abs(val_toxic_ratio - toxic_ratio_total)
    
    print(f"  1. 比例一致性:")
    print(f"     ✓ 测试集有害比例: {test_toxic_ratio:.1%} (总数据: {toxic_ratio_total:.1%}, 差异: {test_ratio_diff:.2%}) "
          f"{'✓ 一致' if test_ratio_diff < 0.01 else '⚠ 有差异'}")
    print(f"     ✓ 验证集有害比例: {val_toxic_ratio:.1%} (总数据: {toxic_ratio_total:.1%}, 差异: {val_ratio_diff:.2%}) "
          f"{'✓ 一致' if val_ratio_diff < 0.01 else '⚠ 有差异'}")
    
    print(f"  2. 有害样本数量:")
    print(f"     ✓ 测试集有害样本: {test_toxic}个 {'✓ 满足最小值' if test_toxic >= min_test_toxic else f'⚠ 低于最小值({min_test_toxic})'}")
    print(f"     ✓ 验证集有害样本: {probe_val_toxic}个 {'✓ 满足最小值' if probe_val_toxic >= min_val_toxic else f'⚠ 低于最小值({min_val_toxic})'}")
    
    print(f"  3. 数据利用率:")
    print(f"     ✓ 总使用率: {usage_rate:.1f}% {'✓ 最大化' if usage_rate >= 99.0 else '⚠ 有未使用数据'}")
    if unused > 0:
        print(f"     ⚠ 未使用: {unused}个样本 (安全={unused_safe}, 有害={unused_toxic})")
    
    print(f"  4. 训练集: 使用全部剩余数据 (安全={probe_train_safe}, 有害={probe_train_toxic})")
    
    if not val_has_toxic:
        print(f"\n[Warning] ⚠️ 验证集没有有害样本！验证指标将无效！")
    
    # 总结
    all_checks_passed = (
        test_ratio_diff < 0.01 and 
        val_ratio_diff < 0.01 and 
        test_toxic >= min_test_toxic and 
        probe_val_toxic >= min_val_toxic and
        usage_rate >= 99.0
    )
    
    if all_checks_passed:
        print(f"\n[Summary] ✓ 所有检查通过：比例一致、有害样本充足、数据利用率最大化")
    else:
        print(f"\n[Summary] ⚠ 部分检查未通过，请查看上述详细信息")
    
    return probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on SALAD base_evaluation dataset.")
    parser.add_argument(
        "--data_file",
        type=Path,
        default=Path("data/salad/raw/base_evaluation.jsonl"),
        help="数据集文件路径（base_evaluation.jsonl格式，自动根据Guard判断标记安全/有害）",
    )
    parser.add_argument(
        "--balance_ratio",
        type=float,
        default=1.0,
        help="安全样本与有害样本的比例（默认1.0，即1:1平衡）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/probes"),
        help="探针输出目录（模型和毒性向量）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数，默认全量",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="提取隐藏态时的批大小",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="分词最大长度",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=80,
        help="每层探针训练轮数（默认80，利于中后层收敛）",
    )
    parser.add_argument(
        "--probe_batch_size",
        type=int,
        default=64,
        help="探针训练批大小（默认64，梯度更稳定）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="学习率（默认3e-3，配合 AdamW 线性探针更快收敛）",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减（默认0.01，减轻过拟合）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.20,
        help="测试集比例（默认 0.20 即 6:2:2 的 20%）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="验证集占总数据的比例（默认 0.20 即 6:2:2 的 20%）",
    )
    parser.add_argument(
        "--use_optimized_split",
        action="store_true",
        help="使用优化的数据划分函数（优先分离验证集，自动计算最优比例，最大化数据利用率）",
    )
    parser.add_argument(
        "--use_ratio_6_2_2",
        action="store_true",
        help="6:2:2 分层划分：Task-Train 60%%→Probe-Train 42%%+Probe-Val 18%%，Task-Test 20%%，每份有害/安全比例与总体相同。需与 --use_optimized_split 同时使用",
    )
    parser.add_argument(
        "--probe_val_ratio_in_train",
        type=float,
        default=0.3,
        help="6:2:2 时 Task-Train 内探针验证集比例，默认 0.3（18%% 总数）。表内 42:18 即 70:30",
    )
    parser.add_argument(
        "--use_6_2_2_recommended_hparams",
        action="store_true",
        help="6:2:2 时采用推荐超参：lr=3e-3, num_epochs=80, probe_batch_size=64, weight_decay=0.01",
    )
    parser.add_argument(
        "--use_ratio_6_2_2_full_train",
        action="store_true",
        help="6:2:2 剩余全作训练集：60%% 训练、20%% 验证、20%% 测试；训练集用满剩余数据，后续过采样至 1.5:1",
    )
    parser.add_argument(
        "--min_test_toxic",
        type=int,
        default=100,
        help="测试集最少有害样本数（使用优化划分时，默认100）",
    )
    parser.add_argument(
        "--min_val_toxic",
        type=int,
        default=100,
        help="验证集最少有害样本数（使用优化划分时，默认100）",
    )
    parser.add_argument(
        "--hidden_states_cache",
        type=Path,
        default=None,
        help="预提取的隐藏态 .npz 路径；若提供则跳过加载数据、LLM 与提取，直接训练。需先用 extract_hidden_states.py 生成。",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.use_ratio_6_2_2 and args.use_6_2_2_recommended_hparams:
        args.lr = 3e-3
        args.num_epochs = 80
        args.probe_batch_size = 64
        args.weight_decay = 0.01  # 减轻过拟合
        print("[6:2:2] 使用推荐超参: lr=3e-3, num_epochs=80, probe_batch_size=64, weight_decay=0.01")

    if args.hidden_states_cache:
        # 从缓存加载，跳过 LLM 与提取
        path = Path(args.hidden_states_cache)
        if not path.exists():
            raise FileNotFoundError(f"隐藏态缓存不存在: {path}")
        print(f"[Cache] 从 {path} 加载...")
        data = np.load(path, allow_pickle=True)
        train_hs = data["train_hs"]
        train_labels = data["train_labels"]
        val_hs = data["val_hs"]
        val_labels = data["val_labels"]
        num_layers = int(data["num_layers"])
        hidden_dim = int(data["hidden_dim"])
        meta = json.loads(data["meta"].item()) if "meta" in data else {}
        hidden_states = [train_hs[i] for i in range(len(train_labels))]
        val_hidden_states = [val_hs[i] for i in range(len(val_labels))]
        labels = train_labels
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from_cache = True
        test_texts, test_labels = [], []
        print(f"[Cache] 训练: {len(hidden_states)}, 验证: {len(val_hidden_states)}, num_layers={num_layers}, hidden_dim={hidden_dim}")
        print(f"[Cache] Device: {device}\n")
    else:
        from_cache = False
        # 加载文本与标签（base_evaluation.jsonl格式）
        print(f"[Data] Loading {args.data_file} ...")
        all_texts, all_labels = load_attack_enhanced(args.data_file, max_samples=args.max_samples)
        if len(all_texts) == 0:
            raise ValueError("未加载到有效样本，请检查数据路径与字段。")

        print(f"[Data] Loaded {len(all_texts)} total samples")
        print(f"[Data] Safe={sum(1 for l in all_labels if l==0)}, Toxic={sum(1 for l in all_labels if l==1)}")

        # 数据划分
        data_split_method = "optimized_default"
        if args.use_optimized_split:
            data_split_method = "use_optimized_split"
            print("\n" + "="*60)
            if args.use_ratio_6_2_2_full_train:
                print("使用 6:2:2 剩余全作训练集：训练 60%、验证 20%、测试 20%，训练集用满剩余数据并过采样至 1.5:1")
            elif args.use_ratio_6_2_2:
                print("使用 6:2:2 分层划分：训练 60%、验证 20%、测试 20%，每份有害/安全比例与总体相同")
            else:
                print("使用优化的数据划分策略（推荐）：")
                print("1. 优先分离验证集（确保验证集平衡且足够大）")
                print("2. 根据实际数据分布自动计算最优比例")
                print("3. 最大化数据利用率")
            print("="*60 + "\n")
            
            probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
                split_data_optimized(
                    texts=all_texts,
                    labels=all_labels,
                    test_ratio=args.test_ratio,
                    val_ratio=args.val_ratio,
                    min_test_toxic=args.min_test_toxic,
                    min_val_toxic=args.min_val_toxic,
                    seed=args.seed,
                    use_ratio_6_2_2=args.use_ratio_6_2_2,
                    probe_val_ratio_in_train=args.probe_val_ratio_in_train,
                    use_ratio_6_2_2_full_train=args.use_ratio_6_2_2_full_train,
                )
        else:
            print("\n" + "="*60)
            print("使用优化的数据划分策略（默认）：")
            print("1. 优先分离验证集与测试集（确保有害样本足够）")
            print("2. 训练集使用全部剩余数据")
            print("="*60 + "\n")

            probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
                split_data_optimized(
                    texts=all_texts,
                    labels=all_labels,
                    test_ratio=args.test_ratio,
                    val_ratio=args.val_ratio,
                    min_test_toxic=args.min_test_toxic,
                    min_val_toxic=args.min_val_toxic,
                    seed=args.seed,
                    use_ratio_6_2_2=False,
                    probe_val_ratio_in_train=args.probe_val_ratio_in_train,
                    use_ratio_6_2_2_full_train=False,
                )

        # 使用探针训练集和验证集
        texts = probe_train_texts
        labels = probe_train_labels
        val_texts = probe_val_texts
        val_labels = probe_val_labels

        # 训练集采用过采样方式：在 6:2:2 划分下，对有害样本过采样，使安全:有害 ≈ 1.5:1
        if args.use_optimized_split and (args.use_ratio_6_2_2 or args.use_ratio_6_2_2_full_train):
            texts, labels = oversample_probe_train(
                texts, labels, target_safe_toxic_ratio=1.5, seed=args.seed
            )

        # 6:2:2 划分时对探针训练集过采样，使安全:有害 = 1.5:1
        if args.use_optimized_split and (args.use_ratio_6_2_2 or args.use_ratio_6_2_2_full_train):
            texts, labels = oversample_probe_train(
                texts, labels, target_safe_toxic_ratio=1.5, seed=args.seed
            )

        # 打乱探针训练集顺序
        combined = list(zip(texts, labels))
        random.Random(args.seed).shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

        # 训练日志：记录划分出的每个数据集个数及训练共使用的样本数
        n_train = len(texts)
        n_train_safe = sum(1 for l in labels if l == 0)
        n_train_toxic = sum(1 for l in labels if l == 1)
        n_val = len(val_texts)
        n_val_safe = sum(1 for l in val_labels if l == 0)
        n_val_toxic = sum(1 for l in val_labels if l == 1)
        n_test = len(test_texts)
        n_test_safe = sum(1 for l in test_labels if l == 0)
        n_test_toxic = sum(1 for l in test_labels if l == 1)
        print("\n[训练日志] 数据集划分")
        print("  集合           | 总样本数 | 安全(0) | 有害(1)")
        print("  ---------------+----------+---------+--------")
        print(f"  探针训练集     | {n_train:8d} | {n_train_safe:7d} | {n_train_toxic:6d}")
        print(f"  探针验证集     | {n_val:8d} | {n_val_safe:7d} | {n_val_toxic:6d}")
        print(f"  测试集         | {n_test:8d} | {n_test_safe:7d} | {n_test_toxic:6d}")
        print(f"  训练共使用样本数: {n_train}（探针训练时每层均使用上述 {n_train} 条数据）")
        print(f"  验证集样本数: {n_val}（安全 {n_val_safe}，有害 {n_val_toxic}，用于调参）\n")

        # 加载模型与分词器（自动使用4-bit量化以节省显存）
        print("[Model] 正在加载LLM模型（使用4-bit量化）...")
        model_manager = ModelManager()
        tokenizer, model = model_manager.load_llm()
        device = next(model.parameters()).device
        print(f"[Model] Device: {device}")

        # 检查是否使用了量化
        if hasattr(model, 'hf_quantizer') or hasattr(model, 'quantization_config'):
            print("[Model] ✓ 模型已使用4-bit量化加载，显存占用显著降低")
        else:
            print("[Model] ⚠ 警告: 模型未使用量化，显存占用较高")

        # 提取隐藏态
        hidden_states = extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        # 维度信息
        num_layers, hidden_dim = hidden_states[0].shape
        print(f"[Hidden] num_layers={num_layers}, hidden_dim={hidden_dim}")

        # 提取验证集的隐藏状态（用于验证探针效果）
        print(f"[Hidden] 提取探针验证集隐藏状态...")
        val_hidden_states = extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            texts=val_texts,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

    # 训练各层探针（使用探针训练集训练，探针验证集验证）
    # 探针训练集使用上游划分/采样后的分布（此处不再额外平衡），探针验证集保持原始分布
    train_indices = list(range(len(labels)))  # 使用全部探针训练集
    
    results = train_layer_probes(
        hidden_states=hidden_states,  # 探针训练集的隐藏状态
        labels=labels,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        train_indices=train_indices,  # 使用全部训练集
        val_indices=[],               # 不使用内部划分，使用外部验证集
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.probe_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_hidden_states=val_hidden_states,  # 外部验证集隐藏状态
        val_labels=val_labels,                # 外部验证集标签
        ensure_accuracy_requirements=True,   # 确保每层达到准确率要求
    )

    # 保存探针（自动过滤准确率<75%的浅层探针）
    save_probes(
        results=results, 
        output_dir=args.output_dir, 
        model_id="llama-3-8b",
        filter_threshold=0.75,  # 按照论文要求，过滤准确率<75%的浅层探针
    )

    # 保存训练配置与数据统计，便于后续加载和分析
    probes_root = Path(args.output_dir) / "probes" / "llama-3-8b"
    probes_root.mkdir(parents=True, exist_ok=True)

    # 本次训练各集样本数（用于每次训练记录）
    run_sample_counts = {
        "训练集": {"总样本数": len(labels), "安全": sum(1 for l in labels if l == 0), "有害": sum(1 for l in labels if l == 1)},
        "验证集": {"总样本数": len(val_labels), "安全": sum(1 for l in val_labels if l == 0), "有害": sum(1 for l in val_labels if l == 1)},
        "测试集": {"总样本数": len(test_texts), "安全": sum(1 for l in test_labels if l == 0), "有害": sum(1 for l in test_labels if l == 1)},
    }

    # 从缓存加载时，data_file/max_samples/batch_size 优先从 meta 取；test 仅在未使用缓存时有值
    _meta = meta if args.hidden_states_cache else {}
    metadata = {
        "model_id": "llama-3-8b",
        "created_at": datetime.utcnow().isoformat(),
        "data_file": _meta.get("data_file", str(args.data_file)),
        "data_split_method": _meta.get("data_split_method", "from_cache") if args.hidden_states_cache else data_split_method,
        "本次训练各集样本数": run_sample_counts,
        "probe_train_samples": int(len(labels)),
        "probe_train_safe": int(run_sample_counts["训练集"]["安全"]),
        "probe_train_toxic": int(run_sample_counts["训练集"]["有害"]),
        "probe_val_samples": int(len(val_labels)),
        "probe_val_safe": int(run_sample_counts["验证集"]["安全"]),
        "probe_val_toxic": int(run_sample_counts["验证集"]["有害"]),
        "test_samples": int(len(test_texts)),
        "test_safe": int(run_sample_counts["测试集"]["安全"]),
        "test_toxic": int(run_sample_counts["测试集"]["有害"]),
        "max_samples": _meta.get("max_samples", args.max_samples),
        "batch_size": _meta.get("batch_size", args.batch_size),
        "probe_batch_size": args.probe_batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "from_cache": bool(args.hidden_states_cache),
    }
    if args.hidden_states_cache:
        metadata["hidden_states_cache"] = str(Path(args.hidden_states_cache).resolve())
    with (probes_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 每次训练追加一条运行记录，便于对比多次训练的样本个数
    runs_log_path = probes_root / "training_runs_log.json"
    run_record = {
        "created_at": metadata["created_at"],
        "本次训练各集样本数": run_sample_counts,
        "probe_train_samples": metadata["probe_train_samples"],
        "probe_val_samples": metadata["probe_val_samples"],
        "test_samples": metadata["test_samples"],
    }
    if runs_log_path.exists():
        try:
            with runs_log_path.open("r", encoding="utf-8") as f:
                runs_log = json.load(f)
        except (json.JSONDecodeError, TypeError):
            runs_log = []
        if not isinstance(runs_log, list):
            runs_log = [runs_log]
    else:
        runs_log = []
    runs_log.append(run_record)
    with runs_log_path.open("w", encoding="utf-8") as f:
        json.dump(runs_log, f, indent=2, ensure_ascii=False)
    print(f"[训练日志] 本次训练样本数已写入 config.json 与 {runs_log_path.name}（共 {len(runs_log)} 次运行记录）")


if __name__ == "__main__":
    main()

