#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练分层线性探针分类器，用于识别模型表征中的有害语义。

当前脚本仅使用 `base_evaluation.jsonl` 作为数据集来源，并面向服务器环境（GPU）训练：
- **自动使用4-bit量化**：通过 ModelManager 加载模型，自动启用4-bit量化以节省显存
- 从 `input.prompt` 读取文本
- 使用 Guard 的 `asr_label` / `verdict` 自动标记安全(0) 与有害(1)
- 按数据集划分文档进行数据划分（6.5:1:1.5方案）
- 探针训练集1:1平衡，验证集保持原始分布
- 每层使用最后 token 的隐藏态，训练 softmax 线性探针
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


def split_data_optimized(
    texts: List[str],
    labels: List[int],
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,  # 验证集占总数据的比例
    balance_train: bool = True,
    train_safe_ratio: float = 1.0,  # 训练集安全/有害比例（1.0=1:1平衡，2.0=2:1，3.0=3:1）
    balance_val: bool = False,  # 验证集是否平衡到1:1（默认False，允许不平衡）
    balance_test: bool = False,  # 测试集是否平衡到1:1（默认False，允许不平衡）
    min_test_toxic: int = 100,  # 测试集最少有害样本数
    min_val_toxic: int = 100,   # 验证集最少有害样本数
    seed: int = 42,
    use_doc_ratios: bool = False,  # 按文档划分：80%训练池+20%测试，训练池内 72.2% 探针训练、11.1% 验证、16.7% 分析
    use_ratio_6_2_2: bool = False,  # 6:2:2 分层划分：Task-Train 60%→Probe-Train 42%+Probe-Val 18%，Task-Test 20%
    probe_val_ratio_in_train: float = 0.3,  # 仅 use_ratio_6_2_2 时有效：Task-Train 内探针验证集比例，0.3→18% 总数
    use_ratio_6_2_2_full_train: bool = False,  # 6:2:2 剩余全作训练集：60% 训练、20% 验证、20% 测试
    train_safe_ratio_622_full: float | None = None,  # 仅 6:2:2 全训练时有效：训练集安全:有害比例，如 1.5 即 1.5:1；None 表示用满 60% 不讲究比例
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    优化的数据划分函数：优先分离测试集和验证集，根据实际数据分布自动计算最优比例，最大化数据利用率
    
    策略：
    1. 先分析数据集的实际分布（安全/有害比例）
    2. 优先为测试集和验证集预留足够的有害样本（确保评估有效）
    3. 测试集和验证集保持与全集一致的有害/安全比例（按文档要求）
    4. 训练集可按 train_safe_ratio 取 1:1 / 2:1 / 3:1 等，有害用满
    5. use_doc_ratios=True 时严格按文档：80% 训练池 + 20% 测试；训练池内 72.2% 探针训练、11.1% 验证、16.7% 神经元分析；验证集保持原始分布
    
    Args:
        texts: 文本列表
        labels: 标签列表，0=安全，1=有害
        test_ratio: 测试集占总数据的比例（默认0.15；use_doc_ratios 时为 0.2）
        val_ratio: 验证集占总数据的比例（默认0.15；use_doc_ratios 时表示占训练池的 0.111）
        balance_train: 训练集是否按 train_safe_ratio 平衡
        train_safe_ratio: 训练集安全/有害比例（1.0=1:1，2.0=2:1，3.0=3:1）
        balance_val: 验证集是否平衡到1:1（默认False，保持有害/安全比例）
        balance_test: 测试集是否平衡到1:1（默认False，保持有害/安全比例）
        min_test_toxic: 测试集最少有害样本数（默认100）
        min_val_toxic: 验证集最少有害样本数（默认100）
        seed: 随机种子
        use_doc_ratios: 是否按文档 6.5:1:1.5 划分（80% 训练池、20% 测试；训练池内 72.2%/11.1%/16.7%）
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
    if not use_doc_ratios and not use_ratio_6_2_2 and not use_ratio_6_2_2_full_train:
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
    
    # 6:2:2 剩余全作训练集：60% 训练、20% 验证、20% 测试，分层；训练集可按 train_safe_ratio_622_full 控制安全:有害比例
    if use_ratio_6_2_2_full_train:
        rng = random.Random(seed)
        safe_indices = [i for i, label in enumerate(labels) if label == 0]
        toxic_indices = [i for i, label in enumerate(labels) if label == 1]
        rng.shuffle(safe_indices)
        random.Random(seed + 1).shuffle(toxic_indices)
        n_safe, n_toxic = len(safe_indices), len(toxic_indices)
        # 分层 60% 训练 / 20% 验证 / 20% 测试，每份有害/安全比例与总体相同
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
        # 训练集比例：None 表示用满 60% 不讲究；否则 安全:有害 = train_safe_ratio_622_full : 1
        if train_safe_ratio_622_full is not None and train_safe_ratio_622_full > 0:
            n_toxic_train = len(train_toxic_idx)
            n_safe_train_want = min(int(round(n_toxic_train * train_safe_ratio_622_full)), len(train_safe_idx))
            train_safe_used = train_safe_idx[:n_safe_train_want]
            probe_train_indices = train_safe_used + train_toxic_idx
            ratio_desc = f"安全:有害={train_safe_ratio_622_full}:1"
            n_train_safe = len(train_safe_used)
        else:
            probe_train_indices = train_safe_idx + train_toxic_idx
            ratio_desc = "不讲究比例(自然分布)"
            n_train_safe = len(train_safe_idx)
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
        print(f"[Split Strategy] 6:2:2 剩余全作训练集 (use_ratio_6_2_2_full_train=True)：训练 {pct_train:.1f}%、验证 {pct_val:.1f}%、测试 {pct_test:.1f}%，训练集 {ratio_desc}")
        print(f"  训练集 ({pct_train:.1f}%): {len(probe_train_texts)} - 安全={n_train_safe}, 有害={len(train_toxic_idx)}")
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
    
    # 按文档划分：80% 训练池 + 20% 测试；训练池内 72.2% 探针训练、11.1% 验证、16.7% 神经元分析；测试/验证均保持有害安全比例
    if use_doc_ratios:
        doc_test_ratio = 0.2
        doc_probe_val_ratio_in_pool = 0.111   # 探针验证集占训练池 11.1%
        doc_probe_train_ratio_in_pool = 0.722  # 探针训练池占训练池 72.2%，剩余 16.7% 为神经元分析集（不返回）
        rng = random.Random(seed)
        safe_indices = [i for i, label in enumerate(labels) if label == 0]
        toxic_indices = [i for i, label in enumerate(labels) if label == 1]
        rng.shuffle(safe_indices)
        random.Random(seed + 1).shuffle(toxic_indices)
        # 第一步：80% 训练池 + 20% 测试，均保持全集有害/安全比例
        test_size = int(total_samples * doc_test_ratio)
        test_toxic_size = max(min_test_toxic, min(int(test_size * toxic_ratio_total), num_toxic_total))
        test_safe_size = min(test_size - test_toxic_size, int(test_toxic_size * safe_ratio_total / toxic_ratio_total) if toxic_ratio_total > 0 else test_size)
        test_toxic_size = min(test_toxic_size, num_toxic_total)
        test_safe_size = min(test_safe_size, num_safe_total)
        if test_safe_size + test_toxic_size > test_size:
            test_safe_size = test_size - test_toxic_size
        test_safe_idx = safe_indices[:test_safe_size]
        test_toxic_idx = toxic_indices[:test_toxic_size]
        pool_safe = safe_indices[test_safe_size:]
        pool_toxic = toxic_indices[test_toxic_size:]
        pool_size = len(pool_safe) + len(pool_toxic)
        pool_toxic_ratio = len(pool_toxic) / pool_size if pool_size > 0 else toxic_ratio_total
        pool_safe_ratio = 1.0 - pool_toxic_ratio
        # 第二步：从训练池中按比例划出 11.1% 验证（保持有害/安全比例）、72.2% 探针训练池（保持比例），剩余 16.7% 为神经元分析集
        val_in_pool_size = int(pool_size * doc_probe_val_ratio_in_pool)
        val_toxic_cnt = max(min_val_toxic, min(int(val_in_pool_size * pool_toxic_ratio), len(pool_toxic)))
        val_safe_cnt = min(val_in_pool_size - val_toxic_cnt, len(pool_safe))
        val_toxic_cnt = min(val_toxic_cnt, len(pool_toxic))
        if val_safe_cnt + val_toxic_cnt > val_in_pool_size:
            val_safe_cnt = val_in_pool_size - val_toxic_cnt
        probe_val_safe_idx = pool_safe[:val_safe_cnt]
        probe_val_toxic_idx = pool_toxic[:val_toxic_cnt]
        train_pool_size = int(pool_size * doc_probe_train_ratio_in_pool)
        train_pool_toxic_cnt = min(int(train_pool_size * pool_toxic_ratio), len(pool_toxic) - val_toxic_cnt)
        train_pool_safe_cnt = min(train_pool_size - train_pool_toxic_cnt, len(pool_safe) - val_safe_cnt)
        train_pool_toxic_cnt = min(train_pool_toxic_cnt, len(pool_toxic) - val_toxic_cnt)
        if train_pool_safe_cnt + train_pool_toxic_cnt > train_pool_size:
            train_pool_safe_cnt = train_pool_size - train_pool_toxic_cnt
        probe_train_pool_safe = pool_safe[val_safe_cnt:val_safe_cnt + train_pool_safe_cnt]
        probe_train_pool_toxic = pool_toxic[val_toxic_cnt:val_toxic_cnt + train_pool_toxic_cnt]
        remaining_toxic_for_train = len(probe_train_pool_toxic)
        remaining_safe_for_train = len(probe_train_pool_safe)
        if remaining_toxic_for_train == 0:
            raise ValueError("按文档划分后探针训练池中无有害样本，请检查数据或放宽 min_val_toxic/min_test_toxic。")
        if balance_train:
            train_safe_size = min(int(remaining_toxic_for_train * train_safe_ratio), remaining_safe_for_train)
            train_toxic_size = remaining_toxic_for_train
        else:
            train_safe_size = remaining_safe_for_train
            train_toxic_size = remaining_toxic_for_train
        probe_train_safe_idx = probe_train_pool_safe[:train_safe_size]
        probe_train_toxic_idx = probe_train_pool_toxic[:train_toxic_size]
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
        probe_val_indices = probe_val_safe_idx + probe_val_toxic_idx
        test_indices = test_safe_idx + test_toxic_idx
        probe_train_texts = [texts[i] for i in probe_train_indices]
        probe_train_labels = [labels[i] for i in probe_train_indices]
        probe_val_texts = [texts[i] for i in probe_val_indices]
        probe_val_labels = [labels[i] for i in probe_val_indices]
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        print(f"[Split Strategy] 按文档划分 (use_doc_ratios=True): 80% 训练池 + 20% 测试；训练池内 72.2% 探针训练、11.1% 验证、16.7% 神经元分析")
        print(f"  测试集 (20%): {len(test_texts)} - 安全={len(test_safe_idx)}, 有害={len(test_toxic_idx)}，保持有害/安全比例")
        print(f"  验证集 (训练池 11.1%): {len(probe_val_texts)} - 安全={len(probe_val_safe_idx)}, 有害={len(probe_val_toxic_idx)}，保持有害/安全比例")
        print(f"  探针训练集 (从训练池 72.2% 中按 train_safe_ratio={train_safe_ratio} 取): {len(probe_train_texts)} - 安全={len(probe_train_safe_idx)}, 有害={len(probe_train_toxic_idx)}")
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
    
    # 第一步：先按15%:85%划分测试集和剩余数据
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
    
    # 剩余数据（85%）
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
    
    # 第二步：从剩余数据（85%）中划分验证集
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
    
    # 第三步：划分训练集（最大化使用所有剩余数据）
    remaining_toxic_for_train = len(train_toxic_indices)
    remaining_safe_for_train = len(train_safe_indices)
    
    print(f"\n[Split Strategy] 第三步：划分训练集")
    print(f"  剩余数据: 安全={remaining_safe_for_train}, 有害={remaining_toxic_for_train}")
    print(f"  说明: 训练集将使用所有剩余的有害样本({remaining_toxic_for_train}个)")
    
    # 检查训练集是否有足够的样本
    if remaining_toxic_for_train == 0:
        error_msg = (
            f"\n[错误] 训练集为空！无法进行训练。\n"
            f"原因: 测试集和验证集的最小值要求占用了所有有害样本。\n"
            f"  - 测试集有害样本: {test_toxic}个 (最小值要求: {min_test_toxic})\n"
            f"  - 验证集有害样本: {val_toxic_needed}个 (最小值要求: {min_val_toxic})\n"
            f"  - 总有害样本: {num_toxic_total}个\n"
            f"  - 剩余有害样本: {remaining_toxic_for_train}个\n\n"
            f"解决方案:\n"
            f"  1. 增加数据量 (--max_samples 建议 >= 5000)\n"
            f"  2. 降低最小值要求 (--min_test_toxic 和 --min_val_toxic，建议根据数据量调整)\n"
            f"  3. 降低训练集比例要求 (--train_safe_ratio，使用更小的比例如 1.0 或 2.0)\n"
        )
        raise ValueError(error_msg)
    
    if balance_train:
        # 训练集使用所有剩余的有害样本，安全样本按比例使用
        train_toxic_size = remaining_toxic_for_train  # 使用所有剩余的有害样本（不会超过）
        train_safe_size = min(
            int(train_toxic_size * train_safe_ratio),  # 按比例计算安全样本数
            remaining_safe_for_train  # 不超过可用数量
        )
        
        # 确保不会超过剩余数量
        assert train_toxic_size <= remaining_toxic_for_train, f"训练集有害样本数({train_toxic_size})不能超过剩余数量({remaining_toxic_for_train})"
        assert train_safe_size <= remaining_safe_for_train, f"训练集安全样本数({train_safe_size})不能超过剩余数量({remaining_safe_for_train})"
        
        probe_train_safe_idx = train_safe_indices[:train_safe_size]
        probe_train_toxic_idx = train_toxic_indices[:train_toxic_size]
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
        
        # 统计未使用的数据
        unused_safe = remaining_safe_for_train - train_safe_size
        unused_toxic = 0  # 所有有害样本都被使用
        
        ratio_desc = f"1:{train_safe_ratio:.1f}" if train_safe_ratio != 1.0 else "1:1"
        print(f"  训练集 (比例={ratio_desc}): 安全={train_safe_size}, 有害={train_toxic_size}, 总计={len(probe_train_indices)}")
        print(f"    使用所有剩余有害样本: {train_toxic_size}/{remaining_toxic_for_train} ✓")
        print(f"    安全/有害比例: {train_safe_size/train_toxic_size:.2f}:1" if train_toxic_size > 0 else "N/A")
        print(f"    说明: 训练集有害样本数 = 剩余有害样本数，不会超过总数据的有害样本数")
        if unused_safe > 0:
            print(f"  未使用安全样本: {unused_safe}个（安全/有害比例={train_safe_ratio:.1f}:1）")
    else:
        # 训练集使用所有剩余数据（保持原始分布）
        probe_train_safe_idx = train_safe_indices
        probe_train_toxic_idx = train_toxic_indices
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
        
        unused_safe = 0
        unused_toxic = 0
    
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
    if balance_train:
        if probe_train_safe == probe_train_toxic:
            print(f"    [平衡到1:1]")
        else:
            print(f"    [比例={train_ratio_actual_val:.1f}:1]")
        print(f"    使用所有剩余有害样本: {probe_train_toxic}/{remaining_toxic_for_train} ✓")
        print(f"    说明: 训练集有害样本数 = 剩余有害样本数，不会超过总数据的有害样本数")
    else:
        train_ratio_final = probe_train_toxic / len(probe_train_texts) if len(probe_train_texts) > 0 else 0
        print(f"    有害比例: {train_ratio_final:.1%}")
    
    # 数据使用情况
    total_used = len(probe_train_texts) + len(probe_val_texts) + len(test_texts)
    unused = total_samples - total_used
    usage_rate = total_used / total_samples * 100
    
    print(f"\n[Usage] 数据使用情况:")
    print(f"  已使用: {total_used}/{total_samples} ({usage_rate:.1f}%)")
    if unused > 0:
        print(f"  未使用: {unused}个样本 (安全={unused_safe}, 有害={unused_toxic})")
        if unused_safe > 0 and balance_train:
            print(f"  [Info] 未使用的样本主要是安全样本（因为训练集需要平衡到1:1）")
        if unused_toxic > 0:
            print(f"  [Warning] 有{unused_toxic}个有害样本未使用！")
    else:
        print(f"  ✓ 数据利用率100%，所有数据都被使用")
    
    # 验证关键指标
    print(f"\n[Validation] 关键指标检查:")
    test_toxic_ratio = test_toxic / len(test_texts) if len(test_texts) > 0 else 0
    val_toxic_ratio = probe_val_toxic / len(probe_val_texts) if len(probe_val_texts) > 0 else 0
    val_has_toxic = probe_val_toxic > 0
    train_balanced = balance_train and probe_train_safe == probe_train_toxic
    
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
    
    print(f"  4. 训练集:")
    print(f"     ✓ 训练集平衡: {'是' if train_balanced else '否'}")
    
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


def split_data_improved(
    texts: List[str],
    labels: List[int],
    test_ratio: float = 0.15,  # 降低测试集比例，使用更多数据
    min_val_samples_per_class: int = 100,  # 验证集每类最少样本数
    balance_train: bool = True,  # 训练集是否平衡
    balance_val: bool = True,    # 验证集是否平衡（推荐True）
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    改进的数据划分函数，解决数据使用率低和验证集不平衡问题
    
    策略：
    1. 先为验证集预留足够的有害样本（确保验证集平衡）
    2. 然后平衡训练集
    3. 使用更多数据（降低测试集比例）
    4. 确保所有数据都被使用（除了测试集）
    
    Args:
        texts: 文本列表
        labels: 标签列表，0=安全，1=有害
        test_ratio: 测试集比例（默认0.15，即15%）
        min_val_samples_per_class: 验证集每类最少样本数（默认100）
        balance_train: 训练集是否平衡到1:1（默认True）
        balance_val: 验证集是否平衡到1:1（默认True，推荐）
        seed: 随机种子
    
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
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio 必须在 (0, 1) 之间，当前值: {test_ratio}")
    
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    
    # 第一步：总体划分 - 训练集 + 测试集
    test_size = int(len(indices) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 提取训练集数据
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    # 分离安全样本和有害样本
    safe_indices = [i for i, label in enumerate(train_labels) if label == 0]
    toxic_indices = [i for i, label in enumerate(train_labels) if label == 1]
    
    num_safe = len(safe_indices)
    num_toxic = len(toxic_indices)
    total_train = len(train_indices)
    
    print(f"[Split] 总体划分: 训练集={total_train}, 测试集={len(test_indices)} (比例={test_ratio:.1%})")
    print(f"[Split] 训练集中: 安全={num_safe} ({num_safe/total_train:.1%}), "
          f"有害={num_toxic} ({num_toxic/total_train:.1%})")
    
    # 打乱
    safe_shuffled = safe_indices.copy()
    toxic_shuffled = toxic_indices.copy()
    rng.shuffle(safe_shuffled)
    random.Random(seed + 1).shuffle(toxic_shuffled)
    
    # 第二步：先为验证集预留样本（优先保证验证集质量）
    if balance_val:
        # 验证集平衡到1:1
        val_per_class = min(min_val_samples_per_class, num_toxic // 2)  # 受限于有害样本数量
        if val_per_class < 50:
            print(f"[Warning] 有害样本太少，验证集每类只能有{val_per_class}个样本")
        
        probe_val_safe_idx = safe_shuffled[:val_per_class]
        probe_val_toxic_idx = toxic_shuffled[:val_per_class]
        probe_val_indices = probe_val_safe_idx + probe_val_toxic_idx
        
        # 剩余数据用于训练
        remaining_safe = safe_shuffled[val_per_class:]
        remaining_toxic = toxic_shuffled[val_per_class:]
    else:
        # 验证集保持原始分布，但确保包含足够的有害样本
        val_toxic_size = min(min_val_samples_per_class, num_toxic)
        val_safe_size = int(val_toxic_size * (num_safe / num_toxic))  # 按原始比例
        
        probe_val_safe_idx = safe_shuffled[:val_safe_size]
        probe_val_toxic_idx = toxic_shuffled[:val_toxic_size]
        probe_val_indices = probe_val_safe_idx + probe_val_toxic_idx
        
        remaining_safe = safe_shuffled[val_safe_size:]
        remaining_toxic = toxic_shuffled[val_toxic_size:]
    
    # 第三步：划分训练集
    if balance_train:
        # 训练集平衡到1:1，使用所有剩余的有害样本
        train_toxic_size = len(remaining_toxic)
        train_safe_size = train_toxic_size  # 平衡到1:1
        
        # 如果安全样本不够，使用所有剩余的安全样本
        if train_safe_size > len(remaining_safe):
            train_safe_size = len(remaining_safe)
            print(f"[Warning] 安全样本不足，训练集将不平衡: {train_safe_size}:{train_toxic_size}")
        
        probe_train_safe_idx = remaining_safe[:train_safe_size]
        probe_train_toxic_idx = remaining_toxic[:train_toxic_size]
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
    else:
        # 训练集保持原始分布，使用所有剩余数据
        probe_train_safe_idx = remaining_safe
        probe_train_toxic_idx = remaining_toxic
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
    
    # 提取数据
    probe_train_texts = [train_texts[i] for i in probe_train_indices]
    probe_train_labels = [train_labels[i] for i in probe_train_indices]
    probe_val_texts = [train_texts[i] for i in probe_val_indices]
    probe_val_labels = [train_labels[i] for i in probe_val_indices]
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # 统计信息
    probe_train_safe = sum(1 for l in probe_train_labels if l == 0)
    probe_train_toxic = sum(1 for l in probe_train_labels if l == 1)
    probe_val_safe = sum(1 for l in probe_val_labels if l == 0)
    probe_val_toxic = sum(1 for l in probe_val_labels if l == 1)
    
    print(f"[Split] 探针训练集: {len(probe_train_texts)} "
          f"(安全={probe_train_safe}, 有害={probe_train_toxic}) "
          f"{'- 已平衡到1:1' if balance_train and probe_train_safe == probe_train_toxic else ''}")
    print(f"[Split] 探针验证集: {len(probe_val_texts)} "
          f"(安全={probe_val_safe} ({probe_val_safe/len(probe_val_texts):.1%}), "
          f"有害={probe_val_toxic} ({probe_val_toxic/len(probe_val_texts):.1%})) "
          f"{'- 已平衡到1:1' if balance_val and probe_val_safe == probe_val_toxic else ''}")
    
    # 验证划分结果
    total_used = len(probe_train_texts) + len(probe_val_texts) + len(test_texts)
    unused = len(texts) - total_used
    usage_rate = total_used / len(texts) * 100
    
    print(f"[Split] 数据使用情况: 已使用={total_used}/{len(texts)} ({usage_rate:.1f}%)")
    if unused > 0:
        print(f"[Split] 未使用数据: {unused}个样本（已保留用于未来扩展）")
    
    # 检查验证集是否有有害样本
    if probe_val_toxic == 0:
        print(f"[Warning] ⚠️ 验证集没有有害样本！验证指标将无效！")
        print(f"[Warning] 建议: 减少min_val_samples_per_class或增加数据集中的有害样本")
    
    return probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels


def split_data_according_to_doc(
    texts: List[str],
    labels: List[int],
    test_ratio: float = 0.2,
    probe_train_ratio: float = 0.722,  # 6.5/(6.5+1+1.5) = 0.722
    probe_val_ratio: float = 0.111,     # 1/(6.5+1+1.5) = 0.111
    balance_probe_train: bool = True,   # 探针训练集是否平衡到1:1
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    按照数据集划分文档进行数据划分
    
    划分流程：
    1. 总体划分：训练集(80%) + 测试集(20%) - 测试集完全隔离
    2. 从训练集中划分：探针训练集(72.2%) + 探针验证集(11.1%) + 神经元分析集(16.7%)
    
    Args:
        texts: 文本列表
        labels: 标签列表，0=安全，1=有害
        test_ratio: 测试集比例（默认0.2，即20%）
        probe_train_ratio: 探针训练集在训练集中的比例（默认0.722，即72.2%）
        probe_val_ratio: 探针验证集在训练集中的比例（默认0.111，即11.1%）
        balance_probe_train: 是否将探针训练集平衡到1:1（默认True）
        seed: 随机种子
    
    Returns:
        (probe_train_texts, probe_train_labels, 
         probe_val_texts, probe_val_labels,
         test_texts, test_labels)
    
    Raises:
        ValueError: 如果数据为空或比例不合理
    """
    # 数据验证
    if len(texts) == 0 or len(labels) == 0:
        raise ValueError("数据列表不能为空")
    if len(texts) != len(labels):
        raise ValueError(f"文本数量({len(texts)})与标签数量({len(labels)})不匹配")
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio 必须在 (0, 1) 之间，当前值: {test_ratio}")
    if not (0 < probe_train_ratio < 1) or not (0 < probe_val_ratio < 1):
        raise ValueError(f"比例必须在 (0, 1) 之间")
    if probe_train_ratio + probe_val_ratio > 1.0:
        raise ValueError(f"探针训练集({probe_train_ratio:.1%}) + 验证集({probe_val_ratio:.1%}) > 100%，"
                        f"剩余空间不足用于神经元分析集")
    # 第一步：总体划分 - 训练集(80%) + 测试集(20%)
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    
    test_size = int(len(indices) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 提取训练集数据
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    # 分离安全样本和有害样本（在训练集中的索引）
    safe_indices = [i for i, label in enumerate(train_labels) if label == 0]
    toxic_indices = [i for i, label in enumerate(train_labels) if label == 1]
    
    num_safe = len(safe_indices)
    num_toxic = len(toxic_indices)
    total_train = len(train_indices)
    
    print(f"[Split] 总体划分: 训练集={total_train}, 测试集={len(test_indices)} (比例={test_ratio:.1%})")
    print(f"[Split] 训练集中: 安全={num_safe} ({num_safe/total_train:.1%}), "
          f"有害={num_toxic} ({num_toxic/total_train:.1%})")
    print(f"[Split] 测试集已隔离，不用于探针训练")
    
    # 第二步：从训练集中划分探针训练集和验证集
    # 计算目标大小
    probe_train_size = int(total_train * probe_train_ratio)
    probe_val_size = int(total_train * probe_val_ratio)
    
    # 探针训练集：平衡到1:1
    if balance_probe_train:
        # 计算每类可用的最大样本数（取较小值，确保1:1平衡）
        min_class_size = min(num_safe, num_toxic)
        probe_train_per_class = min(min_class_size, probe_train_size // 2)
        
        # 随机采样（使用不同的种子确保独立性）
        safe_shuffled = safe_indices.copy()
        toxic_shuffled = toxic_indices.copy()
        rng.shuffle(safe_shuffled)
        random.Random(seed + 1).shuffle(toxic_shuffled)
        
        # 探针训练集：每类取相同数量，确保1:1平衡
        probe_train_safe_idx = safe_shuffled[:probe_train_per_class]
        probe_train_toxic_idx = toxic_shuffled[:probe_train_per_class]
        probe_train_indices = probe_train_safe_idx + probe_train_toxic_idx
        
        # 从剩余数据中划分验证集（保持原始分布）
        remaining_safe = safe_shuffled[probe_train_per_class:]
        remaining_toxic = toxic_shuffled[probe_train_per_class:]
        
        # 按原始比例采样验证集（82.3%:17.7%）
        original_safe_ratio = num_safe / total_train
        safe_val_size = min(int(probe_val_size * original_safe_ratio), len(remaining_safe))
        toxic_val_size = min(probe_val_size - safe_val_size, len(remaining_toxic))
        
        probe_val_safe_idx = remaining_safe[:safe_val_size]
        probe_val_toxic_idx = remaining_toxic[:toxic_val_size]
        probe_val_indices = probe_val_safe_idx + probe_val_toxic_idx
        
        # 提取数据
        probe_train_texts = [train_texts[i] for i in probe_train_indices]
        probe_train_labels = [train_labels[i] for i in probe_train_indices]
        probe_val_texts = [train_texts[i] for i in probe_val_indices]
        probe_val_labels = [train_labels[i] for i in probe_val_indices]
        
        # 统计信息
        probe_train_safe = sum(1 for l in probe_train_labels if l == 0)
        probe_train_toxic = sum(1 for l in probe_train_labels if l == 1)
        probe_val_safe = sum(1 for l in probe_val_labels if l == 0)
        probe_val_toxic = sum(1 for l in probe_val_labels if l == 1)
        
        print(f"[Split] 探针训练集: {len(probe_train_texts)} "
              f"(安全={probe_train_safe}, 有害={probe_train_toxic}) - 已平衡到1:1")
        print(f"[Split] 探针验证集: {len(probe_val_texts)} "
              f"(安全={probe_val_safe} ({probe_val_safe/len(probe_val_texts):.1%}), "
              f"有害={probe_val_toxic} ({probe_val_toxic/len(probe_val_texts):.1%})) - 保持原始分布")
    else:
        # 不平衡模式：直接按比例划分（不推荐）
        train_shuffled = train_indices.copy()
        rng.shuffle(train_shuffled)
        probe_train_indices = train_shuffled[:probe_train_size]
        probe_val_indices = train_shuffled[probe_train_size:probe_train_size + probe_val_size]
        
        probe_train_texts = [train_texts[i] for i in probe_train_indices]
        probe_train_labels = [train_labels[i] for i in probe_train_indices]
        probe_val_texts = [train_texts[i] for i in probe_val_indices]
        probe_val_labels = [train_labels[i] for i in probe_val_indices]
        
        print(f"[Split] 探针训练集: {len(probe_train_texts)} (未平衡)")
        print(f"[Split] 探针验证集: {len(probe_val_texts)} (未平衡)")
    
    # 测试集（完全隔离，不用于训练）
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # 验证划分结果
    total_used = len(probe_train_texts) + len(probe_val_texts) + len(test_texts)
    if total_used != len(texts):
        print(f"[Warning] 数据划分不完整: 已使用={total_used}, 总数={len(texts)}")
    
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
        "--use_improved_split",
        action="store_true",
        help="使用改进的数据划分函数（解决数据使用率低和验证集不平衡问题）",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例（使用改进划分时，默认0.15即15%）",
    )
    parser.add_argument(
        "--min_val_samples_per_class",
        type=int,
        default=100,
        help="验证集每类最少样本数（使用改进划分时，默认100）",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集占总数据的比例（使用优化划分时，默认0.15即15%）",
    )
    parser.add_argument(
        "--use_optimized_split",
        action="store_true",
        help="使用优化的数据划分函数（优先分离验证集，自动计算最优比例，最大化数据利用率）",
    )
    parser.add_argument(
        "--use_doc_ratios",
        action="store_true",
        help="按文档划分：80% 训练池+20% 测试，训练池内 72.2% 探针训练、11.1% 验证、16.7% 神经元分析；测试/验证保持有害安全比例。需与 --use_optimized_split 同时使用",
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
        help="6:2:2 剩余全作训练集：60%% 训练、20%% 验证、20%% 测试；与 --use_optimized_split 同用",
    )
    parser.add_argument(
        "--train_safe_ratio_622_full",
        type=float,
        default=1.5,
        help="6:2:2 全训练时训练集安全:有害比例，默认 1.5；0 表示不讲究比例",
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
        "--train_safe_ratio",
        type=float,
        default=1.0,
        help="训练集安全/有害比例（默认1.0=1:1平衡，2.0=2:1，3.0=3:1，可提高数据利用率）",
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

        # 按照数据集划分文档进行划分
        if args.use_optimized_split:
            print("\n" + "="*60)
            if args.use_ratio_6_2_2_full_train:
                r = f"安全:有害={args.train_safe_ratio_622_full}:1" if args.train_safe_ratio_622_full > 0 else "不讲究比例"
                print(f"使用 6:2:2 剩余全作训练集：训练 60%、验证 20%、测试 20%，训练集 {r}")
            elif args.use_ratio_6_2_2:
                print("使用 6:2:2 分层划分：训练 60%、验证 20%、测试 20%，每份有害/安全比例与总体相同")
            elif args.use_doc_ratios:
                print("使用按文档划分（80% 训练池+20% 测试，训练池内 72.2%/11.1%/16.7%）：")
                print("测试集、验证集均保持与全集一致的有害/安全比例")
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
                    balance_train=True,
                    train_safe_ratio=args.train_safe_ratio,
                    balance_val=False,
                    balance_test=False,
                    min_test_toxic=args.min_test_toxic,
                    min_val_toxic=args.min_val_toxic,
                    seed=args.seed,
                    use_doc_ratios=args.use_doc_ratios,
                    use_ratio_6_2_2=args.use_ratio_6_2_2,
                    probe_val_ratio_in_train=args.probe_val_ratio_in_train,
                    use_ratio_6_2_2_full_train=args.use_ratio_6_2_2_full_train,
                    train_safe_ratio_622_full=args.train_safe_ratio_622_full if args.train_safe_ratio_622_full > 0 else None,
                )
        elif args.use_improved_split:
            print("\n" + "="*60)
            print("使用改进的数据划分策略：")
            print("1. 先为验证集预留足够的有害样本（确保验证集平衡）")
            print("2. 然后平衡训练集")
            print("3. 使用更多数据（降低测试集比例）")
            print("="*60 + "\n")
            print("⚠️  注意: 推荐使用 --use_optimized_split 以获得更好的数据利用率\n")
            
            probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
                split_data_improved(
                    texts=all_texts,
                    labels=all_labels,
                    test_ratio=args.test_ratio,
                    min_val_samples_per_class=args.min_val_samples_per_class,
                    balance_train=True,
                    balance_val=True,  # 验证集也平衡，确保评估有效
                    seed=args.seed,
                )
        else:
            print("\n" + "="*60)
            print("按照数据集划分文档进行数据划分：")
            print("1. 总体划分：训练集(80%) + 测试集(20%)")
            print("2. 从训练集中划分：探针训练集(72.2%, 1:1平衡) + 探针验证集(11.1%, 原始分布)")
            print("="*60 + "\n")
            print("⚠️  注意: 此方法可能导致验证集不平衡，建议使用 --use_improved_split\n")

            probe_train_texts, probe_train_labels, probe_val_texts, probe_val_labels, test_texts, test_labels = \
                split_data_according_to_doc(
                    texts=all_texts,
                    labels=all_labels,
                    test_ratio=0.2,
                    probe_train_ratio=0.722,  # 6.5/(6.5+1+1.5)
                    probe_val_ratio=0.111,    # 1/(6.5+1+1.5)
                    balance_probe_train=True,
                    seed=args.seed,
                )

        # 使用探针训练集和验证集
        texts = probe_train_texts
        labels = probe_train_labels
        val_texts = probe_val_texts
        val_labels = probe_val_labels

        # 打乱探针训练集顺序
        combined = list(zip(texts, labels))
        random.Random(args.seed).shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

        print(f"\n[Final] 探针训练集: {len(texts)} samples")
        print(f"[Final] 探针验证集: {len(val_texts)} samples")
        print(f"[Final] 测试集（已隔离）: {len(test_texts)} samples\n")

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
    # 按照文档：探针训练集(1:1平衡)用于训练，探针验证集(原始分布)用于验证
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

    # 从缓存加载时，data_file/max_samples/batch_size 优先从 meta 取；test 仅在未使用缓存时有值
    _meta = meta if args.hidden_states_cache else {}
    metadata = {
        "model_id": "llama-3-8b",
        "created_at": datetime.utcnow().isoformat(),
        "data_file": _meta.get("data_file", str(args.data_file)),
        "data_split_method": "按照数据集划分文档（6.5:1:1.5方案）",
        "probe_train_samples": int(len(labels)),
        "probe_train_safe": int(sum(1 for l in labels if l == 0)),
        "probe_train_toxic": int(sum(1 for l in labels if l == 1)),
        "probe_val_samples": int(len(val_labels)),
        "probe_val_safe": int(sum(1 for l in val_labels if l == 0)),
        "probe_val_toxic": int(sum(1 for l in val_labels if l == 1)),
        "test_samples": int(len(test_texts)),
        "test_safe": int(sum(1 for l in test_labels if l == 0)),
        "test_toxic": int(sum(1 for l in test_labels if l == 1)),
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


if __name__ == "__main__":
    main()

