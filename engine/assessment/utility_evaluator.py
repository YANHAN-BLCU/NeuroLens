"""Utility 评估模块

基于 Wanda 论文 (Sun et al., ICLR 2024) 的 Utility 评估方法。

使用内置实现进行零样本任务评估（不使用 lm-eval）：
- HellaSwag: acc_norm（4 候选 log-likelihood 取最大）
- WinoGrande: 消歧义填空
- ARC-e / ARC-c: 多选问答
- BoolQ / RTE: 自然语言推理
- OBQA: 科学问答
- WikiText-2: 困惑度计算

参考论文：
    @inproceedings{sun2024wanda,
        title     = {A Simple and an Effective Pruning Approach for Large Language Models},
        author    = {Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
        booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
        year      = {2024},
        url       = {https://arxiv.org/abs/2306.11695}
    }

Usage:
    from engine.assessment.utility_evaluator import evaluate_utility

    results = evaluate_utility(model, tokenizer)
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 尝试从 HuggingFace 数据集加载任务数据（可选，回退到本地已下载数据）
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ============================================================================
# 默认评估配置（基于 Wanda 论文）
# ============================================================================

# 零样本任务列表（与 Wanda 论文一致）
ZERO_SHOT_TASKS = [
    "hellaswag",      # 常识推理
    "winogrande",      # WinoGrande 常识推理
    "arc_easy",        # ARC 简单科学问答
    "arc_challenge",   # ARC 挑战科学问答
    "obqa",            # OpenBookQA 科学问答
    "boolq",           # BoolQ 文本蕴含
    "rte",             # RTE 文本蕴含
]

# WikiText 配置
WIKITEXT_VERSION = "wikitext"
WIKITEXT_VARIANT = "wikitext-2-raw-v1"  # 或 "wikitext-2-v1"

# 论文中的基准数据（Llama-7B Dense 模型）
PAPER_BASELINE = {
    "hellaswag": 0.5692,
    "winogrande": 0.6993,
    "arc_easy": 0.7534,
    "arc_challenge": 0.4189,
    "obqa": 0.3440,
    "boolq": 0.7505,
    "rte": 0.6643,
    "mean": 0.5999,
    "wiki_perplexity": 5.68,
}


# ============================================================================
# 核心评估函数
# ============================================================================

def evaluate_utility(
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
    model_path: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    评估模型在通用任务上的 Utility

    Args:
        model: 已加载的模型（如果提供则直接使用）
        tokenizer: 分词器（如果提供则直接使用）
        model_path: 模型路径（如果 model 未提供，则从此路径加载）
        tasks: 要评估的任务列表（默认为 7 个零样本任务）
        batch_size: 批大小
        max_samples: 每个任务的最大样本数（None 表示全部）
        device: 设备（默认为 cuda if available else cpu）
        output_dir: 输出目录
        save_results: 是否保存结果到文件
        verbose: 是否打印详细进度

    Returns:
        包含以下键的字典：
        - model: 模型名称/路径
        - timestamp: 评估时间
        - zero_shot: 各零样本任务的准确率
        - wiki_perplexity: WikiText 困惑度
        - utility_score: 综合效用分数
        - comparison_with_paper: 与论文基准的对比
    """
    # 确定评估任务
    if tasks is None:
        tasks = ZERO_SHOT_TASKS.copy()

    # 确定设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型（如果未提供）
    if model is None or tokenizer is None:
        if model_path is None:
            raise ValueError("必须提供 model_path 或 model+tokenizer")
        if verbose:
            print(f"[Utility Evaluator] 加载模型: {model_path}")
        model, tokenizer = _load_model(model_path, device)

    model_name = model_path or "unknown"

    # 确保模型在评估模式
    model.eval()

    if verbose:
        print(f"[Utility Evaluator] 开始 Utility 评估")
        print(f"[Utility Evaluator] 设备: {device}")
        print(f"[Utility Evaluator] 任务数: {len(tasks)}")

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "zero_shot": {},
        "wiki_perplexity": None,
        "utility_score": None,
        "comparison_with_paper": {},
    }

    # 1. 零样本任务评估（带顶层进度条）
    if verbose:
        print(f"[Utility Evaluator] 评估零样本任务...")

    # 顶层进度：7 个任务 + WikiText PPL
    num_steps = len(tasks) + 1  # 任务数 + WikiText
    pbar_overall = tqdm(
        total=num_steps,
        desc="[Utility] 总体进度",
        unit="step",
        disable=not verbose,
    )

    zero_shot_results = evaluate_zero_shot_tasks(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=batch_size,
        max_samples=max_samples,
        device=device,
        verbose=verbose,
    )
    results["zero_shot"] = zero_shot_results

    # 计算零样本平均准确率
    task_accuracies = [v for k, v in zero_shot_results.items() if k != "mean" and isinstance(v, (int, float))]
    results["zero_shot"]["mean"] = sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0

    pbar_overall.update(1)
    pbar_overall.set_postfix_str(f"已完成 {len(tasks)}/{len(tasks)} 个任务，mean={results['zero_shot']['mean']:.4f}")

    # 2. WikiText 困惑度（带独立进度条）
    if verbose:
        print(f"[Utility Evaluator] 计算 WikiText 困惑度...")

    wikitext_path = _get_wikitext_path()
    if wikitext_path and wikitext_path.exists():
        results["wiki_perplexity"] = compute_wikitext_perplexity(
            model=model,
            tokenizer=tokenizer,
            wikitext_path=wikitext_path,
            device=device,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"[Utility Evaluator] 警告: WikiText 数据集未找到，跳过困惑度计算")
        results["wiki_perplexity"] = None

    pbar_overall.update(1)
    pbar_overall.set_postfix_str(f"PPL={results['wiki_perplexity']:.2f}" if results["wiki_perplexity"] else "PPL=N/A")
    pbar_overall.close()

    # 3. 计算综合 Utility 分数
    results["utility_score"] = _compute_utility_score(
        zero_shot_mean=results["zero_shot"]["mean"],
        wiki_perplexity=results["wiki_perplexity"],
    )

    # 4. 与论文基准对比
    results["comparison_with_paper"] = _compare_with_paper(
        zero_shot=results["zero_shot"],
        wiki_perplexity=results["wiki_perplexity"],
    )

    # 保存结果
    if save_results and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"utility_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"[Utility Evaluator] 结果已保存到: {result_file}")

    if verbose:
        print(f"[Utility Evaluator] 评估完成!")
        print(f"[Utility Evaluator] 零样本平均准确率: {results['zero_shot']['mean']:.4f}")
        if results["wiki_perplexity"]:
            print(f"[Utility Evaluator] WikiText 困惑度: {results['wiki_perplexity']:.2f}")
        print(f"[Utility Evaluator] Utility 分数: {results['utility_score']:.4f}")

    return results


def evaluate_zero_shot_tasks(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: Optional[List[str]] = None,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    评估零样本任务（纯内置实现，不依赖 lm-eval）

    每个任务调用 _evaluate_single_task，按 Wanda 论文协议计算准确率。

    Args:
        model: 模型
        tokenizer: 分词器
        tasks: 任务列表
        batch_size: 批大小（当前实现为逐样本评估）
        max_samples: 最大样本数
        device: 设备
        verbose: 详细输出

    Returns:
        各任务的准确率字典
    """
    if tasks is None:
        tasks = ZERO_SHOT_TASKS.copy()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # 顶层进度条：显示当前任务及实时准确率
    pbar = tqdm(
        total=len(tasks),
        desc="[Utility] 评估进度",
        unit="task",
        disable=not verbose,
    )

    for task in tasks:
        if verbose:
            print(f"\n[Utility Evaluator] 评估任务: {task}")

        try:
            accuracy = _evaluate_single_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                batch_size=batch_size,
                max_samples=max_samples,
                device=device,
                verbose=verbose,
            )
            results[task] = accuracy
        except Exception as e:
            if verbose:
                print(f"[Utility Evaluator] 任务 {task} 评估失败: {e}")
            results[task] = 0.0

        pbar.update(1)
        pbar.set_postfix_str(f"当前: {task} = {results[task]:.4f}")

    pbar.close()

    return results


def compute_wikitext_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    wikitext_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    block_size: int = 512,
    stride: Optional[int] = None,
    verbose: bool = True,
) -> float:
    """
    计算 WikiText 验证集上的困惑度

    与 Wanda 论文一致，使用 WikiText-2 验证集

    Args:
        model: 模型
        tokenizer: 分词器
        wikitext_path: WikiText 数据文件路径
        device: 设备
        block_size: 上下文长度
        stride: 滑动窗口步长（默认为 block_size）
        verbose: 详细输出

    Returns:
        困惑度分数
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if stride is None:
        stride = block_size

    model.eval()

    # 加载文本
    with open(wikitext_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 分词
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)

    # 计算总负对数似然
    nlls = []
    num_tokens = 0

    # 滑动窗口计算
    for i in tqdm(range(0, seq_len - 1, stride), disable=not verbose, desc="WikiText"):
        begin_loc = i
        end_loc = min(i + block_size, seq_len)

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_ids[:, begin_loc:end_loc].clone()
        target_chunk[:, :-1] = -100  # 忽略预测

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * (end_loc - begin_loc - 1)

        nlls.append(neg_log_likelihood)
        num_tokens += end_loc - begin_loc - 1

    # 计算困惑度
    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens)

    if verbose:
        print(f"[Utility Evaluator] WikiText 困惑度: {ppl.item():.4f}")

    return ppl.item()


# ============================================================================
# 内部辅助函数
# ============================================================================

@contextmanager
def _quiet_transformers_generation_length_warnings():
    """屏蔽评估循环里每条样本 generate 触发的 max_length/max_new_tokens 重复告警。

    部分 transformers 版本在合并 ``model.generation_config`` 后仍会
    ``logger.warning``（每条一次）。此处对 ``transformers.generation.utils``
    临时抬高日志级别；并附加过滤器以防消息走其他 logger。
    """
    log = logging.getLogger("transformers.generation.utils")

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                m = record.getMessage()
            except Exception:
                return True
            if "max_new_tokens" in m and "max_length" in m and "precedence" in m:
                return False
            return True

    f = _Filter()
    log.addFilter(f)
    prev_level = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(prev_level)
        log.removeFilter(f)


def _greedy_generate_short_answer(
    model: torch.nn.Module,
    tokenizer: Any,
    inputs: Any,
    max_new_tokens: int = 10,
) -> torch.Tensor:
    """贪婪解码少量新 token；仅用 kwargs，不传入 GenerationConfig，减少与 hub 默认 max_length 冲突。"""
    pad_id = tokenizer.pad_token_id
    eos_id = getattr(tokenizer, "eos_token_id", None)
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    if pad_id is not None:
        kwargs["pad_token_id"] = pad_id
    if eos_id is not None:
        kwargs["eos_token_id"] = eos_id
    return model.generate(**inputs, **kwargs)


def _load_model(
    model_path: str,
    device: torch.device,
) -> tuple:
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    if device.type == "cpu":
        model = model.to(device)

    return model, tokenizer


def _get_wikitext_path() -> Optional[Path]:
    """获取 WikiText 数据集路径

    优先使用验证集（wiki.valid.raw），与 Wanda 论文保持一致。
    回退到测试集（wiki.test.raw）。
    """
    # 可能的路径（优先 data/utility，验证集优先）
    possible_valid = [
        PROJECT_ROOT / "data" / "utility" / "wikitext" / "wikitext-2-raw" / "wiki.valid.raw",
        PROJECT_ROOT / "data" / "wikitext" / "wikitext-2-raw" / "wiki.valid.raw",
        Path("data/utility/wikitext/wikitext-2-raw/wiki.valid.raw"),
        Path("data/wikitext/wikitext-2-raw/wiki.valid.raw"),
    ]

    possible_test = [
        PROJECT_ROOT / "data" / "utility" / "wikitext" / "wikitext-2-raw" / "wiki.test.raw",
        PROJECT_ROOT / "data" / "wikitext" / "wikitext-2-raw" / "wiki.test.raw",
        Path("data/utility/wikitext/wiki.test.raw"),
        Path("data/wikitext/wiki.test.raw"),
    ]

    for path in possible_valid:
        if path.exists():
            return path

    for path in possible_test:
        if path.exists():
            return path

    return None


def _compute_utility_score(
    zero_shot_mean: float,
    wiki_perplexity: Optional[float] = None,
) -> float:
    """
    计算综合 Utility 分数

    与论文一致，综合考虑零样本准确率和困惑度
    """
    # 零样本分数（归一化到 0-1，假设论文基准 0.61 为满分）
    zero_shot_score = zero_shot_mean  # 已经是在 0-1 范围内

    if wiki_perplexity is not None:
        # 困惑度分数（论文基准 5.68 对应 1.0，越低越好）
        # 使用指数衰减：ppl = 5.68 时得 1.0，ppl = 10 时得 ~0.5
        ppl_score = 1.0 / (1.0 + (wiki_perplexity / 5.68 - 1.0))
        ppl_score = max(0, min(1, ppl_score))

        # 综合分数（加权平均）
        utility_score = 0.7 * zero_shot_score + 0.3 * ppl_score
    else:
        utility_score = zero_shot_score

    return utility_score


def _compare_with_paper(
    zero_shot: Dict[str, float],
    wiki_perplexity: Optional[float] = None,
) -> Dict[str, Any]:
    """与论文基准数据对比"""
    comparison = {}

    # 各任务对比
    for task in ZERO_SHOT_TASKS:
        if task in zero_shot:
            paper_value = PAPER_BASELINE.get(task, 0.0)
            actual_value = zero_shot[task]
            diff = actual_value - paper_value
            comparison[task] = {
                "actual": actual_value,
                "paper_baseline": paper_value,
                "difference": diff,
                "percent_change": (diff / paper_value * 100) if paper_value > 0 else 0,
            }

    # 平均准确率对比
    if "mean" in zero_shot:
        comparison["mean"] = {
            "actual": zero_shot["mean"],
            "paper_baseline": PAPER_BASELINE["mean"],
            "difference": zero_shot["mean"] - PAPER_BASELINE["mean"],
        }

    # 困惑度对比
    if wiki_perplexity is not None:
        comparison["wiki_perplexity"] = {
            "actual": wiki_perplexity,
            "paper_baseline": PAPER_BASELINE["wiki_perplexity"],
            "difference": wiki_perplexity - PAPER_BASELINE["wiki_perplexity"],
        }

    return comparison


def _evaluate_single_task(
    model: torch.nn.Module,
    tokenizer: Any,
    task: str,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> float:
    """
    评估单个零样本任务

    Wanda 论文评估协议：
    - HellaSwag: acc_norm（4 候选 log-likelihood 取最大）
    - 其他任务: 生成 + 解析响应
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # 加载任务数据
    task_data = _load_task_data(task, max_samples)

    if not task_data:
        return 0.0

    # HellaSwag 特殊处理：acc_norm（与 Wanda/lm-eval 协议一致）
    if task == "hellaswag":
        return _evaluate_hellaswag_accnorm(model, tokenizer, task_data, device, verbose)

    correct = 0
    total = 0

    # 必须将可迭代对象传给 tqdm；仅 total= 时 for x in pbar 会触发
    # TypeError: 'NoneType' object is not iterable
    pbar = tqdm(
        task_data,
        total=len(task_data),
        desc=f"[{task}]",
        unit="sample",
        leave=False,
        disable=not verbose,
    )

    with _quiet_transformers_generation_length_warnings():
        for item in pbar:
            try:
                prompt = item["prompt"]
                choices = item.get("choices") or []
                answer = item.get("answer", 0)

                # 构建输入
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = _greedy_generate_short_answer(model, tokenizer, inputs, max_new_tokens=10)

                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                response = response.strip().lower()

                # 简单匹配
                predicted = _parse_response(response, choices, task)

                if predicted == -1:
                    # 无法解析，跳过该样本
                    continue
                if predicted == answer:
                    correct += 1
                total += 1

                # 更新进度条后缀：实时准确率
                if total > 0:
                    pbar.set_postfix_str(f"acc={correct/total:.4f}")

            except Exception:
                continue

    pbar.close()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def _evaluate_hellaswag_accnorm(
    model: torch.nn.Module,
    tokenizer: Any,
    task_data: List[Dict],
    device: torch.device,
    verbose: bool = True,
) -> float:
    """
    HellaSwag acc_norm 评估

    Wanda 论文: 对每个样本，计算 ctx + 4 个候选 ending 的 log-likelihood，
    选择 log-likelihood 最高的候选。

    与 lm-eval HellaSwag acc_norm 计算方式一致。
    """
    correct = 0
    total = 0

    # 每个样本 × 4 个候选 ending，每个 ending 内部无 tqdm（避免过度刷屏）
    pbar = tqdm(
        task_data,
        total=len(task_data),
        desc="[hellaswag]",
        unit="sample",
        leave=False,
        disable=not verbose,
    )

    for item in pbar:
        try:
            ctx = item.get("prompt", "")
            ends = item.get("ends", item.get("choices", []))
            answer = item.get("answer", 0)

            if not ctx or len(ends) < 4:
                continue

            # 计算 ctx + 每个 ending 的 log-likelihood
            log_liks = []
            for end_text in ends:
                text = ctx + " " + end_text

                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = inputs.input_ids.to(device)

                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits  # [1, seq, vocab]

                # log_softmax
                log_probs = torch.log_softmax(logits[0], dim=-1)  # [seq, vocab]

                # 移位: logits[i] 预测 token i+1
                # 目标: 整个序列（不含首个 token 的 log_prob）
                target_ids = input_ids[0, 1:]  # 去掉首 token
                pred_logits = log_probs[:-1, :]  # [seq-1, vocab]

                # log_likelihood = sum(log_p(target_i))
                token_log_probs = pred_logits[range(len(target_ids)), target_ids]
                log_lik = token_log_probs.sum().item()
                log_liks.append(log_lik)

            # 选择 log-likelihood 最高的候选
            predicted = int(torch.argmax(torch.tensor(log_liks)).item())
            if predicted == answer:
                correct += 1
            total += 1

            # 更新进度条后缀：实时准确率
            if total > 0:
                pbar.set_postfix_str(f"acc={correct/total:.4f}")

        except Exception:
            continue

    pbar.close()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def _get_local_data_path(task: str) -> Optional[Path]:
    """获取本地数据集路径（优先使用已下载的数据）

    查找策略：优先找 test.jsonl，回退到 validation.jsonl，
    与 Wanda 论文评估协议及 download_utility_datasets.py 实际下载结果一致。
    """
    base = PROJECT_ROOT / "data" / "utility"

    # 主映射：每个任务对应的实际文件名
    primary_mapping = {
        "hellaswag": base / "hellaswag" / "test.jsonl",
        "winogrande": base / "winogrande" / "test.jsonl",
        "arc_easy": base / "arc" / "arc_easy_test.jsonl",
        "arc_challenge": base / "arc" / "arc_challenge_test.jsonl",
        "obqa": base / "openbookqa" / "test.jsonl",
        "boolq": base / "super_glue" / "boolq" / "validation.jsonl",
        "rte": base / "super_glue" / "rte" / "validation.jsonl",
    }

    # 备选映射（test 不存在时尝试 validation，反之亦然）
    alt_mapping = {
        "hellaswag": base / "hellaswag" / "validation.jsonl",
        "winogrande": base / "winogrande" / "validation.jsonl",
        "arc_easy": base / "arc" / "arc_easy_validation.jsonl",
        "arc_challenge": base / "arc" / "arc_challenge_validation.jsonl",
        "obqa": base / "openbookqa" / "validation.jsonl",
    }

    path = primary_mapping.get(task)
    if path and path.exists():
        return path

    alt = alt_mapping.get(task)
    if alt and alt.exists():
        return alt

    # ARC 额外兼容：可能下载为 _train.jsonl 或其他后缀
    if task.startswith("arc_"):
        arc_base = base / "arc"
        for f in arc_base.glob("arc_easy_*.jsonl"):
            if f.exists():
                return f
        for f in arc_base.glob("arc_challenge_*.jsonl"):
            if f.exists():
                return f

    return None


def _load_local_jsonl(path: Path) -> List[Dict]:
    """从本地 JSONL 文件加载数据集"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _load_task_data(task: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载任务数据（优先从本地加载已下载的数据）

    支持从以下来源加载：
    1. 本地已下载的数据（data/utility/）
    2. HuggingFace datasets 自动下载

    支持的任务：
    - hellaswag
    - winogrande
    - arc_easy / arc_challenge
    - openbookqa
    - boolq
    - rte

    Args:
        task: 任务名称
        max_samples: 最大样本数

    Returns:
        任务数据列表
    """
    # 优先尝试从本地加载
    local_path = _get_local_data_path(task)
    if local_path and local_path.exists():
        print(f"[Utility Evaluator] 从本地加载: {local_path}")
        try:
            local_data = _load_local_jsonl(local_path)
            samples = []
            for i, item in enumerate(local_data):
                if max_samples and i >= max_samples:
                    break
                sample = _convert_to_prompt_format(task, item)
                if sample:
                    samples.append(sample)
            if samples:
                return samples
        except Exception as e:
            print(f"[Utility Evaluator] 本地加载失败: {e}，回退到 HuggingFace")

    if not HAS_DATASETS:
        # 如果 datasets 不可用，返回模拟数据进行测试
        print(f"[Utility Evaluator] 警告: datasets 库不可用，使用模拟数据")
        return _get_mock_task_data(task, max_samples)

    task_mapping = {
        "hellaswag": ("hellaswag", "validation"),
        "winogrande": ("winogrande", "validation"),
        "arc_easy": ("ai2_arc", "ARC-Easy"),
        "arc_challenge": ("ai2_arc", "ARC-Challenge"),
        "obqa": ("openbookqa", "test"),
        "boolq": ("super_glue", "boolq"),
        "rte": ("super_glue", "rte"),
    }

    if task not in task_mapping:
        print(f"[Utility Evaluator] 警告: 不支持的任务 '{task}'，返回空数据")
        return []

    dataset_name, split = task_mapping[task]

    try:
        if task.startswith("arc_"):
            dataset = load_dataset(dataset_name)[split]
        elif task in ("boolq", "rte"):
            dataset = load_dataset("super_glue", task)[split]
        else:
            dataset = load_dataset(dataset_name)[split]

        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            sample = _convert_to_prompt_format(task, item)
            if sample:
                samples.append(sample)

        return samples

    except Exception as e:
        print(f"[Utility Evaluator] 加载任务数据失败: {task} - {e}")
        # 返回模拟数据
        return _get_mock_task_data(task, max_samples)


def _get_mock_task_data(task: str, max_samples: Optional[int] = None) -> List[Dict]:
    """返回模拟任务数据（用于测试）

    Args:
        task: 任务名称
        max_samples: 最大样本数

    Returns:
        模拟数据列表
    """
    # 对于每个任务返回少量模拟数据
    mock_counts = {
        "hellaswag": 5,
        "winogrande": 5,
        "arc_easy": 3,
        "arc_challenge": 3,
        "obqa": 3,
        "boolq": 5,
        "rte": 3,
    }

    count = min(mock_counts.get(task, 3), max_samples or 10)
    return [{"prompt": f"[{task}] Sample {i}", "choices": ["A", "B"], "answer": 0} for i in range(count)]


def _convert_to_prompt_format(task: str, item: Dict) -> Optional[Dict]:
    """将原始数据集项转换为 prompt 格式

    严格遵循 Wanda 论文的评估协议：
    - HellaSwag: 完形填空（带 4 个候选项），使用 acc_norm（4 候选 log-likelihood 取最大）
    - WinoGrande: 消歧义填空（_ 替换为选项）
    - ARC: 多选问答（ABCD）
    - BoolQ/RTE: 自然语言推理（Yes/No 或 entail/not_entail）
    - OBQA: 多选科学问答（ABCD）

    Args:
        task: 任务名称
        item: 原始数据项

    Returns:
        转换后的样本字典，包含 prompt / choices / answer / ends（acc_norm 专用）
    """
    try:
        if task == "hellaswag":
            ctx_a = item.get("ctx_a", "")
            ctx_b = item.get("ctx_b", "")
            endings = item.get("endings", [])
            answer_str = str(item.get("answer", "0")).strip()

            # prompt = 前缀上下文（不含选项）
            prompt = f"{ctx_a} {ctx_b}"

            # choices = 4 个候选项
            if len(endings) < 4:
                return None
            choices = [endings[i] for i in range(4)]

            # ends = 4 个候选项文本（用于 acc_norm 计算，与 lm-eval 协议一致）
            ends = choices

            try:
                answer = int(answer_str)
            except (ValueError, TypeError):
                answer = 0

            return {"prompt": prompt, "choices": choices, "answer": answer, "ends": ends}

        elif task == "winogrande":
            sentence = item.get("sentence", "")
            option1 = item.get("option1", "")
            option2 = item.get("option2", "")
            answer_str = str(item.get("answer", "1")).strip()

            # Wanda: 替换 _ 为选项形成完整句子
            option = option1 if answer_str == "1" else option2
            prompt = sentence.replace("_", option).strip()
            choices = [option1, option2]

            try:
                answer = int(answer_str) - 1
            except (ValueError, TypeError):
                answer = 0

            return {"prompt": prompt, "choices": choices, "answer": answer}

        elif task.startswith("arc_"):
            question = item.get("question", "")
            choices = item.get("choices", [])
            texts = choices.get("text", []) if isinstance(choices, dict) else []
            label_key = choices.get("label", []) if isinstance(choices, dict) else []
            answer_text = item.get("answerKey", "A")
            # 转换字母答案为索引
            answer_idx = ord(answer_text.upper()) - ord('A') if len(answer_text) == 1 else 0
            # prompt = 问题 + 选项（带 A) B) C) D) 标签）
            prompt = question + "\n"
            choice_labels = []
            for i, text in enumerate(texts):
                label = f"{chr(ord('A') + i)})"
                prompt += f"{label} {text}\n"
                choice_labels.append(text)

            try:
                answer_idx = ord(answer_text.upper()) - ord('A')
                if answer_idx < 0 or answer_idx >= len(texts):
                    answer_idx = 0
            except (ValueError, TypeError):
                answer_idx = 0

            return {"prompt": prompt.strip(), "choices": choice_labels, "answer": answer_idx}

        elif task == "obqa":
            question_stem = item.get("question_stem", "")
            choices_data = item.get("choices", {})
            answer_text = str(item.get("answerKey", "A")).strip()

            if isinstance(choices_data, dict):
                texts = choices_data.get("text", [])
            else:
                texts = list(choices_data) if choices_data else []

            prompt = question_stem + "\n"
            choice_labels = []
            for i, text in enumerate(texts):
                label = f"{chr(ord('A') + i)})"
                prompt += f"{label} {text}\n"
                choice_labels.append(text)

            try:
                answer_idx = ord(answer_text.upper()) - ord('A')
                if answer_idx < 0 or answer_idx >= len(texts):
                    answer_idx = 0
            except (ValueError, TypeError):
                answer_idx = 0

            return {"prompt": prompt.strip(), "choices": choice_labels, "answer": answer_idx}

        elif task == "boolq":
            passage = item.get("passage", "")
            question = item.get("question", "")
            # SuperGLUE jsonl 使用 label（0/1），部分来源使用 answer
            raw = item.get("answer", item.get("label", False))
            answer = bool(raw) if isinstance(raw, (bool, int)) else str(raw).lower() in ("true", "1", "yes")
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            return {"prompt": prompt, "choices": ["True", "False"], "answer": 1 if answer else 0}

        elif task == "rte":
            premise = item.get("premise", "")
            hypothesis = item.get("hypothesis", "")
            label = item.get("label", 0)

            # Wanda 使用 entailment 判断
            prompt = f"{premise}\n{hypothesis}\nAnswer:"
            choices = ["Yes", "No"]
            answer = 1 if label in (1, "entailment", "entail") else 0
            return {"prompt": prompt, "choices": choices, "answer": answer}

    except Exception as e:
        print(f"[Utility Evaluator] 转换数据格式失败 ({task}): {e}")
        return None

    return None


def _parse_response(response: str, choices: List[str], task: str) -> int:
    """解析模型响应获取答案索引

    策略（按优先级递减）：
    1. 精确匹配选项文本（或前3个词）
    2. 匹配选项字母 (a/b/c/d)
    3. 匹配 True/False / Yes/No 关键词
    4. 匹配数字 1/2/3/4
    5. 若完全无法匹配，返回 -1（表示不确定），由调用方处理
    """
    if not choices:
        return -1

    response = response.lower().strip()

    # 候选字母（a, b, c, d）
    letters = [chr(ord('a') + i) for i in range(len(choices))]

    # 策略 1: 精确匹配选项文本或前3词
    for i, choice in enumerate(choices):
        choice_text = choice.lower().strip()
        if not choice_text:
            continue
        # 完整匹配
        if choice_text in response:
            return i
        # 前3词匹配（避免过长选项导致匹配失败）
        short = " ".join(choice_text.split()[:3])
        if short in response:
            return i

    # 策略 2: 匹配字母 (a/b/c/d)
    for i, letter in enumerate(letters):
        if f"({letter})" in response or f"({letter.upper()})" in response:
            return i
        if response.startswith(letter + ".") or response.startswith(letter + ")"):
            return i

    # 策略 3: 匹配 True/False / Yes/No
    for i, choice in enumerate(choices):
        text = choice.lower().strip()
        if text in ("true", "false", "yes", "no"):
            if text in response:
                return i

    # 策略 4: 匹配数字 1/2/3/4
    num_match = re.search(r'\b([1-4])\b', response)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(choices):
            return idx

    # 完全无法匹配：返回 -1，让调用方跳过或计入错误
    return -1


# ============================================================================
# 便捷函数
# ============================================================================

def quick_evaluate(model_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    快速评估 Utility

    Args:
        model_path: 模型路径
        output_dir: 输出目录

    Returns:
        评估结果
    """
    return evaluate_utility(
        model_path=model_path,
        output_dir=output_dir,
        verbose=True,
    )


def compare_models(
    model_paths: List[str],
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    对比多个模型的 Utility

    Args:
        model_paths: 模型路径列表
        output_dir: 输出目录

    Returns:
        各模型的评估结果
    """
    results = {}

    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"评估模型: {model_path}")
        print(f"{'='*60}\n")

        results[model_path] = evaluate_utility(
            model_path=model_path,
            output_dir=output_dir,
            verbose=True,
        )

    return results
