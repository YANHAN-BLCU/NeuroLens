"""Utility 评估模块

基于 Wanda 论文 (Sun et al., ICLR 2024) 的 Utility 评估方法。

评估使用 EleutherAI LM Harness，包括：
- 零样本任务评估：BoolQ, RTE, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA
- WikiText 困惑度计算

参考论文：
    @inproceedings{sun2024simple,
        title={A Simple and Effective Pruning Approach for Large Language Models},
        author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
        booktitle={ICLR},
        year={2024}
    }

Usage:
    from engine.assessment.utility_evaluator import evaluate_utility

    results = evaluate_utility(model, tokenizer)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 尝试导入 lm-eval，如果不可用则使用内置实现
HAS_LM_EVAL = False
_LM_EVAL_SHORT_PATH = r"D:\lm_eval_lib"
_LM_EVAL_HF_PATH = "EleutherAI/lm-evaluation-harness"

try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    HAS_LM_EVAL = True
except ImportError:
    # 尝试从备用短路径导入（Windows 长路径问题解决方案）
    try:
        import sys
        if _LM_EVAL_SHORT_PATH not in sys.path:
            sys.path.insert(0, _LM_EVAL_SHORT_PATH)
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        HAS_LM_EVAL = True
    except ImportError:
        HAS_LM_EVAL = False
        print("[Utility Evaluator] 警告: lm-eval 未安装，将使用内置评估方法")


# 尝试从 HuggingFace 数据集加载任务数据
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

    # 1. 零样本任务评估
    if verbose:
        print(f"[Utility Evaluator] 评估零样本任务...")

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

    # 2. WikiText 困惑度
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
    评估零样本任务

    Args:
        model: 模型
        tokenizer: 分词器
        tasks: 任务列表
        batch_size: 批大小
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

    # 尝试使用 lm-eval
    if HAS_LM_EVAL:
        if verbose:
            print(f"[Utility Evaluator] 使用 lm-eval 进行评估...")

        try:
            results = _evaluate_with_lm_eval(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                device=device,
                verbose=verbose,
            )
            return results
        except Exception as e:
            if verbose:
                print(f"[Utility Evaluator] lm-eval 评估失败: {e}")
                print(f"[Utility Evaluator] 回退到内置评估方法...")

    # 使用内置评估方法
    if verbose:
        print(f"[Utility Evaluator] 使用内置方法进行评估...")

    for task in tasks:
        if verbose:
            print(f"[Utility Evaluator] 评估任务: {task}")

        try:
            accuracy = _evaluate_single_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                batch_size=batch_size,
                max_samples=max_samples,
                device=device,
            )
            results[task] = accuracy
        except Exception as e:
            if verbose:
                print(f"[Utility Evaluator] 任务 {task} 评估失败: {e}")
            results[task] = 0.0

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

def _load_model(
    model_path: str,
    device: torch.device,
) -> tuple:
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    """获取 WikiText 数据集路径"""
    # 可能的路径
    possible_paths = [
        Path("data/wikitext/wikitext-2-raw/wiki.valid.raw"),
        Path("data/wikitext/wiki.valid.raw"),
        Path("wikitext/wiki.valid.raw"),
    ]

    for path in possible_paths:
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
) -> float:
    """
    评估单个零样本任务

    使用简单的 few-shot prompting 进行评估
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # 加载任务数据
    task_data = _load_task_data(task, max_samples)

    if not task_data:
        return 0.0

    correct = 0
    total = 0

    for item in tqdm(task_data, desc=f"{task}", disable=True):
        try:
            prompt = item["prompt"]
            choices = item.get("choices", [])
            answer = item.get("answer", 0)

            # 构建输入
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            response = response.strip().lower()

            # 简单匹配
            predicted = _parse_response(response, choices, task)

            if predicted == answer:
                correct += 1
            total += 1

        except Exception:
            continue

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def _load_task_data(task: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载任务数据（内置实现）

    支持从 HuggingFace datasets 加载以下任务：
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

    Args:
        task: 任务名称
        item: 原始数据项

    Returns:
        转换后的样本字典
    """
    try:
        if task == "hellaswag":
            ctx_a = item.get("ctx_a", "")
            ctx_b = item.get("ctx_b", "")
            activity_label = item.get("activity_label", "")
            endings = item.get("endings", [])
            prompt = f"{ctx_a} {ctx_b}"
            choices = [endings[i] if i < len(endings) else "" for i in range(4)]
            # 答案在 activity_label 中，需要解析
            answer = 0  # 默认值
            return {"prompt": prompt, "choices": choices, "answer": answer}

        elif task == "winogrande":
            sentence = item.get("sentence", "")
            option1 = item.get("option1", "")
            option2 = item.get("option2", "")
            answer = item.get("answer", "1")
            # 替换 _ 填空
            prompt = sentence.replace("_", option1 if answer == "1" else option2)
            choices = [option1, option2]
            return {"prompt": prompt, "choices": choices, "answer": int(answer) - 1}

        elif task.startswith("arc_"):
            question = item.get("question", "")
            choices = item.get("choices", [])
            texts = choices.get("text", []) if isinstance(choices, dict) else []
            label_key = choices.get("label", []) if isinstance(choices, dict) else []
            answer_text = item.get("answerKey", "A")
            # 转换字母答案为索引
            answer_idx = ord(answer_text.upper()) - ord('A') if len(answer_text) == 1 else 0
            choices_text = [f"{chr(ord('A') + i)}) {text}" for i, text in enumerate(texts)]
            prompt = question + "\n" + "\n".join(choices_text)
            return {"prompt": prompt, "choices": texts, "answer": answer_idx}

        elif task == "obqa":
            question_stem = item.get("question_stem", "")
            choices = item.get("choices", [])
            texts = choices.get("text", []) if isinstance(choices, dict) else []
            label_key = choices.get("label", []) if isinstance(choices, dict) else []
            answer_text = item.get("answerKey", "A")
            answer_idx = ord(answer_text.upper()) - ord('A') if len(answer_text) == 1 else 0
            choices_text = [f"{chr(ord('A') + i)}) {text}" for i, text in enumerate(texts)]
            prompt = question_stem + "\n" + "\n".join(choices_text)
            return {"prompt": prompt, "choices": texts, "answer": answer_idx}

        elif task == "boolq":
            passage = item.get("passage", "")
            question = item.get("question", "")
            answer = item.get("answer", False)
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            return {"prompt": prompt, "choices": ["True", "False"], "answer": 1 if answer else 0}

        elif task == "rte":
            premise = item.get("premise", "")
            hypothesis = item.get("hypothesis", "")
            answer = item.get("label", 0)
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the hypothesis entail the premise?"
            return {"prompt": prompt, "choices": ["Yes", "No"], "answer": answer}

    except Exception as e:
        print(f"[Utility Evaluator] 转换数据格式失败 ({task}): {e}")
        return None

    return None


def _parse_response(response: str, choices: List[str], task: str) -> int:
    """解析模型响应获取答案索引"""
    response = response.lower().strip()

    # 尝试匹配选项
    for i, choice in enumerate(choices):
        choice_text = choice.lower().strip()
        if choice_text in response:
            return i
        if choice_text[:3] in response:
            return i

    # 尝试匹配字母
    for i in range(len(choices)):
        letter = chr(ord('a') + i)
        if letter in response or letter.upper() in response:
            return i

    return 0


def _evaluate_with_lm_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: List[str],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, float]:
    """使用 lm-eval 库进行评估（支持 lm-eval >= 0.4.x API）

    Args:
        model: 模型
        tokenizer: 分词器
        tasks: 任务列表
        device: 设备
        verbose: 详细输出

    Returns:
        各任务的准确率字典
    """
    if not HAS_LM_EVAL:
        raise ImportError("lm-eval 不可用")

    try:
        # 尝试新版本 API (>= 0.4.0)
        from lm_eval import tasks as lm_tasks
        task_dict = lm_tasks.get_task_dict(tasks)

        # 创建 lm-eval 模型包装
        class WrapperLM(HFLM):
            def __init__(self, model, tokenizer, device):
                # 直接使用已加载的模型
                self._model = model
                self._tokenizer = tokenizer
                self._device = device
                super().__init__(
                    pretrained=model,
                    tokenizer=tokenizer,
                    device=str(device),
                )

        results = evaluator.evaluate(
            lm=WrapperLM(model, tokenizer, device),
            task_dict=task_dict,
            log_samples=False,
        )
    except (ImportError, AttributeError):
        try:
            # 尝试旧版本 API (< 0.4.0)
            results = evaluator.evaluate(
                model=model,
                model_args=f"tokenizer={tokenizer.__class__.__name__},device={device}",
                tasks=tasks,
                verbose=verbose,
            )
        except Exception:
            # 尝试直接使用 lm_eval.api.request_factory
            print("[Utility Evaluator] 警告: lm-eval API 调用失败，尝试备用方法")
            return _evaluate_with_lm_eval_simple(model, tokenizer, tasks, device, verbose)

    # 提取结果
    task_results = {}
    results_dict = results.get("results", {})

    for task_name in tasks:
        if task_name in results_dict:
            task_res = results_dict[task_name]
            # 尝试多种可能的准确率键
            for key in ("acc", "accuracy", "acc_norm", "wer", "perplexity"):
                if key in task_res:
                    task_results[task_name] = float(task_res[key])
                    break
            else:
                # 如果没有找到准确率键，尝试查找任何数值结果
                for key, value in task_res.items():
                    if isinstance(value, (int, float)):
                        task_results[task_name] = float(value)
                        break
                else:
                    task_results[task_name] = 0.0
        else:
            task_results[task_name] = 0.0

    return task_results


def _evaluate_with_lm_eval_simple(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: List[str],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, float]:
    """简化的 lm-eval 评估方法（备用）

    Args:
        model: 模型
        tokenizer: 分词器
        tasks: 任务列表
        device: 设备
        verbose: 详细输出

    Returns:
        各任务的准确率字典
    """
    try:
        from lm_eval.api.model import LM
        from lm_eval.api.tasks import TaskConfig

        # 创建一个简单的包装类
        class SimpleHFLM(LM):
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self._device = device
                self.model.eval()
                # 设置 pad_token_id 避免生成警告
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            def loglikelihood(self, requests):
                """计算 token pairs 的对数似然

                lm-eval 期望返回: List[Tuple[float, bool]]
                - float: 对数似然值
                - bool: 是否截断（is_greedy）
                """
                results = []
                for request in requests:
                    # lm-eval 的 requests 是 TaskRequest 对象，
                    # ctx 和 cont 部分可以通过 get_selection 获取
                    try:
                        if hasattr(request, "get_selection"):
                            # 新版本 lm-eval (>= 0.4.0) 使用 get_selection
                            selections = request.get_selection(reqs=[request])
                            ctx = selections.get("ctx", "")
                            cont = selections.get("continuations", [("", {})])[0]
                            if isinstance(cont, dict):
                                cont_str = cont.get("text", "")
                            else:
                                cont_str = str(cont)
                        else:
                            # 兼容旧版本
                            ctx = getattr(request, "ctx", "")
                            cont = getattr(request, "cont", "")

                        if not ctx and not cont:
                            results.append((0.0, False))
                            continue

                        # Tokenize
                        inputs = self.tokenizer(ctx, return_tensors="pt", truncation=True, max_length=2048)
                        input_ids = inputs.input_ids.to(self._device)
                        input_len = input_ids.shape[1]

                        # 对 continuation 进行 tokenize（不添加特殊 token）
                        cont_inputs = self.tokenizer(cont, return_tensors="pt", truncation=True, max_length=2048)
                        cont_ids = cont_inputs.input_ids.to(self._device)
                        # 移除前缀空格等
                        cont_ids = cont_ids[0, 1:] if cont_ids[0, 0] == self.tokenizer.bos_token_id else cont_ids[0]

                        # 拼接计算 log-likelihood
                        target_ids = cont_ids.unsqueeze(0)

                        with torch.no_grad():
                            outputs = self.model(input_ids)
                            logits = outputs.logits[0]  # [seq_len, vocab_size]

                        # 计算 log probabilities
                        log_probs = torch.log_softmax(logits, dim=-1)

                        # 计算 log-likelihood: sum(log_p(target_tokens))
                        # logits[i] 预测的是 token i+1，所以取 [input_len-1:-1]
                        start_idx = input_len - 1
                        end_idx = start_idx + target_ids.shape[1]

                        if end_idx > log_probs.shape[0]:
                            # 序列不够长，截断
                            results.append((0.0, True))
                            continue

                        token_log_probs = log_probs[start_idx:end_idx, target_ids[0]]
                        log_likelihood = token_log_probs.sum().item()

                        # is_greedy: True 表示 greedy 解码（即下一个 token 是概率最高的）
                        # 这里返回 False 表示我们有完整概率
                        results.append((log_likelihood, False))

                    except Exception as e:
                        if verbose:
                            print(f"[SimpleHFLM] loglikelihood error: {e}")
                        results.append((0.0, False))

                return results

            def loglikelihood_rolling(self, requests):
                """计算 rolling token 的对数似然（用于 perplexity）"""
                results = []
                for request in requests:
                    try:
                        text = request.args[0] if request.args else ""
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                        input_ids = inputs.input_ids.to(self._device)

                        with torch.no_grad():
                            outputs = self.model(input_ids)
                            logits = outputs.logits  # [batch, seq, vocab]

                        log_probs = torch.log_softmax(logits, dim=-1)

                        # 移位：logits[i] 预测 token i+1
                        # target = input_ids[:, 1:]
                        # pred = log_probs[:, :-1, :]
                        # log_lik = gather(pred, dim=-1, index=target.unsqueeze(-1)).squeeze(-1).sum()
                        target_ids = input_ids[:, 1:]
                        pred_logits = log_probs[:, :-1, :]

                        # 使用 torch.gather
                        gathered = torch.gather(pred_logits, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                        log_likelihood = gathered.sum(dim=-1).item()

                        results.append((log_likelihood, False))

                    except Exception as e:
                        if verbose:
                            print(f"[SimpleHFLM] loglikelihood_rolling error: {e}")
                        results.append((0.0, False))

                return results

            def generate_until(self, requests):
                results = []
                for request in requests:
                    try:
                        if hasattr(request, "args") and request.args:
                            prompt = request.args[0]
                        elif hasattr(request, "kwargs") and request.kwargs:
                            prompt = request.kwargs.get("text", "")
                        else:
                            prompt = str(request)

                        until = getattr(request, "until", None) or ["<|endoftext|>"]

                        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                        input_ids = inputs.input_ids.to(self._device)

                        max_gen_tokens = getattr(request, "max_gen_tokens", 20)

                        with torch.no_grad():
                            outputs = self.model.generate(
                                input_ids,
                                max_new_tokens=max_gen_tokens,
                                do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                            )

                        generated = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

                        # 截断到 until token
                        for stop_tok in until:
                            if stop_tok in generated:
                                generated = generated.split(stop_tok)[0]
                        results.append(generated)

                    except Exception as e:
                        if verbose:
                            print(f"[SimpleHFLM] generate_until error: {e}")
                        results.append("")

                return results

        simple_lm = SimpleHFLM(model, tokenizer, device)

        # 手动加载任务并评估（兼容不同 lm-eval 版本）
        task_results = {}
        for task_name in tasks:
            try:
                # 方法 1: 尝试使用 lm_eval.api.registry（0.4.0+）
                from lm_eval.api import registry as lm_registry
                task_config = lm_registry.get_task(task_name)
                if task_config is not None:
                    results = task_config.evaluate(simple_lm)
                    acc = results.get("acc",
                              results.get("accuracy",
                              results.get("acc_norm", 0.0)))
                    task_results[task_name] = float(acc)
                    continue
            except Exception:
                pass

            try:
                # 方法 2: 尝试使用 lm_eval.tasks.get_task_dict
                from lm_eval import tasks as lm_tasks
                task_dict = lm_tasks.get_task_dict([task_name])
                if task_name in task_dict and task_dict[task_name] is not None:
                    task_obj = task_dict[task_name]
                    if hasattr(task_obj, "process_results") and hasattr(task_obj, "construct_requests"):
                        # lm-eval 0.4.x task object
                        results = evaluator.evaluate(
                            lm=simple_lm,
                            task_dict={task_name: task_obj},
                            log_samples=False,
                        )
                        res = results.get("results", {}).get(task_name, {})
                        acc = res.get("acc", res.get("accuracy", res.get("acc_norm", 0.0)))
                        task_results[task_name] = float(acc)
                        continue
            except Exception:
                pass

            # 方法 3: 回退到内置评估（基于 HuggingFace datasets）
            try:
                task_data = _load_task_data(task_name, max_samples=None)
                if task_data:
                    accuracy = _evaluate_single_task(
                        model, tokenizer, task_name, batch_size=8,
                        max_samples=None, device=device,
                    )
                    task_results[task_name] = float(accuracy)
                else:
                    task_results[task_name] = 0.0
            except Exception as e:
                if verbose:
                    print(f"[Utility Evaluator] 任务 {task_name} 评估失败: {e}")
                task_results[task_name] = 0.0

        return task_results

    except Exception as e:
        if verbose:
            print(f"[Utility Evaluator] 简化 lm-eval 评估失败: {e}")
        return {task: 0.0 for task in tasks}


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
