#!/usr/bin/env python3
r"""
VA+TSFT 端到端串联脚本

完整流程：
    Step 1  [可选]  四象限分类（若 quadrant_file 已存在则跳过）
    Step 2           从四象限结果自动拆分 S-A- 和 S+A- 神经元
    Step 3           提取 / 构建 Refusal Templates
    Step 4           构建 Refusal-Guided 训练数据集
    Step 5           执行 VA+TSFT 两阶段微调（安全神经元 + 脆弱神经元）
    Step 6  [可选]  评估微调前后模型（ASR 对比）

推荐使用方式（已有四象限结果）：
    python scripts/run_vatsft_pipeline.py ^
        --quadrant-results outputs/quadrant_classification/quadrant_classification.json ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --evaluation-log logs/base_evaluation.jsonl ^
        --output outputs/vatsft_pipeline

最小命令（标准 TSFT，不含脆弱神经元反转）：
    python scripts/run_vatsft_pipeline.py ^
        --quadrant-results outputs/quadrant_classification/quadrant_classification.json ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --evaluation-log logs/base_evaluation.jsonl ^
        --output outputs/vatsft_pipeline ^
        --method tsft

指定自定义安全 / 脆弱神经元文件（跳过自动拆分）：
    python scripts/run_vatsft_pipeline.py ^
        --safety-neurons outputs/tsft_neurons/dedicated_safety_neurons.json ^
        --vulnerable-neurons outputs/tsft_neurons/vulnerable_neurons.json ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --evaluation-log logs/base_evaluation.jsonl ^
        --output outputs/vatsft_pipeline ^
        --method va-tsft

跳过模型加载和训练，仅预览拆分结果（dry-run）：
    python scripts/run_vatsft_pipeline.py ^
        --quadrant-results outputs/quadrant_classification/quadrant_classification.json ^
        --output outputs/vatsft_pipeline ^
        --dry-run
"""

import sys
import os
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- 路径处理 ----
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if _PROJECT_ROOT.exists() and (_PROJECT_ROOT / "engine").exists():
    sys.path.insert(0, str(_PROJECT_ROOT))
else:
    _ws = os.getenv("WORKSPACE_PATH", "/workspace")
    if os.path.exists(_ws):
        sys.path.insert(0, _ws)

# ---- 核心模块导入 ----
from engine.fine_tuning import (
    extract_refusal_templates,
    save_refusal_templates,
    load_refusal_templates,
    build_refusal_guided_dataset,
    load_dedicated_safety_neurons,
    tsft_finetune,
    vatft_finetune,
    identify_vulnerable_neurons,
)
from engine.fine_tuning.dataset_builder import save_dataset


# =============================================================================
# Step 1: 四象限结果加载与拆分
# =============================================================================

def _load_quadrant_as_dict(quadrant_file: str) -> Dict[Tuple[int, int], Dict]:
    """将四象限 JSON 加载为 {(layer, neuron): data} 格式。"""
    path = Path(quadrant_file)
    if not path.exists():
        raise FileNotFoundError(f"四象限文件不存在: {quadrant_file}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败 ({path}): {e}")

    neurons: Dict[Tuple[int, int], Dict] = {}
    for key, val in raw.items():
        if key in ("_statistics", "_metadata", "metadata", "statistics"):
            continue

        li = ni = None
        if isinstance(val, dict):
            li = val.get("layer_idx") or val.get("layer")
            ni = val.get("neuron_idx") or val.get("neuron")

        if li is None or ni is None:
            parts = key.split("_")
            if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "neuron":
                try:
                    li, ni = int(parts[1]), int(parts[3])
                except (ValueError, IndexError):
                    continue
            elif len(parts) == 2:
                try:
                    li, ni = int(parts[0]), int(parts[1])
                except ValueError:
                    continue

        if li is not None and ni is not None:
            neurons[(int(li), int(ni))] = val

    return neurons


def split_quadrant_to_files(
    quadrant_file: str,
    output_dir: Path,
    safety_only: bool = False,
    safety_align_min: Optional[float] = None,
    safety_align_max: Optional[float] = None,
    vulnerable_align_min: Optional[float] = None,
    vulnerable_align_max: Optional[float] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    将四象限结果拆分为 TSFT 可用的神经元 JSON 文件。

    Returns:
        (safety_neurons_path, vulnerable_neurons_path)
        若 safety_only=True，vulnerable_neurons_path 为 None。
    """
    neurons = _load_quadrant_as_dict(quadrant_file)
    print(f"[Pipeline] 四象限加载完成: {len(neurons)} 个神经元")

    safety_out: Dict[Tuple[int, int], Dict] = {}
    vulnerable_out: Dict[Tuple[int, int], Dict] = {}

    for (layer, neuron), data in neurons.items():
        q = data.get("quadrant", "")
        align = data.get("alignment", data.get("cosine_similarity", 0.0))

        if q == "S-A-":
            if safety_align_min is not None and align < safety_align_min:
                continue
            if safety_align_max is not None and align > safety_align_max:
                continue
            safety_out[(layer, neuron)] = data

        if not safety_only and q == "S+A-":
            if vulnerable_align_min is not None and align < vulnerable_align_min:
                continue
            if vulnerable_align_max is not None and align > vulnerable_align_max:
                continue
            vulnerable_out[(layer, neuron)] = data

    output_dir.mkdir(parents=True, exist_ok=True)

    def _serialize(src: Dict[Tuple[int, int], Dict], quadrant: str) -> Dict[str, Dict]:
        return {
            f"layer_{l}_neuron_{n}": {"layer_idx": l, "neuron_idx": n, **d}
            for (l, n), d in src.items()
        }

    def _save(sub: Dict, path: Path, quadrant: str, desc: str,
              neurons_key: str = "dedicated_safety_neurons"):
        if not sub:
            print(f"[Pipeline] 警告: 没有找到 {quadrant} 神经元，跳过")
            return None
        payload = {
            "metadata": {"source": quadrant, "num_neurons": len(sub), "description": desc},
            neurons_key: _serialize(sub, quadrant),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Pipeline] 保存 {len(sub)} 个 {quadrant} 神经元 → {path}")
        return path

    s_path = _save(
        safety_out,
        output_dir / "dedicated_safety_neurons.json",
        "S-A-",
        "专用安全神经元，用于标准 TSFT 和 VA+TSFT 阶段一",
    )
    v_path = _save(
        vulnerable_out,
        output_dir / "vulnerable_neurons.json",
        "S+A-",
        "脆弱神经元，用于 VA+TSFT 阶段二（负梯度反转）",
        neurons_key="vulnerable_neurons",
    ) if not safety_only else None

    # 打印统计摘要
    q_counts: Dict[str, int] = {}
    for d in [safety_out, vulnerable_out]:
        for v in d.values():
            q = v.get("quadrant", "?")
            q_counts[q] = q_counts.get(q, 0) + 1
    print(f"[Pipeline] 象限分布: {dict(sorted(q_counts.items()))}")
    print(f"[Pipeline] S-A- 安全神经元: {len(safety_out)}, S+A- 脆弱神经元: {len(vulnerable_out)}")

    return s_path, v_path


# =============================================================================
# Step 2: Refusal Templates 提取 / 加载
# =============================================================================

def prepare_refusal_templates(
    evaluation_log: str,
    templates_path: Optional[str],
    output_dir: Path,
    min_frequency: int = 2,
) -> str:
    """提取或加载 refusal templates，返回模板文件路径。"""
    if templates_path and Path(templates_path).exists():
        print(f"[Pipeline] 使用已有 refusal templates: {templates_path}")
        templates = load_refusal_templates(templates_path)
        return templates_path
    else:
        print(f"[Pipeline] 从评估日志提取 refusal templates: {evaluation_log}")
        templates = extract_refusal_templates(
            evaluation_log,
            min_frequency=min_frequency,
        )
        out = output_dir / "refusal_templates.json"
        save_refusal_templates(templates, str(out))
        print(f"[Pipeline] 保存 {len(templates)} 个 templates → {out}")
        return str(out)


# =============================================================================
# Step 3: 构建训练数据集
# =============================================================================

def build_training_dataset(
    evaluation_log: str,
    templates_path: str,
    output_dir: Path,
    only_successful: bool = True,
    min_tpl: int = 1,
    max_tpl: int = 3,
    dataset_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[List[Dict], Path]:
    """构建 refusal-guided 训练数据集。"""
    if dataset_path and Path(dataset_path).exists():
        print(f"[Pipeline] 使用已有数据集: {dataset_path}")
        from engine.fine_tuning.dataset_builder import load_dataset
        dataset = load_dataset(dataset_path)
        return dataset, Path(dataset_path)

    print(f"[Pipeline] 构建 refusal-guided 数据集...")
    templates = load_refusal_templates(templates_path)
    dataset = build_refusal_guided_dataset(
        evaluation_log_path=evaluation_log,
        refusal_templates=templates,
        output_path=str(output_dir / "dataset.jsonl"),
        only_successful_jailbreaks=only_successful,
        min_templates_per_prompt=min_tpl,
        max_templates_per_prompt=max_tpl,
        seed=seed,
    )

    if not dataset:
        raise RuntimeError("数据集构建失败（样本为空）")

    return dataset, output_dir / "dataset.jsonl"


# =============================================================================
# Step 4: 执行微调
# =============================================================================

def run_tsft(
    model_path: str,
    tokenizer,
    model,
    dataset: List[Dict],
    safety_neurons_path: str,
    output_dir: Path,
    device: torch.device,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    save_steps: int = 100,
    logging_steps: int = 10,
    warmup_steps: int = 100,
    grad_accum: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    save_only_delta: bool = True,
) -> Dict:
    """执行标准 TSFT。"""
    safety_neurons = load_dedicated_safety_neurons(safety_neurons_path)
    print(f"[Pipeline] 加载 {len(safety_neurons)} 个安全神经元")

    log = tsft_finetune(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        safety_neurons=safety_neurons,
        output_dir=str(output_dir / "model"),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,
        fp16=fp16,
        bf16=bf16,
        device=device,
        save_only_delta=save_only_delta,
    )
    return log


def run_vatsft(
    model_path: str,
    tokenizer,
    model,
    dataset: List[Dict],
    safety_neurons_path: str,
    vulnerable_neurons_path: str,
    output_dir: Path,
    device: torch.device,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    save_steps: int = 100,
    logging_steps: int = 10,
    warmup_steps: int = 100,
    grad_accum: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    save_only_delta: bool = True,
    reversal_lr_factor: float = 1.0,
) -> Dict:
    """执行 VA+TSFT 两阶段微调。"""
    safety_neurons = load_dedicated_safety_neurons(safety_neurons_path)
    vulnerable_neurons = load_dedicated_safety_neurons(vulnerable_neurons_path)
    print(f"[Pipeline] 加载 {len(safety_neurons)} 个安全神经元, {len(vulnerable_neurons)} 个脆弱神经元")

    log = vatft_finetune(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        dedicated_safety_neurons=safety_neurons,
        vulnerable_neurons=vulnerable_neurons,
        output_dir=str(output_dir / "model"),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,
        fp16=fp16,
        bf16=bf16,
        reversal_lr_factor=reversal_lr_factor,
        device=device,
        save_only_delta=save_only_delta,
    )
    return log


# =============================================================================
# CLI
# =============================================================================

def _add_common_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("模型与路径")
    g.add_argument("--model-path", type=str, required=True, help="模型路径")
    g.add_argument("--evaluation-log", type=str, required=True, help="评估日志（JSONL）")
    g.add_argument("--output", type=str, required=True, help="输出根目录")

    g = p.add_argument_group("神经元来源（四选一）")
    g.add_argument("--quadrant-results", type=str, help="四象限分类结果（会自动拆分）")
    g.add_argument("--safety-neurons", type=str, help="专用安全神经元 JSON（跳过拆分）")
    g.add_argument("--vulnerable-neurons", type=str, help="脆弱神经元 JSON（需要 --safety-neurons）")
    g.add_argument("--neurons-dir", type=str, help="预拆分目录（含 dedicated_safety_neurons.json 和 vulnerable_neurons.json）")

    g = p.add_argument_group("Refusal Templates")
    g.add_argument("--refusal-templates", type=str, default=None, help="已有 templates JSON（不存在则从评估日志提取）")

    g = p.add_argument_group("数据集")
    g.add_argument("--dataset", type=str, default=None, help="已有训练数据集 JSONL（不存在则构建）")
    g.add_argument("--only-successful", action="store_true", default=True, help="只使用 successful jailbreak 样本")
    g.add_argument("--min-templates-per-prompt", type=int, default=1)
    g.add_argument("--max-templates-per-prompt", type=int, default=3)

    g = p.add_argument_group("微调参数")
    g.add_argument("--method", choices=["tsft", "va-tsft"], default="va-tsft", help="TSFT 或 VA+TSFT（默认 VA+TSFT）")
    g.add_argument("--num-epochs", type=int, default=3)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--learning-rate", type=float, default=5e-5)
    g.add_argument("--max-length", type=int, default=512)
    g.add_argument("--gradient-accumulation-steps", type=int, default=4)
    g.add_argument("--warmup-steps", type=int, default=100)
    g.add_argument("--save-steps", type=int, default=100)
    g.add_argument("--logging-steps", type=int, default=10)
    g.add_argument("--reversal-lr-factor", type=float, default=1.0, help="VA+TSFT 脆弱神经元学习率倍率")
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--bf16", action="store_true")
    g.add_argument("--save-only-delta", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("运行控制")
    g.add_argument("--dry-run", action="store_true", help="只预览拆分结果，不加载模型和训练")
    g.add_argument("--device", type=str, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VA+TSFT 端到端串联脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_common_args(parser)
    return parser.parse_args()


def _resolve_neurons(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[str, Optional[str]]:
    """
    根据输入参数解析安全神经元和脆弱神经元文件路径。

    Returns:
        (safety_neurons_path, vulnerable_neurons_path)
    """
    # 优先顺序：显式文件 > 预拆分目录 > 四象限拆分
    if args.safety_neurons:
        safety_path = args.safety_neurons
        print(f"[Pipeline] 使用显式安全神经元文件: {safety_path}")
    elif args.neurons_dir:
        p = Path(args.neurons_dir) / "dedicated_safety_neurons.json"
        if not p.exists():
            raise FileNotFoundError(f"neurons-dir 中缺少 dedicated_safety_neurons.json: {p}")
        safety_path = str(p)
        print(f"[Pipeline] 从 neurons-dir 加载安全神经元: {safety_path}")
    elif args.quadrant_results:
        # 自动拆分，输出到 output_dir / neurons
        neurons_dir = output_dir / "neurons"
        s_path, v_path = split_quadrant_to_files(
            args.quadrant_results,
            neurons_dir,
            safety_only=(args.method == "tsft"),
        )
        if s_path is None:
            raise RuntimeError("拆分失败：没有找到 S-A- 安全神经元")
        safety_path = str(s_path)
        # 保存脆弱神经元路径供后续使用
        if v_path is not None:
            args._vulnerable_path = str(v_path)
        else:
            args._vulnerable_path = None
        return safety_path, args._vulnerable_path
    else:
        raise ValueError(
            "必须指定 --quadrant-results, --safety-neurons 或 --neurons-dir 之一"
        )

    vulnerable_path: Optional[str] = None
    if args.vulnerable_neurons:
        vulnerable_path = args.vulnerable_neurons
    elif args.neurons_dir and args.method == "va-tsft":
        p = Path(args.neurons_dir) / "vulnerable_neurons.json"
        if p.exists():
            vulnerable_path = str(p)
            print(f"[Pipeline] 从 neurons-dir 加载脆弱神经元: {vulnerable_path}")

    return safety_path, vulnerable_path


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 设备 ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Pipeline] 设备: {device}")

    # ---- 保存配置 ----
    config = {
        "method": args.method,
        "model_path": args.model_path,
        "evaluation_log": args.evaluation_log,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "save_only_delta": args.save_only_delta,
    }
    with open(output_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---- Dry-run：只做拆分 ----
    if args.dry_run:
        if not args.quadrant_results:
            print("[Pipeline] dry-run 模式需要 --quadrant-results")
            return 1
        print("\n" + "=" * 60)
        print("[Pipeline] DRY-RUN 模式（不加载模型，不执行训练）")
        print("=" * 60)
        s_path, v_path = split_quadrant_to_files(
            args.quadrant_results,
            output_dir / "neurons",
            safety_only=(args.method == "tsft"),
        )
        print(f"\n[Pipeline] 预览完成。输出目录: {output_dir / 'neurons'}")
        print(f"  安全神经元文件: {s_path}")
        print(f"  脆弱神经元文件: {v_path}")
        return 0

    # ===================================================================
    # Step A: 解析神经元来源
    # ===================================================================
    print("\n" + "=" * 60)
    print("[Pipeline Step A] 解析神经元来源")
    print("=" * 60)
    safety_neurons_path, vulnerable_neurons_path = _resolve_neurons(args, output_dir)

    # 对于 va-tsft，检查脆弱神经元是否存在
    if args.method == "va-tsft" and not vulnerable_neurons_path:
        print("[Pipeline] 警告: --method va-tsft 但未找到脆弱神经元，自动降级为 tsft")
        args.method = "tsft"

    # ===================================================================
    # Step B: Refusal Templates
    # ===================================================================
    print("\n" + "=" * 60)
    print("[Pipeline Step B] 准备 Refusal Templates")
    print("=" * 60)
    templates_path = prepare_refusal_templates(
        args.evaluation_log,
        args.refusal_templates,
        output_dir,
    )

    # ===================================================================
    # Step C: 构建训练数据集
    # ===================================================================
    print("\n" + "=" * 60)
    print("[Pipeline Step C] 构建训练数据集")
    print("=" * 60)
    dataset, dataset_path = build_training_dataset(
        args.evaluation_log,
        templates_path,
        output_dir,
        only_successful=args.only_successful,
        min_tpl=args.min_templates_per_prompt,
        max_tpl=args.max_templates_per_prompt,
        dataset_path=args.dataset,
        seed=args.seed,
    )

    # ===================================================================
    # Step D: 加载模型
    # ===================================================================
    print("\n" + "=" * 60)
    print(f"[Pipeline Step D] 加载模型: {args.model_path}")
    print("=" * 60)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)

    # ===================================================================
    # Step E: 执行微调
    # ===================================================================
    print("\n" + "=" * 60)
    print(f"[Pipeline Step E] 执行微调: {args.method.upper()}")
    print("=" * 60)
    print(f"  方法        : {args.method}")
    print(f"  安全神经元  : {safety_neurons_path}")
    print(f"  脆弱神经元  : {vulnerable_neurons_path}")
    print(f"  数据集大小  : {len(dataset)}")
    print(f"  Epochs      : {args.num_epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  学习率      : {args.learning_rate}")
    print(f"  保存模式    : {'Delta (~几 MB)' if args.save_only_delta else 'Full (~几 GB)'}")

    if args.method == "tsft":
        training_log = run_tsft(
            model_path=args.model_path,
            tokenizer=tokenizer,
            model=model,
            dataset=dataset,
            safety_neurons_path=safety_neurons_path,
            output_dir=output_dir,
            device=device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            warmup_steps=args.warmup_steps,
            grad_accum=args.gradient_accumulation_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            save_only_delta=args.save_only_delta,
        )
    else:
        training_log = run_vatsft(
            model_path=args.model_path,
            tokenizer=tokenizer,
            model=model,
            dataset=dataset,
            safety_neurons_path=safety_neurons_path,
            vulnerable_neurons_path=vulnerable_neurons_path,
            output_dir=output_dir,
            device=device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            warmup_steps=args.warmup_steps,
            grad_accum=args.gradient_accumulation_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            save_only_delta=args.save_only_delta,
            reversal_lr_factor=args.reversal_lr_factor,
        )

    # ===================================================================
    # Step F: 汇总报告
    # ===================================================================
    print("\n" + "=" * 60)
    print("[Pipeline] 完成！")
    print("=" * 60)
    print(f"  输出目录          : {output_dir}")
    print(f"  Refusal Templates : {templates_path}")
    print(f"  训练数据集        : {dataset_path}")
    print(f"  安全神经元        : {safety_neurons_path}")
    if vulnerable_neurons_path:
        print(f"  脆弱神经元        : {vulnerable_neurons_path}")
    print(f"  微调模型          : {output_dir / 'model'}")
    if args.save_only_delta:
        print(f"  Delta 权重        : {output_dir / 'model' / 'delta_weights.pt'}")
    print(f"  训练日志          : {output_dir / 'model' / 'training_log.json'}")

    if args.method == "tsft":
        print(f"\n  训练损失 (最终)   : {training_log.get('train_loss', 'N/A')}")
    else:
        print(f"\n  阶段一损失        : {training_log.get('stage1_loss', 'N/A')}")
        print(f"  阶段二损失        : {training_log.get('stage2_loss', 'N/A')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
