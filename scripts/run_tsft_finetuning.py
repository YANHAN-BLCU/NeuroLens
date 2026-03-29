#!/usr/bin/env python3
r"""
Targeted Safety Fine-tuning (TSFT) 训练脚本

根据Zhao et al. (2025)论文，实现基于dedicated safety neurons的targeted safety fine-tuning。

使用方法：
    python scripts/run_tsft_finetuning.py ^
        --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct ^
        --evaluation-log logs/base_evaluation.jsonl ^
        --safety-neurons outputs/dedicated_safety_neurons.json ^
        --output outputs/tsft_finetuning

默认（Delta 模式，约几 MB）

python scripts/run_tsft_finetuning.py `
    --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct `
    --evaluation-log logs/base_evaluation.jsonl `
    --safety-neurons outputs/dedicated_safety_neurons.json `
    --output outputs/tsft_finetuning


完整模型保存（约几 GB）

python scripts/run_tsft_finetuning.py `
    --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct `
    --evaluation-log logs/base_evaluation.jsonl `
    --safety-neurons outputs/dedicated_safety_neurons.json `
    --output outputs/tsft_finetuning `
    --save-only-delta False


自定义训练参数
python scripts/run_tsft_finetuning.py `
    --model-path D:\NeuroLens-master\ms_models\LLM-Research\Meta-Llama-3-8B-Instruct `
    --evaluation-log logs/base_evaluation.jsonl `
    --safety-neurons outputs/dedicated_safety_neurons.json `
    --output outputs/tsft_finetuning `
    --num-epochs 5 `
    --batch-size 8 `
    --learning-rate 1e-5


Delta 权重保存（默认）：
    Delta 模式只保存修改的权重差异，文件约几 MB，而非完整的几 GB 模型。
    python scripts/run_tsft_finetuning.py ... --save-only-delta True

完整模型保存：
    python scripts/run_tsft_finetuning.py ... --save-only-delta False
"""

import sys
import os
import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加工作目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.exists() and (PROJECT_ROOT / 'engine').exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    workspace_path = os.getenv('WORKSPACE_PATH', '/workspace')
    if os.path.exists(workspace_path):
        sys.path.insert(0, workspace_path)
    else:
        cwd = Path.cwd()
        if (cwd / 'engine').exists():
            sys.path.insert(0, str(cwd))
        else:
            sys.path.insert(0, '/workspace')

from engine.fine_tuning import (
    extract_refusal_templates,
    save_refusal_templates,
    load_refusal_templates,
    build_refusal_guided_dataset,
    save_dataset,
    load_dedicated_safety_neurons,
    tsft_finetune,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Targeted Safety Fine-tuning (TSFT) 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必需参数
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径（本地路径或 HuggingFace 模型ID）"
    )
    
    parser.add_argument(
        "--evaluation-log",
        type=str,
        required=True,
        help="评估日志文件路径（JSONL格式）"
    )
    
    parser.add_argument(
        "--safety-neurons",
        type=str,
        required=True,
        help="Dedicated safety neurons文件路径（JSON格式）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    # 可选参数：数据集构建
    parser.add_argument(
        "--refusal-templates-path",
        type=str,
        default=None,
        help="Refusal templates文件路径（如果已存在，将直接使用；否则从evaluation-log提取）"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="已构建的数据集路径（如果已存在，将直接使用；否则从evaluation-log构建）"
    )
    
    parser.add_argument(
        "--min-template-frequency",
        type=int,
        default=2,
        help="Refusal template的最小出现频率（默认2）"
    )
    
    parser.add_argument(
        "--min-templates-per-prompt",
        type=int,
        default=1,
        help="每个prompt使用的最少template数量（默认1）"
    )
    
    parser.add_argument(
        "--max-templates-per-prompt",
        type=int,
        default=3,
        help="每个prompt使用的最多template数量（默认3）"
    )
    
    # 训练参数
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="训练轮数（默认3）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批大小（默认4）"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="学习率（默认5e-5）"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="最大序列长度（默认512）"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="梯度累积步数（默认4）"
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup步数（默认100）"
    )
    
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="保存步数间隔（默认100）"
    )
    
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="日志步数间隔（默认10）"
    )
    
    # 其他参数
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用FP16精度"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="使用BF16精度"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（'cuda' 或 'cpu'，默认自动检测）"
    )

    parser.add_argument(
        "--save-only-delta",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="是否只保存权重差异（True/False，默认True，文件约几 MB）"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 确定设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[TSFT Training] 使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = {
        "model_path": args.model_path,
        "evaluation_log": args.evaluation_log,
        "safety_neurons": args.safety_neurons,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "save_only_delta": args.save_only_delta,
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 1. 加载模型和分词器
    print(f"[TSFT Training] 加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    # 注意：这里不要使用 device_map="auto"，否则模型会通过 accelerate 做分片 / offload，
    # 而 Hugging Face Trainer 在初始化时会再次尝试 model.to(device)，从而触发
    # "You can't move a model that has some modules offloaded to cpu or disk." 错误。
    #
    # 为了兼容 Trainer，这里加载一个未使用 accelerate hooks 的模型，然后再根据
    # 上面的 device 设置显式移动到对应设备。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    
    # 2. 加载dedicated safety neurons
    print(f"[TSFT Training] 加载dedicated safety neurons: {args.safety_neurons}")
    safety_neurons = load_dedicated_safety_neurons(args.safety_neurons)
    
    # 3. 构建数据集
    dataset_path = output_dir / "dataset.jsonl"
    
    if args.dataset_path and Path(args.dataset_path).exists():
        print(f"[TSFT Training] 使用已存在的数据集: {args.dataset_path}")
        from engine.fine_tuning.dataset_builder import load_dataset
        dataset = load_dataset(args.dataset_path)
    else:
        print(f"[TSFT Training] 构建refusal-guided数据集...")
        
        # 3.1 提取或加载refusal templates
        if args.refusal_templates_path and Path(args.refusal_templates_path).exists():
            print(f"[TSFT Training] 使用已存在的refusal templates: {args.refusal_templates_path}")
            refusal_templates = load_refusal_templates(args.refusal_templates_path)
        else:
            print(f"[TSFT Training] 从评估日志提取refusal templates...")
            refusal_templates = extract_refusal_templates(
                args.evaluation_log,
                min_frequency=args.min_template_frequency,
            )
            
            # 保存templates
            templates_path = output_dir / "refusal_templates.json"
            save_refusal_templates(refusal_templates, str(templates_path))
        
        if not refusal_templates:
            raise ValueError("无法提取refusal templates！请检查evaluation-log文件")
        
        # 3.2 构建数据集
        dataset = build_refusal_guided_dataset(
            evaluation_log_path=args.evaluation_log,
            refusal_templates=refusal_templates,
            output_path=str(dataset_path),
            only_successful_jailbreaks=True,
            min_templates_per_prompt=args.min_templates_per_prompt,
            max_templates_per_prompt=args.max_templates_per_prompt,
            seed=args.seed,
        )
        
        if not dataset:
            raise ValueError("无法构建数据集！请检查evaluation-log文件")
    
    print(f"[TSFT Training] 数据集大小: {len(dataset)}")
    
    # 4. 执行TSFT fine-tuning
    model_output_dir = output_dir / "model"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    training_log = tsft_finetune(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        safety_neurons=safety_neurons,
        output_dir=str(model_output_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        device=device,
        save_only_delta=args.save_only_delta,
    )
    
    print(f"[TSFT Training] 训练完成！")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 模型目录: {model_output_dir}")
    print(f"  - 数据集: {dataset_path}")
    print(f"  - 训练日志: {model_output_dir / 'training_log.json'}")
    print(f"  - 保存模式: {'Delta (差异，约几 MB)' if args.save_only_delta else 'Full (完整，约几 GB)'}")

    if args.save_only_delta:
        print(f"  - Delta 权重: {model_output_dir / 'delta_weights.pt'}")
        print(f"  - 使用方式: load_delta_weights(base_model_path, delta_weights_path)")
    else:
        print(f"  - 模型权重: {model_output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
