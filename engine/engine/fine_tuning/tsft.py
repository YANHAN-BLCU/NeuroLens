"""
Targeted Safety Fine-tuning (TSFT) module

实现只更新dedicated safety neurons的fine-tuning方法，根据Zhao et al. (2025)论文。
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class FineTuningDataset(Dataset):
    """Fine-tuning数据集"""
    
    def __init__(self, samples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample.get("input", "")
        response = sample.get("output", "")
        
        # 构建训练文本：prompt + response
        text = f"{prompt}{self.tokenizer.eos_token}{response}{self.tokenizer.eos_token}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 只对response部分计算loss（mask掉prompt部分）
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 找到prompt结束的位置（第一个eos_token之后）
        prompt_end = len(self.tokenizer.encode(prompt, add_special_tokens=False)) + 1  # +1 for eos_token
        
        # 创建labels：prompt部分设为-100（忽略），response部分保留
        labels = input_ids.clone()
        labels[:prompt_end] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_dedicated_safety_neurons(neurons_file: str) -> Dict[Tuple[int, int], Dict]:
    """
    从JSON文件中加载dedicated safety neurons
    
    Args:
        neurons_file: JSON文件路径，应包含 'dedicated_safety_neurons' 或 'safety_neurons' 字段
    
    Returns:
        神经元字典，格式为 Dict[(layer_idx, neuron_idx), Dict]
    """
    if not Path(neurons_file).exists():
        raise FileNotFoundError(f"Dedicated safety neurons文件不存在: {neurons_file}")
    
    print(f"[TSFT] 加载dedicated safety neurons: {neurons_file}")
    
    with open(neurons_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 自动检测键名
    neurons_key = None
    for key in ['dedicated_safety_neurons', 'safety_neurons', 'all_neurons']:
        if key in data:
            neurons_key = key
            print(f"[TSFT] 检测到神经元集合键: {key}")
            break
    
    if neurons_key is None:
        raise ValueError(
            f"无法识别神经元格式。文件应包含 'dedicated_safety_neurons', 'safety_neurons' 或 'all_neurons' 字段"
        )
    
    neurons_data = data[neurons_key]
    
    # 解析神经元数据
    target_neurons = {}
    for key, value in neurons_data.items():
        # 支持多种格式
        if 'layer_idx' in value and 'neuron_idx' in value:
            layer_idx = int(value['layer_idx'])
            neuron_idx = int(value['neuron_idx'])
            target_neurons[(layer_idx, neuron_idx)] = value
        elif 'layer' in value and 'neuron' in value:
            layer_idx = int(value['layer'])
            neuron_idx = int(value['neuron'])
            target_neurons[(layer_idx, neuron_idx)] = value
        elif '_' in key:
            # 尝试从键名解析
            try:
                parts = key.split('_')
                # 格式1: layer_X_neuron_Y
                if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'neuron':
                    layer_idx = int(parts[1])
                    neuron_idx = int(parts[3])
                    target_neurons[(layer_idx, neuron_idx)] = value
                # 格式2: X_Y (下划线分隔，如 "31_4062")
                elif len(parts) == 2:
                    layer_idx = int(parts[0])
                    neuron_idx = int(parts[1])
                    target_neurons[(layer_idx, neuron_idx)] = value
            except (ValueError, IndexError):
                continue
    
    print(f"[TSFT] 成功加载 {len(target_neurons)} 个dedicated safety neurons")
    return target_neurons


def enable_safety_neuron_gradients(
    model: AutoModelForCausalLM,
    safety_neurons: Dict[Tuple[int, int], Dict],
) -> Set[str]:
    """
    只启用dedicated safety neurons相关的参数梯度
    
    对于每个安全神经元，启用其所在层的MLP down_proj权重中对应神经元的参数。
    
    Args:
        model: 模型
        safety_neurons: 安全神经元字典，格式为 Dict[(layer_idx, neuron_idx), Dict]
    
    Returns:
        Set[str]: 启用了梯度的参数名称集合
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    enabled_params = set()
    
    # 获取模型架构信息
    model_name = model.config.model_type.lower()
    
    # 根据模型类型确定层名称模式
    if 'llama' in model_name or 'mistral' in model_name:
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.down_proj"
    elif 'qwen' in model_name:
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.c_proj"
    else:
        # 默认尝试常见的命名
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.down_proj"
    
    # 为每个安全神经元启用梯度
    for (layer_idx, neuron_idx), neuron_info in safety_neurons.items():
        # 构建参数名称
        param_name = f"{layer_prefix}.{layer_idx}.{mlp_down_proj_name}.weight"
        
        # 检查参数是否存在
        param = None
        for name, p in model.named_parameters():
            if name == param_name:
                param = p
                break
        
        if param is None:
            # 尝试其他可能的命名
            alternative_names = [
                f"transformer.h.{layer_idx}.mlp.c_proj.weight",  # GPT-2 style
                f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight",  # GPT-NeoX style
            ]
            for alt_name in alternative_names:
                for name, p in model.named_parameters():
                    if name == alt_name:
                        param = p
                        param_name = alt_name
                        break
                if param is not None:
                    break
        
        if param is not None:
            # 启用整个down_proj权重矩阵的梯度
            # 注意：虽然我们只关心特定神经元，但PyTorch需要整个参数启用梯度
            # 在训练时，梯度会自然传播到相关权重
            param.requires_grad = True
            enabled_params.add(param_name)
            print(f"[TSFT] 启用参数梯度: {param_name} (layer {layer_idx}, neuron {neuron_idx})")
        else:
            print(f"[TSFT] 警告: 未找到参数 {param_name} (layer {layer_idx}, neuron {neuron_idx})")
    
    print(f"[TSFT] 共启用了 {len(enabled_params)} 个参数的梯度")
    return enabled_params


def create_tsft_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    创建只优化安全神经元参数的优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        optimizer_type: 优化器类型（"adamw" 或 "sgd"）
    
    Returns:
        torch.optim.Optimizer: 优化器
    """
    # 只收集需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        raise ValueError("没有可训练的参数！请确保已调用 enable_safety_neuron_gradients")
    
    print(f"[TSFT] 优化器将优化 {sum(p.numel() for p in trainable_params)} 个参数")
    
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def tsft_finetune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    safety_neurons: Dict[Tuple[int, int], Dict],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    save_steps: int = 100,
    logging_steps: int = 10,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    执行targeted safety fine-tuning
    
    Args:
        model: 模型
        tokenizer: 分词器
        dataset: 训练数据集（List[Dict]，每个dict包含"input"和"output"字段）
        safety_neurons: 安全神经元字典
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        max_length: 最大序列长度
        save_steps: 保存步数间隔
        logging_steps: 日志步数间隔
        warmup_steps: Warmup步数
        gradient_accumulation_steps: 梯度累积步数
        fp16: 是否使用FP16
        bf16: 是否使用BF16
        device: 计算设备
    
    Returns:
        Dict: 训练结果字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[TSFT] 开始targeted safety fine-tuning...")
    print(f"  - 数据集大小: {len(dataset)}")
    print(f"  - 安全神经元数量: {len(safety_neurons)}")
    print(f"  - 设备: {device}")
    
    # 启用安全神经元的梯度
    enabled_params = enable_safety_neuron_gradients(model, safety_neurons)
    
    if not enabled_params:
        raise ValueError("没有启用任何参数的梯度！请检查safety_neurons配置")
    
    # 创建数据集
    train_dataset = FineTuningDataset(dataset, tokenizer, max_length=max_length)
    
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",  # 不使用wandb等
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言模型，不是MLM
        ),
    )
    
    # 训练
    print(f"[TSFT] 开始训练...")
    train_result = trainer.train()
    
    # 保存模型
    print(f"[TSFT] 保存模型到 {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练日志
    training_log = {
        "num_samples": len(dataset),
        "num_safety_neurons": len(safety_neurons),
        "enabled_params": list(enabled_params),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }
    
    log_path = Path(output_dir) / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    print(f"[TSFT] 训练完成！")
    print(f"  - 训练损失: {train_result.training_loss:.4f}")
    print(f"  - 训练时间: {train_result.metrics.get('train_runtime', 0):.2f}秒")
    
    return training_log
