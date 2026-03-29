"""
Targeted Safety Fine-tuning (TSFT) module

实现只更新dedicated safety neurons的fine-tuning方法，根据Zhao et al. (2025)论文。
"""
#使用 Delta 模式保存后，需要原始模型 + delta 文件才能还原完整权重。
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
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


@dataclass
class VulnerableAwareConfig:
    """VA+TSFT 配置类

    用于配置 Vulnerable-Aware Targeted Safety Fine-Tuning 的训练参数。

    Attributes:
        dedicated_safety_neurons: 专用安全神经元 D(p,q)，格式为 Dict[(layer_idx, neuron_idx), Dict]
        vulnerable_neurons: S+A- 象限的脆弱神经元，需要功能反转，格式同 dedicated_safety_neurons
        reversal_lr_factor: 脆弱神经元学习率倍率（默认1.0）
        reversal_grad_sign: 梯度反转符号（-1.0 表示负梯度，默认-1.0）
        learning_rate: 学习率（将在 train 时由参数传入，此处仅作为占位）
    """
    dedicated_safety_neurons: Dict[Tuple[int, int], Dict]
    vulnerable_neurons: Dict[Tuple[int, int], Dict]
    reversal_lr_factor: float = 1.0
    reversal_grad_sign: float = -1.0
    learning_rate: float = 5e-5


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
    
    try:
        with open(neurons_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败 ({neurons_file}): {e}")
    
    # 自动检测键名（优先级从高到低）
    # dedicated_safety_neurons: 标准安全神经元
    # safety_neurons: 安全神经元（SNIP 筛选后）
    # all_neurons: 所有神经元
    # vulnerable_neurons: 脆弱神经元（S+A-，用于 VA+TSFT 阶段二）
    neurons_key = None
    for key in ['dedicated_safety_neurons', 'safety_neurons', 'all_neurons', 'vulnerable_neurons']:
        if key in data:
            neurons_key = key
            print(f"[TSFT] 检测到神经元集合键: {key}")
            break
    
    if neurons_key is None:
        raise ValueError(
            f"无法识别神经元格式。文件应包含 'dedicated_safety_neurons', 'safety_neurons', "
            f"'all_neurons' 或 'vulnerable_neurons' 字段"
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
    stage: str = "only",
) -> Set[str]:
    """
    只启用 dedicated safety neurons 相关的参数梯度

    对于每个安全神经元，启用其所在层的 MLP down_proj 权重中对应神经元的参数。

    Args:
        model: 模型
        safety_neurons: 安全神经元字典，格式为 Dict[(layer_idx, neuron_idx), Dict]
        stage: 梯度启用模式
            - "only": 冻结所有参数，只启用 safety_neurons 对应的层（默认）
            - "add": 不冻结，在现有基础上添加启用 safety_neurons 对应的层
            - "vulnerable_only": 冻结所有参数，只启用 vulnerable_neurons 对应的层
              （需要将 vulnerable_neurons 传入 safety_neurons 参数位）

    Returns:
        Set[str]: 启用了梯度的参数名称集合
    """
    # 根据 stage 决定是否冻结所有参数
    if stage in ("only", "vulnerable_only"):
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
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.down_proj"

    # 为每个目标神经元启用梯度
    for (layer_idx, neuron_idx), neuron_info in safety_neurons.items():
        param_name = f"{layer_prefix}.{layer_idx}.{mlp_down_proj_name}.weight"

        param = None
        for name, p in model.named_parameters():
            if name == param_name:
                param = p
                break

        if param is None:
            alternative_names = [
                f"transformer.h.{layer_idx}.mlp.c_proj.weight",
                f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight",
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
            param.requires_grad = True
            enabled_params.add(param_name)
        else:
            print(f"[TSFT] 警告: 未找到参数 {param_name} (layer {layer_idx}, neuron {neuron_idx})")

    print(f"[TSFT] 共启用了 {len(enabled_params)} 个参数的梯度 (stage={stage})")
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
    save_only_delta: bool = True,
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
        save_only_delta: 是否只保存权重差异（默认 True，文件约几 MB）

    Returns:
        Dict: 训练结果字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[TSFT] 开始targeted safety fine-tuning...")
    print(f"  - 数据集大小: {len(dataset)}")
    print(f"  - 安全神经元数量: {len(safety_neurons)}")
    print(f"  - 设备: {device}")
    print(f"  - 保存模式: {'Delta (差异)' if save_only_delta else 'Full (完整)'}")

    # 保存原始权重（用于 delta 计算）
    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    print(f"[TSFT] 已保存原始权重状态，共 {len(original_state_dict)} 个参数层")

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
        report_to="none",
        optim="adamw_torch",
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    # 训练
    print(f"[TSFT] 开始训练...")
    train_result = trainer.train()

    # 保存模型（Delta 或 Full）
    save_tsft_checkpoint(
        model=model,
        tokenizer=tokenizer,
        original_state_dict=original_state_dict,
        output_dir=output_dir,
        save_only_delta=save_only_delta,
    )

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
        "save_mode": "delta" if save_only_delta else "full",
    }

    log_path = Path(output_dir) / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"[TSFT] 训练完成！")
    print(f"  - 训练损失: {train_result.training_loss:.4f}")
    print(f"  - 训练时间: {train_result.metrics.get('train_runtime', 0):.2f}秒")

    return training_log


def _get_model_param_name(
    model: AutoModelForCausalLM,
    layer_idx: int,
    neuron_idx: int,
) -> Optional[str]:
    """获取指定神经元对应的参数名称

    Args:
        model: 模型
        layer_idx: 层索引
        neuron_idx: 神经元索引

    Returns:
        参数名称，如果未找到则返回 None
    """
    model_name = model.config.model_type.lower()

    if 'llama' in model_name or 'mistral' in model_name:
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.down_proj"
    elif 'qwen' in model_name:
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.c_proj"
    else:
        layer_prefix = "model.layers"
        mlp_down_proj_name = "mlp.down_proj"

    param_name = f"{layer_prefix}.{layer_idx}.{mlp_down_proj_name}.weight"

    for name, _ in model.named_parameters():
        if name == param_name:
            return param_name

    alternative_names = [
        f"transformer.h.{layer_idx}.mlp.c_proj.weight",
        f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight",
    ]
    for alt_name in alternative_names:
        for name, _ in model.named_parameters():
            if name == alt_name:
                return alt_name

    return None


def save_delta_weights(
    original_state_dict: Dict[str, torch.Tensor],
    current_state_dict: Dict[str, torch.Tensor],
    output_path: str,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """只保存权重的差异部分（Delta），大幅减小输出文件大小

    只保存被修改的权重（original vs current 的差异），未修改的权重不保存。
    文件大小从 ~几 GB 降到 ~几 MB。

    Args:
        original_state_dict: 训练前的原始模型权重
        current_state_dict: 训练后的当前模型权重
        output_path: 保存路径（.pt 或 .safetensors）

    Returns:
        (delta_state_dict, num_modified_params): 差异权重字典和修改的参数数量
    """
    delta_state_dict = {}

    for name in original_state_dict:
        if name not in current_state_dict:
            continue

        original_param = original_state_dict[name]
        current_param = current_state_dict[name]

        if not torch.equal(original_param, current_param):
            delta_state_dict[name] = current_param - original_param

    num_modified = len(delta_state_dict)

    print(f"[TSFT] Delta 保存: {num_modified} 个参数层被修改")

    if num_modified > 0:
        total_size_mb = sum(
            t.numel() * t.element_size() / 1024 / 1024
            for t in delta_state_dict.values()
        )
        print(f"[TSFT] Delta 文件大小: ~{total_size_mb:.2f} MB")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta_state_dict, output_path)
    print(f"[TSFT] Delta 权重已保存到: {output_path}")

    return delta_state_dict, num_modified


def load_delta_weights(
    base_model_path: str,
    delta_weights_path: str,
    device: Optional[torch.device] = None,
) -> AutoModelForCausalLM:
    """加载基础模型并应用 Delta 权重

    Args:
        base_model_path: 原始基础模型路径或模型对象
        delta_weights_path: Delta 权重文件路径（.pt）
        device: 加载设备

    Returns:
        应用了 delta 权重后的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[TSFT] 加载基础模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model.to(device)
    model.eval()

    original_state = model.state_dict()
    delta_state = torch.load(delta_weights_path, map_location=device)

    print(f"[TSFT] 应用 {len(delta_state)} 个 delta 权重层")

    for name, delta in delta_state.items():
        if name in original_state:
            original_state[name] = original_state[name] + delta
        else:
            print(f"[TSFT] 警告: delta 中有未知参数 {name}")

    model.load_state_dict(original_state)
    print(f"[TSFT] Delta 权重已应用，模型已更新")

    return model


def save_tsft_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    original_state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    save_only_delta: bool = True,
) -> Tuple[str, Optional[Dict]]:
    """保存 TSFT/VA+TSFT 检查点

    Args:
        model: 微调后的模型
        tokenizer: 分词器
        original_state_dict: 训练前的原始权重（用于计算 delta）
        output_dir: 输出目录
        save_only_delta: 是否只保存 delta（默认 True）

    Returns:
        (checkpoint_info, delta_path): 检查点信息和 delta 路径
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_info = {"save_mode": "full"}

    if save_only_delta:
        delta_path = Path(output_dir) / "delta_weights.pt"
        current_state = model.state_dict()

        delta_state, num_modified = save_delta_weights(
            original_state_dict, current_state, str(delta_path)
        )

        checkpoint_info = {
            "save_mode": "delta",
            "delta_path": str(delta_path),
            "num_modified_layers": num_modified,
        }

        total_delta_mb = sum(
            t.numel() * t.element_size() / 1024 / 1024
            for t in delta_state.values()
        )
        checkpoint_info["delta_size_mb"] = round(total_delta_mb, 2)

        model_type = model.config.model_type
        checkpoint_info["base_model_type"] = model_type
        checkpoint_info["requires_base_model"] = True

        meta_path = Path(output_dir) / "checkpoint_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)
        print(f"[TSFT] 检查点元信息已保存: {meta_path}")
    else:
        print(f"[TSFT] 保存完整模型到: {output_dir}")
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    return checkpoint_info, delta_path if save_only_delta else None


class _VATSFTTrainer(Trainer):
    """内部 Trainer 子类：支持分组学习率的 VA+TSFT 训练

    重写 _create_optimizer() 以注入分组学习率：
    - 安全神经元（D(p,q)）：使用基础学习率（正常梯度）
    - 脆弱神经元（S+A-）：使用 reversal_lr_factor * reversal_grad_sign * 基础学习率（梯度反转）
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        tokenizer,
        safety_param_names: Set[str],
        vulnerable_param_names: Set[str],
        reversal_lr_factor: float,
        reversal_grad_sign: float,
        base_learning_rate: float,
        original_state_dict: Dict[str, torch.Tensor],
        **kwargs,
    ):
        super().__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)
        self._safety_param_names = safety_param_names
        self._vulnerable_param_names = vulnerable_param_names
        self._reversal_lr_factor = reversal_lr_factor
        self._reversal_grad_sign = reversal_grad_sign
        self._base_lr = base_learning_rate
        self._original_state_dict = original_state_dict

    def _create_optimizer(self):
        """重写以使用分组学习率"""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad
                    and n in self._safety_param_names
                    and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self._base_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad
                    and n in self._safety_param_names
                    and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self._base_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad
                    and n in self._vulnerable_param_names
                    and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self._base_lr * self._reversal_lr_factor * self._reversal_grad_sign,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if p.requires_grad
                    and n in self._vulnerable_param_names
                    and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self._base_lr * self._reversal_lr_factor * self._reversal_grad_sign,
            },
        ]

        # 过滤空组
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if g["params"]
        ]

        if self.args.optim == TrainingArguments.OPTIM.ADAMW_TORCH:
            from torch.optim import AdamW
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        elif self.args.optim == TrainingArguments.OPTIM.ADAMW_HF:
            from transformers.optimization import AdamW
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        else:
            self.optimizer = self._get_optimizer_signature()(optimizer_grouped_parameters)

        return self.optimizer

    def training_step(self, model, inputs):
        """重写：在反向传播后对脆弱神经元应用梯度归一化"""
        model.train()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        # 对脆弱神经元应用梯度归一化，防止过度干预
        # 梯度归一化到单位范数，防止引入额外误差
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and n in self._vulnerable_param_names:
                    grad_norm = p.grad.norm()
                    if grad_norm > 1e-8:
                        p.grad = p.grad / grad_norm

        return loss.detach()


class VATSFTTrainer:
    """VA+TSFT: Vulnerable-Aware Targeted Safety Fine-Tuning

    两阶段定向安全微调：
    - 阶段一（Stage 1）：对 D(p,q) 中的安全神经元应用正常梯度更新
    - 阶段二（Stage 2）：对 S+A- 象限的脆弱神经元应用负梯度反转

    与标准 TSFT 的区别：
    1. 除了更新安全神经元 D(p,q)（应用正常梯度）
    2. 还更新 S+A- 象限的脆弱神经元（应用负梯度反转功能）
    3. 对脆弱神经元应用梯度归一化，防止过度干预

    论文参考：Section 5.5, 分工方案 3.7 节
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: VulnerableAwareConfig,
    ):
        """初始化 VA+TSFT 训练器

        Args:
            model: 因果语言模型
            tokenizer: 分词器
            config: VA+TSFT 配置
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _enable_gradients(
        self,
        target: str = "both",
    ) -> Tuple[Set[str], Set[str]]:
        """启用安全神经元和/或脆弱神经元的梯度

        Args:
            target: 启用目标
                - "both": 冻结全部，启用安全+脆弱（两阶段一起启用时）
                - "safety": 只启用安全神经元
                - "vulnerable": 只启用脆弱神经元

        Returns:
            (safety_param_names, vulnerable_param_names): 两组参数的名称集合
        """
        if target == "safety":
            # 阶段一：只启用安全神经元
            safety_param_names = enable_safety_neuron_gradients(
                self.model, self.config.dedicated_safety_neurons, stage="only"
            )
            vulnerable_param_names = set()
        elif target == "vulnerable":
            # 阶段二：只启用脆弱神经元
            safety_param_names = set()
            vulnerable_param_names = enable_safety_neuron_gradients(
                self.model, self.config.vulnerable_neurons, stage="vulnerable_only"
            )
        else:
            # 两阶段一起启用（目前不再使用，但保留以防需要）
            safety_param_names = enable_safety_neuron_gradients(
                self.model, self.config.dedicated_safety_neurons, stage="only"
            )
            vulnerable_param_names = enable_safety_neuron_gradients(
                self.model, self.config.vulnerable_neurons, stage="add"
            )

        return safety_param_names, vulnerable_param_names

    def _run_single_phase(
        self,
        dataset,
        output_dir,
        num_epochs,
        batch_size,
        learning_rate,
        max_length,
        save_steps,
        logging_steps,
        warmup_steps,
        gradient_accumulation_steps,
        fp16,
        bf16,
        save_only_delta,
        phase_name: str,
        safety_param_names: Set[str],
        vulnerable_param_names: Set[str],
    ) -> Tuple:
        """运行单阶段训练（安全神经元或脆弱神经元）

        Args:
            phase_name: 阶段名称（"Stage 1: D(p,q)" 或 "Stage 2: Vulnerable"）
            safety_param_names: 安全神经元对应的参数名集合
            vulnerable_param_names: 脆弱神经元对应的参数名集合

        Returns:
            (train_result, original_state_dict): 训练结果和原始权重
        """
        print(f"[VA+TSFT] === {phase_name} ===")

        original_state_dict = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}

        train_dataset = FineTuningDataset(dataset, self.tokenizer, max_length=max_length)

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
            report_to="none",
            optim="adamw_torch",
        )

        trainer = _VATSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            safety_param_names=safety_param_names,
            vulnerable_param_names=vulnerable_param_names,
            reversal_lr_factor=self.config.reversal_grad_sign,
            reversal_grad_sign=self.config.reversal_grad_sign,
            base_learning_rate=learning_rate,
            original_state_dict=original_state_dict,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )

        train_result = trainer.train()

        return train_result, original_state_dict

    def train(
        self,
        dataset: List[Dict],
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
        save_only_delta: bool = True,
    ) -> Dict:
        """执行 VA+TSFT 两阶段训练

        Args:
            dataset: 训练数据集
            output_dir: 输出目录
            num_epochs: 阶段一的训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            max_length: 最大序列长度
            save_steps: 保存步数间隔
            logging_steps: 日志步数间隔
            warmup_steps: Warmup步数
            gradient_accumulation_steps: 梯度累积步数
            fp16: 是否使用FP16
            bf16: 是否使用BF16
            save_only_delta: 是否只保存权重差异（默认 True，文件约几 MB）

        Returns:
            Dict: 训练结果字典
        """
        self.config.learning_rate = learning_rate

        print(f"[VA+TSFT] 开始两阶段训练...")
        print(f"  - 数据集大小: {len(dataset)}")
        print(f"  - 安全神经元 D(p,q): {len(self.config.dedicated_safety_neurons)} 个")
        print(f"  - 脆弱神经元 S+A-: {len(self.config.vulnerable_neurons)} 个")
        print(f"  - 保存模式: {'Delta (差异)' if save_only_delta else 'Full (完整)'}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stage1_dir = Path(output_dir) / "stage1_safety"
        stage2_dir = Path(output_dir) / "stage2_vulnerable"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)

        # ═══════════════════════════════════════════
        # 阶段一：训练安全神经元 D(p,q)（正常梯度）
        # ═══════════════════════════════════════════
        safety_param_names, _ = self._enable_gradients(target="safety")
        print(f"[VA+TSFT] 启用 {len(safety_param_names)} 个安全神经元参数层")

        train_result1, original_state_dict = self._run_single_phase(
            dataset=dataset,
            output_dir=str(stage1_dir),
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            save_steps=save_steps,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            bf16=bf16,
            save_only_delta=False,  # 中间检查点不保存 delta
            phase_name="Stage 1: D(p,q) 安全神经元（正常梯度）",
            safety_param_names=safety_param_names,
            vulnerable_param_names=set(),
        )
        print(f"[VA+TSFT] 阶段一完成，最终损失: {train_result1.training_loss:.4f}")

        # 保存阶段一检查点
        save_tsft_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            original_state_dict=original_state_dict,
            output_dir=str(stage1_dir),
            save_only_delta=False,
        )

        # ═══════════════════════════════════════════
        # 阶段二：训练脆弱神经元 S+A-（负梯度反转）
        # ═══════════════════════════════════════════
        if self.config.vulnerable_neurons:
            num_epochs_stage2 = max(1, num_epochs // 2)
            learning_rate_stage2 = learning_rate * self.config.reversal_lr_factor * 0.5

            print(f"[VA+TSFT] 切换到脆弱神经元（阶段二）...")
            print(f"[VA+TSFT]   学习率调整: {learning_rate} → {learning_rate_stage2:.2e}")
            print(f"[VA+TSFT]   梯度反转: reversal_sign={self.config.reversal_grad_sign}")

            _, vulnerable_param_names = self._enable_gradients(target="vulnerable")
            print(f"[VA+TSFT] 启用 {len(vulnerable_param_names)} 个脆弱神经元参数层")

            train_result2, _ = self._run_single_phase(
                dataset=dataset,
                output_dir=str(stage2_dir),
                num_epochs=num_epochs_stage2,
                batch_size=batch_size,
                learning_rate=learning_rate_stage2,
                max_length=max_length,
                save_steps=save_steps,
                logging_steps=logging_steps,
                warmup_steps=min(warmup_steps // 2, 10),
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=fp16,
                bf16=bf16,
                save_only_delta=False,
                phase_name=f"Stage 2: S+A- 脆弱神经元（梯度反转, lr={learning_rate_stage2:.2e}）",
                safety_param_names=set(),
                vulnerable_param_names=vulnerable_param_names,
            )
            print(f"[VA+TSFT] 阶段二完成，最终损失: {train_result2.training_loss:.4f}")

            # 保存阶段二检查点
            save_tsft_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                original_state_dict=original_state_dict,
                output_dir=str(stage2_dir),
                save_only_delta=False,
            )
        else:
            print("[VA+TSFT] 无脆弱神经元，跳过阶段二")
            vulnerable_param_names = set()

        # ═══════════════════════════════════════════
        # 保存最终模型（Delta 模式）
        # ═══════════════════════════════════════════
        save_tsft_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            original_state_dict=original_state_dict,
            output_dir=output_dir,
            save_only_delta=save_only_delta,
        )

        training_log = {
            "method": "VA+TSFT",
            "num_samples": len(dataset),
            "num_safety_neurons": len(self.config.dedicated_safety_neurons),
            "num_vulnerable_neurons": len(self.config.vulnerable_neurons),
            "enabled_safety_params": list(safety_param_names),
            "enabled_vulnerable_params": list(vulnerable_param_names),
            "reversal_lr_factor": self.config.reversal_lr_factor,
            "reversal_grad_sign": self.config.reversal_grad_sign,
            "stage1_epochs": num_epochs,
            "stage1_loss": train_result1.training_loss,
            "stage1_params": list(safety_param_names),
            "stage2_epochs": max(1, num_epochs // 2) if self.config.vulnerable_neurons else 0,
            "stage2_loss": train_result2.training_loss if self.config.vulnerable_neurons else None,
            "stage2_params": list(vulnerable_param_names),
            "batch_size": batch_size,
            "learning_rate_stage1": learning_rate,
            "learning_rate_stage2": learning_rate * self.config.reversal_lr_factor * 0.5
                if self.config.vulnerable_neurons else 0,
            "train_runtime_stage1": train_result1.metrics.get("train_runtime", 0),
            "train_runtime_stage2": train_result2.metrics.get("train_runtime", 0)
                if self.config.vulnerable_neurons else 0,
            "save_mode": "delta" if save_only_delta else "full",
        }

        log_path = Path(output_dir) / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

        print(f"[VA+TSFT] 训练完成！")
        print(f"  - 阶段一损失: {train_result1.training_loss:.4f}")
        if self.config.vulnerable_neurons:
            print(f"  - 阶段二损失: {train_result2.training_loss:.4f}")

        return training_log


def vatft_finetune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    dedicated_safety_neurons: Dict[Tuple[int, int], Dict],
    vulnerable_neurons: Dict[Tuple[int, int], Dict],
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
    reversal_lr_factor: float = 1.0,
    device: Optional[torch.device] = None,
    save_only_delta: bool = True,
) -> Dict:
    """执行 VA+TSFT (Vulnerable-Aware Targeted Safety Fine-Tuning)

    VA+TSFT 在标准 TSFT 基础上，额外对 S+A- 象限的脆弱神经元应用负梯度反转其功能，
    以达到更优的防御效果。

    Args:
        model: 因果语言模型
        tokenizer: 分词器
        dataset: 训练数据集（List[Dict]，每个dict包含"input"和"output"字段）
        dedicated_safety_neurons: 专用安全神经元 D(p,q)
        vulnerable_neurons: S+A- 象限的脆弱神经元（需要功能反转）
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
        reversal_lr_factor: 脆弱神经元学习率倍率（默认1.0）
        device: 计算设备
        save_only_delta: 是否只保存权重差异（默认 True，文件约几 MB）

    Returns:
        Dict: 训练结果字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[VA+TSFT] 使用设备: {device}")
    print(f"[VA+TSFT] 保存模式: {'Delta (差异)' if save_only_delta else 'Full (完整)'}")
    model.to(device)

    config = VulnerableAwareConfig(
        dedicated_safety_neurons=dedicated_safety_neurons,
        vulnerable_neurons=vulnerable_neurons,
        reversal_lr_factor=reversal_lr_factor,
        reversal_grad_sign=-1.0,
    )

    trainer = VATSFTTrainer(model, tokenizer, config)

    return trainer.train(
        dataset=dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        bf16=bf16,
        save_only_delta=save_only_delta,
    )


def identify_vulnerable_neurons(
    quadrant_results: Dict[Tuple[int, int], Dict],
) -> Dict[Tuple[int, int], Dict]:
    """从四象限分类结果中识别需要功能反转的脆弱神经元

    S+A- 象限神经元：参数对齐为正（S+），激活投影为负（A-）
    这些神经元在参数空间中与毒性向量对齐，但激活时反而抑制毒性
    通过负梯度反转，使其从"促进毒性"变为"抑制毒性"

    Args:
        quadrant_results: 四象限分类结果，来自 classify_neuron_quadrants()
            格式为 Dict[(layer_idx, neuron_idx), {
                'quadrant': 'S+A+' | 'S-A+' | 'S+A-' | 'S-A-',
                'alignment': float,
                'activation_projection': float,
                ...
            }]

    Returns:
        Dict[Tuple[int, int], Dict]: 脆弱神经元字典，只包含 S+A- 象限的神经元
    """
    vulnerable = {}

    for (layer_idx, neuron_idx), data in quadrant_results.items():
        if data.get('quadrant') == 'S+A-':
            vulnerable[(layer_idx, neuron_idx)] = data

    print(f"[VA+TSFT] 识别到 {len(vulnerable)} 个 S+A- 脆弱神经元")

    if vulnerable:
        print("[VA+TSFT] S+A- 象限含义：参数对齐为正（S+），激活投影为负（A-）")
        print("[VA+TSFT]   → 这些神经元参数方向与毒性对齐，但激活时抑制毒性（伪安全）")
        print("[VA+TSFT]   → 通过负梯度反转，使参数方向与激活方向一致，真正发挥抑制毒性作用")

    return vulnerable
