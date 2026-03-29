"""
Refusal-guided dataset builder module

构建用于fine-tuning的数据集，通过组合refusal templates和categorized prompts生成safety-aligned responses。
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .refusal_templates import load_refusal_templates
from .salad_taxonomy import (
    extract_prompt,
    get_category_specific_refusal,
    get_prompt_category,
)


def build_refusal_guided_dataset(
    evaluation_log_path: str,
    refusal_templates: List[str],
    output_path: Optional[str] = None,
    only_successful_jailbreaks: bool = True,
    min_templates_per_prompt: int = 1,
    max_templates_per_prompt: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    构建refusal-guided fine-tuning数据集
    
    从评估日志中提取successful jailbreak prompts，为每个prompt：
    1. 获取其类别（从SALAD taxonomy）
    2. 随机选择一个或多个refusal templates
    3. 组合生成safety response
    4. 生成训练样本对：(jailbreak_prompt, safety_response)
    
    Args:
        evaluation_log_path: 评估日志文件路径（JSONL格式）
        refusal_templates: Refusal templates列表
        output_path: 可选，保存数据集的输出路径
        only_successful_jailbreaks: 是否只使用successful jailbreak prompts（guard.jailbreak_success == true）
        min_templates_per_prompt: 每个prompt使用的最少template数量
        max_templates_per_prompt: 每个prompt使用的最多template数量
        seed: 随机种子
    
    Returns:
        List[Dict]: 训练样本列表，每个样本格式为：
            {
                "input": "jailbreak_prompt",
                "output": "safety_response",
                "category": "violence",
                "templates_used": ["template1", "template2"],
            }
    """
    random.seed(seed)
    
    if not refusal_templates:
        print("[Dataset Builder] 警告: refusal_templates为空，无法构建数据集")
        return []
    
    print(f"[Dataset Builder] 构建refusal-guided数据集...")
    print(f"  - Refusal templates数量: {len(refusal_templates)}")
    print(f"  - 只使用successful jailbreaks: {only_successful_jailbreaks}")
    
    dataset = []
    
    if not Path(evaluation_log_path).exists():
        print(f"[Dataset Builder] 警告: 文件不存在: {evaluation_log_path}")
        return dataset
    
    with open(evaluation_log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Dataset Builder] 警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
            
            # 检查是否为successful jailbreak
            guard = obj.get("guard", {})
            jailbreak_success = guard.get("jailbreak_success", False)
            
            if only_successful_jailbreaks and not jailbreak_success:
                continue
            
            # 提取prompt（使用与SALAD taxonomy一致的鲁棒抽取逻辑）
            prompt = extract_prompt(obj)
            
            if not prompt:
                continue
            
            # 获取类别
            category = get_prompt_category(obj)
            if not category:
                continue
            
            # 随机选择templates
            num_templates = random.randint(min_templates_per_prompt, max_templates_per_prompt)
            selected_templates = random.sample(refusal_templates, min(num_templates, len(refusal_templates)))
            
            # 组合生成safety response
            safety_response = combine_template_with_prompt(
                selected_templates,
                prompt,
                category,
            )
            
            if not safety_response:
                continue
            
            # 创建训练样本
            sample = {
                "input": prompt,
                "output": safety_response,
                "category": category,
                "templates_used": selected_templates,
                "sample_id": obj.get("sample_id"),
            }
            
            dataset.append(sample)
    
    print(f"[Dataset Builder] 构建了 {len(dataset)} 个训练样本")
    
    # 按类别统计
    category_counts = {}
    for sample in dataset:
        cat = sample.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"[Dataset Builder] 类别分布:")
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count} 个样本")
    
    # 保存到文件（如果指定了输出路径）
    if output_path:
        save_dataset(dataset, output_path)
    
    return dataset


def combine_template_with_prompt(
    templates: List[str],
    prompt: str,
    category: str,
) -> Optional[str]:
    """
    组合template和prompt生成safety response
    
    Args:
        templates: Refusal templates列表
        prompt: Jailbreak prompt
        category: Prompt的类别
    
    Returns:
        Optional[str]: 生成的safety response，如果生成失败则返回None
    """
    if not templates:
        return None
    
    # 获取类别特定的refusal说明
    category_refusal = get_category_specific_refusal(category)
    
    # 组合templates（使用第一个template作为主要refusal，其他作为补充）
    main_template = templates[0]
    
    # 如果template中包含占位符，替换为类别特定的说明
    if "{category}" in main_template:
        main_template = main_template.replace("{category}", category_refusal)
    elif "content that" not in main_template.lower() and "information" not in main_template.lower():
        # 如果template没有指定具体内容，添加类别说明
        main_template = f"{main_template} {category_refusal}"
    
    # 构建完整的response
    response_parts = [main_template]
    
    # 添加额外的templates（如果有）
    if len(templates) > 1:
        for template in templates[1:]:
            if template != main_template:
                response_parts.append(template)
    
    # 添加礼貌的结尾
    response_parts.append("Is there anything else I can help you with?")
    
    # 组合成完整response
    response = " ".join(response_parts)
    
    # 清理多余的标点和空格
    response = response.replace("  ", " ").strip()
    
    return response


def save_dataset(dataset: List[Dict], output_path: str, format: str = "jsonl") -> None:
    """
    保存数据集到文件
    
    Args:
        dataset: 数据集列表
        output_path: 输出文件路径
        format: 输出格式（"jsonl"或"json"）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        # JSONL格式：每行一个JSON对象
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    else:
        # JSON格式：单个JSON数组
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"[Dataset Builder] 保存了 {len(dataset)} 个样本到 {output_path} (格式: {format})")


def load_dataset(input_path: str) -> List[Dict]:
    """
    从文件加载数据集
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        List[Dict]: 数据集列表
    """
    if not Path(input_path).exists():
        print(f"[Dataset Builder] 警告: 文件不存在: {input_path}")
        return []
    
    dataset = []
    
    # 自动检测格式
    if input_path.endswith(".jsonl"):
        # JSONL格式
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    dataset.append(sample)
                except json.JSONDecodeError as e:
                    print(f"[Dataset Builder] 警告: JSON解析失败: {e}")
                    continue
    else:
        # JSON格式
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                dataset = data
            elif isinstance(data, dict) and "dataset" in data:
                dataset = data["dataset"]
    
    print(f"[Dataset Builder] 从 {input_path} 加载了 {len(dataset)} 个样本")
    
    return dataset


def build_dataset_from_taxonomy(
    taxonomy_path: str,
    refusal_templates: List[str],
    output_path: Optional[str] = None,
    only_successful: bool = True,
    max_samples_per_category: Optional[int] = None,
    min_samples_per_category: Optional[int] = None,
    upsample_rare_categories: bool = True,
    seed: int = 42,
) -> List[Dict]:
    """
    从已保存的taxonomy文件构建数据集
    
    Args:
        taxonomy_path: Taxonomy文件路径
        refusal_templates: Refusal templates列表
        output_path: 可选，保存数据集的输出路径
        only_successful: 是否只使用successful jailbreak prompts
        max_samples_per_category: 每个generic类别的最大样本数（>0时对大类进行下采样）
        min_samples_per_category: 每个generic类别的最小样本数（>0时对冷门类进行过采样）
        upsample_rare_categories: 是否对冷门类别进行过采样（仅在min_samples_per_category生效时使用）
        seed: 随机种子
    
    Returns:
        List[Dict]: 训练样本列表
    """
    from .salad_taxonomy import load_salad_taxonomy_from_file
    
    taxonomy = load_salad_taxonomy_from_file(taxonomy_path)
    
    if not taxonomy:
        print("[Dataset Builder] 警告: taxonomy为空")
        return []
    
    random.seed(seed)
    dataset: List[Dict] = []

    # 先按successful过滤 & 去掉空prompt，得到每个类别的“可用样本池”
    filtered_by_category: Dict[str, List[Dict]] = {}
    for category, samples in taxonomy.items():
        available: List[Dict] = []
        for sample in samples:
            # 检查是否为successful jailbreak（如果taxonomy样本中包含该字段）。
            # 早期保存的taxonomy并不会记录jailbreak_success，因此这里采取“仅当显式为False时才过滤”的策略，
            # 避免把缺失字段的样本全部过滤掉。
            jailbreak_success = sample.get("jailbreak_success", None)
            if only_successful and jailbreak_success is False:
                continue

            prompt = sample.get("prompt", "")
            if not prompt:
                continue

            available.append(sample)

        if available:
            filtered_by_category[category] = available

    # 在过滤后的基础上做“每类上限 + 冷门类过采样”
    final_counts: Dict[str, int] = {}

    for category, samples in sorted(filtered_by_category.items()):
        orig_n = len(samples)
        chosen_samples = list(samples)

        # 对大类做下采样（无放回）
        if max_samples_per_category is not None and max_samples_per_category > 0:
            if orig_n > max_samples_per_category:
                chosen_samples = random.sample(chosen_samples, max_samples_per_category)

        # 对冷门类做过采样（有放回）
        if (
            min_samples_per_category is not None
            and min_samples_per_category > 0
            and upsample_rare_categories
            and len(chosen_samples) < min_samples_per_category
        ):
            if not chosen_samples:
                # 理论上不会发生，因为available非空才会进入filtered_by_category
                pass
            else:
                need = min_samples_per_category - len(chosen_samples)
                extra = random.choices(chosen_samples, k=need)
                chosen_samples = chosen_samples + extra

        # 记录最终该类别会生成多少数据
        final_counts[category] = len(chosen_samples)

        for sample in chosen_samples:
            prompt = sample.get("prompt", "")
            if not prompt:
                continue

            # 随机选择template
            template = random.choice(refusal_templates) if refusal_templates else None
            if not template:
                continue

            # 生成safety response
            safety_response = combine_template_with_prompt([template], prompt, category)
            if not safety_response:
                continue

            dataset.append(
                {
                    "input": prompt,
                    "output": safety_response,
                    "category": category,
                    "templates_used": [template],
                    "sample_id": sample.get("sample_id"),
                }
            )

    print(f"[Dataset Builder] 从taxonomy构建了 {len(dataset)} 个训练样本")
    print("[Dataset Builder] 按类别的最终样本数：")
    for cat, count in sorted(final_counts.items()):
        print(f"  - {cat}: {count} 个样本")
    
    if output_path:
        save_dataset(dataset, output_path)
    
    return dataset
