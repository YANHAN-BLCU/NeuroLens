"""
SALAD taxonomy integration module

从SALAD-Bench数据集中获取jailbreak prompts的fine-grained类别信息。
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GENERIC_CATEGORIES: Tuple[str, ...] = (
    "violence",
    "hate",
    "self-harm",
    "illegal",
    "harassment",
    "fraud",
    "property",
    "information",
    "malicious",
    "autonomy",
    "toxicity",
    "sexual",
    "other",
)


# Base mappings from SALAD-Bench taxonomy fields to a stable set of generic buckets.
# NOTE: In our evaluation logs we often see 1/2/3-category together; 2-category is
# much smaller-cardinality than 3-category, so it is usually the most robust base.
SALAD_2_CATEGORY_TO_GENERIC: Dict[str, str] = {
    # Toxic / harassment
    "O1: Toxic Content": "toxicity",
    "O2: Unfair Representation": "hate",
    "O11: Defamation": "harassment",
    "O4: Harassment": "harassment",
    "O9: Privacy Infringement": "harassment",
    # Illegal / weapons / threats
    "O14: Illegal Activities": "illegal",
    "O10: Security Threats": "malicious",
    # Deception / misinformation
    "O12: Fraud or Deceptive Action": "fraud",
    "O13: Influence Operations": "fraud",
    "O4: Erosion of Trust in Public Information": "fraud",
    "O5: Propagating Misconceptions/False Beliefs": "fraud",
    "O6: Risky Financial Practices": "fraud",
    # Property
    "O16: Violation of Personal Property": "property",
    "O65: Plagiarism": "property",
    # Information hazards
    "O8: Dissemination of Dangerous Information": "information",
    # Adult / sexual content
    "O3: Adult Content": "sexual",
    # Autonomy / manipulation
    "O15: Persuasion and Manipulation": "autonomy",
}


SALAD_1_CATEGORY_TO_GENERIC: Dict[str, str] = {
    "O5: Malicious Use": "malicious",
    "O1: Representation & Toxicity": "toxicity",
    "O2: Misinformation Harms": "fraud",
    "O6: Human Autonomy & Integrity": "autonomy",
    "O4: Information & Safety": "information",
    "O3: Socioeconomic Harms": "other",
}


# Some common 3-category labels we have seen in evaluation logs. This mapping is
# intentionally partial; we prefer 2-category/1-category when available.
SALAD_3_CATEGORY_TO_GENERIC: Dict[str, str] = {
    # Violence
    "O56: Violent Crimes": "violence",
    "O5: Violent Content": "violence",
    "O35: Weapon Generation and Transportation": "violence",
    # Hate / toxicity
    "O1: Hate Speech": "hate",
    "O7: Insult": "toxicity",
    "O3: Enjoying someone else's Pain": "toxicity",
    "O11: Gender Stereotyping": "hate",
    "O8: Racial and Ethnic Stereotyping": "hate",
    # Harassment / bullying
    "O2: Harass, Threaten, or Bully An Individual": "harassment",
    "O33: Inferring Personal Information": "harassment",
    "O34: Leaking Private Information": "harassment",
    # Illegal
    "O57: Theft": "illegal",
    "O53: Financial Crimes": "illegal",
    "O19: Illegal Drugs and Regulated/Controlled Substances": "illegal",
    "O67: Forge Identity": "illegal",
    "O60: Traffic and Driving Offenses": "illegal",
    # Fraud / disinfo
    "O47: Disinformation": "fraud",
    "O29: Fake News": "fraud",
    "O30: Social Media Rumors": "fraud",
    "O45: Fake Review Generation": "fraud",
    "O46: Fake Online Content": "fraud",
    "O48: Spam": "fraud",
    "O50: Manipulation of Public Opinion": "fraud",
    "O51: Campaign Materials": "fraud",
    "O21: Historical Revisionism": "fraud",
    "O22: Science Denialism": "fraud",
    # Property / academic dishonesty
    "O44: Academic Dishonesty": "property",
    "O65: Plagiarism": "property",
    "O64: Mimicking Writing Style": "property",
    "O66: Copyright Infringement": "property",
    # Sexual
    "O14: Erotic Chat": "sexual",
    "O16: Pornography": "sexual",
    "O6: Child Abuse": "violence",
    # Autonomy / manipulation
    "O63: Psychological Manipulation": "autonomy",
    # Information / safety-adjacent
    "O31: Government Secrets": "information",
    "O20: Medical Advice": "information",
    # Fraud / scams / shady finance
    "O26: Gambling": "fraud",
    "O24: Multi-level Marketing": "fraud",
    "O25: Paypal Lending": "fraud",
    # Self-harm
    "O2: Self-Harm": "self-harm",
    "O3: Suicide": "self-harm",
    "O62: Self-Harm": "self-harm",
    # Malicious / security threats
    "O38: Cyber Attack": "malicious",
    "O37: Malware Generation": "malicious",
    "O36: Management or Operation of Critical Infrastructure": "malicious",
    # Violent extremism
    "O4: Terrorism": "violence",
    "O23: Radicalization": "violence",
}


def _normalize_generic_category(category: str) -> str:
    """
    Normalize an already-generic label (or a close variant) to our canonical set.
    """
    c = (category or "").strip()
    if not c:
        return "other"
    c_low = c.lower().strip()
    aliases = {
        "self harm": "self-harm",
        "self_harm": "self-harm",
        "selfharm": "self-harm",
        "hate speech": "hate",
        "tox": "toxicity",
        "sex": "sexual",
    }
    c_low = aliases.get(c_low, c_low)
    for g in GENERIC_CATEGORIES:
        if c_low == g:
            return g
    return c


def map_salad_category_to_generic(category: str) -> str:
    """
    Map SALAD taxonomy strings (1/2/3-category) into a stable generic bucket.
    If the input is already a generic bucket, it is normalized and returned.
    """
    c = (category or "").strip()
    if not c:
        return "other"

    # Already-generic labels
    c_norm = _normalize_generic_category(c)
    if c_norm in GENERIC_CATEGORIES:
        return c_norm

    # Try exact SALAD field mappings (3 → 2 → 1)
    if c in SALAD_3_CATEGORY_TO_GENERIC:
        return SALAD_3_CATEGORY_TO_GENERIC[c]
    if c in SALAD_2_CATEGORY_TO_GENERIC:
        return SALAD_2_CATEGORY_TO_GENERIC[c]
    if c in SALAD_1_CATEGORY_TO_GENERIC:
        return SALAD_1_CATEGORY_TO_GENERIC[c]

    # Heuristic fallbacks (keyword-based)
    c_low = c.lower()
    if any(k in c_low for k in ("porn", "erotic", "sexual", "adult")):
        return "sexual"
    if any(k in c_low for k in ("theft", "drugs", "illegal", "crime", "financial", "forge identity", "traffic", "driving")):
        return "illegal"
    if any(k in c_low for k in ("weapon", "violent", "violence", "terror", "radical", "child abuse", "harm")):
        return "violence"
    if any(k in c_low for k in ("harass", "bully", "threat", "defam", "privacy", "surveillance", "inferring personal", "leaking private")):
        return "harassment"
    if any(k in c_low for k in ("fake news", "disinfo", "misinfo", "decept", "fraud", "scam", "spam", "fake review", "manipulation of public opinion")):
        return "fraud"
    if any(k in c_low for k in ("stereotyp", "hate", "rac", "gender")):
        return "hate"
    if any(k in c_low for k in ("malware", "cyber", "critical infrastructure")):
        return "malicious"
    if any(k in c_low for k in ("self-harm", "suicide")):
        return "self-harm"
    if any(k in c_low for k in ("psychological manipulation", "manipulation", "coercion")):
        return "autonomy"
    if any(k in c_low for k in ("government secrets", "medical advice")):
        return "information"

    return "other"


def extract_prompt(evaluation_sample: Dict) -> str:
    """
    Extract prompt text from an evaluation sample with robust fallbacks.
    """
    input_data = evaluation_sample.get("input", {})
    if isinstance(input_data, dict):
        prompt = input_data.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

        # Some pipelines store chat messages instead of a single prompt string
        messages = input_data.get("messages")
        if isinstance(messages, list) and messages:
            parts: List[str] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
            if parts:
                return "\n".join(parts)

        text = input_data.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    if isinstance(input_data, str):
        return input_data.strip()

    return ""


def get_prompt_category(
    evaluation_sample: Dict,
    prefer_guard_categories: bool = True,
    original_field_priority: Tuple[str, ...] = ("3-category", "2-category", "1-category"),
    map_to_generic: bool = True,
) -> Optional[str]:
    """
    从评估样本中获取jailbreak prompt的类别
    
    优先使用guard.categories，如果没有则使用original_sample的(1/2/3-category)字段。
    
    Args:
        evaluation_sample: 评估日志中的样本（JSON对象）
        prefer_guard_categories: 是否优先使用guard.categories
        original_field_priority: original_sample中用于回退的字段顺序
        map_to_generic: 是否将SALAD类别映射到通用bucket（violence/illegal/sexual/...）
    
    Returns:
        Optional[str]: 类别名称（generic bucket或原始类别），如果无法确定则返回None
    """
    def _keep_only_known_generic_ids(items: Iterable[str]) -> List[str]:
        kept: List[str] = []
        for x in items:
            g = _normalize_generic_category(x)
            if g in GENERIC_CATEGORIES and g != "other":
                kept.append(g)
        return kept

    # 优先使用guard.categories
    if prefer_guard_categories:
        guard = evaluation_sample.get("guard", {})
        categories = guard.get("categories", []) if isinstance(guard, dict) else []
        
        if categories and isinstance(categories, list) and len(categories) > 0:
            # 允许两种格式：
            # - [{"id"/"label": str, "score": float}, ...]
            # - ["illegal", "violence", ...]
            if all(isinstance(x, str) for x in categories):
                kept = _keep_only_known_generic_ids([c for c in categories if isinstance(c, str)])
                if len(kept) == 1:
                    return kept[0] if map_to_generic else kept[0]
                # If guard returns multiple generic ids with no ranking, treat as ambiguous and fall back.

            dict_cats = [x for x in categories if isinstance(x, dict)]
            if dict_cats:
                scored: List[Tuple[str, float]] = []
                for d in dict_cats:
                    category_id = d.get("id") or d.get("label") or d.get("category")
                    if not isinstance(category_id, str) or not category_id.strip():
                        continue
                    g = _normalize_generic_category(category_id.strip())
                    if g not in GENERIC_CATEGORIES or g == "other":
                        continue
                    score = d.get("score", 0.0)
                    try:
                        score_f = float(score)
                    except (TypeError, ValueError):
                        score_f = 0.0
                    scored.append((g, score_f))

                if scored:
                    scored.sort(key=lambda x: x[1], reverse=True)
                    # If the guard scores are tied/near-tied, the result is ambiguous.
                    if len(scored) >= 2 and abs(scored[0][1] - scored[1][1]) < 1e-9:
                        scored = []
                    if scored:
                        return scored[0][0] if map_to_generic else scored[0][0]
    
    # 使用original_sample的回退字段
    input_data = evaluation_sample.get("input", {})
    original_sample = input_data.get("original_sample", {})
    
    if isinstance(original_sample, dict):
        for field in original_field_priority:
            cat = original_sample.get(field, "")
            if isinstance(cat, str) and cat.strip():
                cat = cat.strip()
                return map_salad_category_to_generic(cat) if map_to_generic else cat
    
    return None


def load_salad_taxonomy(
    evaluation_log_path: str,
    output_path: Optional[str] = None,
    original_field_priority: Tuple[str, ...] = ("3-category", "2-category", "1-category"),
    map_to_generic: bool = True,
) -> Dict[str, List[Dict]]:
    """
    从评估日志中加载SALAD taxonomy映射
    
    为每个类别收集相关的prompts和样本信息。
    
    Args:
        evaluation_log_path: 评估日志文件路径（JSONL格式）
        output_path: 可选，保存taxonomy映射的输出路径
        original_field_priority: original_sample中用于回退的字段顺序
        map_to_generic: 是否将类别映射到通用bucket（violence/illegal/sexual/...）
    
    Returns:
        Dict[str, List[Dict]]: 按类别分组的样本字典，格式为：
            {
                "violence": [{"prompt": "...", "sample_id": 0, ...}, ...],
                "hate": [{"prompt": "...", "sample_id": 1, ...}, ...],
                ...
            }
    """
    taxonomy = {}
    
    print(f"[SALAD Taxonomy] 从 {evaluation_log_path} 加载taxonomy...")
    
    if not Path(evaluation_log_path).exists():
        print(f"[SALAD Taxonomy] 警告: 文件不存在: {evaluation_log_path}")
        return taxonomy
    
    with open(evaluation_log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[SALAD Taxonomy] 警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
            
            # 获取类别
            category = get_prompt_category(
                obj,
                original_field_priority=original_field_priority,
                map_to_generic=map_to_generic,
            )
            if not category:
                continue
            
            # 提取prompt
            prompt = extract_prompt(obj)
            
            if not prompt:
                continue

            input_data = obj.get("input", {})
            original_sample = input_data.get("original_sample", {}) if isinstance(input_data, dict) else {}
            
            # 添加到taxonomy
            if category not in taxonomy:
                taxonomy[category] = []
            
            taxonomy[category].append({
                "prompt": prompt,
                "sample_id": obj.get("sample_id"),
                "category": category,
                "original_category_1": original_sample.get("1-category") if isinstance(original_sample, dict) else None,
                "original_category_2": original_sample.get("2-category") if isinstance(original_sample, dict) else None,
                "original_category_3": original_sample.get("3-category") if isinstance(original_sample, dict) else None,
                "mapped_to_generic": map_to_generic,
            })
    
    # 统计信息
    total_samples = sum(len(samples) for samples in taxonomy.values())
    print(f"[SALAD Taxonomy] 加载了 {len(taxonomy)} 个类别，共 {total_samples} 个样本")
    for category, samples in sorted(taxonomy.items()):
        print(f"  - {category}: {len(samples)} 个样本")
    
    # 保存到文件（如果指定了输出路径）
    if output_path:
        save_salad_taxonomy(taxonomy, output_path)
    
    return taxonomy


def save_salad_taxonomy(taxonomy: Dict[str, List[Dict]], output_path: str) -> None:
    """
    保存SALAD taxonomy映射到JSON文件
    
    Args:
        taxonomy: Taxonomy字典
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "taxonomy": taxonomy,
        "category_counts": {cat: len(samples) for cat, samples in taxonomy.items()},
        "total_samples": sum(len(samples) for samples in taxonomy.values()),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[SALAD Taxonomy] 保存taxonomy到 {output_path}")


def load_salad_taxonomy_from_file(input_path: str) -> Dict[str, List[Dict]]:
    """
    从JSON文件加载SALAD taxonomy映射
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        Dict[str, List[Dict]]: Taxonomy字典
    """
    if not Path(input_path).exists():
        print(f"[SALAD Taxonomy] 警告: 文件不存在: {input_path}")
        return {}
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    taxonomy = data.get("taxonomy", {})
    print(f"[SALAD Taxonomy] 从 {input_path} 加载了 {len(taxonomy)} 个类别")
    
    return taxonomy


def get_category_specific_refusal(category: str) -> str:
    """
    根据类别生成类别特定的refusal说明
    
    Args:
        category: 类别名称（如"violence", "hate"等）
    
    Returns:
        str: 类别特定的refusal说明
    """
    category_refusals = {
        "violence": "content that promotes violence or harm",
        "hate": "content that promotes hate speech or discrimination",
        "self-harm": "content related to self-harm or suicide",
        "illegal": "illegal activities or content",
        "harassment": "content that involves harassment or privacy violations",
        "fraud": "content related to fraud or deception",
        "property": "content that violates intellectual property or personal property",
        "information": "dangerous or harmful information",
        "malicious": "malicious or harmful content",
        "autonomy": "content that violates human autonomy or integrity",
        "toxicity": "toxic or harmful content",
        "sexual": "explicit sexual content",
    }
    
    return category_refusals.get(category, "harmful or inappropriate content")
