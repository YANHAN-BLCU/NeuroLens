"""
Refusal template extraction module

从LLM对 jailbreak 攻击的响应中提取 refusal templates，用于构建 refusal-guided fine-tuning 数据集。

Notes (based on mined results from `logs/base_evaluation.jsonl`):
- 真实输出里经常出现 “such as … / including …” 这类举例子分句，里面可能包含具体有害细节。
  在构建训练模板时，通常希望保留“拒绝骨架”，而去掉后面的举例内容。
- 之前实现的“按频率降序”排序实际上没有生效（Counter.items() 的顺序不是频率顺序）。
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Normalize common unicode punctuation seen in model outputs so regex patterns
# can match reliably (especially on Windows + mixed tokenizers).
def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    trans = str.maketrans(
        {
            0x2019: 0x27,  # ’ -> '
            0x2018: 0x27,  # ‘ -> '
            0x201C: 0x22,  # “ -> "
            0x201D: 0x22,  # ” -> "
            0x00A0: 0x20,  # NBSP -> space
        }
    )
    text = text.translate(trans)
    # Collapse whitespace for cleaner extraction
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Cut off example/specification clauses that often contain concrete harmful details.
# We want templates to be reusable “refusal skeletons”.
_SANITIZE_CUT_RE = re.compile(
    r"\b("
    r"such as|including|like|for example|for instance|e\.g\.?|eg\.?"
    r")\b"
    r"|(?:,|，)\s*(?:such as|including|like)\b"
    r"|(?:例如|比如|包括|如)\b",
    re.IGNORECASE,
)

# Split on common sentence terminators (English + Chinese). Note: we normalize whitespace,
# so this mostly guards against outputs with multiple sentences.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _sanitize_template(template: str) -> str:
    template = _normalize_text(template)
    if not template:
        return ""

    m = _SANITIZE_CUT_RE.search(template)
    if m:
        template = template[: m.start()].strip()

    # Remove trailing separators introduced by truncation
    template = template.rstrip(" ,，:;；-–—")

    # Ensure the result is a standalone sentence-ish string
    if template and template[-1] not in ".!?。！？":
        template = template + "."

    return template


def _is_incomplete_template(template: str) -> bool:
    """
    Heuristics for filtering obviously truncated templates like:
      "I cannot assist with illegal activities such"
    """
    t = _normalize_text(template).lower()
    if not t:
        return True

    # Ends with an unfinished example lead-in
    bad_endings = (
        "such",
        "such as",
        "including",
        "like",
        "for example",
        "for instance",
        "e.g",
        "e.g.",
        "eg",
        "eg.",
        "例如",
        "比如",
        "包括",
        "如",
    )
    if any(t.endswith(be) for be in bad_endings):
        return True

    # Ends with a dangling preposition (usually means the object got cut)
    dangling = (" on", " about", " regarding", " for", " to", " with", " in")
    if any(t.endswith(d) for d in dangling) and t[-1] not in ".!?。！？":
        return True

    return False


# 常见的refusal patterns（用于识别和提取refusal templates）
REFUSAL_PATTERNS = [
    # Direct inability / refusal (common)
    r"I cannot (?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I'm not able to (?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I can't (?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I'm unable to (?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I do not (?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I don't (?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I must (?:decline|refuse|not) (?:to )?(?:create|provide|generate|assist with|help with|support|offer guidance on)",
    r"I (?:cannot|can't|must not|should not) (?:create|provide|generate|assist with|help with|support|offer guidance on) (?:content|information|guidance|advice|assistance) (?:that|which)",
    r"I (?:cannot|can't|must not|should not) (?:help|assist|support) (?:with|you with|you in)",
    r"I (?:cannot|can't|must not|should not) (?:engage|participate) (?:in|with)",
    r"It is (?:not|inappropriate|illegal|unethical|harmful) (?:to|for me to)",
    r"This (?:is|would be) (?:inappropriate|illegal|unethical|harmful|against)",
    r"I (?:cannot|can't|must not|should not) (?:provide|give|offer) (?:advice|guidance|information|assistance) (?:on|about|regarding)",

    # Polite-preface refusals ("I'm sorry..., but ...")
    r"(?:I'm|I am) sorry(?:,)? (?:but )?I (?:cannot|can't|won't) ",
    r"(?:I'?m|I am) afraid I (?:cannot|can't|won't) ",

    # Policy / permission framing
    r"(?:I'm|I am) not (?:allowed|permitted) to ",
    r"(?:I|we) (?:won't|will not) (?:help|assist|provide|comply)",
    r"(?:I|we) (?:can't|cannot) comply with",
    r"(?:I|we) (?:can't|cannot) fulfill (?:that )?request",
    r"(?:I|we) (?:can't|cannot) help with that",
    r"(?:I|we) (?:can't|cannot) assist with that",
    r"(?:I|we) (?:can't|cannot) provide (?:instructions|steps|details|guidance) (?:for|on|about)",
    r"I (?:can't|cannot) provide (?:any )?(?:information|guidance|instructions|details) (?:on|about|for)",
    r"I (?:can't|cannot) provide (?:any )?(?:information|guidance|instructions|details) (?:regarding)",

    # Identity / capability framing used by many assistants
    r"As an AI(?: language model)?, I (?:cannot|can't|won't) ",
    r"(?:I|we) (?:do|does) not have the ability to ",
    r"(?:I|we) (?:don't|do not) have access to ",

    # Safety / legality / ethics explicit language
    r"(?:I|we) (?:can't|cannot) help with (?:illegal|harmful|dangerous) (?:activities|behavior|actions)",
    r"(?:I|we) must decline to (?:provide|share|assist with)",
    r"(?:I|we) have to refuse to (?:provide|share|assist with)",
]


def extract_refusal_templates(
    evaluation_log_path: str,
    min_length: int = 20,
    max_length: int = 200,
    min_frequency: int = 2,
    sanitize: bool = True,
    drop_incomplete: bool = True,
    dedupe_case_insensitive: bool = True,
) -> List[str]:
    """
    从评估日志中提取refusal templates
    
    从`guard.verdict == "allow"`的样本中提取模型响应，识别refusal patterns并提取templates。
    
    Args:
        evaluation_log_path: 评估日志文件路径（JSONL格式）
        min_length: 最小template长度（字符数）
        max_length: 最大template长度（字符数）
        min_frequency: 最小出现频率（出现次数少于此次数的template会被过滤）
    
    Returns:
        List[str]: 提取的refusal templates列表
    """
    template_counter = Counter()
    
    print(f"[Refusal Templates] 从 {evaluation_log_path} 提取refusal templates...")
    
    if not Path(evaluation_log_path).exists():
        print(f"[Refusal Templates] 警告: 文件不存在: {evaluation_log_path}")
        return []
    
    with open(evaluation_log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Refusal Templates] 警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
            
            # 只处理guard.verdict == "allow"的样本（模型成功拒绝的jailbreak）
            guard = obj.get("guard", {})
            verdict = guard.get("verdict", "").lower()
            
            if verdict != "allow":
                continue
            
            # 提取模型响应
            inference = obj.get("inference", {})
            output = inference.get("output", "")
            
            if not output or not isinstance(output, str):
                continue
            
            # 从响应中提取refusal templates
            extracted = _extract_templates_from_text(
                output,
                min_length=min_length,
                max_length=max_length,
                sanitize=sanitize,
                drop_incomplete=drop_incomplete,
            )
            template_counter.update(extracted)
    
    # 过滤低频templates
    filtered_items: List[Tuple[str, int]] = [
        (template, count) for template, count in template_counter.items() if count >= min_frequency
    ]

    # 按频率降序（tie-breaker：字典序，方便可复现）
    filtered_items.sort(key=lambda kv: (-kv[1], kv[0].lower()))
    sorted_templates = [t for t, _ in filtered_items]

    # 去重（可选：大小写不敏感）
    if dedupe_case_insensitive:
        seen = set()
        unique_templates: List[str] = []
        for t in sorted_templates:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            unique_templates.append(t)
    else:
        unique_templates = list(dict.fromkeys(sorted_templates))
    
    print(f"[Refusal Templates] 提取了 {len(unique_templates)} 个唯一的refusal templates")
    print(f"[Refusal Templates] 总提取次数: {sum(template_counter.values())}")
    
    return unique_templates


def extract_refusal_template_counts(
    evaluation_log_path: str,
    min_length: int = 20,
    max_length: int = 200,
    min_frequency: int = 2,
    sanitize: bool = True,
    drop_incomplete: bool = True,
) -> Dict[str, int]:
    """
    与 `extract_refusal_templates` 相同的抽取逻辑，但返回 {template: count}。
    """
    template_counter = Counter()

    if not Path(evaluation_log_path).exists():
        return {}

    with open(evaluation_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            guard = obj.get("guard", {}) or {}
            verdict = (guard.get("verdict", "") or "").lower()
            if verdict != "allow":
                continue
            output = ((obj.get("inference", {}) or {}).get("output")) or ""
            if not isinstance(output, str) or not output:
                continue
            extracted = _extract_templates_from_text(
                output,
                min_length=min_length,
                max_length=max_length,
                sanitize=sanitize,
                drop_incomplete=drop_incomplete,
            )
            template_counter.update(extracted)

    return {t: c for t, c in template_counter.items() if c >= min_frequency}


def _extract_templates_from_text(
    text: str,
    min_length: int = 20,
    max_length: int = 200,
    sanitize: bool = True,
    drop_incomplete: bool = True,
) -> List[str]:
    """
    从文本中提取refusal templates
    
    Args:
        text: 输入文本
        min_length: 最小template长度
        max_length: 最大template长度
    
    Returns:
        List[str]: 提取的templates列表
    """
    templates: List[str] = []
    text = _normalize_text(text)

    # 按句子分割（使用句号、问号、感叹号；含中文标点）
    sentences = _SENTENCE_SPLIT_RE.split(text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 检查是否包含refusal pattern
        for pattern in REFUSAL_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                # 提取从匹配位置开始到句子结束的部分
                start_pos = match.start()
                template = sentence[start_pos:].strip()
                
                # 清理template（移除多余的标点、空格）
                template = re.sub(r"\s+", " ", template).strip()

                if sanitize:
                    template = _sanitize_template(template)

                if drop_incomplete and _is_incomplete_template(template):
                    break
                
                # 检查长度
                if min_length <= len(template) <= max_length:
                    templates.append(template)
                    break  # 每个句子只提取一个template
    
    return templates


def analyze_refusal_patterns(
    templates: List[str],
    top_k: int = 20,
    template_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    分析并分类refusal patterns
    
    Args:
        templates: Refusal templates列表
        top_k: 返回top-k最常见的patterns
    
    Returns:
        Dict: 分析结果，包含：
            - "patterns": 最常见的patterns列表
            - "categories": 按类别分组的patterns
            - "statistics": 统计信息
    """
    if not templates:
        return {
            "patterns": [],
            "categories": {},
            "statistics": {},
        }
    
    # 统计patterns（优先使用计数信息）
    if template_counts:
        pattern_counter = Counter(template_counts)
    else:
        pattern_counter = Counter(templates)

    top_patterns = [pattern for pattern, _ in pattern_counter.most_common(top_k)]
    
    # 按类别分组（基于关键词）
    categories = {
        "cannot_create": [],
        "cannot_provide": [],
        "cannot_help": [],
        "illegal_unethical": [],
        "other": [],
    }
    
    for template in templates:
        template_lower = template.lower()
        if re.search(r"\b(can(?:not|'t))\s+create\b", template_lower):
            categories["cannot_create"].append(template)
        elif re.search(r"\b(can(?:not|'t))\s+provide\b", template_lower):
            categories["cannot_provide"].append(template)
        elif re.search(r"\b(can(?:not|'t))\s+(help|assist|support)\b", template_lower):
            categories["cannot_help"].append(template)
        elif any(word in template_lower for word in ["illegal", "unethical", "inappropriate", "harmful"]):
            categories["illegal_unethical"].append(template)
        else:
            categories["other"].append(template)
    
    # 去重每个类别
    for key in categories:
        categories[key] = list(dict.fromkeys(categories[key]))
    
    statistics = {
        "total_templates": len(templates),
        "unique_templates": len(pattern_counter),
        "most_common": dict(pattern_counter.most_common(10)),
        "category_counts": {key: len(value) for key, value in categories.items()},
    }
    
    return {
        "patterns": top_patterns,
        "categories": categories,
        "statistics": statistics,
    }


def save_refusal_templates(
    templates: List[str],
    output_path: str,
    template_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    保存refusal templates到JSON文件
    
    Args:
        templates: Templates列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data: Dict[str, Any] = {"refusal_templates": templates, "count": len(templates)}
    if template_counts:
        # Keep backward compatibility: still write `refusal_templates` as List[str]
        # but also store counts for inspection/reproducibility.
        data["template_counts"] = template_counts
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[Refusal Templates] 保存了 {len(templates)} 个templates到 {output_path}")


def load_refusal_templates(input_path: str) -> List[str]:
    """
    从JSON文件加载refusal templates
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        List[str]: Templates列表
    """
    if not Path(input_path).exists():
        print(f"[Refusal Templates] 警告: 文件不存在: {input_path}")
        return []
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    templates = data.get("refusal_templates", [])
    print(f"[Refusal Templates] 从 {input_path} 加载了 {len(templates)} 个templates")
    
    return templates
