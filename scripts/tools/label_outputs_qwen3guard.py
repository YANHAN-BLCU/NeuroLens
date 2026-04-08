"""
使用 Qwen3Guard-Gen-8B 对 data_set_output 下的五个输出文件打安全标签。
每条记录输入 prompt + generated_output，输出 Safe / Controversial / Unsafe。
结果只保存 original_index 和 label，写入与输入同名的 JSONL 标签文件。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = Path("/root/autodl-tmp/neurolens")
GUARD_PATH = ROOT / "ms_models/Qwen/Qwen3Guard-Gen-8B"
OUTPUT_DIR = ROOT / "outputs/data_set_output"
LABEL_DIR  = ROOT / "outputs/data_set_output/labels"
DATA_DIR   = ROOT / "data/salad/raw"

LABEL_DIR.mkdir(parents=True, exist_ok=True)

# 五个输入文件及其对应的原始数据集（用于取 prompt）
FILE_CONFIG = [
    {
        "output_file": "attack_enhanced_outputs.jsonl",
        "source_file": DATA_DIR / "attack_enhanced_set_train.jsonl",
        "prompt_field": "augq",      # 使用增强攻击 prompt
    },
    {
        "output_file": "base_set_outputs_0_4999.jsonl",
        "source_file": DATA_DIR / "base_set_train.jsonl",
        "prompt_field": "question",
    },
    {
        "output_file": "base_set_outputs_5000_9999.jsonl",
        "source_file": DATA_DIR / "base_set_train.jsonl",
        "prompt_field": "question",
    },
    {
        "output_file": "base_set_outputs_10000_14999.jsonl",
        "source_file": DATA_DIR / "base_set_train.jsonl",
        "prompt_field": "question",
    },
    {
        "output_file": "base_set_outputs_15000_21316.jsonl",
        "source_file": DATA_DIR / "base_set_train.jsonl",
        "prompt_field": "question",
    },
]

# ── 工具函数 ──────────────────────────────────────────────────────────────────
SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)


def extract_label(content: str) -> str:
    """从模型输出中提取 Safe / Controversial / Unsafe，解析失败返回 unknown。"""
    m = SAFE_PATTERN.search(content)
    if m:
        return m.group(1).capitalize()   # 统一首字母大写
    return "unknown"


def load_source(source_file: Path, prompt_field: str) -> list[str]:
    """按行加载数据集，返回以行号为索引的 prompt 列表。"""
    prompts: list[str] = []
    with open(source_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append(item.get(prompt_field, ""))
    return prompts


# ── 加载模型 ──────────────────────────────────────────────────────────────────
print(f"[Guard] 加载模型: {GUARD_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(GUARD_PATH))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    str(GUARD_PATH),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
print("[Guard] 模型加载完成")


def moderate(prompt: str, response: str) -> str:
    """调用 Qwen3Guard-Gen 进行 response moderation，返回标签字符串。"""
    messages = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return extract_label(content)


# ── 主循环 ────────────────────────────────────────────────────────────────────
for cfg in FILE_CONFIG:
    out_path   = OUTPUT_DIR / cfg["output_file"]
    label_path = LABEL_DIR  / cfg["output_file"]

    if label_path.exists():
        print(f"[跳过] {cfg['output_file']} 标签文件已存在，如需重新生成请删除后运行。")
        continue

    print(f"\n[处理] {cfg['output_file']}")
    prompts = load_source(cfg["source_file"], cfg["prompt_field"])

    with open(out_path, encoding="utf-8", errors="replace") as fin, \
         open(label_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [警告] 第 {i} 行 JSON 解析失败，跳过: {e}")
                continue
            idx      = item["original_index"]
            response = item["generated_output"]

            # 用行序号 i 取对应 prompt（original_index 是全局序号，i 是文件内行号）
            prompt = prompts[idx] if idx < len(prompts) else ""

            label = moderate(prompt, response)

            fout.write(json.dumps({"original_index": idx, "label": label}, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1} 条")

    print(f"[完成] 标签已写入 {label_path}")

print("\n全部文件处理完毕。")
