#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新处理 attack_enhanced_outputs.jsonl 中空输出的样本，
检测空输出是由模型行为还是代码问题导致的。

诊断内容：
  1. 模型实际生成了什么（原始 token ID、长度）
  2. EOS 检测是否正确
  3. tokenizer 解码结果
  4. 若重新生成后仍然为空，则很可能是模型行为；
     若重新生成后有输出，则说明原代码存在 bug
"""

import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.pop("NPU_VISIBLE_DEVICES", None)
os.environ.pop("ASCEND_VISIBLE_DEVICES", None)

import json
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 保持与 extract_salad_hidden_states.py 完全一致的生成参数
MAX_NEW_TOKENS = 16384
GEN_TEMPERATURE = 0.3
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.05


def load_jsonl(file_path: Path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    output_dir = PROJECT_ROOT / "outputs" / "salad_extraction"
    dataset_file = PROJECT_ROOT / "data" / "salad" / "raw" / "attack_enhanced_set_train.jsonl"
    outputs_file = output_dir / "attack_enhanced_outputs.jsonl"

    # --- 1. 找出所有空输出的 original_index ---
    empty_indices = []
    all_outputs = []
    with open(outputs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["generated_output"] == "":
                    empty_indices.append(rec["original_index"])
                all_outputs.append(rec)

    print(f"总输出条数: {len(all_outputs)}")
    print(f"空输出条数: {len(empty_indices)}")
    print(f"空输出索引: {empty_indices}")
    print()

    # --- 2. 加载原始数据集 ---
    raw_data = load_jsonl(dataset_file)
    print(f"数据集总条数: {len(raw_data)}")
    print()

    # --- 3. 加载模型 ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        from engine.models import ModelManager
        mm = ModelManager()
        tokenizer, model = mm.load_llm()
        device = next(model.parameters()).device
        print(f"LLM 加载成功, Device: {device}")
    except Exception as e:
        print(f"LLM 加载失败: {e}")
        sys.exit(1)

    # --- 4. 对每个空输出样本进行诊断性重新生成 ---
    results = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for idx in tqdm(empty_indices, desc="诊断空输出"):
        sample = raw_data[idx]
        input_text = sample.get("augq", "")

        # 构建 prompt（与原代码完全一致）
        formatted = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

        inputs = tokenizer(
            [formatted],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=False,
            max_length=16384,
        ).to(device)

        attention_mask = inputs["attention_mask"]
        inp_len = (attention_mask[0] == 1).sum().item()

        # 诊断生成
        with torch.inference_mode():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=GEN_TEMPERATURE > 0,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REPETITION_PENALTY,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )

        generated_ids = gen_out.sequences[0]
        raw_output_ids = generated_ids[inp_len:]
        gen_len = raw_output_ids.shape[0]

        # EOS 检测
        eos_positions = (raw_output_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            eos_first = eos_positions[0].item()
            if eos_first == 0:
                last_pos = len(generated_ids) - 1
            else:
                last_pos = min(inp_len + eos_first - 1, len(generated_ids) - 1)
        else:
            last_pos = len(generated_ids) - 1

        # 解码
        decoded_raw = tokenizer.decode(raw_output_ids, skip_special_tokens=True).strip()
        decoded_eos = tokenizer.decode(generated_ids[inp_len:last_pos + 1], skip_special_tokens=True).strip()

        result = {
            "original_index": idx,
            "baseq": sample.get("baseq", ""),
            "method": sample.get("method", ""),
            "input_len": inp_len,
            "total_gen_len": len(generated_ids),
            "raw_gen_len": gen_len,
            "has_eos": len(eos_positions) > 0,
            "eos_first_pos": int(eos_positions[0].item()) if len(eos_positions) > 0 else None,
            "last_pos": last_pos,
            "raw_gen_ids": raw_output_ids.tolist(),
            "raw_gen_ids_first_20": raw_output_ids[:20].tolist(),
            "decoded_raw": decoded_raw,
            "decoded_eos": decoded_eos,
            "output_is_empty": len(decoded_eos) == 0,
        }
        results.append(result)

    # --- 5. 输出诊断报告 ---
    report_path = output_dir / "empty_output_diagnosis.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n诊断报告已保存: {report_path}")

    # --- 6. 统计摘要 ---
    still_empty = [r for r in results if r["output_is_empty"]]
    has_output = [r for r in results if not r["output_is_empty"]]

    print(f"\n===== 诊断摘要 =====")
    print(f"重新生成后仍有空输出: {len(still_empty)} / {len(empty_indices)}")
    print(f"重新生成后有输出:    {len(has_output)} / {len(empty_indices)}")

    if has_output:
        print(f"\n以下样本重新生成后有输出（说明原代码可能有问题）:")
        for r in has_output:
            print(f"  索引 {r['original_index']} | 方法 {r['method']} | 原始输入: {r['baseq'][:60]}...")
            print(f"    raw_gen_len={r['raw_gen_len']}, has_eos={r['has_eos']}, decoded[:100]: {r['decoded_eos'][:100]}")

    if still_empty:
        print(f"\n以下样本重新生成后仍为空（说明是模型自身行为）:")
        for r in still_empty:
            print(f"  索引 {r['original_index']} | 方法 {r['method']} | raw_gen_len={r['raw_gen_len']}, has_eos={r['has_eos']}")
            print(f"    raw_gen_ids[:20]={r['raw_gen_ids_first_20']}")
            print(f"    decoded_raw[:100]={r['decoded_raw'][:100]!r}")


if __name__ == "__main__":
    main()
