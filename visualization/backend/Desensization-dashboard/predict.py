"""加载微调 NER 模型，在验证集上评估或对新文本预测。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

LABELS = ("PER", "ADDR")


def load_ner(model_dir: Path, device: int):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )


def entities_from_ner(text: str, ner) -> list[dict]:
    return [
        {
            "start": int(e["start"]),
            "end": int(e["end"]),
            "label": e["entity_group"],
            "text": text[int(e["start"]) : int(e["end"])],
        }
        for e in ner(text)
    ]


def char_labels_from_entities(text: str, entities: list[dict]) -> list[str]:
    labels = ["O"] * len(text)
    for ent in sorted(entities, key=lambda x: x["start"]):
        tag = ent["label"].upper()
        start, end = ent["start"], ent["end"]
        labels[start] = f"B-{tag}"
        for i in range(start + 1, end):
            labels[i] = f"I-{tag}"
    return labels


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate(ner, valid_path: Path) -> None:
    samples = load_jsonl(valid_path)
    true_labels, pred_labels = [], []

    print(f"验证集: {valid_path} ({len(samples)} 条)\n")
    for sample in samples:
        text = sample["text"]
        gold = sample.get("entities", [])
        pred = entities_from_ner(text, ner)
        true_labels.append(char_labels_from_entities(text, gold))
        pred_labels.append(char_labels_from_entities(text, pred))

        status = "OK" if gold == [
            {"start": e["start"], "end": e["end"], "label": e["label"]} for e in pred
        ] else "DIFF"
        print(f"[{status}] {text}")
        if gold:
            print(f"  标注: {gold}")
        if pred:
            print(f"  预测: {pred}")
        if status == "DIFF" and not pred and gold:
            print("  预测: (无)")

    print("\n--- 整体指标 ---")
    print(f"Precision: {precision_score(true_labels, pred_labels):.3f}")
    print(f"Recall:    {recall_score(true_labels, pred_labels):.3f}")
    print(f"F1:        {f1_score(true_labels, pred_labels):.3f}")
    print("\n--- 分类报告 ---")
    print(classification_report(true_labels, pred_labels, digits=3))


def predict_texts(ner, texts: list[str]) -> None:
    for text in texts:
        ents = entities_from_ner(text, ner)
        print(f"输入: {text}")
        if not ents:
            print("  (未识别到 PER/ADDR)")
        for e in ents:
            print(f"  [{e['label']}] {e['text']} ({e['start']}-{e['end']})")
        print()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Evaluate or predict with Chinese PER/ADDR NER")
    parser.add_argument("--model", type=Path, default=root / "models" / "chinese-ner-per-addr-rbt3")
    parser.add_argument("--valid", type=Path, default=root / "data" / "valid.jsonl")
    parser.add_argument("--text", action="append", default=[], help="待预测的单条文本，可重复传入")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = 0 if torch.cuda.is_available() else -1
    print(f"模型: {args.model}")
    print(f"设备: {'cuda' if device == 0 else 'cpu'}\n")

    ner = load_ner(args.model, device)

    if args.text:
        predict_texts(ner, args.text)
    else:
        evaluate(ner, args.valid)


if __name__ == "__main__":
    main()
