"""中文 NER 微调：识别姓名(PER) 与详细地址(ADDR)"""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ADDR", "I-ADDR"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def char_labels_from_entities(text: str, entities: list[dict]) -> list[str]:
    labels = ["O"] * len(text)
    for ent in sorted(entities, key=lambda x: x["start"]):
        tag = ent["label"].upper()
        if tag not in {"PER", "ADDR"}:
            continue
        start, end = ent["start"], ent["end"]
        if start < 0 or end > len(text) or start >= end:
            continue
        labels[start] = f"B-{tag}"
        for i in range(start + 1, end):
            labels[i] = f"I-{tag}"
    return labels


def tokenize_and_align_labels(
    samples: list[dict],
    tokenizer,
    max_length: int,
) -> Dataset:
    texts = [s["text"] for s in samples]
    char_labels_list = [
        char_labels_from_entities(s["text"], s.get("entities", [])) for s in samples
    ]

    encodings = tokenizer(
        [list(text) for text in texts],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels: list[list[int]] = []
    for i, char_labels in enumerate(char_labels_list):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids: list[int] = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(LABEL2ID[char_labels[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)

    encodings["labels"] = all_labels
    return Dataset.from_dict(encodings)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels, pred_labels = [], []
    for pred_row, label_row in zip(preds, labels):
        true_seq, pred_seq = [], []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            true_seq.append(ID2LABEL[int(label_id)])
            pred_seq.append(ID2LABEL[int(pred_id)])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def default_pretrained_model(root: Path) -> str:
    local = root / "models" / "pretrained" / "rbt3"
    if (local / "pytorch_model.bin").exists() or (local / "model.safetensors").exists():
        return str(local)
    return "hfl/rbt3"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Train Chinese PER/ADDR NER model")
    parser.add_argument("--model", default=default_pretrained_model(root))
    parser.add_argument("--train", type=Path, default=root / "data" / "train.jsonl")
    parser.add_argument("--valid", type=Path, default=root / "data" / "valid.jsonl")
    parser.add_argument("--output", type=Path, default=root / "models" / "chinese-ner-per-addr-rbt3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train_samples = load_jsonl(args.train)
    valid_samples = load_jsonl(args.valid)
    if not train_samples:
        raise SystemExit(f"训练集为空: {args.train}")
    if not valid_samples:
        raise SystemExit(f"验证集为空: {args.valid}")

    print(f"训练样本: {len(train_samples)}, 验证样本: {len(valid_samples)}")
    print(f"预训练模型: {args.model}")
    print(f"输出目录: {args.output}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = tokenize_and_align_labels(train_samples, tokenizer, args.max_length)
    valid_ds = tokenize_and_align_labels(valid_samples, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("验证集指标:", metrics)

    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"模型已保存至 {args.output}")


if __name__ == "__main__":
    main()
