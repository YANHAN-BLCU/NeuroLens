#!/usr/bin/env python3
"""
Utility task evaluation script

Usage:
    python scripts/eval_utility.py --model meta-llama/Llama-3.2-3B-Instruct \
      --task csqa --max_samples 500 \
      --output logs/baseline/utility_csqa.jsonl
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets package required")
    print("Run: pip install datasets")
    sys.exit(1)


def load_utility_dataset(task: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load utility task dataset
    
    Args:
        task: Task name (csqa|piqa|race)
        max_samples: Maximum samples to load
    
    Returns:
        List of samples
    """
    dataset_map = {
        "csqa": "commonsense_qa",
        "piqa": "piqa",
        "race": "race",
    }
    
    dataset_name = dataset_map.get(task)
    if not dataset_name:
        raise ValueError(f"Unknown task: {task}")
    
    print(f"Loading dataset: {dataset_name}")
    
    if task == "race":
        dataset = load_dataset("race", "all")
        split = "test"
    else:
        dataset = load_dataset(dataset_name)
        split = "validation" if "validation" in dataset else "test"
    
    data = dataset[split]
    samples = []
    
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        
        sample = {
            "id": f"{task}_{i}",
            "task": task,
            "question": item.get("question", ""),
            "choices": item.get("choices", []),
            "answer": item.get("answerKey", item.get("answer", "")),
        }
        samples.append(sample)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def evaluate_sample(
    model,
    tokenizer,
    sample: Dict,
    device: torch.device,
) -> Dict:
    """
    Evaluate a single utility task sample
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        sample: Sample dictionary
        device: Device to run on
    
    Returns:
        Evaluation result
    """
    question = sample.get("question", "")
    choices = sample.get("choices", [])
    correct_answer = sample.get("answer", "")
    
    if not question or not choices:
        return {
            "sample_id": sample.get("id", "unknown"),
            "error": "Invalid sample",
        }
    
    start_time = time.time()
    
    try:
        # Format prompt
        prompt = question + "\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Extract answer (first letter or word)
        predicted_answer = response[0].upper() if response else ""
        if predicted_answer not in [chr(65+i) for i in range(len(choices))]:
            predicted_answer = ""
        
        is_correct = predicted_answer == correct_answer.upper()
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "sample_id": sample.get("id", "unknown"),
            "task": sample.get("task", "unknown"),
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "latency_ms": latency_ms,
            "num_tokens": len(outputs[0]),
        }
        
    except Exception as e:
        return {
            "sample_id": sample.get("id", "unknown"),
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate utility tasks")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--task", type=str, required=True, choices=["csqa", "piqa", "race"], help="Task name")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16 if args.precision == "fp16" else torch.float32,
        device_map="auto",
    )
    
    # Load data
    samples = load_utility_dataset(args.task, max_samples=args.max_samples)
    
    # Evaluate
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    correct_count = 0
    
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"Processing sample {i+1}/{len(samples)}")
        
        result = evaluate_sample(model, tokenizer, sample, device)
        result["timestamp"] = datetime.now().isoformat()
        results.append(result)
        
        if result.get("is_correct"):
            correct_count += 1
        
        # Write incrementally
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    accuracy = (correct_count / len(results)) * 100 if results else 0
    
    print(f"\nEvaluation complete. Results saved to: {output_path}")
    print(f"Total samples: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

