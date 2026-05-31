#!/usr/bin/env python3
"""最小测试 - 只测5条样本，验证模型能跑通"""
import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-1.5B-Instruct"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"模型: {model_path}")
    print(f"样本数: {num_samples}")
    
    # 加载数据
    data_file = Path(__file__).parent.parent / "data" / "salad" / "raw" / "attack_enhanced_set_train.jsonl"
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条样本")
    
    # 加载模型
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"加载模型...")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # CPU用float32更稳定
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型加载完成 ({time.time()-start:.1f}s)")
    
    # 逐条推理
    for i, sample in enumerate(samples):
        prompt = sample.get("augq") or sample.get("baseq", "")
        print(f"\n--- 样本 {i+1}/{len(samples)} ---")
        print(f"Prompt: {prompt[:100].encode('utf-8', errors='replace').decode('utf-8')}...")
        
        t0 = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # greedy最快
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - t0
        
        print(f"耗时: {elapsed:.1f}s")
        print(f"Response: {response[:200].encode('utf-8', errors='replace').decode('utf-8')}")
    
    print(f"\n✅ 测试完成！共 {len(samples)} 条，总耗时 {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
