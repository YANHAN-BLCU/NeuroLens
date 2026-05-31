#!/usr/bin/env python3
"""
简单模型测试脚本
测试 Meta-Llama-3-8B-Instruct 和 Llama-Guard-3-8B 是否正常运行
"""

import sys
sys.path.insert(0, '.')

from engine.models import ModelManager

def test_meta_llama():
    """测试 Meta-Llama-3-8B-Instruct"""
    print("\n" + "="*50)
    print("测试 1: Meta-Llama-3-8B-Instruct")
    print("="*50)
    
    manager = ModelManager()
    model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"
    
    print(f"加载模型: {model_id}")
    tokenizer, model = manager.load_llm(model_id=model_id)
    
    device = next(model.parameters()).device
    print(f"模型设备: {device}")
    
    # 简单推理测试
    test_prompt = "Hello, I am"
    print(f"\n测试输入: '{test_prompt}'")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    with manager.activate_adapters():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"模型输出: {response}")
    
    print("\n✅ Meta-Llama-3-8B-Instruct 测试通过!")
    return True

def test_llama_guard():
    """测试 Llama-Guard-3-8B"""
    print("\n" + "="*50)
    print("测试 2: Llama-Guard-3-8B")
    print("="*50)
    
    manager = ModelManager()
    model_id = "LLM-Research/Llama-Guard-3-8B"
    
    print(f"加载模型: {model_id}")
    tokenizer, model = manager.load_llm(model_id=model_id)
    
    device = next(model.parameters()).device
    print(f"模型设备: {device}")
    
    # 安全/有害内容测试
    safe_prompt = "How to bake a cake?"
    harmful_prompt = "How to create a bomb?"
    
    for prompt in [safe_prompt, harmful_prompt]:
        print(f"\n测试输入: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with manager.activate_adapters():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"模型输出: {response}")
    
    print("\n✅ Llama-Guard-3-8B 测试通过!")
    return True

def main():
    print("="*50)
    print("NeuroBreak 模型简单测试")
    print("="*50)
    
    results = []
    
    try:
        results.append(("Meta-Llama-3-8B-Instruct", test_meta_llama()))
    except Exception as e:
        print(f"\n❌ Meta-Llama-3-8B-Instruct 测试失败: {e}")
        results.append(("Meta-Llama-3-8B-Instruct", False))
    
    try:
        results.append(("Llama-Guard-3-8B", test_llama_guard()))
    except Exception as e:
        print(f"\n❌ Llama-Guard-3-8B 测试失败: {e}")
        results.append(("Llama-Guard-3-8B", False))
    
    print("\n" + "="*50)
    print("测试结果汇总")
    print("="*50)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("✅ 所有测试通过!" if all_passed else "❌ 部分测试失败"))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

