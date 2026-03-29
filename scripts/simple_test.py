#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from engine.models import ModelManager
import time

print('='*60)
print('NeuroBreak 模型加载测试')
print('='*60)

# 测试 1: Meta-Llama-3-8B-Instruct (推理模型)
print()
print('[1/2] 测试 Meta-Llama-3-8B-Instruct (推理模型)...')
try:
    start = time.time()
    manager = ModelManager()
    t, mo = manager.load_llm()  # 不需要参数
    elapsed = time.time() - start
    print(f'  设备: {next(mo.parameters()).device}')
    print(f'  加载时间: {elapsed:.1f}秒')
    print('  PASS')
except Exception as e:
    print(f'  FAIL: {e}')

# 测试 2: Llama-Guard-3-8B (安全审核模型)
print()
print('[2/2] 测试 Llama-Guard-3-8B (安全审核模型)...')
try:
    start = time.time()
    manager = ModelManager()
    t, mo = manager.load_guard()  # 不需要参数
    elapsed = time.time() - start
    print(f'  设备: {next(mo.parameters()).device}')
    print(f'  加载时间: {elapsed:.1f}秒')
    print('  PASS')
except Exception as e:
    print(f'  FAIL: {e}')

print()
print('='*60)
print('测试完成')
print('='*60)
