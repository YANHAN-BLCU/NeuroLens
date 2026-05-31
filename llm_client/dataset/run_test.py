"""
TestRunner 快速测试脚本
"""

import sys
sys.path.insert(0, 'd:/Project_code/Neurolens')
sys.path.insert(0, 'd:/Project_code/Neurolens/llm_client')

from dataset import CSVLoader, TestRunner
from llm_client import create_client
import json

def main():
    # 创建客户端
    print("Creating LLM client...")
    client = create_client(
        provider='qwen',
        api_key='sk-07645060ef8b48ca941ff5d7dfb369ea',
        model_name='qwen-turbo'
    )

    # 加载数据集
    print("Loading dataset...")
    loader = CSVLoader('d:/Project_code/Neurolens/JailBench.csv')
    loader.load()
    print(f"Loaded {len(loader.dataset)} samples")

    # 创建测试运行器
    runner = TestRunner(
        model_client=client,
        use_safeguard=True,
        max_retries=3,
        retry_delay=2.0
    )

    # 运行测试（只测2条）
    print("\nRunning test on 2 samples...")
    results = runner.run(loader, n=2, shuffle=False)

    # 打印结果
    print('\n' + '='*60)
    print('Results:')
    print('='*60)
    for r in results:
        print(f"\nIndex: {r['Index']}")
        print(f"Method: {r['Method']['name']}")
        print(f"Prompt: {r['Base Jailbreak']['text'][:60]}...")
        print(f"Response: {r['Output']['text'][:100] if r['Output']['text'] else 'No response'}")
        print(f"Has Refusal: {r['Output']['has_refusal']}")
        print(f"Contains Sensitive: {r['Output']['contains_sensitive']}")
        print(f"Successful Jailbreak: {r['Output']['is_successful_jailbreak']}")
        if r['Output']['error']:
            print(f"Error: {r['Output']['error']}")

    # 打印摘要
    TestRunner.print_summary(results)

    # 保存结果
    output_file = 'd:/Project_code/Neurolens/test_runner_results.json'
    TestRunner.save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
