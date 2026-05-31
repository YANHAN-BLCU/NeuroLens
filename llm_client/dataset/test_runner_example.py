"""
TestRunner 使用示例
"""

import sys
sys.path.insert(0, 'd:/Project_code/Neurolens')
sys.path.insert(0, 'd:/Project_code/Neurolens/llm_client')

from dataset import CSVLoader, TestRunner
from llm_client import create_client


def example_basic_usage():
    """基础用法示例"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    client = create_client(
        provider="qwen",
        api_key="sk-07645060ef8b48ca941ff5d7dfb369ea",
        model_name="qwen-turbo"
    )

    runner = TestRunner(model_client=client, use_safeguard=True)

    print(f"\nRunning test on {len(loader.dataset)} samples...")
    results = runner.run(loader, n=3, shuffle=False)

    print(f"\nGot {len(results)} results")

    print("\nFirst result structure:")
    import json
    print(json.dumps(results[0], ensure_ascii=False, indent=2))

    TestRunner.print_summary(results)


def example_with_analysis():
    """带结果分析的示例"""
    print("\n" + "=" * 60)
    print("Example 2: With Analysis")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    client = create_client(
        provider="qwen",
        api_key="sk-07645060ef8b48ca941ff5d7dfb369ea",
        model_name="qwen-turbo"
    )

    runner = TestRunner(model_client=client, use_safeguard=True)

    print("\nRunning test...")
    results = runner.run(loader, n=5, shuffle=True, seed=42)

    stats = runner.analyze_results(results)

    print("\nAnalysis:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Successful jailbreak: {stats['successful_jailbreak']}")
    print(f"  Refusal count: {stats['refusal_count']}")
    print(f"  Jailbreak rate: {stats['jailbreak_rate']:.2%}")


def example_filter_by_method():
    """按方法过滤的示例"""
    print("\n" + "=" * 60)
    print("Example 3: Filter by Method")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    client = create_client(
        provider="qwen",
        api_key="sk-07645060ef8b48ca941ff5d7dfb369ea",
        model_name="qwen-turbo"
    )

    runner = TestRunner(model_client=client, use_safeguard=True)

    print("\nTesting role_play method only...")
    results = runner.run(loader, n=3, method="role_play", shuffle=True, seed=42)

    TestRunner.print_summary(results)


def example_save_and_load():
    """保存和加载结果的示例"""
    print("\n" + "=" * 60)
    print("Example 4: Save and Load Results")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    client = create_client(
        provider="qwen",
        api_key="sk-07645060ef8b48ca941ff5d7dfb369ea",
        model_name="qwen-turbo"
    )

    runner = TestRunner(model_client=client, use_safeguard=True)

    print("\nRunning test...")
    results = runner.run(loader, n=2, shuffle=False)

    output_file = "d:/Project_code/Neurolens/test_results.json"
    TestRunner.save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")

    loaded = TestRunner.load_results(output_file)
    print(f"Loaded {len(loaded)} results from file")

    TestRunner.print_summary(loaded)


def example_output_structure():
    """输出结构详解"""
    print("\n" + "=" * 60)
    print("Example 5: Output Structure Explanation")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    client = create_client(
        provider="qwen",
        api_key="sk-07645060ef8b48ca941ff5d7dfb369ea",
        model_name="qwen-turbo"
    )

    runner = TestRunner(model_client=client, use_safeguard=True)

    print("\nRunning single sample test...")
    results = runner.run(loader, n=1, shuffle=False)

    print("\nResult structure:")
    result = results[0]

    print("\n  Index:", result["Index"])

    print("\n  Base Jailbreak:")
    print("    text:", result["Base Jailbreak"]["text"][:80], "...")
    print("    enhanced_text:", result["Base Jailbreak"].get("enhanced_text", "None"))

    print("\n  Method:")
    print("    name:", result["Method"]["name"])
    print("    category:", result["Method"]["category"])
    print("    description:", result["Method"]["description"])

    if result.get("Enhanced Jailbreak"):
        print("\n  Enhanced Jailbreak:")
        print("    text:", result["Enhanced Jailbreak"].get("text", ""))

    print("\n  Output:")
    print("    text:", result["Output"]["text"][:80], "...")
    print("    output_length:", result["Output"]["output_length"])
    print("    has_refusal:", result["Output"]["has_refusal"])
    print("    contains_sensitive:", result["Output"]["contains_sensitive"])
    print("    is_successful_jailbreak:", result["Output"]["is_successful_jailbreak"])
    print("    latency_ms:", result["Output"]["latency_ms"])
    print("    error:", result["Output"]["error"])


if __name__ == "__main__":
    print("=" * 60)
    print("TestRunner Examples")
    print("=" * 60)

    example_output_structure()
    example_basic_usage()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)