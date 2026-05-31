"""
LLM 安全评测框架 - 完整测试示例

该脚本展示如何使用框架进行越狱攻击测试

用法:
    python run_security_test.py --provider qwen --api-key YOUR_KEY --model qwen-turbo --samples 20
"""

import argparse
import os
from llm_client import create_client, LLMClientFactory
from llm_client.dataset import (
    CSVLoader, TestRunner, evaluate,
    ResultFormatter, apply_safeguard
)
import time


def run_security_test(provider: str, api_key: str, model_name: str, n_samples: int = 20):
    """运行安全评测的完整流程"""

    print("=" * 60)
    print("LLM 安全评测框架 - 越狱攻击测试")
    print(f"模型: {model_name} | 样本数: {n_samples}")
    print("=" * 60)

    # ========================================
    # 第1步：加载数据集
    # ========================================
    print("\n[1] 加载数据集...")
    
    # 数据集路径
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "JailBench", "JailBench.csv")
    loader = CSVLoader(dataset_path)
    dataset = loader.load()

    # 取指定数量样本
    test_samples = dataset.samples[:n_samples]
    print(f"    已加载 {len(dataset.samples)} 条样本，测试使用 {len(test_samples)} 条")

    # ========================================
    # 第2步：创建 LLM 客户端
    # ========================================
    print("\n[2] 创建 LLM 客户端...")

    # 创建真实 LLM 客户端
    client = create_client(provider=provider, api_key=api_key, model_name=model_name)
    print(f"    使用 {provider} 客户端进行测试")

    # ========================================
    # 第3步：创建测试运行器
    # ========================================
    print("\n[3] 创建测试运行器...")

    runner = TestRunner(
        model_client=client,
        use_safeguard=True,         # 是否使用 safeguard
        max_workers=1,              # 并发数
        max_retries=2               # 失败重试次数
    )
    print(f"    运行器已就绪 (并发: {runner.max_workers}, 重试: {runner.max_retries})")

    # ========================================
    # 第4步：运行测试
    # ========================================
    print("\n[4] 开始测试...")

    start_time = time.time()

    try:
        results = runner.run(dataset, n=n_samples, shuffle=False)
    except Exception as e:
        print(f"\n    测试出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - start_time

    print(f"    测试完成! 耗时: {elapsed:.1f}秒")
    print(f"    结果数量: {len(results)}")

    # ========================================
    # 第5步：评估结果
    # ========================================
    print("\n[5] 评估结果...")

    evaluations = []
    for result in results:
        output = result.get("Output", {})
        if output and output.get("text"):
            # 使用 Evaluator 评估
            eval_result = evaluate(output["text"])
            eval_result["Index"] = result.get("Index", 0)
            evaluations.append(eval_result)
        else:
            evaluations.append({
                "Index": result.get("Index", 0),
                "Jailbroken": 0,
                "Jailbroken_Label": "Safe",
                "Score": {"risk_score": 0.0, "risk_level": "LOW"}
            })

    print(f"    已评估 {len(evaluations)} 条结果")

    # ========================================
    # 第6步：格式化输出
    # ========================================
    print("\n[6] 格式化输出...")

    # 转换为 ResultFormatter 格式
    formatted_results = []
    for result, eval_result in zip(results, evaluations):
        base_jailbreak = result.get("Base Jailbreak", {})
        method = result.get("Method", {})
        enhanced = result.get("Enhanced Jailbreak", {})
        output = result.get("Output", {})

        formatted = {
            "Index": result.get("Index", 0),
            "Base Jailbreak": {"text": base_jailbreak.get("content", "")[:100] + "..." if base_jailbreak.get("content") else ""},
            "Method": {"name": method.get("name", ""), "category": method.get("category", "")},
            "Enhanced Jailbreak": {"text": enhanced.get("content", "")[:100] + "..." if enhanced.get("content") else ""},
            "Output": {
                "text": output.get("text", "")[:100] + "..." if output.get("text") else "",
                "output_length": output.get("output_length", 0),
                "has_refusal": output.get("has_refusal", False),
                "contains_sensitive": output.get("contains_sensitive", False),
                "is_successful_jailbreak": output.get("is_successful_jailbreak", False)
            },
            "Jailbroken": eval_result["Jailbroken"],
            "Jailbroken_Label": eval_result["Jailbroken_Label"],
            "Score": eval_result["Score"]
        }
        formatted_results.append(formatted)

    # ========================================
    # 第7步：保存结果
    # ========================================
    print("\n[7] 保存结果...")

    ResultFormatter.save_json(formatted_results, "security_test_results.json")
    ResultFormatter.save_csv(formatted_results, "security_test_results.csv")

    print("    JSON 保存至: security_test_results.json")
    print("    CSV 保存至: security_test_results.csv")

    # ========================================
    # 第8步：打印摘要
    # ========================================
    print("\n[8] 结果摘要:")
    print("-" * 60)
    ResultFormatter.print_summary(formatted_results)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return formatted_results


def show_supported_providers():
    """显示支持的 LLM 厂商"""
    print("\n支持的 LLM 厂商:")
    for name in LLMClientFactory.list_providers():
        info = LLMClientFactory.get_provider_info(name)
        print(f"  - {name}: {info}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LLM 安全评测框架")
    parser.add_argument("--provider", type=str, default="qwen", help="LLM 提供商")
    parser.add_argument("--api-key", type=str, default="", help="API Key")
    parser.add_argument("--model", type=str, default="qwen-turbo", help="模型名称")
    parser.add_argument("--samples", type=int, default=20, help="测试样本数量")
    
    args = parser.parse_args()

    # 显示支持的厂商
    show_supported_providers()

    print("\n" + "=" * 60)
    print("开始运行安全测试...")
    print("=" * 60)

    # 运行测试
    run_security_test(
        provider=args.provider,
        api_key=args.api_key,
        model_name=args.model,
        n_samples=args.samples
    )
