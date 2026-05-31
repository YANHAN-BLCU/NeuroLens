"""
CSV 数据集测试
"""

import sys
sys.path.insert(0, 'd:/Project_code/Neurolens')
sys.path.insert(0, 'd:/Project_code/Neurolens/llm_client')

from dataset import CSVLoader


def test_jailbench():
    """测试 JailBench.csv"""
    print("=" * 60)
    print("Test: JailBench.csv")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    dataset = loader.load()

    print(f"\nDataset: {dataset.name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Methods: {dataset.get_methods()}")

    stats = loader.stats()
    print(f"\nStatistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Categories: {list(stats.get('category_counts', {}).keys())}")
    print(f"  Methods: {stats['methods']}")

    print("\nSample unified output (first 3):")
    for item in loader.get_unified_output(n=3, shuffle=False):
        print(f"\n  ID: {item['id']}")
        print(f"  Method: {item['method']}")
        print(f"  Category1: {item['category1']}")
        print(f"  Category2: {item['category2']}")
        print(f"  Prompt: {item['prompt'][:80]}...")

    print("\n[PASS] JailBench test")


def test_jailbench_seed():
    """测试 JailBench-seed.csv"""
    print("\n" + "=" * 60)
    print("Test: JailBench-seed.csv")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench-seed.csv")
    dataset = loader.load(prompt_column="seed", is_seed_file=True)

    print(f"\nDataset: {dataset.name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Methods: {dataset.get_methods()}")

    stats = loader.stats()
    print(f"\nStatistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Methods: {stats['methods']}")

    print("\nSample unified output (first 3):")
    for item in loader.get_unified_output(n=3, shuffle=False):
        print(f"\n  ID: {item['id']}")
        print(f"  Method: {item['method']}")
        print(f"  Category1: {item['category1']}")
        print(f"  Prompt: {item['prompt'][:80]}...")

    print("\n[PASS] JailBench-seed test")


def test_random_sampling():
    """测试随机抽样"""
    print("\n" + "=" * 60)
    print("Test: Random Sampling")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    print("\nRandom sample 5 items (seed=42):")
    samples = loader.get_samples(n=5, shuffle=True, seed=42)
    for s in samples:
        print(f"  [{s.index}] {s.method.name}: {s.get_prompt()[:50]}...")

    print("\nSame seed, should get same result:")
    samples = loader.get_samples(n=5, shuffle=True, seed=42)
    for s in samples:
        print(f"  [{s.index}] {s.method.name}")

    print("\n[PASS] Random sampling test")


def test_filter_by_method():
    """测试按方法过滤"""
    print("\n" + "=" * 60)
    print("Test: Filter by Method")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    method = "role_play"
    samples = loader.get_samples(method=method)
    print(f"\nFound {len(samples)} samples with method='{method}'")

    if samples:
        s = samples[0]
        print(f"\nFirst sample:")
        print(f"  Method: {s.method.name}")
        print(f"  Category1: {s.method.category}")
        print(f"  Prompt: {s.get_prompt()[:60]}...")

    print("\n[PASS] Filter by method test")


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 60)
    print("Test: Batch Processing")
    print("=" * 60)

    loader = CSVLoader("d:/Project_code/Neurolens/JailBench.csv")
    loader.load()

    print("\nBatch processing (batch_size=10, total 20):")
    for batch_idx, batch in enumerate(loader.get_batch_output(batch_size=10, shuffle=False)):
        print(f"\n  Batch {batch_idx + 1}: {len(batch)} items")
        for item in batch[:3]:
            print(f"    - ID {item['id']}, {item['method']}")

    print("\n[PASS] Batch processing test")


if __name__ == "__main__":
    print("=" * 60)
    print("CSV Dataset Tests")
    print("=" * 60)

    test_jailbench()
    test_jailbench_seed()
    test_random_sampling()
    test_filter_by_method()
    test_batch_processing()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)