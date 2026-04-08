#!/usr/bin/env python3
"""
下载 Utility 评估所需的数据集

支持多种下载方式：
    1. HuggingFace 官方源（自动重试）
    2. HuggingFace 镜像源（hf-mirror.com）
    3. 单个数据集下载
    4. 检查已下载的数据

使用方法：
    python scripts/download_utility_datasets.py              # 自动下载全部
    python scripts/download_utility_datasets.py --check   # 仅检查已下载
    python scripts/download_utility_datasets.py --mirror   # 优先使用镜像
    python scripts/download_utility_datasets.py --single wikitext  # 仅下载单个

依赖：
    pip install datasets requests
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# HuggingFace 镜像源列表（按优先级排序）
HF_MIRRORS = [
    "https://hf-mirror.com",            # 镜像（通常更快）
    "https://huggingface.co",           # 官方源
]


def test_mirror(mirror_url: str, timeout: int = 5) -> bool:
    """测试镜像是否可用"""
    try:
        import requests
        response = requests.get(f"{mirror_url}/", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def find_working_mirror() -> str:
    """找到可用的镜像源"""
    print("正在测试 HuggingFace 镜像源...")
    for mirror in HF_MIRRORS:
        print(f"  测试: {mirror}...", end=" ")
        if test_mirror(mirror):
            print("✓ 可用")
            return mirror
        else:
            print("✗ 不可用")
    # 如果都不可用，返回官方源
    print("警告: 所有镜像源均不可用，使用官方源")
    return HF_MIRRORS[1]


def download_with_retry(
    load_fn,
    dataset_name: str,
    max_retries: int = 3,
    retry_delay: int = 10
) -> Tuple[bool, any]:
    """
    带重试和镜像切换的下载函数

    Args:
        load_fn: 下载函数
        dataset_name: 数据集名称
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）

    Returns:
        (是否成功, 数据集或None)
    """
    mirror_idx = 0
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}...")
            os.environ["HF_ENDPOINT"] = HF_MIRRORS[mirror_idx % len(HF_MIRRORS)]
            print(f"  使用源: {os.environ['HF_ENDPOINT']}")

            dataset = load_fn()
            print(f"  ✓ {dataset_name} 下载成功！")
            return True, dataset

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # 判断是否是网络错误
            is_network_error = any(keyword in error_str for keyword in [
                "connection", "timeout", "network", "ssl", "remote",
                "certificate", "status_code", "http"
            ])

            if is_network_error:
                print(f"  ✗ 网络错误: {e}")
                mirror_idx += 1  # 切换镜像
            else:
                print(f"  ✗ 错误: {e}")

            if attempt < max_retries - 1:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)

    print(f"  ✗ {dataset_name} 下载失败（已重试 {max_retries} 次）")
    if last_error:
        print(f"  最后错误: {last_error}")
    return False, None


def save_jsonl(dataset, output_path: Path, split: str = "validation"):
    """保存数据集为 JSONL 格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset[split]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def download_wikitext(use_mirror: bool = False) -> bool:
    """下载 WikiText-2 数据集"""
    print("\n" + "=" * 60)
    print("下载 WikiText-2 数据集")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "data" / "utility" / "wikitext" / "wikitext-2-raw"
    val_path = output_dir / "wiki.valid.raw"

    # 检查是否已存在
    if val_path.exists():
        print(f"  ✓ 已存在: {val_path}")
        return True

    def load_fn():
        from datasets import load_dataset
        return load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

    success, dataset = download_with_retry(load_fn, "WikiText-2")

    if not success:
        return False

    try:
        print(f"  - 训练集: {len(dataset['train'])} 条")
        print(f"  - 验证集: {len(dataset['validation'])} 条")
        print(f"  - 测试集: {len(dataset['test'])} 条")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存验证集（用于困惑度计算）
        with open(val_path, "w", encoding="utf-8") as f:
            for item in dataset["validation"]:
                f.write(item["text"] + "\n")
        print(f"  验证集已保存: {val_path}")

        # 保存训练集
        train_path = output_dir / "wiki.train.raw"
        with open(train_path, "w", encoding="utf-8") as f:
            for item in dataset["train"]:
                f.write(item["text"] + "\n")
        print(f"  训练集已保存: {train_path}")

        # 保存测试集
        test_path = output_dir / "wiki.test.raw"
        with open(test_path, "w", encoding="utf-8") as f:
            for item in dataset["test"]:
                f.write(item["text"] + "\n")
        print(f"  测试集已保存: {test_path}")

        return True

    except Exception as e:
        print(f"  保存数据失败: {e}")
        return False


def download_hellaswag(use_mirror: bool = False) -> bool:
    """下载 HellaSwag 数据集"""
    print("\n" + "=" * 60)
    print("下载 HellaSwag 数据集")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "data" / "utility" / "hellaswag"
    sample_path = output_dir / "validation.jsonl"

    if sample_path.exists():
        print(f"  ✓ 已存在: {sample_path}")
        return True

    def load_fn():
        from datasets import load_dataset
        return load_dataset("hellaswag", trust_remote_code=True)

    success, dataset = download_with_retry(load_fn, "HellaSwag")

    if not success:
        return False

    try:
        print(f"  - 训练集: {len(dataset['train'])} 条")
        print(f"  - 验证集: {len(dataset['validation'])} 条")
        print(f"  - 测试集: {len(dataset['test'])} 条")

        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "validation", "test"]:
            out_path = output_dir / f"{split}.jsonl"
            save_jsonl(dataset, out_path, split)
            print(f"  {split} 集已保存: {out_path}")

        return True

    except Exception as e:
        print(f"  保存数据失败: {e}")
        return False


def download_winogrande(use_mirror: bool = False) -> bool:
    """下载 WinoGrande 数据集"""
    print("\n" + "=" * 60)
    print("下载 WinoGrande 数据集")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "data" / "utility" / "winogrande"
    sample_path = output_dir / "validation.jsonl"

    if sample_path.exists():
        print(f"  ✓ 已存在: {sample_path}")
        return True

    def load_fn():
        from datasets import load_dataset
        return load_dataset("winogrande", "winogrande_xl", trust_remote_code=True)

    success, dataset = download_with_retry(load_fn, "WinoGrande")

    if not success:
        return False

    try:
        print(f"  - 训练集: {len(dataset['train'])} 条")
        print(f"  - 验证集: {len(dataset['validation'])} 条")
        print(f"  - 测试集: {len(dataset['test'])} 条")

        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "validation", "test"]:
            out_path = output_dir / f"{split}.jsonl"
            save_jsonl(dataset, out_path, split)
            print(f"  {split} 集已保存: {out_path}")

        return True

    except Exception as e:
        print(f"  保存数据失败: {e}")
        return False


def download_arc(use_mirror: bool = False) -> bool:
    """下载 ARC 数据集"""
    print("\n" + "=" * 60)
    print("下载 ARC 数据集")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "data" / "utility" / "arc"
    sample_path = output_dir / "arc_easy_train.jsonl"

    if sample_path.exists():
        print(f"  ✓ 已存在: {sample_path}")
        return True

    def load_fn():
        from datasets import load_dataset
        # ARC 数据集需要分别加载 ARC-Easy 和 ARC-Challenge
        return load_dataset("ai2_arc", "ARC-Easy"), load_dataset("ai2_arc", "ARC-Challenge")

    success, datasets = download_with_retry(load_fn, "ARC")

    if not success:
        return False

    try:
        dataset_easy, dataset_challenge = datasets

        print(f"  - ARC-Easy 训练集: {len(dataset_easy['train'])} 条")
        print(f"  - ARC-Easy 测试集: {len(dataset_easy['test'])} 条")
        print(f"  - ARC-Challenge 训练集: {len(dataset_challenge['train'])} 条")
        print(f"  - ARC-Challenge 测试集: {len(dataset_challenge['test'])} 条")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 ARC-Easy
        for split in ["train", "test"]:
            out_path = output_dir / f"arc_easy_{split}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in dataset_easy[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  ARC-Easy {split} 集已保存: {out_path}")

        # 保存 ARC-Challenge
        for split in ["train", "test"]:
            out_path = output_dir / f"arc_challenge_{split}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in dataset_challenge[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  ARC-Challenge {split} 集已保存: {out_path}")

        return True

    except Exception as e:
        print(f"  保存数据失败: {e}")
        return False


def download_openbookqa(use_mirror: bool = False) -> bool:
    """下载 OpenBookQA 数据集"""
    print("\n" + "=" * 60)
    print("下载 OpenBookQA 数据集")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "data" / "utility" / "openbookqa"
    sample_path = output_dir / "validation.jsonl"

    if sample_path.exists():
        print(f"  ✓ 已存在: {sample_path}")
        return True

    def load_fn():
        from datasets import load_dataset
        return load_dataset("openbookqa", trust_remote_code=True)

    success, dataset = download_with_retry(load_fn, "OpenBookQA")

    if not success:
        return False

    try:
        print(f"  - 训练集: {len(dataset['train'])} 条")
        print(f"  - 验证集: {len(dataset['validation'])} 条")
        print(f"  - 测试集: {len(dataset['test'])} 条")

        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "validation", "test"]:
            out_path = output_dir / f"{split}.jsonl"
            save_jsonl(dataset, out_path, split)
            print(f"  {split} 集已保存: {out_path}")

        return True

    except Exception as e:
        print(f"  保存数据失败: {e}")
        return False


def download_super_glue(use_mirror: bool = False) -> bool:
    """下载 SuperGLUE 数据集 (BoolQ, RTE)"""
    print("\n" + "=" * 60)
    print("下载 SuperGLUE 数据集 (BoolQ, RTE)")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "data" / "utility" / "super_glue"
    sample_path = output_dir / "boolq" / "validation.jsonl"

    if sample_path.exists():
        print(f"  ✓ 已存在: {sample_path}")
        return True

    all_success = True

    for task in ["boolq", "rte"]:
        print(f"\n  下载 {task.upper()}...")

        task_dir = output_dir / task
        task_sample = task_dir / "validation.jsonl"

        if task_sample.exists():
            print(f"    ✓ 已存在: {task_sample}")
            continue

        def load_fn():
            from datasets import load_dataset
            return load_dataset("super_glue", task, trust_remote_code=True)

        success, dataset = download_with_retry(load_fn, task.upper())

        if not success:
            all_success = False
            continue

        try:
            print(f"    - 训练集: {len(dataset['train'])} 条")
            print(f"    - 验证集: {len(dataset['validation'])} 条")
            print(f"    - 测试集: {len(dataset['test'])} 条")

            task_dir.mkdir(parents=True, exist_ok=True)

            for split in ["train", "validation", "test"]:
                out_path = task_dir / f"{split}.jsonl"
                save_jsonl(dataset, out_path, split)
                print(f"    {split} 集已保存: {out_path}")

        except Exception as e:
            print(f"    保存 {task} 数据失败: {e}")
            all_success = False

    return all_success


def check_existing_datasets() -> Dict[str, bool]:
    """检查已下载的数据集"""
    print("=" * 60)
    print("检查已下载的数据集")
    print("=" * 60)

    checks = {
        "WikiText-2": PROJECT_ROOT / "data" / "utility" / "wikitext" / "wikitext-2-raw" / "wiki.valid.raw",
        "HellaSwag": PROJECT_ROOT / "data" / "utility" / "hellaswag" / "validation.jsonl",
        "WinoGrande": PROJECT_ROOT / "data" / "utility" / "winogrande" / "validation.jsonl",
        "ARC": PROJECT_ROOT / "data" / "utility" / "arc" / "arc_easy_train.jsonl",
        "OpenBookQA": PROJECT_ROOT / "data" / "utility" / "openbookqa" / "validation.jsonl",
        "BoolQ": PROJECT_ROOT / "data" / "utility" / "super_glue" / "boolq" / "validation.jsonl",
        "RTE": PROJECT_ROOT / "data" / "utility" / "super_glue" / "rte" / "validation.jsonl",
    }

    results = {}
    for name, path in checks.items():
        exists = path.exists()
        results[name] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")

    return results


def check_and_install_packages():
    """检查并安装必要的包"""
    print("=" * 60)
    print("检查依赖")
    print("=" * 60)

    missing = []
    suggestions = []

    # 检查 datasets
    try:
        import datasets
        print(f"✓ datasets: {datasets.__version__}")
    except ImportError:
        print("✗ datasets: 未安装")
        missing.append("datasets")
        suggestions.append("pip install datasets")

    # 检查 requests（用于测试镜像）
    try:
        import requests
        print(f"✓ requests: {requests.__version__}")
    except ImportError:
        print("✗ requests: 未安装（建议安装以测试镜像）")
        suggestions.append("pip install requests")

    # 检查 torch
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
    except ImportError:
        print("✗ torch: 未安装")
        missing.append("torch")
        suggestions.append("pip install torch")

    # 检查 transformers
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers: 未安装")
        missing.append("transformers")
        suggestions.append("pip install transformers")

    if missing:
        print(f"\n请安装缺失的包:")
        for s in suggestions:
            print(f"  {s}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="下载 Utility 评估所需的数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_utility_datasets.py              # 下载全部
  python scripts/download_utility_datasets.py --check       # 仅检查已下载
  python scripts/download_utility_datasets.py --mirror     # 优先使用镜像
  python scripts/download_utility_datasets.py --single wikitext  # 下载单个数据集

可用数据集:
  wikitext, hellaswag, winogrande, arc, openbookqa, superglue, all
        """
    )

    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="仅检查已下载的数据集，不下载"
    )

    parser.add_argument(
        "--mirror", "-m",
        action="store_true",
        help="优先使用镜像源（hf-mirror.com）"
    )

    parser.add_argument(
        "--single", "-s",
        type=str,
        choices=["wikitext", "hellaswag", "winogrande", "arc", "openbookqa", "superglue", "all"],
        default="all",
        help="下载单个数据集（默认：all）"
    )

    parser.add_argument(
        "--retry", "-r",
        type=int,
        default=3,
        help="最大重试次数（默认：3）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Utility 评估数据集下载脚本")
    print("=" * 60)
    print()

    # 如果使用镜像，设置环境变量
    if args.mirror:
        working_mirror = find_working_mirror()
        os.environ["HF_ENDPOINT"] = working_mirror
        print(f"已设置镜像: {working_mirror}\n")

    # 检查依赖
    if not check_and_install_packages():
        sys.exit(1)

    # 仅检查模式
    if args.check:
        results = check_existing_datasets()
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        print(f"\n已下载: {success_count}/{total_count}")
        sys.exit(0)

    print("=" * 60)
    print("开始下载数据集")
    print("=" * 60)

    # 下载映射
    downloaders = {
        "wikitext": download_wikitext,
        "hellaswag": download_hellaswag,
        "winogrande": download_winogrande,
        "arc": download_arc,
        "openbookqa": download_openbookqa,
        "superglue": download_super_glue,
    }

    # 下载单个或全部
    if args.single == "all":
        targets = list(downloaders.keys())
    else:
        targets = [args.single]

    results = {}
    for target in targets:
        if target in downloaders:
            results[target] = downloaders[target](use_mirror=args.mirror)

    # 总结
    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"\n成功: {success_count}/{total_count}")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    # 显示数据集保存位置
    print("\n" + "=" * 60)
    print("数据集保存位置")
    print("=" * 60)
    print(f"  data/utility/wikitext/wikitext-2-raw/  - WikiText-2")
    print(f"  data/utility/hellaswag/                 - HellaSwag")
    print(f"  data/utility/winogrande/                 - WinoGrande")
    print(f"  data/utility/arc/                        - ARC")
    print(f"  data/utility/openbookqa/                 - OpenBookQA")
    print(f"  data/utility/super_glue/               - SuperGLUE (BoolQ, RTE)")
    print()

    if success_count == total_count:
        print("所有数据集下载成功！现在可以运行 Utility 评估：")
        print("  python scripts/run_evaluate_utility.py --model <model_path>")
    else:
        print("部分数据集下载失败。")
        print("可以尝试以下方法：")
        print("  1. 使用镜像源: python scripts/download_utility_datasets.py --mirror")
        print("  2. 检查网络连接后重试")
        print("  3. 单独下载失败的数据集: python scripts/download_utility_datasets.py --single <dataset_name>")


if __name__ == "__main__":
    main()
