import argparse
import os
import torch

from engine.neurons.snip_scorer import (
    compute_snip_scores,
    rank_and_annotate_snip_scores,
)


# ===== 这里根据你自己项目实际情况修改 =====

def load_utility_model_and_dataset():
    """
    从你现有的 run_utility_identifier 中复用加载逻辑。
    建议在 scripts/run_utility_identifier.py 中写一个
    load_model_and_datasets() 然后这里直接 import 使用。
    """
    from scripts.run_utility_identifier import load_model_and_datasets
    return load_model_and_datasets()


def load_safety_model_and_dataset():
    """
    同理，从 run_safety_identifier 中复用加载逻辑。
    """
    from scripts.run_safety_identifier import load_model_and_datasets
    return load_model_and_datasets()


# =========================================


def compute_and_save(
    mode: str,
    batch_size: int,
    num_samples: int | None,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    if mode in ("utility", "both"):
        print("[SNIP] 计算效用 SNIP 分数...")
        model, tokenizer, utility_dataset, device, loss_fn = (
            load_utility_model_and_dataset()
        )

        utility_scores = compute_snip_scores(
            model=model,
            tokenizer=tokenizer,
            dataset=utility_dataset,
            device=device,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_samples=num_samples,
        )

        utility_annotated = rank_and_annotate_snip_scores(utility_scores)
        utility_path = os.path.join(output_dir, "utility_snip_annotated.pt")
        torch.save(utility_annotated, utility_path)
        print(
            f"[SNIP] 已保存效用 SNIP 到 {utility_path}，"
            f"总神经元数: {len(utility_annotated)}"
        )

    if mode in ("safety", "both"):
        print("[SNIP] 计算安全 SNIP 分数...")
        model, tokenizer, benign_dataset, device, loss_fn = (
            load_safety_model_and_dataset()
        )

        safety_scores = compute_snip_scores(
            model=model,
            tokenizer=tokenizer,
            dataset=benign_dataset,
            device=device,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_samples=num_samples,
        )

        safety_annotated = rank_and_annotate_snip_scores(safety_scores)
        safety_path = os.path.join(output_dir, "safety_snip_annotated.pt")
        torch.save(safety_annotated, safety_path)
        print(
            f"[SNIP] 已保存安全 SNIP 到 {safety_path}，"
            f"总神经元数: {len(safety_annotated)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="一次性计算 SNIP 分数并离线保存（效用 / 安全）"
    )
    parser.add_argument(
        "--mode",
        choices=["utility", "safety", "both"],
        default="both",
        help="计算哪一种 SNIP（utility / safety / both）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="SNIP 计算时的 batch size（按显存调整）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="使用多少个样本计算 SNIP，默认 None 表示用全部数据",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="保存 annotated 结果的目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compute_and_save(
        mode=args.mode,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()