import argparse
import os
import math
import torch

from engine.neurons.snip_scorer import select_top_percent_neurons


def parse_args():
    parser = argparse.ArgumentParser(
        description="从离线 SNIP 结果中按 p / q 选取前百分比神经元"
    )

    parser.add_argument(
        "--utility-file",
        type=str,
        default="outputs/utility_snip_annotated.pt",
        help="效用 SNIP annotated 文件路径",
    )
    parser.add_argument(
        "--safety-file",
        type=str,
        default="outputs/safety_snip_annotated.pt",
        help="安全 SNIP annotated 文件路径",
    )
    parser.add_argument(
        "-p",
        "--utility-top-percent",
        type=float,
        required=True,
        help="效用神经元前 top 百分比（小数形式，如 0.005 表示 0.5%%）",
    )
    parser.add_argument(
        "-q",
        "--safety-top-percent",
        type=float,
        required=True,
        help="安全神经元前 top 百分比（小数形式，如 0.010 表示 1%%）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="保存 U(p), S(q) 的输出目录",
    )

    return parser.parse_args()


def float_to_tag(x: float) -> str:
    """
    把 0.005 这样的浮点数变成字符串标签 0p005，方便写到文件名中。
    """
    # 保留 4 位小数
    s = f"{x:.4f}"
    # 例如 0.0050 -> 0p0050
    return s.replace(".", "p")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[LOAD] 从 {args.utility_file} 载入效用 SNIP annotated ...")
    utility_annotated = torch.load(args.utility_file, map_location="cpu")

    print(f"[LOAD] 从 {args.safety_file} 载入安全 SNIP annotated ...")
    safety_annotated = torch.load(args.safety_file, map_location="cpu")

    p = args.utility_top_percent
    q = args.safety_top_percent

    print(
        f"[SELECT] 选择效用前 p={p*100:.3f}% 与安全前 q={q*100:.3f}% 的神经元..."
    )

    U_p = select_top_percent_neurons(utility_annotated, top_percent=p)
    S_q = select_top_percent_neurons(safety_annotated, top_percent=q)

    print(
        f"[RESULT] 总效用神经元: {len(utility_annotated)}, "
        f"U(p) 大小: {len(U_p)} (p={p*100:.3f}%)"
    )
    print(
        f"[RESULT] 总安全神经元: {len(safety_annotated)}, "
        f"S(q) 大小: {len(S_q)} (q={q*100:.3f}%)"
    )

    p_tag = float_to_tag(p)
    q_tag = float_to_tag(q)

    U_path = os.path.join(args.output_dir, f"utility_U_p_{p_tag}.pt")
    S_path = os.path.join(args.output_dir, f"safety_S_q_{q_tag}.pt")

    torch.save(U_p, U_path)
    torch.save(S_q, S_path)

    print(f"[SAVE] 已保存 U(p) 到 {U_path}")
    print(f"[SAVE] 已保存 S(q) 到 {S_path}")


if __name__ == "__main__":
    main()