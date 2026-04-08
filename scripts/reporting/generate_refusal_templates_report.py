"""
Generate a Markdown report from a refusal template extraction JSON.

Input JSON schema (see outputs/tmp_refusal_templates/refusal_templates.json):
  - refusal_templates: List[str]
  - count: int
  - template_counts: Dict[str, int]  (optional)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _coverage(counter: Counter[str], total: int, k: int) -> float:
    if total <= 0 or k <= 0:
        return 0.0
    return sum(v for _, v in counter.most_common(k)) / total


def _quantile(values_desc: List[int], q: float) -> int | None:
    if not values_desc:
        return None
    q = max(0.0, min(1.0, q))
    idx = int(math.ceil(q * len(values_desc))) - 1
    idx = max(0, min(idx, len(values_desc) - 1))
    return values_desc[idx]


def _detect_quality_flags(templates: List[str]) -> Dict[str, List[str]]:
    # Lightweight heuristics to surface suspicious templates (usually truncation artifacts)
    # without being overly opinionated.
    truncated_suffixes = (
        " or.",
        " illegal or harmful.",
        " illegal or.",
        " illegal.",
        " on.",
        " or harmful.",
        " or harmful",
        " or",
        " or,",
    )
    truncated = [t for t in templates if t.strip().endswith(truncated_suffixes)]
    lowercase_start = [t for t in templates if t and t[0].islower()]
    return {
        "potentially_truncated": truncated,
        "lowercase_start": lowercase_start,
    }


def build_report(
    *,
    input_path: Path,
    templates: List[str],
    template_counts: Dict[str, int],
    top_k: int,
) -> str:
    c = Counter(template_counts)
    total_all = sum(c.values())
    values_desc = sorted(c.values(), reverse=True)
    template_set = set(templates)
    count_key_set = set(c.keys())
    only_in_templates = sorted(template_set - count_key_set)
    only_in_counts = sorted(count_key_set - template_set)
    total_in_list = sum(c[t] for t in templates if t in c)

    buckets = {
        ">=1000": sum(1 for v in values_desc if v >= 1000),
        "100-999": sum(1 for v in values_desc if 100 <= v <= 999),
        "10-99": sum(1 for v in values_desc if 10 <= v <= 99),
        "2-9": sum(1 for v in values_desc if 2 <= v <= 9),
        "==1": sum(1 for v in values_desc if v == 1),
    }

    flags = _detect_quality_flags(templates)

    lines: List[str] = []
    lines.append("# Refusal Templates Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Source JSON**: `{input_path.as_posix()}`")
    lines.append(f"- **Templates listed (`refusal_templates`)**: **{len(templates)}**")
    lines.append(f"- **Templates with counts (`template_counts`)**: **{len(c)}**")
    lines.append(f"- **Total extracted occurrences (sum of all `template_counts`)**: **{total_all}**")
    lines.append(f"- **Total occurrences restricted to listed templates**: **{total_in_list}**")
    if only_in_templates or only_in_counts:
        lines.append(f"- **Keys mismatch**: templates-only={len(only_in_templates)}, counts-only={len(only_in_counts)}")
    lines.append("")

    lines.append("## Coverage (by frequency)")
    lines.append("| Top-K | Coverage |")
    lines.append("|---:|---:|")
    for k in [1, 5, 10, 20, 50, 100]:
        lines.append(f"| {k} | {_coverage(c, total_all, k):.2%} |")
    lines.append("")

    lines.append("## Frequency distribution")
    lines.append("| Frequency | #templates |")
    lines.append("|---:|---:|")
    for k in [">=1000", "100-999", "10-99", "2-9", "==1"]:
        lines.append(f"| {k} | {buckets[k]} |")
    lines.append("")

    lines.append("## Frequency quantiles")
    lines.append("| Quantile | Count |")
    lines.append("|---:|---:|")
    for q in [0.50, 0.90, 0.95]:
        v = _quantile(values_desc, q)
        lines.append(f"| {q:.2f} | {v if v is not None else ''} |")
    lines.append("")

    lines.append(f"## Top {top_k} templates")
    lines.append("| Rank | Count | Share | Template |")
    lines.append("|---:|---:|---:|---|")
    for i, (t, v) in enumerate(c.most_common(top_k), 1):
        share = (v / total_all * 100.0) if total_all else 0.0
        safe = t.replace("\n", " ").strip()
        lines.append(f"| {i} | {v} | {share:.2f}% | {safe} |")
    lines.append("")

    if only_in_templates or only_in_counts:
        lines.append("## Count/list mismatches")
        lines.append(
            "- This can happen if you dedupe templates (e.g., case-insensitive) when building the list, "
            "but keep raw counts without the same dedupe logic."
        )
        lines.append(f"- **templates-only (missing in `template_counts`)**: **{len(only_in_templates)}**")
        if only_in_templates:
            for t in only_in_templates[:10]:
                lines.append(f"  - `{t}`")
        lines.append(f"- **counts-only (missing in `refusal_templates`)**: **{len(only_in_counts)}**")
        if only_in_counts:
            for t in only_in_counts[:10]:
                lines.append(f"  - `{t}`")
        lines.append("")

    lines.append("## Quality flags (heuristics)")
    lines.append(f"- **Potentially truncated / incomplete endings**: **{len(flags['potentially_truncated'])}**")
    if flags["potentially_truncated"]:
        lines.append("  - Examples:")
        for t in flags["potentially_truncated"][:10]:
            lines.append(f"    - `{t}`")
    lines.append(f"- **Templates starting with lowercase**: **{len(flags['lowercase_start'])}**")
    if flags["lowercase_start"]:
        lines.append("  - Examples:")
        for t in flags["lowercase_start"][:10]:
            lines.append(f"    - `{t}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("- Counts come from `template_counts` (typically frequency >= a chosen threshold during extraction).")
    lines.append("- If you want a cleaner refusal “skeleton”, consider filtering or tightening incomplete-template detection during extraction.")
    lines.append("")

    return "\n".join(lines)


def load_input(input_path: Path) -> Tuple[List[str], Dict[str, int]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    templates = data.get("refusal_templates") or []
    if not isinstance(templates, list):
        raise ValueError("Expected `refusal_templates` to be a list")

    template_counts = data.get("template_counts") or {}
    if template_counts and not isinstance(template_counts, dict):
        raise ValueError("Expected `template_counts` to be a dict if present")

    # If counts are missing, default to counting each listed template once.
    if not template_counts:
        template_counts = {t: 1 for t in templates if isinstance(t, str)}

    # Normalize: only keep str->int pairs with int>=0
    normalized: Dict[str, int] = {}
    for k, v in template_counts.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, int):
            try:
                v = int(v)
            except Exception:
                continue
        if v < 0:
            continue
        normalized[k] = v

    templates = [t for t in templates if isinstance(t, str)]
    return templates, normalized


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Markdown report from refusal_templates.json")
    ap.add_argument(
        "--input",
        default="outputs/tmp_refusal_templates/refusal_templates.json",
        help="Path to refusal_templates.json",
    )
    ap.add_argument(
        "--output",
        default="outputs/tmp_refusal_templates/refusal_templates_report.md",
        help="Output Markdown path",
    )
    ap.add_argument("--top-k", type=int, default=30, help="Show Top-K templates table")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    templates, template_counts = load_input(input_path)
    report = build_report(
        input_path=input_path,
        templates=templates,
        template_counts=template_counts,
        top_k=max(1, args.top_k),
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

