#!/usr/bin/env python3
"""
Mine common refusal phrasing from an evaluation JSONL log.

This is a small utility for inspecting `logs/base_evaluation.jsonl`-style files.
It is intentionally robust on Windows terminals (smart quotes, NBSP, encoding).

Examples:
  python scripts/mine_refusal_phrases.py --log logs/base_evaluation.jsonl
  python scripts/mine_refusal_phrases.py --log logs/base_evaluation.jsonl --verdict allow --top-k 30
  # Only count outputs that start with common refusal cues (filters out MCQ-style outputs)
  python scripts/mine_refusal_phrases.py --log logs/base_evaluation.jsonl --verdict allow --startswith-cue
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    trans = str.maketrans(
        {
            0x2019: 0x27,  # ’ -> '
            0x2018: 0x27,  # ‘ -> '
            0x201C: 0x22,  # “ -> "
            0x201D: 0x22,  # ” -> "
            0x00A0: 0x20,  # NBSP -> space
        }
    )
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _iter_jsonl(path: Path, max_lines: Optional[int] = None) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if max_lines and i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    # Make stdout tolerant on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="Path to evaluation JSONL log")
    ap.add_argument("--verdict", default="allow", help="Guard verdict to filter (default: allow)")
    ap.add_argument("--max-lines", type=int, default=None, help="Only read first N lines")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K prefixes to print")
    ap.add_argument("--prefix-chars", type=int, default=100, help="Prefix length (chars) to count")
    ap.add_argument(
        "--startswith-cue",
        action="store_true",
        help="Only include samples whose normalized output starts with a known refusal cue",
    )
    ap.add_argument(
        "--exclude-start-regex",
        action="append",
        default=[],
        help="Regex (repeatable) to exclude samples whose normalized output matches at start (case-insensitive)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save refusal templates JSON",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[mine_refusal_phrases] Log not found: {log_path}")
        return 2

    verdict_target = (args.verdict or "").lower()

    prefixes = Counter()
    cues = Counter()
    allow_with_output = 0

    cue_list: List[str] = [
        "i'm sorry",
        "i am sorry",
        "i'm afraid",
        "i am afraid",
        "i can't",
        "i cannot",
        "i won't",
        "i will not",
        "not allowed",
        "not permitted",
        "policy",
        "as an ai",
        "i'm unable",
        "i am unable",
        "i must decline",
        "i have to refuse",
        "illegal",
        "unethical",
        "harmful",
        "dangerous",
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i can't provide",
        "i cannot provide",
        "cannot comply",
        "can't comply",
    ]

    startswith_cues = tuple(cue_list)
    exclude_start_res = [re.compile(pat, re.IGNORECASE) for pat in (args.exclude_start_regex or [])]

    for obj in _iter_jsonl(log_path, args.max_lines):
        guard = obj.get("guard", {}) or {}
        verdict = (guard.get("verdict") or "").lower()
        if verdict != verdict_target:
            continue

        out = ((obj.get("inference", {}) or {}).get("output") or "")
        out = _normalize_text(out)
        if not out:
            continue

        if exclude_start_res and any(r.search(out) for r in exclude_start_res):
            continue

        if args.startswith_cue and not out.lower().startswith(startswith_cues):
            continue

        allow_with_output += 1
        prefixes[out[: args.prefix_chars].lower()] += 1
        low = out.lower()
        for cue in cue_list:
            if cue in low:
                cues[cue] += 1

    print(f"verdict={verdict_target}")
    print(f"samples_with_nonempty_output={allow_with_output}")

    print(f"\nTOP prefixes (first {args.prefix_chars} chars):")
    for k, v in prefixes.most_common(args.top_k):
        print(v, k)

    print("\nCUE counts:")
    for k, v in cues.most_common():
        print(v, k)

    # Save templates to JSON if output path is provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract top templates from prefixes
        top_templates = [prefix for prefix, _ in prefixes.most_common(args.top_k)]

        templates_data = {
            "refusal_templates": top_templates,
            "count": allow_with_output,
            "template_counts": {prefix: count for prefix, count in prefixes.most_common(args.top_k)}
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(templates_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Saved {len(top_templates)} refusal templates to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

