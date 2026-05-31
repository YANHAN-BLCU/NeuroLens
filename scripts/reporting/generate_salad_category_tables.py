import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
LOG_FILES: Tuple[Path, ...] = (
    ROOT / "logs" / "base_evaluation.jsonl",
    ROOT / "logs" / "fixed_corrected_v2.jsonl",
    ROOT / "logs" / "salad_safety_samples.jsonl",
)


def parse_categories() -> Dict[str, Set[str]]:
    """
    Parse all 1/2/3-level SALAD categories from the configured log files.

    Returns:
        Dict[str, Set[str]]: keys are "1", "2", "3" for each level.
    """
    levels: Dict[str, Set[str]] = {"1": set(), "2": set(), "3": set()}
    keys = {
        "1": "1-category",
        "2": "2-category",
        "3": "3-category",
    }

    for log_path in LOG_FILES:
        if not log_path.exists():
            continue

        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                input_data = obj.get("input", {})
                if not isinstance(input_data, dict):
                    continue

                original_sample = input_data.get("original_sample", {})
                if not isinstance(original_sample, dict):
                    continue

                for level, field in keys.items():
                    val = original_sample.get(field)
                    if isinstance(val, str):
                        val = val.strip()
                        if val:
                            levels[level].add(val)

    return levels


def _sort_key(label: str) -> Tuple[int, int, str]:
    """
    Sort SALAD labels by numeric code (O1, O2, ...) when possible.
    Falls back to lexicographic order.
    """
    m = re.match(r"^\s*O(\d+)\s*:", label)
    if m:
        try:
            num = int(m.group(1))
        except ValueError:
            num = 0
        return (0, num, label)
    return (1, 0, label)


def split_code_name(label: str) -> Tuple[str, str]:
    """
    Split a label like 'O5: Malicious Use' into ('O5', 'Malicious Use').
    """
    parts = label.split(":", 1)
    if len(parts) == 2:
        code = parts[0].strip()
        name = parts[1].strip()
        return code, name
    return "", label.strip()


def build_markdown(levels: Dict[str, Set[str]]) -> str:
    """
    Build a Markdown table summarizing all 1/2/3 level categories.
    """
    lines: List[str] = []

    def add_section(title: str, level_key: str) -> None:
        labels = sorted(levels.get(level_key, []), key=_sort_key)
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Code | Name | Full Label |")
        lines.append("| ---- | ---- | ---------- |")
        for lbl in labels:
            code, name = split_code_name(lbl)
            # Escape pipe characters in Markdown cells
            full = lbl.replace("|", "\\|")
            name_cell = name.replace("|", "\\|")
            lines.append(f"| {code} | {name_cell} | {full} |")
        lines.append("")

    lines.append("## SALAD Taxonomy 1/2/3-Level Categories")
    lines.append("")
    add_section("Level 1 Categories (`1-category`)", "1")
    add_section("Level 2 Categories (`2-category`)", "2")
    add_section("Level 3 Categories (`3-category`)", "3")

    return "\n".join(lines)


def main() -> None:
    levels = parse_categories()
    output_dir = ROOT / "outputs" / "salad_taxonomy"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "salad_1_2_3_categories.md"

    md = build_markdown(levels)
    output_path.write_text(md, encoding="utf-8")

    total_counts = {k: len(v) for k, v in levels.items()}
    print(f"Wrote SALAD category table to: {output_path}")
    print("Counts per level:", total_counts)


if __name__ == "__main__":
    main()

