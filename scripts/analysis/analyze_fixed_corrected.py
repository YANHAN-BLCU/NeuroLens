import json
import sys
from collections import Counter
from pathlib import Path


def analyze(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    n = 0
    verdict_fixed = Counter()
    severity_fixed = Counter()
    verdict_orig = Counter()
    severity_orig = Counter()
    fix_types = Counter()
    category_counts = Counter()
    changed_verdict = 0
    changed_severity = 0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            d = json.loads(line)

            guard = d.get("guard", {}) or {}
            v_fix = guard.get("verdict")
            s_fix = guard.get("severity")
            if v_fix is not None:
                verdict_fixed[v_fix] += 1
            if s_fix is not None:
                severity_fixed[s_fix] += 1
            for c in guard.get("categories", []) or []:
                label = c.get("label") or c.get("id")
                if label:
                    category_counts[label] += 1

            rv = d.get("reevaluation", {}) or {}
            if rv:
                og = rv.get("original_guard", {}) or {}
                v_org = og.get("verdict")
                s_org = og.get("severity")
                if v_org is not None:
                    verdict_orig[v_org] += 1
                if s_org is not None:
                    severity_orig[s_org] += 1
                if v_org is not None and v_fix is not None and v_org != v_fix:
                    changed_verdict += 1
                if s_org is not None and s_fix is not None and s_org != s_fix:
                    changed_severity += 1

            fix = d.get("fix_applied")
            if fix:
                fix_types[fix] += 1

    return {
        "path": str(p),
        "total_samples": n,
        "verdict_fixed": verdict_fixed,
        "verdict_orig": verdict_orig,
        "severity_fixed": severity_fixed,
        "severity_orig": severity_orig,
        "changed_verdict": changed_verdict,
        "changed_severity": changed_severity,
        "fix_types": fix_types,
        "top_categories": category_counts.most_common(20),
    }


def to_jsonable(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "logs/fixed_corrected_v2.jsonl"
    summary = analyze(input_path)
    json_summary = to_jsonable(summary)
    json.dump(json_summary, sys.stdout, ensure_ascii=False, indent=2)
    print()

