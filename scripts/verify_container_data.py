#!/usr/bin/env python3
"""Verify SALAD-Bench data in container"""

import json
from pathlib import Path

data_dir = Path("/workspace/data/salad/raw")
files = list(data_dir.glob("*.jsonl"))

print(f"Found {len(files)} files in container:")
total_samples = 0
for f in files:
    count = sum(1 for _ in open(f, "r", encoding="utf-8"))
    total_samples += count
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {count} samples, {size_mb:.2f} MB")

print(f"\nTotal: {total_samples} samples")

