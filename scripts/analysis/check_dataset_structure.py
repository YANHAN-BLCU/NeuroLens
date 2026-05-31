#!/usr/bin/env python3
"""Check SALAD-Bench dataset structure"""

from datasets import load_dataset

dataset_name = "OpenSafetyLab/Salad-Data"
configs = ["base_set", "attack_enhanced_set", "defense_enhanced_set", "mcq_set"]

print("Checking dataset structure...")
print("=" * 60)

for config in configs:
    try:
        ds = load_dataset(dataset_name, config)
        splits = list(ds.keys())
        print(f"\nConfig: {config}")
        print(f"  Splits: {splits}")
        for split in splits:
            size = len(ds[split])
            print(f"  {split}: {size} samples")
            if size > 0:
                # Show first sample structure
                sample = ds[split][0]
                print(f"    Sample keys: {list(sample.keys())}")
    except Exception as e:
        print(f"\nConfig: {config} - Error: {e}")

