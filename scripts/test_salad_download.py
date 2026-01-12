#!/usr/bin/env python3
"""Test SALAD-Data dataset loading"""

from datasets import load_dataset

print("Testing dataset load...")
try:
    ds = load_dataset("OpenSafetyLab/Salad-Data")
    print(f"Success! Keys: {list(ds.keys())}")
    for key in ds.keys():
        print(f"  {key}: {len(ds[key])} samples")
        if len(ds[key]) > 0:
            print(f"    First sample keys: {list(ds[key][0].keys())}")
except Exception as e:
    print(f"Error: {e}")

