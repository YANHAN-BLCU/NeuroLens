#!/usr/bin/env python3
"""Test download functionality"""

import json
from pathlib import Path
from datasets import load_dataset

output_dir = Path("data/salad/raw")
output_dir.mkdir(parents=True, exist_ok=True)

dataset_name = "OpenSafetyLab/Salad-Data"
configs = ["base_set", "attack_enhanced_set", "defense_enhanced_set", "mcq_set"]

print("Testing download...")
for config in configs:
    try:
        print(f"\nDownloading {config}...")
        ds = load_dataset(dataset_name, config)
        train_data = ds['train']
        print(f"  Loaded {len(train_data)} samples")
        
        output_file = output_dir / f"{config}_train.jsonl"
        print(f"  Saving to {output_file}...")
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"    Saved {count} samples...")
        
        print(f"  [OK] Saved {count} samples to {output_file}")
        
        # Verify file exists
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  [OK] File exists: {size_mb:.2f} MB")
        else:
            print(f"  [ERROR] File does not exist!")
            
    except Exception as e:
        print(f"  [ERROR] Error: {e}")

print("\nChecking output directory...")
files = list(output_dir.glob("*.jsonl"))
print(f"Found {len(files)} files:")
for f in files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {size_mb:.2f} MB")

