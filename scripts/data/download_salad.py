#!/usr/bin/env python3
"""
Download SALAD-Bench dataset script

Usage:
    python scripts/download_salad.py --split all --output data/salad/raw
    python scripts/download_salad.py --split analysis --output data/salad/raw
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets package required")
    print("Run: pip install datasets")
    sys.exit(1)


def download_from_github(output_dir: Path) -> bool:
    """
    Download SALAD-Bench from GitHub repository (code only, data is on HuggingFace)
    
    Args:
        output_dir: Output directory
    
    Returns:
        Success status
    """
    github_repo = "https://github.com/OpenSafetyLab/SALAD-BENCH.git"
    temp_dir = output_dir.parent / "salad_bench_repo"
    
    try:
        print(f"Note: GitHub repo contains code only. Data should be downloaded from HuggingFace.")
        print(f"Repository location: {temp_dir}")
        
        # Check if example file exists
        example_file = temp_dir / "examples" / "example_qa.jsonl"
        if example_file.exists():
            print(f"Found example file: {example_file}")
            # Copy example as reference
            import shutil
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example_file, output_dir / "example_qa.jsonl")
            print(f"Example file copied to {output_dir}")
            return True
        else:
            print(f"Warning: Example file not found. Please download data from HuggingFace:")
            print(f"  Dataset: OpenSafetyLab/Salad-Data")
            print(f"  URL: https://huggingface.co/datasets/OpenSafetyLab/Salad-Data")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def download_from_huggingface(split: str, output_dir: Path) -> bool:
    """
    Download SALAD-Bench from HuggingFace Hub
    
    Args:
        split: Data split name (will be mapped to configs)
        output_dir: Output directory
    
    Returns:
        Success status
    """
    try:
        # Correct dataset name from README: OpenSafetyLab/Salad-Data
        # The dataset requires a config name and only has 'train' split
        dataset_name = "OpenSafetyLab/Salad-Data"
        configs = ["base_set", "attack_enhanced_set", "defense_enhanced_set", "mcq_set"]
        
        # Map split names to configs (if needed)
        # For now, download all configs since dataset only has 'train' split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = False
        for config in configs:
            try:
                print(f"\nDownloading config: {config}")
                # Load dataset with specific config
                full_dataset = load_dataset(dataset_name, config)
                
                # Check available splits (should be 'train')
                available_splits = list(full_dataset.keys())
                print(f"Available splits for {config}: {available_splits}")
                
                # Use 'train' split (the only available split)
                if 'train' in available_splits:
                    dataset = full_dataset['train']
                    print(f"Loaded {len(dataset)} samples from {config}")
                    
                    # Save dataset with config name as part of filename
                    output_file = output_dir / f"{config}_{split}.jsonl"
                    print(f"Saving to {output_file}...")
                    
                    sample_count = 0
                    with open(output_file, "w", encoding="utf-8") as f:
                        for item in dataset:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            sample_count += 1
                            if sample_count % 1000 == 0:
                                print(f"  Saved {sample_count} samples...")
                    
                    print(f"Saved {sample_count} samples to {output_file}")
                    success = True
                else:
                    print(f"Warning: 'train' split not found in {config}")
                    
            except Exception as e:
                print(f"Failed to download {config}: {e}")
                continue
        
        if not success:
            print("Could not load any dataset from HuggingFace, trying GitHub...")
            return download_from_github(output_dir)
        
        return True
        
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return download_from_github(output_dir)


def download_salad_split(split: str, output_dir: Path, source: str = "auto") -> bool:
    """
    Download specified split of SALAD-Bench data
    
    Args:
        split: Data split (analysis|eval|finetune|all)
        output_dir: Output directory
        source: Data source (auto|huggingface|github)
    
    Returns:
        Success status
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading SALAD-Bench {split} data...")
    print(f"Output directory: {output_dir}")
    
    if source == "github" or (source == "auto" and not download_from_huggingface(split, output_dir)):
        return download_from_github(output_dir)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SALAD-Bench dataset")
    parser.add_argument(
        "--split",
        type=str,
        choices=["analysis", "eval", "finetune", "all"],
        default="all",
        help="Data split to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/salad/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["auto", "huggingface", "github"],
        default="auto",
        help="Data source (auto tries HuggingFace first, then GitHub)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.split == "all":
        splits = ["analysis", "eval", "finetune"]
    else:
        splits = [args.split]
    
    success = True
    for split in splits:
        if not download_salad_split(split, output_dir, args.source):
            success = False
    
    if success:
        print(f"\nAll data downloaded successfully to: {output_dir}")
    else:
        print(f"\nSome data downloads failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

