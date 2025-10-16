#!/usr/bin/env python3
"""
Quick script to verify dataset splits are correct.
Run this after creating splits to ensure everything is aligned.
"""
import json
from pathlib import Path
from collections import Counter

def verify_splits():
    """Verify all dataset splits are correctly created."""
    splits_dir = Path("data/cifar100_lt_if100_splits")
    
    print("="*70)
    print("DATASET SPLIT VERIFICATION")
    print("="*70)
    
    # Expected splits and their properties
    expected = {
        'train': {'size': 10847, 'if_target': 100.0},
        'val_lt': {'size': 2169, 'if_target': 100.0},
        'tuneV': {'size': 2169, 'if_target': 100.0},
        'test_lt': {'size': 10847, 'if_target': 100.0},
        'test_balanced': {'size': 6500, 'if_target': 1.0}
    }
    
    all_ok = True
    
    for split_name, props in expected.items():
        filepath = splits_dir / f"{split_name}_indices.json"
        
        if not filepath.exists():
            print(f"\n❌ {split_name}: FILE NOT FOUND")
            all_ok = False
            continue
        
        # Load indices
        with open(filepath, 'r') as f:
            indices = json.load(f)
        
        # Check size
        actual_size = len(indices)
        expected_size = props['size']
        size_ok = (actual_size == expected_size)
        
        status = "[OK]" if size_ok else "[FAIL]"
        print(f"\n{status} {split_name.upper()}")
        print(f"  Size: {actual_size:,} (expected {expected_size:,})")
        print(f"  Target IF: {props['if_target']}")
        print(f"  File: {filepath.name}")
        
        if not size_ok:
            all_ok = False
            print(f"  [WARNING] SIZE MISMATCH!")
    
    print("\n" + "="*70)
    
    if all_ok:
        print("[SUCCESS] ALL SPLITS VERIFIED - READY TO RUN PIPELINE!")
        print("\nNext step:")
        print("  python -m src.train.train_expert")
    else:
        print("[ERROR] SOME SPLITS HAVE ISSUES - RE-RUN DATASET CREATION")
        print("\nFix by running:")
        print('  python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"')
    
    print("="*70)
    
    return all_ok

if __name__ == "__main__":
    verify_splits()

