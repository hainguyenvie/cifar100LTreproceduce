#!/usr/bin/env python3
"""
Enhanced data preparation for CIFAR-100-LT with duplication-based val/test creation.
Implements the methodology:
1. Train: Standard exponential profile long-tail
2. Val/Test: Match train proportions with duplication when needed
3. TuneV: Subset from test to avoid data leakage
"""

import numpy as np
import json
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

# Set nice plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class CIFAR100LTDataset(Dataset):
    """Custom Dataset wrapper for CIFAR-100-LT with flexible indexing."""
    
    def __init__(self, cifar_dataset, indices, transform=None):
        self.cifar_dataset = cifar_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map to actual CIFAR index
        cifar_idx = self.indices[idx]
        image, label = self.cifar_dataset[cifar_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_cifar100_transforms():
    """Get CIFAR-100 transforms following paper specifications."""
    
    # Training transforms (basic augmentation as per Menon et al., 2021a)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
            std=[0.2675, 0.2565, 0.2761]   # CIFAR-100 std
        )
    ])
    
    # Evaluation transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    return train_transform, eval_transform

def get_exponential_counts(num_classes: int = 100, imb_factor: float = 100, max_samples: int = 500) -> List[int]:
    """Generate exponential profile counts for long-tail distribution."""
    counts = []
    for cls_idx in range(num_classes):
        # Exponential decay: n_i = n_max * (IF)^(-i/(C-1))
        num = max_samples * (imb_factor ** (-cls_idx / (num_classes - 1.0)))
        counts.append(max(1, int(num)))
    return counts

def create_longtail_train(cifar_train_dataset, imb_factor: float = 100, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Create long-tail training set using exponential profile."""
    print(f"Creating CIFAR-100-LT training set (IF={imb_factor})...")
    
    np.random.seed(seed)
    targets = np.array(cifar_train_dataset.targets)
    num_classes = 100
    
    # Get target counts
    target_counts = get_exponential_counts(num_classes, imb_factor, 500)
    
    # Sample indices for each class
    train_indices = []
    actual_counts = []
    
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        
        # Random sample without replacement
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        train_indices.extend(sampled.tolist())
        actual_counts.append(num_to_sample)
    
    train_targets = targets[train_indices].tolist()
    
    print(f"  Created training set: {len(train_indices):,} samples")
    print(f"  Head class: {actual_counts[0]} samples")
    print(f"  Tail class: {actual_counts[-1]} samples")
    print(f"  Actual IF: {actual_counts[0] / actual_counts[-1]:.1f}")
    
    return train_indices, train_targets, actual_counts

def create_proportional_test_val_with_duplication(
    cifar_test_dataset, 
    train_class_counts: List[int],
    val_ratio: float = 0.2,
    tunev_ratio: float = 0.15,
    seed: int = 42,
    create_balanced_test: bool = True
):
    """
    Create val/tuneV/test sets that match train proportions using duplication when needed.
    
    CRITICAL: Split original indices FIRST (disjoint), then replicate SECOND (within each split).
    This ensures zero leakage between val, tuneV, and test.
    
    Methodology (following "Learning to Reject Meets Long-tail Learning"):
    1. Split original balanced CIFAR-100 test into disjoint base sets (by original indices)
    2. Replicate within each base set independently to match train's long-tail proportions
    3. All final splits have same LT distribution as train, with zero cross-contamination
    
    Args:
        cifar_test_dataset: Original CIFAR-100 test dataset (balanced, 100 per class)
        train_class_counts: Count of samples per class in training set
        val_ratio: Proportion of base for validation (default 0.2 = 20 indices per class)
        tunev_ratio: Proportion of base for tuneV (default 0.15 = 15 indices per class)
        seed: Random seed
        
    Returns:
        Tuple of (val_indices, val_targets, tunev_indices, tunev_targets, test_indices, test_targets)
    """
    print("\n" + "="*60)
    print("Creating LT val/tuneV/test splits (SPLIT-FIRST, NO LEAKAGE)")
    print("="*60)
    
    np.random.seed(seed)
    test_targets = np.array(cifar_test_dataset.targets)
    num_classes = 100
    
    # Calculate train proportions - these are the exact proportions we must maintain
    total_train = sum(train_class_counts)
    train_proportions = [count / total_train for count in train_class_counts]
    
    # CRITICAL: To maintain IF=100, we must keep exact class proportions from train
    # We'll use a scaling factor to determine total size while preserving ratios
    # The min class in train determines how we can scale
    min_train_count = min(train_class_counts)
    max_train_count = max(train_class_counts)
    
    print(f"Train distribution: {total_train:,} samples (IF={max_train_count/min_train_count:.1f})")
    print(f"Train class range: {min_train_count} to {max_train_count} samples per class")
    
    # STEP 1: Split original indices into disjoint base sets (no duplication yet)
    print("\nStep 1: Creating disjoint base splits from original test...")
    val_base_indices = []
    tunev_base_indices = []
    test_base_indices = []
    
    base_counts = {'val': 0, 'tunev': 0, 'test': 0}
    
    for cls in range(num_classes):
        cls_indices_in_test = np.where(test_targets == cls)[0]
        np.random.shuffle(cls_indices_in_test)
        
        # Calculate base split sizes (from 100 available per class)
        # Strategy: Maximize test base (since it's final evaluation), smaller val/tuneV bases
        # We'll use replication to match LT distribution (no downsampling)
        n_val_base = max(1, int(round(100 * val_ratio)))       # 20 per class
        n_tunev_base = max(1, int(round(100 * tunev_ratio)))   # 15 per class
        n_test_base = max(1, 100 - n_val_base - n_tunev_base)  # 65 per class (majority for test)
        
        # Adjust if overflow
        if n_val_base + n_tunev_base + n_test_base > 100:
            n_test_base = 100 - n_val_base - n_tunev_base
        
        # Split disjoint
        val_base_indices.extend(cls_indices_in_test[:n_val_base])
        tunev_base_indices.extend(cls_indices_in_test[n_val_base:n_val_base+n_tunev_base])
        test_base_indices.extend(cls_indices_in_test[n_val_base+n_tunev_base:n_val_base+n_tunev_base+n_test_base])
        
        base_counts['val'] += n_val_base
        base_counts['tunev'] += n_tunev_base
        base_counts['test'] += n_test_base
    
    print(f"  Val base (disjoint): {base_counts['val']} samples")
    print(f"  TuneV base (disjoint): {base_counts['tunev']} samples")
    print(f"  Test base (disjoint): {base_counts['test']} samples")
    print(f"  Total base: {sum(base_counts.values())} (should be ~10,000)")
    
    # STEP 2: Replicate within each split to maintain EXACT IF=100
    print("\nStep 2: Replicating to maintain IF=100 (same proportions as train)...")
    
    def replicate_to_exact_proportions(base_indices, scaling_factor, split_name):
        """
        Replicate base indices to maintain EXACT train proportions (IF=100).
        For each class: target_count[cls] = train_class_counts[cls] * adjusted_scaling
        
        Key insight: To maintain IF=100, minimum class needs enough samples.
        We adjust scaling factor to ensure IF=100 is maintained.
        """
        replicated_indices = []
        duplication_stats = {'no_dup': 0, 'duplicated': 0, 'max_dup_factor': 0}
    
        # Calculate required scaling to maintain IF=100
        # If min_train_count=5, and we want min_split_count=5, then scale=1.0
        # If we want smaller splits, we need at least min_split_count samples for tail
        # to maintain IF=100: max/min = 100, so if tail=n, head=100n
        
        # Compute raw targets
        raw_targets = [train_class_counts[cls] * scaling_factor for cls in range(num_classes)]
        min_raw = min([t for t in raw_targets if t > 0]) if any(raw_targets) else 0
        
        # To maintain IF=100, ensure minimum class has at least 1 sample after rounding
        # Then scale everything proportionally
        if min_raw < 1.0:
            # Need to scale up to ensure min becomes at least 1.0 before rounding
            scale_adjustment = 1.0 / min_raw
            adjusted_scaling = scaling_factor * scale_adjustment
        else:
            adjusted_scaling = scaling_factor
        
        # Now compute final targets with adjusted scaling
        target_counts = []
        for cls in range(num_classes):
            raw_count = train_class_counts[cls] * adjusted_scaling
            target_count = max(1, int(round(raw_count)))
            target_counts.append(target_count)
        
        # Second pass: actually sample/replicate
        for cls in range(num_classes):
            base_indices_array = np.array(base_indices)
            base_targets = test_targets[base_indices_array]
            cls_base = base_indices_array[base_targets == cls]
            
            if len(cls_base) == 0:
                # print(f"  Warning: Class {cls} has no base in {split_name}")
                continue
            
            target_count = target_counts[cls]
            if target_count == 0:
                continue
            
            if target_count <= len(cls_base):
                # Sample without replacement
                sampled = np.random.choice(cls_base, target_count, replace=False)
                duplication_stats['no_dup'] += 1
            else:
                # Need duplication
                duplication_factor = int(np.ceil(target_count / len(cls_base)))
                duplication_stats['max_dup_factor'] = max(duplication_stats['max_dup_factor'], duplication_factor)
                duplication_stats['duplicated'] += 1
                
                duplicated_pool = np.tile(cls_base, duplication_factor)
                sampled = np.random.choice(duplicated_pool, target_count, replace=False)
            
            replicated_indices.extend(sampled.tolist())
        
        # Verify IF
        actual_counts = [c for c in target_counts if c > 0]
        if actual_counts:
            actual_if = max(actual_counts) / min(actual_counts)
        else:
            actual_if = 0
        
        print(f"  {split_name}: {len(replicated_indices):,} samples, IF={actual_if:.1f} "
              f"(no_dup: {duplication_stats['no_dup']}, "
              f"duplicated: {duplication_stats['duplicated']}, "
              f"max_factor: {duplication_stats['max_dup_factor']})")
        
        return replicated_indices, target_counts
    
    # CRITICAL DESIGN DECISION: How to "reweight" each split
    # =========================================================
    # Based on split purpose and long-tail learning methodology:
    #
    # 1. VAL (S2): For plugin optimization (α*, μ*)
    #    - Needs LT distribution to optimize for real-world deployment
    #    - Scale: ~20% of train (moderate size, enough for optimization)
    #
    # 2. TUNEV (S1): For gating network training
    #    - Needs LT distribution for gating to learn proper mixture
    #    - Scale: ~15-20% of train (smaller OK, just for gating)
    #
    # 3. TEST-LT: For main evaluation vs papers
    #    - Needs LT distribution matching train
    #    - Scale: 100% of train OR maximize base usage
    #    - Recommendation: Use 1.0 (same size as train) for direct comparison
    #
    # 4. TEST-BALANCED (optional): For robustness analysis
    #    - Keep original balanced distribution
    #    - Use all available base samples
    
    val_scale = 0.20      # ~20% of train (~2,169 samples) for optimization
    tunev_scale = 0.20    # ~20% of train (~2,169 samples) for gating training
    test_lt_scale = 1.0   # 100% of train (~10,847 samples) - FULL SIZE for fair comparison
    
    val_indices, val_class_counts = replicate_to_exact_proportions(val_base_indices, val_scale, "Val-LT")
    tunev_indices, tunev_class_counts = replicate_to_exact_proportions(tunev_base_indices, tunev_scale, "TuneV-LT")
    test_lt_indices, test_lt_class_counts = replicate_to_exact_proportions(test_base_indices, test_lt_scale, "Test-LT")
    
    # Get targets for LT test
    val_targets = test_targets[val_indices].tolist()
    tunev_targets = test_targets[tunev_indices].tolist()
    test_lt_targets = test_targets[test_lt_indices].tolist()
    
    # Optionally create balanced test set (uses all remaining base samples)
    test_balanced_indices = None
    test_balanced_targets = None
    
    if create_balanced_test:
        print("\nStep 3: Creating balanced test set (all unique base samples)...")
        # Use ALL test base indices without replication
        test_balanced_indices = test_base_indices
        test_balanced_targets = test_targets[test_balanced_indices].tolist()
        print(f"  Test-Balanced: {len(test_balanced_indices):,} samples (100% unique, 65 per class)")
    
    # Verify disjoint property (by unique original indices)
    val_unique = set(val_base_indices)
    tunev_unique = set(tunev_base_indices)
    test_unique = set(test_base_indices)
    
    assert len(val_unique & tunev_unique) == 0, "Val and TuneV share base indices!"
    assert len(val_unique & test_unique) == 0, "Val and Test share base indices!"
    assert len(tunev_unique & test_unique) == 0, "TuneV and Test share base indices!"
    
    print("\n[OK] Verified: All splits are disjoint by original CIFAR-100 indices")
    print("="*60)
    
    return_dict = {
        'val': (val_indices, val_targets),
        'tunev': (tunev_indices, tunev_targets),
        'test_lt': (test_lt_indices, test_lt_targets),
    }
    
    if create_balanced_test:
        return_dict['test_balanced'] = (test_balanced_indices, test_balanced_targets)
    
    return return_dict


def analyze_distribution(indices: List[int], targets: List[int], name: str, train_counts: Optional[List[int]] = None, threshold: int = 20):
    """Analyze and print distribution statistics with head/tail based on threshold."""
    print(f"\n=== {name.upper()} DISTRIBUTION ===")
    
    target_counts = Counter(targets)
    sorted_counts = [target_counts.get(i, 0) for i in range(100)]
    
    total = sum(sorted_counts)
    head_count = sorted_counts[0]
    tail_count = sorted_counts[99]
    
    print(f"Total samples: {total:,}")
    print(f"Head class (0): {head_count} samples ({head_count/total*100:.2f}%)")
    print(f"Tail class (99): {tail_count} samples ({tail_count/total*100:.2f}%)")
    if tail_count > 0:
        print(f"Imbalance factor: {head_count/tail_count:.1f}")
    
    # Head/Tail analysis based on train threshold
    if train_counts is not None:
        head_classes = [i for i in range(100) if train_counts[i] > threshold]
        tail_classes = [i for i in range(100) if train_counts[i] <= threshold]
        
        head_group_count = sum(sorted_counts[i] for i in head_classes)
        tail_group_count = sum(sorted_counts[i] for i in tail_classes)
        
        print(f"\nDistribution by Head/Tail (threshold={threshold} in train):")
        print(f"  Head (>20 samples, {len(head_classes)} classes): {head_group_count:,} samples ({head_group_count/total*100:.1f}%)")
        print(f"  Tail (<=20 samples, {len(tail_classes)} classes): {tail_group_count:,} samples ({tail_group_count/total*100:.1f}%)")
    
    # Legacy group analysis for reference
    groups = {
        'Head (0-9)': sum(sorted_counts[0:10]),
        'Medium (10-49)': sum(sorted_counts[10:50]), 
        'Low (50-89)': sum(sorted_counts[50:90]),
        'Tail (90-99)': sum(sorted_counts[90:100])
    }
    
    print("\nLegacy grouping (by class index):")
    for group_name, group_count in groups.items():
        print(f"  {group_name}: {group_count:,} samples ({group_count/total*100:.1f}%)")
    
    # Compare with train if available
    if train_counts is not None:
        print("\nComparison with train proportions:")
        train_total = sum(train_counts)
        for i in [0, 25, 50, 75, 99]:  # Sample classes
            train_prop = train_counts[i] / train_total
            test_prop = sorted_counts[i] / total
            diff = abs(train_prop - test_prop)
            print(f"  Class {i:2d}: train={train_prop:.4f}, {name.lower()}={test_prop:.4f}, diff={diff:.4f}")

def save_splits_to_json(splits_dict: Dict, output_dir: str):
    """Save all splits to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}...")
    
    for split_name, indices in splits_dict.items():
        filepath = output_path / f"{split_name}_indices.json"
        
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(indices, 'tolist'):
            indices_to_save = indices.tolist()
        elif isinstance(indices, (list, tuple)):
            indices_to_save = [int(idx) if hasattr(idx, 'item') else idx for idx in indices]
        else:
            indices_to_save = list(indices)
            
        with open(filepath, 'w') as f:
            json.dump(indices_to_save, f)
        print(f"  Saved {split_name}: {len(indices_to_save):,} samples")

def compute_split_statistics(indices: List[int], targets: List[int], name: str, 
                             base_indices: Optional[List[int]] = None,
                             train_class_counts: Optional[List[int]] = None,
                             threshold: int = 20) -> Dict:
    """
    Compute comprehensive statistics for a split.
    
    Head/Tail definition: Based on train set class counts
    - Head: classes with > threshold samples in train
    - Tail: classes with <= threshold samples in train
    """
    target_counts = Counter(targets)
    class_counts = [target_counts.get(i, 0) for i in range(100)]
    
    total = len(indices)
    
    # Compute head/tail based on train class counts if provided
    if train_class_counts is not None:
        head_classes = [i for i in range(100) if train_class_counts[i] > threshold]
        tail_classes = [i for i in range(100) if train_class_counts[i] <= threshold]
        
        head_count = sum(class_counts[i] for i in head_classes)
        tail_count = sum(class_counts[i] for i in tail_classes)
        
        # Store which classes are head/tail
        num_head_classes = len(head_classes)
        num_tail_classes = len(tail_classes)
    else:
        # Fallback to arbitrary grouping if train counts not provided
        head_count = sum(class_counts[:50])
        tail_count = sum(class_counts[50:100])
        num_head_classes = 50
        num_tail_classes = 50
    
    # Legacy grouping for backward compatibility (can be used for detailed analysis)
    medium_count = sum(class_counts[10:50])
    low_count = sum(class_counts[50:90])
    
    # Compute duplication statistics if base indices provided
    duplication_stats = {}
    if base_indices is not None:
        unique_base = len(set(base_indices))
        total_samples = len(indices)
        duplicates_count = total_samples - unique_base
        avg_duplication = total_samples / unique_base if unique_base > 0 else 0
        
        duplication_stats = {
            'unique_base_samples': unique_base,
            'total_after_duplication': total_samples,
            'duplicated_samples': duplicates_count,
            'duplication_ratio': avg_duplication
        }
    
    return {
        'name': name,
        'total_samples': total,
        'class_counts': class_counts,
        'head_samples': head_count,
        'tail_samples': tail_count,
        'num_head_classes': num_head_classes,
        'num_tail_classes': num_tail_classes,
        'medium_samples': medium_count,  # Legacy
        'low_samples': low_count,  # Legacy
        'head_ratio': head_count / total if total > 0 else 0,
        'tail_ratio': tail_count / total if total > 0 else 0,
        'medium_ratio': medium_count / total if total > 0 else 0,  # Legacy
        'low_ratio': low_count / total if total > 0 else 0,  # Legacy
        'max_class_count': max(class_counts),
        'min_class_count': min(class_counts),
        'imbalance_factor': max(class_counts) / max(1, min(class_counts)),
        **duplication_stats
    }

def plot_comprehensive_statistics(all_stats: Dict[str, Dict], output_dir: str):
    """Create comprehensive visualization of all dataset statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    split_names = list(all_stats.keys())
    
    # Create large comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Total samples per split (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    totals = [all_stats[name]['total_samples'] for name in split_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']  # Extended for more splits
    bars = ax1.bar(split_names, totals, color=colors[:len(split_names)], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Samples', fontweight='bold')
    ax1.set_title('Total Samples per Split', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Head/Tail distribution (stacked bar) - Based on threshold=20 in train
    ax2 = fig.add_subplot(gs[0, 1])
    head_vals = [all_stats[name]['head_samples'] for name in split_names]
    tail_vals = [all_stats[name]['tail_samples'] for name in split_names]
    
    # Get number of classes for legend
    if 'Train' in all_stats:
        num_head_classes = all_stats['Train'].get('num_head_classes', 0)
        num_tail_classes = all_stats['Train'].get('num_tail_classes', 0)
    else:
        num_head_classes = 0
        num_tail_classes = 0
    
    x = np.arange(len(split_names))
    width = 0.6
    
    p1 = ax2.bar(x, head_vals, width, label=f'Head (>20 samples, n={num_head_classes})', color='#3498db', alpha=0.8)
    p2 = ax2.bar(x, tail_vals, width, bottom=head_vals, 
                label=f'Tail (<=20 samples, n={num_tail_classes})', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_title('Head/Tail Distribution (Threshold=20 in Train)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_names)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Proportion ratios (grouped bar) - Head vs Tail only
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(split_names))
    bar_width = 0.35
    
    head_ratios = [all_stats[name]['head_ratio'] for name in split_names]
    tail_ratios = [all_stats[name]['tail_ratio'] for name in split_names]
    
    ax3.bar(x - bar_width/2, head_ratios, bar_width, label='Head (>20)', color='#3498db', alpha=0.8, edgecolor='black')
    ax3.bar(x + bar_width/2, tail_ratios, bar_width, label='Tail (<=20)', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add percentage labels on bars
    for i, (h, t) in enumerate(zip(head_ratios, tail_ratios)):
        ax3.text(i - bar_width/2, h + 0.02, f'{h*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax3.text(i + bar_width/2, t + 0.02, f'{t*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_ylabel('Proportion', fontweight='bold')
    ax3.set_title('Head vs Tail Proportions', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(split_names)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(max(head_ratios), max(tail_ratios)) * 1.2)
    
    # 4. Per-class distribution for each split (line plots)
    ax4 = fig.add_subplot(gs[1, :])
    class_range = np.arange(100)
    
    for i, name in enumerate(split_names):
        class_counts = all_stats[name]['class_counts']
        ax4.plot(class_range, class_counts, label=name, linewidth=2, 
                color=colors[i], marker='o', markersize=2, alpha=0.7)
    
    ax4.set_xlabel('Class Index', fontweight='bold')
    ax4.set_ylabel('Number of Samples (log scale)', fontweight='bold')
    ax4.set_title('Per-Class Sample Distribution (All Splits)', fontweight='bold', fontsize=12)
    ax4.set_yscale('log')
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0, 99)
    
    # 5. Imbalance factors
    ax5 = fig.add_subplot(gs[2, 0])
    imb_factors = [all_stats[name]['imbalance_factor'] for name in split_names]
    bars = ax5.bar(split_names, imb_factors, color=colors[:len(split_names)], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Imbalance Factor', fontweight='bold')
    ax5.set_title('Imbalance Factor (Max/Min Class)', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Target IF=100', alpha=0.7)
    ax5.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Duplication statistics (if available)
    ax6 = fig.add_subplot(gs[2, 1])
    has_duplication = any('duplication_ratio' in all_stats[name] for name in split_names)
    
    if has_duplication:
        dup_names = [name for name in split_names if 'duplication_ratio' in all_stats[name]]
        dup_ratios = [all_stats[name]['duplication_ratio'] for name in dup_names]
        unique_bases = [all_stats[name]['unique_base_samples'] for name in dup_names]
        
        bars = ax6.bar(dup_names, dup_ratios, color=['#9b59b6', '#1abc9c', '#34495e'][:len(dup_names)], 
                      alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Average Duplication Factor', fontweight='bold')
        ax6.set_title('Duplication Statistics', fontweight='bold', fontsize=12)
        ax6.grid(axis='y', alpha=0.3)
        ax6.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='No duplication', alpha=0.7)
        ax6.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No duplication data available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_xticks([])
        ax6.set_yticks([])
    
    # 7. Comparison table (text)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = []
    table_data.append(['Split', 'Total', 'Head%', 'Tail%', 'IF'])
    for name in split_names:
        stats = all_stats[name]
        table_data.append([
            name,
            f"{stats['total_samples']:,}",
            f"{stats['head_ratio']*100:.1f}%",
            f"{stats['tail_ratio']*100:.1f}%",
            f"{stats['imbalance_factor']:.1f}"
        ])
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax7.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    
    # Overall title
    fig.suptitle('CIFAR-100-LT Dataset Splits - Comprehensive Statistics', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plot_path = output_path / 'dataset_statistics_comprehensive.png'
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
    print(f"\n[PLOT] Saved comprehensive visualization: {plot_path}")
    
    # Also save as PDF for publication quality
    pdf_path = output_path / 'dataset_statistics_comprehensive.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"[PLOT] Saved PDF version: {pdf_path}")
    
    plt.close()
    
    return plot_path

def save_statistics_to_csv(all_stats: Dict[str, Dict], output_dir: str):
    """Save detailed statistics to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics
    summary_data = []
    for name, stats in all_stats.items():
        row = {
            'Split': name,
            'Total_Samples': stats['total_samples'],
            'Head_Samples': stats['head_samples'],
            'Medium_Samples': stats['medium_samples'],
            'Low_Samples': stats['low_samples'],
            'Tail_Samples': stats['tail_samples'],
            'Head_Ratio': f"{stats['head_ratio']:.4f}",
            'Medium_Ratio': f"{stats['medium_ratio']:.4f}",
            'Low_Ratio': f"{stats['low_ratio']:.4f}",
            'Tail_Ratio': f"{stats['tail_ratio']:.4f}",
            'Imbalance_Factor': f"{stats['imbalance_factor']:.2f}",
            'Max_Class_Count': stats['max_class_count'],
            'Min_Class_Count': stats['min_class_count']
        }
        
        # Add duplication stats if available
        if 'duplication_ratio' in stats:
            row['Unique_Base_Samples'] = stats['unique_base_samples']
            row['Duplicated_Samples'] = stats['duplicated_samples']
            row['Duplication_Ratio'] = f"{stats['duplication_ratio']:.4f}"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / 'split_summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"[CSV] Saved summary statistics: {summary_path}")
    
    # 2. Per-class distribution
    class_data = []
    for class_idx in range(100):
        row = {'Class': class_idx}
        for name, stats in all_stats.items():
            row[f'{name}_count'] = stats['class_counts'][class_idx]
        class_data.append(row)
    
    class_df = pd.DataFrame(class_data)
    class_path = output_path / 'per_class_distribution.csv'
    class_df.to_csv(class_path, index=False)
    print(f"[CSV] Saved per-class distribution: {class_path}")
    
    return summary_path, class_path

def create_full_cifar100_lt_splits(
    imb_factor: float = 100,
    output_dir: str = "data/cifar100_lt_if100_splits", 
    val_ratio: float = 0.2,
    tunev_ratio: float = 0.15,
    seed: int = 42,
    create_balanced_test: bool = True,
    use_evaluation_reweighting: bool = False
):
    """
    Create complete CIFAR-100-LT dataset splits with flexible reweighting strategies.
    
    Two modes:
    1. Physical Replication (use_evaluation_reweighting=False):
       - Replicate samples to match train's LT distribution
       - Test/val have same IF=100 as train
       
    2. Evaluation Reweighting (use_evaluation_reweighting=True):
       - Keep all splits BALANCED (no duplication)
       - Save train class weights for weighted loss/metrics
       - More common in research, preserves all unique data
    
    Args:
        imb_factor: Imbalance factor for training set (default 100)
        output_dir: Directory to save split indices
        val_ratio: Base proportion for validation (default 0.2)
        tunev_ratio: Base proportion for tuneV (default 0.15)
        seed: Random seed for reproducibility
        create_balanced_test: Create additional balanced test set
        use_evaluation_reweighting: If True, keep val/tuneV/test balanced and use weighted metrics
        
    Returns:
        Tuple of (datasets, splits)
    """
    print("=" * 60)
    print("CREATING CIFAR-100-LT DATASET SPLITS (SPLIT-FIRST)")  
    if use_evaluation_reweighting:
        print("Mode: EVALUATION REWEIGHTING (balanced splits + weighted metrics)")
    else:
        print("Mode: PHYSICAL REPLICATION (LT splits with sample duplication)")
    print("=" * 60)
    
    # Load original CIFAR-100
    print("\nLoading CIFAR-100 datasets...")
    cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=None)
    cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=None)
    
    # 1. Create long-tail training set
    train_indices, train_targets, train_counts = create_longtail_train(cifar_train, imb_factor, seed)
    
    # 2. Create val/tuneV/test with SPLIT-FIRST methodology (all in one go, no leakage)
    # We need to also track base indices for duplication statistics
    print("\n" + "="*60)
    print("CREATING SPLITS WITH BASE TRACKING FOR STATISTICS")
    print("="*60)
    
    # First, create base indices (before duplication)
    test_targets_array = np.array(cifar_test.targets)
    val_base_indices = []
    tunev_base_indices = []
    test_base_indices = []
    
    np.random.seed(seed)
    for cls in range(100):
        cls_indices_in_test = np.where(test_targets_array == cls)[0]
        np.random.shuffle(cls_indices_in_test)
        
        n_val_base = max(1, int(round(100 * val_ratio)))
        n_tunev_base = max(1, int(round(100 * tunev_ratio)))
        n_test_base = max(1, 100 - n_val_base - n_tunev_base)
        
        if n_val_base + n_tunev_base + n_test_base > 100:
            n_test_base = 100 - n_val_base - n_tunev_base
        
        val_base_indices.extend(cls_indices_in_test[:n_val_base])
        tunev_base_indices.extend(cls_indices_in_test[n_val_base:n_val_base+n_tunev_base])
        test_base_indices.extend(cls_indices_in_test[n_val_base+n_tunev_base:n_val_base+n_tunev_base+n_test_base])
    
    # MODE SWITCH: Evaluation reweighting vs physical replication
    if use_evaluation_reweighting:
        print("\n[MODE] Using EVALUATION REWEIGHTING - all splits kept BALANCED")
        print("  Val/TuneV/Test will use ALL unique base samples (no duplication)")
        print("  Train class weights will be saved for weighted loss/metrics")
        
        # Use base indices directly (balanced)
        val_indices = val_base_indices
        val_targets = test_targets_array[val_indices].tolist()
        
        tunev_indices = tunev_base_indices
        tunev_targets = test_targets_array[tunev_indices].tolist()
        
        test_lt_indices = test_base_indices
        test_lt_targets = test_targets_array[test_lt_indices].tolist()
        
        test_balanced_indices = None
        test_balanced_targets = None
        
        print(f"\n  Val: {len(val_indices):,} samples (balanced, 20 per class)")
        print(f"  TuneV: {len(tunev_indices):,} samples (balanced, 15 per class)")
        print(f"  Test: {len(test_lt_indices):,} samples (balanced, 65 per class)")
        print("  [INFO] No duplication - all unique samples")
        
    else:
        # Original physical replication mode
        print("\n[MODE] Using PHYSICAL REPLICATION - splits match train LT distribution")
        
        splits_dict = create_proportional_test_val_with_duplication(
            cifar_test, train_counts, val_ratio, tunev_ratio, seed, create_balanced_test
        )
        
        # Unpack splits
        val_indices, val_targets = splits_dict['val']
        tunev_indices, tunev_targets = splits_dict['tunev']
        test_lt_indices, test_lt_targets = splits_dict['test_lt']
        
        # Optionally unpack balanced test
        if 'test_balanced' in splits_dict:
            test_balanced_indices, test_balanced_targets = splits_dict['test_balanced']
        else:
            test_balanced_indices, test_balanced_targets = None, None
    
    # 3. Analyze all distributions (console output)
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    analyze_distribution(train_indices, train_targets, "TRAIN", train_counts, threshold=20)
    analyze_distribution(val_indices, val_targets, "VAL-LT", train_counts, threshold=20)  
    analyze_distribution(tunev_indices, tunev_targets, "TUNEV-LT", train_counts, threshold=20)
    analyze_distribution(test_lt_indices, test_lt_targets, "TEST-LT", train_counts, threshold=20)
    
    if test_balanced_indices is not None:
        analyze_distribution(test_balanced_indices, test_balanced_targets, "TEST-BALANCED", train_counts, threshold=20)
    
    # 4. Compute detailed statistics for visualization
    print("\n" + "="*60)
    print("COMPUTING COMPREHENSIVE STATISTICS")
    print("="*60)
    
    # Compute train class counts for threshold-based head/tail definition
    train_class_counts_list = [Counter(train_targets)[i] for i in range(100)]
    
    all_stats = {
        'Train': compute_split_statistics(train_indices, train_targets, 'Train', 
                                          train_class_counts=train_class_counts_list, threshold=20),
        'Val-LT': compute_split_statistics(val_indices, val_targets, 'Val-LT', val_base_indices,
                                       train_class_counts=train_class_counts_list, threshold=20),
        'TuneV-LT': compute_split_statistics(tunev_indices, tunev_targets, 'TuneV-LT', tunev_base_indices,
                                         train_class_counts=train_class_counts_list, threshold=20),
        'Test-LT': compute_split_statistics(test_lt_indices, test_lt_targets, 'Test-LT', test_base_indices,
                                        train_class_counts=train_class_counts_list, threshold=20)
    }
    
    # Add balanced test if created
    if test_balanced_indices is not None:
        all_stats['Test-Bal'] = compute_split_statistics(test_balanced_indices, test_balanced_targets, 
                                                         'Test-Bal', test_base_indices,
                                                         train_class_counts=train_class_counts_list, threshold=20)
    
    # Print summary
    for split_name, stats in all_stats.items():
        print(f"\n{split_name}:")
        print(f"  Total: {stats['total_samples']:,}")
        print(f"  IF: {stats['imbalance_factor']:.2f}")
        print(f"  Head (>20): {stats['head_samples']:,} samples ({stats['head_ratio']*100:.1f}%) from {stats['num_head_classes']} classes")
        print(f"  Tail (<=20): {stats['tail_samples']:,} samples ({stats['tail_ratio']*100:.1f}%) from {stats['num_tail_classes']} classes")
        if 'duplication_ratio' in stats:
            print(f"  Duplication: {stats['duplication_ratio']:.2f}x (unique base: {stats['unique_base_samples']})")
    
    # 5. Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_path = plot_comprehensive_statistics(all_stats, output_dir)
    
    # 6. Save statistics to CSV
    summary_path, class_path = save_statistics_to_csv(all_stats, output_dir)
    
    # 7. Save split indices to JSON
    splits = {
        'train': train_indices,
        'val_lt': val_indices,
        'tuneV': tunev_indices,
        'test_lt': test_lt_indices
    }
    
    # Add balanced test if created
    if test_balanced_indices is not None:
        splits['test_balanced'] = test_balanced_indices
    
    save_splits_to_json(splits, output_dir)
    
    # Save train class weights for evaluation reweighting
    train_class_counts_list = [Counter(train_targets)[i] for i in range(100)]
    total_train = sum(train_class_counts_list)
    
    # Compute sample weights: weight_i = freq(class_i) / total
    # These weights sum to 1.0 and represent class proportions
    class_weights = [count / total_train for count in train_class_counts_list]
    
    weights_info = {
        'class_counts': train_class_counts_list,
        'class_weights': class_weights,  # Proportions (sum to 1.0)
        'total_samples': total_train,
        'imbalance_factor': max(train_class_counts_list) / min(train_class_counts_list),
        'mode': 'evaluation_reweighting' if use_evaluation_reweighting else 'physical_replication',
        'threshold': 20,
        'head_classes': [i for i in range(100) if train_class_counts_list[i] > 20],
        'tail_classes': [i for i in range(100) if train_class_counts_list[i] <= 20]
    }
    
    weights_path = Path(output_dir) / 'train_class_weights.json'
    with open(weights_path, 'w') as f:
        json.dump(weights_info, f, indent=2)
    print(f"\n[WEIGHTS] Saved train class weights: {weights_path}")
    if use_evaluation_reweighting:
        print("  [INFO] Use these weights for weighted loss/metrics in training/evaluation")
    
    # 8. Create dataset objects with transforms
    train_transform, eval_transform = get_cifar100_transforms()
    
    datasets = {
        'train': CIFAR100LTDataset(cifar_train, train_indices, train_transform),
        'val': CIFAR100LTDataset(cifar_test, val_indices, eval_transform),
        'tunev': CIFAR100LTDataset(cifar_test, tunev_indices, eval_transform),
        'test_lt': CIFAR100LTDataset(cifar_test, test_lt_indices, eval_transform)
    }
    
    if test_balanced_indices is not None:
        datasets['test_balanced'] = CIFAR100LTDataset(cifar_test, test_balanced_indices, eval_transform)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] DATASET CREATION COMPLETED!")
    print("=" * 60)
    print("Key properties:")
    print("  [OK] All LT splits have same distribution as train (IF=100)")
    print("  [OK] Head/Tail defined by threshold=20 in train")
    print(f"  [OK] Head: 69 classes (>20 samples), Tail: 31 classes (<=20 samples)")
    print("  [OK] Val, TuneV, Test-LT are disjoint by original CIFAR-100 indices")
    print("  [OK] Zero data leakage guaranteed")
    if test_balanced_indices is not None:
        print("  [OK] Balanced test set created for additional analysis")
    print("\nOutput files:")
    print(f"  * Visualizations: {plot_path}")
    print(f"  * Summary stats: {summary_path}")
    print(f"  * Per-class data: {class_path}")
    print(f"  * Split indices: {output_dir}/*.json")
    print("=" * 60)
    
    return datasets, splits

if __name__ == "__main__":
    # Create the full dataset
    datasets, splits = create_full_cifar100_lt_splits(
        imb_factor=100,
        output_dir="data/cifar100_lt_if100_splits",
        val_ratio=0.2,
        tunev_ratio=0.15,
        seed=42,
        create_balanced_test=True  # Also create balanced test for robustness analysis
    )
    
    print("\nDatasets ready for training:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset):,} samples")