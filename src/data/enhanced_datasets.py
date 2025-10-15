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
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
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
    
    # Calculate train proportions
    total_train = sum(train_class_counts)
    train_proportions = [count / total_train for count in train_class_counts]
    
    # Target total sizes for each split (after replication)
    # Scale based on train total to match proportions
    target_val_total = int(total_train * 0.15)      # ~15% of train size
    target_tunev_total = int(total_train * 0.12)    # ~12% of train size
    target_test_total = int(total_train * 0.20)     # ~20% of train size
    
    print(f"Train distribution: {total_train:,} samples")
    print(f"Target val total: {target_val_total:,}")
    print(f"Target tuneV total: {target_tunev_total:,}")
    print(f"Target test total: {target_test_total:,}")
    
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
        # Ensure at least 1 per split and sum to 100
        n_val_base = max(1, int(round(100 * val_ratio)))
        n_tunev_base = max(1, int(round(100 * tunev_ratio)))
        n_test_base = max(1, 100 - n_val_base - n_tunev_base)
        
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
    
    # STEP 2: Replicate within each split independently to match train proportions
    print("\nStep 2: Replicating within each split to match train LT distribution...")
    
    def replicate_to_match_proportions(base_indices, target_total, split_name):
        """Replicate base indices to match train proportions for this split."""
        replicated_indices = []
        duplication_stats = {'no_dup': 0, 'duplicated': 0, 'max_dup_factor': 0}
        
        for cls in range(num_classes):
            # Get base indices for this class in this split
            base_indices_array = np.array(base_indices)
            base_targets = test_targets[base_indices_array]
            cls_base = base_indices_array[base_targets == cls]
            
            if len(cls_base) == 0:
                print(f"  ⚠️  Warning: Class {cls} has no base samples in {split_name}")
                continue
            
            # Target count for this class based on train proportions
            target_count = max(1, int(round(train_proportions[cls] * target_total)))
            
            if target_count <= len(cls_base):
                # Sample without replacement (downsample from base)
                sampled = np.random.choice(cls_base, target_count, replace=False)
                duplication_stats['no_dup'] += 1
            else:
                # Need duplication
                duplication_factor = int(np.ceil(target_count / len(cls_base)))
                duplication_stats['max_dup_factor'] = max(duplication_stats['max_dup_factor'], duplication_factor)
                duplication_stats['duplicated'] += 1
                
                # Replicate base and sample
                duplicated_pool = np.tile(cls_base, duplication_factor)
                sampled = np.random.choice(duplicated_pool, target_count, replace=False)
            
            replicated_indices.extend(sampled.tolist())
        
        print(f"  {split_name}: {len(replicated_indices):,} samples "
              f"(no_dup: {duplication_stats['no_dup']}, "
              f"duplicated: {duplication_stats['duplicated']}, "
              f"max_factor: {duplication_stats['max_dup_factor']})")
        
        return replicated_indices
    
    val_indices = replicate_to_match_proportions(val_base_indices, target_val_total, "Val")
    tunev_indices = replicate_to_match_proportions(tunev_base_indices, target_tunev_total, "TuneV")
    test_indices = replicate_to_match_proportions(test_base_indices, target_test_total, "Test")
    
    # Get targets
    val_targets = test_targets[val_indices].tolist()
    tunev_targets = test_targets[tunev_indices].tolist()
    test_targets_final = test_targets[test_indices].tolist()
    
    # Verify disjoint property (by unique original indices)
    val_unique = set(val_base_indices)
    tunev_unique = set(tunev_base_indices)
    test_unique = set(test_base_indices)
    
    assert len(val_unique & tunev_unique) == 0, "Val and TuneV share base indices!"
    assert len(val_unique & test_unique) == 0, "Val and Test share base indices!"
    assert len(tunev_unique & test_unique) == 0, "TuneV and Test share base indices!"
    
    print("\n✅ Verified: All splits are disjoint by original CIFAR-100 indices")
    print("="*60)
    
    return val_indices, val_targets, tunev_indices, tunev_targets, test_indices, test_targets_final


def analyze_distribution(indices: List[int], targets: List[int], name: str, train_counts: Optional[List[int]] = None):
    """Analyze and print distribution statistics."""
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
    
    # Group analysis
    groups = {
        'Head (0-9)': sum(sorted_counts[0:10]),
        'Medium (10-49)': sum(sorted_counts[10:50]), 
        'Low (50-89)': sum(sorted_counts[50:90]),
        'Tail (90-99)': sum(sorted_counts[90:100])
    }
    
    print("Distribution by groups:")
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
                             base_indices: Optional[List[int]] = None) -> Dict:
    """Compute comprehensive statistics for a split."""
    target_counts = Counter(targets)
    class_counts = [target_counts.get(i, 0) for i in range(100)]
    
    total = len(indices)
    head_count = sum(class_counts[:10])
    medium_count = sum(class_counts[10:50])
    low_count = sum(class_counts[50:90])
    tail_count = sum(class_counts[90:100])
    
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
        'medium_samples': medium_count,
        'low_samples': low_count,
        'tail_samples': tail_count,
        'head_ratio': head_count / total if total > 0 else 0,
        'medium_ratio': medium_count / total if total > 0 else 0,
        'low_ratio': low_count / total if total > 0 else 0,
        'tail_ratio': tail_count / total if total > 0 else 0,
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
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax1.bar(split_names, totals, color=colors[:len(split_names)], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Samples', fontweight='bold')
    ax1.set_title('Total Samples per Split', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Head/Medium/Low/Tail distribution (stacked bar)
    ax2 = fig.add_subplot(gs[0, 1])
    head_vals = [all_stats[name]['head_samples'] for name in split_names]
    medium_vals = [all_stats[name]['medium_samples'] for name in split_names]
    low_vals = [all_stats[name]['low_samples'] for name in split_names]
    tail_vals = [all_stats[name]['tail_samples'] for name in split_names]
    
    x = np.arange(len(split_names))
    width = 0.6
    
    p1 = ax2.bar(x, head_vals, width, label='Head (0-9)', color='#3498db', alpha=0.8)
    p2 = ax2.bar(x, medium_vals, width, bottom=head_vals, label='Medium (10-49)', color='#2ecc71', alpha=0.8)
    p3 = ax2.bar(x, low_vals, width, bottom=np.array(head_vals)+np.array(medium_vals), 
                label='Low (50-89)', color='#f39c12', alpha=0.8)
    p4 = ax2.bar(x, tail_vals, width, 
                bottom=np.array(head_vals)+np.array(medium_vals)+np.array(low_vals),
                label='Tail (90-99)', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_title('Head/Medium/Low/Tail Distribution', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_names)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Proportion ratios (grouped bar)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(split_names))
    bar_width = 0.2
    
    head_ratios = [all_stats[name]['head_ratio'] for name in split_names]
    medium_ratios = [all_stats[name]['medium_ratio'] for name in split_names]
    low_ratios = [all_stats[name]['low_ratio'] for name in split_names]
    tail_ratios = [all_stats[name]['tail_ratio'] for name in split_names]
    
    ax3.bar(x - 1.5*bar_width, head_ratios, bar_width, label='Head', color='#3498db', alpha=0.8)
    ax3.bar(x - 0.5*bar_width, medium_ratios, bar_width, label='Medium', color='#2ecc71', alpha=0.8)
    ax3.bar(x + 0.5*bar_width, low_ratios, bar_width, label='Low', color='#f39c12', alpha=0.8)
    ax3.bar(x + 1.5*bar_width, tail_ratios, bar_width, label='Tail', color='#e74c3c', alpha=0.8)
    
    ax3.set_ylabel('Proportion', fontweight='bold')
    ax3.set_title('Group Proportions by Split', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(split_names)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
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
    print(f"\n📊 Saved comprehensive visualization: {plot_path}")
    
    # Also save as PDF for publication quality
    pdf_path = output_path / 'dataset_statistics_comprehensive.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"📊 Saved PDF version: {pdf_path}")
    
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
    print(f"📄 Saved summary statistics: {summary_path}")
    
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
    print(f"📄 Saved per-class distribution: {class_path}")
    
    return summary_path, class_path

def create_full_cifar100_lt_splits(
    imb_factor: float = 100,
    output_dir: str = "data/cifar100_lt_if100_splits", 
    val_ratio: float = 0.2,
    tunev_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create complete CIFAR-100-LT dataset splits with SPLIT-FIRST methodology.
    
    All splits (val, tuneV, test) have the same long-tail distribution as train.
    Guarantees zero leakage by splitting original indices first, then replicating.
    
    Following "Learning to Reject Meets Long-tail Learning" (Cao et al., 2024):
    - Train: Long-tail with exponential imbalance
    - Val/TuneV/Test: Match train's long-tail proportions via replication
    - All splits are disjoint by original CIFAR-100 test indices
    
    Args:
        imb_factor: Imbalance factor for training set (default 100)
        output_dir: Directory to save split indices
        val_ratio: Base proportion for validation (default 0.2 = 20% of original test)
        tunev_ratio: Base proportion for tuneV (default 0.15 = 15% of original test)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (datasets, splits) where datasets contains train/val/test/tunev Dataset objects
        and splits contains the indices for each split
    """
    print("=" * 60)
    print("CREATING CIFAR-100-LT DATASET SPLITS (SPLIT-FIRST)")  
    print("Following 'Learning to Reject Meets Long-tail Learning' methodology")
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
    
    # Now create the replicated splits
    val_indices, val_targets, tunev_indices, tunev_targets, test_indices, test_targets = \
        create_proportional_test_val_with_duplication(
            cifar_test, train_counts, val_ratio, tunev_ratio, seed
        )
    
    # 3. Analyze all distributions (console output)
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    analyze_distribution(train_indices, train_targets, "TRAIN")
    analyze_distribution(val_indices, val_targets, "VALIDATION", train_counts)  
    analyze_distribution(tunev_indices, tunev_targets, "TUNEV", train_counts)
    analyze_distribution(test_indices, test_targets, "TEST", train_counts)
    
    # 4. Compute detailed statistics for visualization
    print("\n" + "="*60)
    print("COMPUTING COMPREHENSIVE STATISTICS")
    print("="*60)
    
    all_stats = {
        'Train': compute_split_statistics(train_indices, train_targets, 'Train'),
        'Val': compute_split_statistics(val_indices, val_targets, 'Val', val_base_indices),
        'TuneV': compute_split_statistics(tunev_indices, tunev_targets, 'TuneV', tunev_base_indices),
        'Test': compute_split_statistics(test_indices, test_targets, 'Test', test_base_indices)
    }
    
    # Print summary
    for split_name, stats in all_stats.items():
        print(f"\n{split_name}:")
        print(f"  Total: {stats['total_samples']:,}")
        print(f"  IF: {stats['imbalance_factor']:.2f}")
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
        'train_indices': train_indices,
        'val_lt_indices': val_indices,
        'tuneV_indices': tunev_indices,
        'test_lt_indices': test_indices
    }
    
    save_splits_to_json(splits, output_dir)
    
    # 8. Create dataset objects with transforms
    train_transform, eval_transform = get_cifar100_transforms()
    
    datasets = {
        'train': CIFAR100LTDataset(cifar_train, train_indices, train_transform),
        'val': CIFAR100LTDataset(cifar_test, val_indices, eval_transform),
        'tunev': CIFAR100LTDataset(cifar_test, tunev_indices, eval_transform),
        'test': CIFAR100LTDataset(cifar_test, test_indices, eval_transform)
    }
    
    print("\n" + "=" * 60)
    print("✅ DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Key properties:")
    print("  ✓ All splits have same long-tail distribution as train")
    print("  ✓ Val, TuneV, and Test are disjoint by original CIFAR-100 indices")
    print("  ✓ Zero data leakage guaranteed")
    print("  ✓ Follows methodology from 'Learning to Reject Meets Long-tail Learning'")
    print("\n📁 Output files:")
    print(f"  • Visualizations: {plot_path}")
    print(f"  • Summary stats: {summary_path}")
    print(f"  • Per-class data: {class_path}")
    print(f"  • Split indices: {output_dir}/*.json")
    print("=" * 60)
    
    return datasets, splits

if __name__ == "__main__":
    # Create the full dataset
    datasets, splits = create_full_cifar100_lt_splits(
        imb_factor=100,
        output_dir="data/cifar100_lt_if100_splits",
        val_ratio=0.2,
        tunev_ratio=0.15,
        seed=42
    )
    
    print("\nDatasets ready for training:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset):,} samples")