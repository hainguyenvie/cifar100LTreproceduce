# CIFAR-100-LT Data Splitting Methodology

## Overview

This document explains how we create long-tail splits for CIFAR-100 following the methodology from **"Learning to Reject Meets Long-tail Learning" (Cao et al., 2024)** and related long-tail learning research.

## Core Principle: SPLIT-FIRST, Then Replicate

**Critical for avoiding data leakage:** We split the original CIFAR-100 test set into disjoint subsets FIRST, then replicate within each subset independently to match the training distribution.

## Methodology

### Step 1: Create Long-Tail Training Set

- Start with CIFAR-100 train set (50,000 samples, balanced)
- Apply exponential imbalance: `n_i = n_max * (IF)^(-i/(C-1))`
  - `n_max = 500` (head class)
  - `IF = 100` (imbalance factor)
  - Results in ~5 samples per tail class
- Sample without replacement for each class

**Output:** `train_indices.json` (~12,000 samples, long-tail)

### Step 2: Split Original Test Indices (Disjoint)

Start with CIFAR-100 test (10,000 samples, 100 per class, balanced):

For each class, split the 100 original indices into three disjoint groups:
- **Val base:** 20 indices (20%)
- **TuneV base:** 15 indices (15%)
- **Test base:** 65 indices (65%)

**Key property:** These base sets are completely disjoint by original CIFAR-100 indices.

### Step 3: Replicate Within Each Split

For each split independently:
1. Compute target counts based on train proportions
   - Example targets: Val ~15% of train size, TuneV ~12%, Test ~20%
2. For each class:
   - If base samples ≥ target: downsample (no duplication)
   - If base samples < target: replicate base indices to reach target
3. Sample from the (possibly replicated) pool

**Key property:** Replication happens within each split, so duplicates never cross split boundaries.

### Final Splits

All splits share the same long-tail distribution as training:

| Split | Base Indices | Target Size | Distribution |
|-------|--------------|-------------|--------------|
| **Train** | From CIFAR train | ~12,000 | Long-tail (IF=100) |
| **Val** | 20 per class from test | ~15% of train | Matches train LT |
| **TuneV** | 15 per class from test | ~12% of train | Matches train LT |
| **Test** | 65 per class from test | ~20% of train | Matches train LT |

## Leakage Prevention

### Why This Works

1. **Disjoint base sets:** Original CIFAR-100 test indices are split first into non-overlapping groups
2. **Independent replication:** Each split replicates only its own base indices
3. **No cross-contamination:** Even with duplication, the same original image never appears in multiple splits

### Verification

The code includes assertions to verify:
```python
assert len(val_unique & tunev_unique) == 0, "Val and TuneV share base indices!"
assert len(val_unique & test_unique) == 0, "Val and Test share base indices!"
assert len(tunev_unique & test_unique) == 0, "TuneV and Test share base indices!"
```

## Why Long-Tail Test Sets?

### Research Justification

From **"Learning to Reject Meets Long-tail Learning"**:
> "The train, test and validations. tunev samples all follow the same long-tailed label distributions. We re-weight the samples in the original test set to replicate the same label proportions as the training set."

### Rationale

1. **Real-world alignment:** In practice, test distributions often match training (deployment matches historical data)
2. **Fair evaluation:** Methods that improve tail performance should be evaluated on tail-heavy test sets
3. **Worst-group metrics:** Long-tail test sets are essential for measuring worst-group error and coverage

## Comparison: Old vs New

### Old Approach (❌ Leakage Risk)

```
1. Duplicate CIFAR test indices to match train proportions
2. Split the duplicated pool into val and test
3. Remove tuneV from test

Problem: Same original image can appear in val, tuneV, and test
```

### New Approach (✅ No Leakage)

```
1. Split original test indices into disjoint val/tuneV/test bases
2. Replicate within each base independently to match train proportions
3. All splits have LT distribution, but are disjoint by original indices
```

## Usage

### Creating Splits

```python
from src.data.enhanced_datasets import create_full_cifar100_lt_splits

datasets, splits = create_full_cifar100_lt_splits(
    imb_factor=100,           # Training imbalance factor
    output_dir="data/cifar100_lt_if100_splits",
    val_ratio=0.2,            # 20% of original test for val base
    tunev_ratio=0.15,         # 15% of original test for tuneV base
    seed=42
)
```

### Output Files

- `data/cifar100_lt_if100_splits/train_indices.json`
- `data/cifar100_lt_if100_splits/val_lt_indices.json`
- `data/cifar100_lt_if100_splits/tuneV_indices.json`
- `data/cifar100_lt_if100_splits/test_lt_indices.json`

### Using in Training Pipeline

```python
# Step 1: Create splits
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"

# Step 2: Train experts on train split
python -m src.train.train_expert

# Step 3: Train gating on tuneV split
python -m src.train.train_gating_only --mode pretrain

# Step 4: Selective training using tuneV (S1) and val_lt (S2)
python -m src.train.train_gating_only --mode selective

# Step 5: Final evaluation on test_lt split
python -m src.train.eval_gse_plugin
```

## References

1. Cao, K., et al. (2024). "Learning to Reject Meets Long-tail Learning"
2. Ren, J., et al. (2020). "Balanced Meta-Softmax for Long-Tailed Visual Recognition"
3. Menon, A., et al. (2021). "Long-tail Learning via Logit Adjustment"

## Key Takeaways

✅ **All splits have same long-tail distribution as training**  
✅ **Val, TuneV, and Test are disjoint by original CIFAR-100 indices**  
✅ **Zero data leakage guaranteed**  
✅ **Follows established methodology from long-tail learning research**

