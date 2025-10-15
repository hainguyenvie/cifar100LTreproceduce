# Pipeline Alignment Summary

## ✅ All Scripts Are Now Aligned!

All scripts in your pipeline now correctly reference the new dataset paths and expert names.

---

## Complete Pipeline Commands

```bash
# Step 0: Create dataset splits with visualizations
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"

# Step 1: Train 3 experts and export logits
python -m src.train.train_expert

# Step 2: Pretrain gating network (mixture CE)
python -m src.train.train_gating_only --mode pretrain

# Step 3: Selective gating training (with Pinball loss)
python -m src.train.train_gating_only --mode selective

# Step 4: Plugin optimization (worst-group EG-outer)
python run_improved_eg_outer.py

# Step 5: Evaluate final model
python -m src.train.eval_gse_plugin
```

---

## Data Flow

### Step 0: Dataset Creation
**Script:** `src/data/enhanced_datasets.py`

**Creates:**
```
data/cifar100_lt_if100_splits/
├── train_indices.json          # Long-tail training set
├── val_lt_indices.json         # Validation (matches train LT distribution)
├── tuneV_indices.json          # Gating training (matches train LT distribution)
├── test_lt_indices.json        # Test set (matches train LT distribution)
├── dataset_statistics_comprehensive.png  # Visualization
├── dataset_statistics_comprehensive.pdf  # Publication quality
├── split_summary_statistics.csv         # Summary stats
└── per_class_distribution.csv           # Per-class details
```

**Key Properties:**
- ✅ All splits have same long-tail distribution (IF=100)
- ✅ Val, TuneV, Test are disjoint by original CIFAR-100 indices
- ✅ Zero data leakage guaranteed
- ✅ Comprehensive visualizations and statistics

---

### Step 1: Expert Training
**Script:** `src/train/train_expert.py`

**Trains 3 Experts:**
1. `ce_baseline` - Standard Cross-Entropy
2. `logitadjust_baseline` - Logit Adjustment (tail-friendly)
3. `balsoftmax_baseline` - Balanced Softmax (tail-friendly)

**Exports Logits to:**
```
outputs/logits/cifar100_lt_if100/
├── ce_baseline/
│   ├── train_logits.pt
│   ├── tuneV_logits.pt
│   ├── val_lt_logits.pt
│   └── test_lt_logits.pt
├── logitadjust_baseline/
│   ├── train_logits.pt
│   ├── tuneV_logits.pt
│   ├── val_lt_logits.pt
│   └── test_lt_logits.pt
└── balsoftmax_baseline/
    ├── train_logits.pt
    ├── tuneV_logits.pt
    ├── val_lt_logits.pt
    └── test_lt_logits.pt
```

**Configuration:**
```python
CONFIG = {
    'dataset': {
        'splits_dir': './data/cifar100_lt_if100_splits',  ✅
    },
    'output': {
        'logits_dir': './outputs/logits',  ✅
    }
}

EXPERT_CONFIGS = {
    'ce': {'name': 'ce_baseline', ...},                    ✅
    'logitadjust': {'name': 'logitadjust_baseline', ...}, ✅
    'balsoftmax': {'name': 'balsoftmax_baseline', ...},   ✅
}
```

---

### Step 2: Gating Pretrain
**Script:** `src/train/train_gating_only.py --mode pretrain`

**Reads:**
- Logits from: `outputs/logits/cifar100_lt_if100/{ce,logitadjust,balsoftmax}_baseline/tuneV_logits.pt`

**Trains:**
- Gating network only (α=1, μ=0 fixed)
- Optimizes mixture cross-entropy on tuneV split

**Saves:**
```
checkpoints/gating_pretrained/cifar100_lt_if100/
└── gating_pretrained.ckpt
```

**Configuration:**
```python
CONFIG = {
    'dataset': {
        'splits_dir': './data/cifar100_lt_if100_splits',  ✅
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],  ✅
        'logits_dir': './outputs/logits/',  ✅
    }
}
```

---

### Step 3: Selective Gating
**Script:** `src/train/train_gating_only.py --mode selective`

**Reads:**
- tuneV logits (S1) for training
- val_lt logits (S2) for μ sweep

**Trains:**
- Selective gating with per-group learnable thresholds (Pinball loss)
- Updates α via fixed-point iteration
- Sweeps μ on validation set
- Stage A: Warm-up
- Stage B: Alternating optimization (gating → α → μ)

**Saves:**
```
checkpoints/gating_pretrained/cifar100_lt_if100/
└── gating_selective.ckpt
    (contains: gating_net, alpha, mu, t_param, temperatures, cycle_logs)
```

**Configuration:**
```python
CONFIG = {
    'dataset': {
        'splits_dir': './data/cifar100_lt_if100_splits',  ✅
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],  ✅
        'logits_dir': './outputs/logits/',  ✅
    }
}
```

---

### Step 4: Plugin Optimization
**Script:** `run_improved_eg_outer.py` → calls `src/train/gse_balanced_plugin.py`

**Reads:**
- val_lt logits (S1) for optimization
- Pretrained gating checkpoint (optional, can train from scratch)

**Optimizes:**
- α* and μ* for worst-group error minimization
- Uses EG-outer with all improvements:
  - Anti-collapse β with floor and momentum
  - Reduced step size (xi=0.2)
  - Error centering
  - Early stopping
  - Blended alpha updates

**Saves:**
```
checkpoints/argse_worst_eg_improved/cifar100_lt_if100/
└── gse_balanced_plugin.ckpt
    (contains: gating_net, alpha*, mu*, t* or t_group*, class_to_group)
```

**Configuration:**
```python
CONFIG = {
    'dataset': {
        'splits_dir': './data/cifar100_lt_if100_splits',  ✅
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],  ✅
        'logits_dir': './outputs/logits/',  ✅
    }
}
```

---

### Step 5: Evaluation
**Script:** `src/train/eval_gse_plugin.py`

**Reads:**
- test_lt logits
- Plugin checkpoint with optimal (α*, μ*, t*)

**Evaluates:**
- Risk-Coverage (RC) curves
- AURC (Area Under Risk-Coverage curve)
  - Full range [0, 1]
  - Practical range [0.2, 1.0]
- Metrics at fixed coverages
- ECE (Expected Calibration Error)
- Selective risk at different rejection costs
- Per-group performance analysis

**Saves:**
```
results_worst_eg_improved/cifar100_lt_if100/
├── metrics.json                      # All metrics
├── rc_curve.csv                      # Risk-coverage data
├── rc_curve_02_10.csv               # RC data (0.2-1.0 range)
├── aurc_detailed_results.csv        # AURC sweep results
├── aurc_curves.png                  # AURC visualization
└── rc_curve_comparison.(png|pdf)    # RC curves
```

**Configuration:**
```python
CONFIG = {
    'dataset': {
        'splits_dir': './data/cifar100_lt_if100_splits',  ✅
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],  ✅
        'logits_dir': './outputs/logits',  ✅
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',  ✅
}
```

---

## Key Alignments Fixed

### 1. Dataset Paths ✅
All scripts now use:
```python
'splits_dir': './data/cifar100_lt_if100_splits'
```

### 2. Expert Names ✅
All scripts now use the same 3 experts:
```python
'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
```

**Previous issues (FIXED):**
- ❌ Some scripts had `['logitadjust_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']` (duplicate)
- ✅ Now all use `['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']`

### 3. Split Files ✅
Removed references to non-existent splits:
- ❌ `val_small_indices.json` (removed)
- ❌ `calib_indices.json` (removed)

Only uses existing splits:
- ✅ `train_indices.json`
- ✅ `tuneV_indices.json`
- ✅ `val_lt_indices.json`
- ✅ `test_lt_indices.json`

### 4. Logits Directory ✅
All scripts use:
```python
'logits_dir': './outputs/logits'
```

---

## Quick Verification

Run this to verify all paths exist after dataset creation:

```bash
# Check dataset splits exist
ls data/cifar100_lt_if100_splits/*.json

# Expected output:
# train_indices.json
# tuneV_indices.json
# val_lt_indices.json
# test_lt_indices.json
```

After running `train_expert`:

```bash
# Check expert logits exist
ls outputs/logits/cifar100_lt_if100/*/tuneV_logits.pt

# Expected output:
# outputs/logits/cifar100_lt_if100/ce_baseline/tuneV_logits.pt
# outputs/logits/cifar100_lt_if100/logitadjust_baseline/tuneV_logits.pt
# outputs/logits/cifar100_lt_if100/balsoftmax_baseline/tuneV_logits.pt
```

---

## Summary

✅ **All 4 pipeline scripts are now aligned:**
1. `train_expert.py` - Exports 3 experts with correct names
2. `train_gating_only.py` - Uses all 3 experts correctly
3. `gse_balanced_plugin.py` (via `run_improved_eg_outer.py`) - Uses all 3 experts
4. `eval_gse_plugin.py` - Uses all 3 experts

✅ **Dataset paths match across all scripts**

✅ **No references to non-existent splits**

✅ **Expert names consistent throughout pipeline**

You can now run the complete pipeline without path or configuration mismatches!

