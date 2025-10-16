# ✅ Pipeline Ready to Run - Complete Guide

## 🎯 Dataset Configuration Summary

Your dataset is now correctly configured with **TWO test sets**:

### Long-Tail Splits (Main Pipeline)
```
Train:    10,847 samples, IF=100 (LT distribution)
Val-LT:    2,169 samples, IF=100 (20% of train, 1.08x duplication)
TuneV-LT:  2,169 samples, IF=100 (20% of train, 1.45x duplication)
Test-LT:  10,847 samples, IF=100 (100% of train, 1.67x duplication) ← Main evaluation
```

### Balanced Test (Supplementary)
```
Test-Balanced: 6,500 samples, IF=1.0 (all unique, 65/class) ← Additional analysis
```

### Head/Tail Definition (Threshold=20)
```
Head: 69 classes with >20 samples in train  (97.0% of data)
Tail: 31 classes with ≤20 samples in train  (3.0% of data)
```

---

## 🚀 Complete Pipeline Commands

```bash
# Step 0: Create dataset splits
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"

# Step 1: Train 3 experts and export logits
python -m src.train.train_expert

# Step 2: Pretrain gating network
python -m src.train.train_gating_only --mode pretrain

# Step 3: Selective gating training
python -m src.train.train_gating_only --mode selective

# Step 4: Plugin optimization
python run_improved_eg_outer.py

# Step 5: Evaluate on Test-LT
python -m src.train.eval_gse_plugin
```

---

## 📁 Files Created by Step 0

### JSON Indices (5 files)
```
data/cifar100_lt_if100_splits/
├── train_indices.json           (10,847 indices)
├── val_lt_indices.json          (2,169 indices)
├── tuneV_indices.json           (2,169 indices)
├── test_lt_indices.json         (10,847 indices) ← Main test
└── test_balanced_indices.json   (6,500 indices)  ← Supplementary test
```

### Visualizations & Statistics
```
data/cifar100_lt_if100_splits/
├── dataset_statistics_comprehensive.png  (visualization with all 5 splits)
├── dataset_statistics_comprehensive.pdf  (publication quality)
├── split_summary_statistics.csv
└── per_class_distribution.csv
```

---

## 🔧 What Each Script Does

### Step 1: `train_expert.py`
**Reads:**
- `train_indices.json` (for training)
- `val_lt_indices.json` (for validation)
- `tuneV_indices.json`, `test_lt_indices.json` (for logit export)

**Exports logits for all 4 main splits:**
```
outputs/logits/cifar100_lt_if100/
├── ce_baseline/{train,tuneV,val_lt,test_lt}_logits.pt
├── logitadjust_baseline/{train,tuneV,val_lt,test_lt}_logits.pt
└── balsoftmax_baseline/{train,tuneV,val_lt,test_lt}_logits.pt
```

**Expected sizes:**
- train_logits.pt: [10847, 100]
- tuneV_logits.pt: [2169, 100]
- val_lt_logits.pt: [2169, 100]
- test_lt_logits.pt: [10847, 100]

### Step 2: `train_gating_only.py --mode pretrain`
**Reads:**
- `tuneV_logits.pt` from all 3 experts

**Trains:**
- Gating network with mixture CE loss
- Uses tuneV split (2,169 samples, IF=100)

**Saves:**
- `checkpoints/gating_pretrained/cifar100_lt_if100/gating_pretrained.ckpt`

### Step 3: `train_gating_only.py --mode selective`
**Reads:**
- `tuneV_logits.pt` (S1 for training)
- `val_lt_logits.pt` (S2 for μ sweep)

**Trains:**
- Selective gating with Pinball loss
- Learns per-group thresholds
- Updates α and μ

**Saves:**
- `checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt`

### Step 4: `run_improved_eg_outer.py`
**Reads:**
- `val_lt_logits.pt` for optimization
- Optionally loads pretrained gating

**Optimizes:**
- α* and μ* using worst-group EG-outer
- Learns optimal threshold t*

**Saves:**
- `checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt`

### Step 5: `eval_gse_plugin.py`
**Reads:**
- `test_lt_logits.pt` (10,847 samples)
- Plugin checkpoint (α*, μ*, t*)

**Evaluates:**
- Risk-Coverage curves
- AURC metrics
- Per-group performance
- Saves plots and metrics

**Outputs:**
- `results_worst_eg_improved/cifar100_lt_if100/` (metrics, plots)

---

## ✅ All Scripts Are Aligned

I've verified that all scripts correctly use:

### Dataset Paths ✓
```python
'splits_dir': './data/cifar100_lt_if100_splits'
```

### Expert Names ✓
```python
'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
```

### Split Files ✓
- `train_indices.json` (10,847, IF=100)
- `val_lt_indices.json` (2,169, IF=100)
- `tuneV_indices.json` (2,169, IF=100)
- `test_lt_indices.json` (10,847, IF=100) ← **Main test**
- `test_balanced_indices.json` (6,500, IF=1.0) ← Optional

### Logit Sizes ✓
All scripts expect:
- tuneV: 2,169 samples
- val_lt: 2,169 samples
- test_lt: 10,847 samples

---

## 🔍 Key Changes from Previous Version

### Test Set Sizing
**Before:** Test had 2,169 samples (0.33x downsampling) ❌
**Now:** Test has 10,847 samples (1.67x replication) ✅

**Why this matters:**
- Uses ALL available test data efficiently
- Same size as training set (fair comparison)
- No wasteful downsampling
- Follows "Learning to Reject Meets Long-tail Learning" methodology

### Dataset Structure
**Before:** 4 splits (train, val_lt, tuneV, test_lt)
**Now:** 5 splits (+ test_balanced for robustness)

### Head/Tail Definition
**Before:** Arbitrary index-based (0-49 head, 50-99 tail)
**Now:** Threshold-based (>20 samples = head, ≤20 = tail)
- Head: 69 classes (97% of data)
- Tail: 31 classes (3% of data)

---

## 📊 Expected Behavior

### After Step 0 (Dataset Creation)
```bash
$ ls data/cifar100_lt_if100_splits/*.json
train_indices.json          # 10,847 samples
val_lt_indices.json         # 2,169 samples
tuneV_indices.json          # 2,169 samples
test_lt_indices.json        # 10,847 samples ← NOTE: Same size as train!
test_balanced_indices.json  # 6,500 samples
```

### After Step 1 (Expert Training)
```bash
$ ls outputs/logits/cifar100_lt_if100/*/test_lt_logits.pt
# Should see 3 files, each with shape [10847, 100]
```

### During Step 2-3 (Gating Training)
```
Loaded tuneV: 2,169 samples ✓
Loaded val_lt: 2,169 samples ✓
```

### During Step 5 (Evaluation)
```
Loaded test_lt: 10,847 samples ✓
Test coverage: ~0.X
```

---

## ⚠️ Important Notes

### 1. All Logits Must Be Re-exported
Since test_lt changed from 2,169 → 10,847 samples, you MUST re-run:
```bash
python -m src.train.train_expert
```

**Old logits will cause errors:**
```
RuntimeError: Size mismatch - expected [10847, 100], got [2169, 100]
```

### 2. Batch Size Consideration
Current config has `batch_size=2` which is very small. Consider updating:

```python
# In train_gating_only.py CONFIG
'gating_params': {
    'batch_size': 64,  # Increase from 2 for better training
    ...
}
```

### 3. Test Set to Use
- **For paper comparison:** Use `test_lt` (10,847 samples, IF=100)
- **For additional analysis:** Use `test_balanced` (6,500 samples, IF=1.0)

---

## ✅ Verification Checklist

Before running the pipeline, verify:

- [ ] Dataset splits created (5 JSON files)
- [ ] Visualization shows all splits with IF=100 (except test_balanced)
- [ ] Test-LT has 10,847 samples (not 2,169)
- [ ] Test-LT duplication is 1.67x (not 0.33x)
- [ ] Head/Tail shows 97% / 3% split for all LT sets

After Step 1 (expert training):

- [ ] 3 experts trained successfully
- [ ] Logits exported for all 4 splits
- [ ] test_lt_logits.pt has shape [10847, 100]

---

## 🎯 Ready to Run!

All scripts are aligned. Your pipeline will now:

1. ✅ Create splits with correct LT distribution (IF=100)
2. ✅ Train experts on 10,847 train samples
3. ✅ Export logits with correct sizes
4. ✅ Train gating on 2,169 tuneV samples
5. ✅ Optimize on 2,169 val_lt samples
6. ✅ Evaluate on 10,847 test_lt samples ← Full-size test!
7. ✅ Optional: Also evaluate on 6,500 balanced test

**No more size mismatches!**
**No more downsampling waste!**
**Ready for fair comparison with papers!** 🚀

---

## 📖 Documentation Reference

- `REWEIGHTING_METHODOLOGY_EXPLAINED.md` - How "reweighting" works
- `DATA_SPLITTING_METHODOLOGY.md` - Technical methodology
- `DATA_FIXES_SUMMARY.md` - What was fixed
- `VISUALIZATION_OUTPUT_GUIDE.md` - Understanding the plots
- `QUICK_PIPELINE_REFERENCE.md` - Quick commands reference

