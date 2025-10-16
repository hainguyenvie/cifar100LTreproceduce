# Evaluation Reweighting - Implementation Status

## ✅ What's Been Implemented

### 1. Dataset Creation (COMPLETE)
**File:** `src/data/enhanced_datasets.py`

```python
create_full_cifar100_lt_splits(use_evaluation_reweighting=True)
```

**Creates:**
- **Balanced splits:** Val (2,000), TuneV (1,500), Test (6,500) - all unique, 100% data preserved
- **train_class_weights.json:** Class frequency weights for weighted loss/metrics

**Result:**
```
Train:  10,847 samples, IF=100  (long-tail)
Val:     2,000 samples, IF=1.0  (balanced, 20/class)
TuneV:   1,500 samples, IF=1.0  (balanced, 15/class)
Test:    6,500 samples, IF=1.0  (balanced, 65/class)
```

### 2. Reweighting Utilities (COMPLETE)
**File:** `src/data/reweighting_utils.py` (NEW)

**Functions:**
- `load_train_class_weights()` - Load weights from JSON
- `get_sample_weights(labels, class_weights)` - Get per-sample weights
- `weighted_cross_entropy_loss()` - Weighted CE loss
- `weighted_accuracy()` - Weighted accuracy metric
- `compute_weighted_group_metrics()` - Head/tail weighted metrics
- `apply_weighted_selective_metrics()` - Weighted selective classification metrics

### 3. Gating Pretrain (COMPLETE)
**File:** `src/train/train_gating_only.py` (pretrain mode)

**Changes:**
- Loads `train_class_weights.json`
- Applies weighted loss during training
- Works on balanced tuneV (1,500 samples)

**Usage:**
```bash
python -m src.train.train_gating_only --mode pretrain
```

---

## ⏳ What Still Needs Modification

### 4. Selective Gating (TODO)
**File:** `src/train/train_gating_only.py` (selective mode)

**Needed changes:**
- Load class weights in `run_selective_mode()`
- Apply weights in selective loss functions
- Update coverage/error metrics to use weighted versions

### 5. Plugin Optimization (TODO)
**File:** `src/train/gse_balanced_plugin.py`

**Needed changes:**
- Load class weights
- Apply weights during μ/α optimization  
- Use weighted error for worst-group objective

### 6. Evaluation (TODO)
**File:** `src/train/eval_gse_plugin.py`

**Needed changes:**
- Load class weights
- Compute weighted AURC
- Report both weighted and unweighted metrics
- Add weighted RC curves

---

## 🎯 Your Options

### Option A: Complete Full Implementation (Recommended)
I continue modifying scripts 4-6 to fully support evaluation reweighting across entire pipeline.

**Pros:**
- ✅ Complete solution
- ✅ Consistent methodology
- ✅ Both weighted and unweighted metrics

**Time:** ~10-15 more modifications

### Option B: Hybrid Approach
Keep current implementation (pretrain done), run rest of pipeline with physical replication.

**Pros:**
- ✅ Can start running now
- ✅ Less testing needed

**Cons:**
- ⚠️ Inconsistent (pretrain weighted, rest not)

### Option C: Switch Back to Physical Replication
Revert to physical replication for entire pipeline.

**Pros:**
- ✅ Simpler (no weight handling)
- ✅ Already mostly implemented

**Cons:**
- ⚠️ Less data efficient
- ⚠️ Duplicates in test set

---

## 📋 Recommended Next Steps

I recommend **Option A** - let me complete the full implementation. Here's what I'll do:

1. ✅ Modify `run_selective_mode()` in `train_gating_only.py`
2. ✅ Modify `gse_balanced_plugin.py` for weighted optimization
3. ✅ Modify `eval_gse_plugin.py` for weighted evaluation
4. ✅ Add config flags for easy switching between modes
5. ✅ Create comparison script showing both weighted/unweighted results

**Estimated:** 30 minutes of modifications

**Result:** Complete pipeline supporting evaluation reweighting with proper weighted metrics throughout!

---

## 🚀 Current Status

**Ready to use NOW:**
```bash
# These work with evaluation reweighting:
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits(use_evaluation_reweighting=True)"
python -m src.train.train_expert
python -m src.train.train_gating_only --mode pretrain  ← Weighted loss applied!
```

**Need modifications:**
```bash
# These need updates for evaluation reweighting:
python -m src.train.train_gating_only --mode selective  ← TODO
python run_improved_eg_outer.py  ← TODO
python -m src.train.eval_gse_plugin  ← TODO
```

---

## 💬 Your Decision

**Would you like me to:**

**A)** Complete the full evaluation reweighting implementation (modify remaining 3 scripts)?

**B)** Keep hybrid (pretrain weighted, rest using physical replication)?

**C)** Switch everything back to physical replication?

Let me know and I'll proceed accordingly!

