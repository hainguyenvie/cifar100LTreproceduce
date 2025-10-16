# Evaluation Reweighting Implementation Guide

## 🎯 What is Evaluation Reweighting?

Instead of physically replicating samples to match training distribution, we:

1. **Keep all datasets BALANCED** (use all unique samples, no duplication)
2. **Apply class frequency weights** during training and evaluation
3. **Compute weighted metrics** that reflect the long-tail distribution

---

## 📊 Dataset Changes (Evaluation Reweighting Mode)

### Created When You Run:
```python
create_full_cifar100_lt_splits(use_evaluation_reweighting=True)
```

### Splits Created (All BALANCED)

| Split | Samples | Samples/Class | IF | Unique |
|-------|---------|---------------|-----|--------|
| Train | 10,847 | Variable (5-500) | 100.0 | All unique (LT sampled) |
| Val | 2,000 | 20 each | 1.0 | All unique (balanced) |
| TuneV | 1,500 | 15 each | 1.0 | All unique (balanced) |
| Test | 6,500 | 65 each | 1.0 | All unique (balanced) |

### Key File: `train_class_weights.json`
```json
{
  "class_counts": [500, 497, ..., 5],  // Train counts per class
  "class_weights": [0.0461, 0.0458, ..., 0.0005],  // Proportions (sum to 1.0)
  "mode": "evaluation_reweighting",
  "head_classes": [0, 1, ..., 68],  // 69 classes with >20 samples
  "tail_classes": [69, 70, ..., 99]  // 31 classes with <=20 samples
}
```

---

## 🔧 How to Apply Weights (Step by Step)

### 1️⃣ **Gating Training** (`train_gating_only.py`)

**Current:** Uses balanced tuneV (1,500 samples)
**Modification:** Apply class frequency weights to loss

```python
# Load weights
from src.data.reweighting_utils import load_train_class_weights, weighted_cross_entropy_loss

weights_info = load_train_class_weights()
class_weights = weights_info['class_weights']

# In training loop:
for expert_logits, labels in train_loader:
    # Compute gating weights
    w = model.gating_net(...)
    
    # Apply weighted loss
    loss = weighted_cross_entropy_loss(
        mixture_logits, labels, class_weights, reduction='mean'
    )
    loss.backward()
```

**Effect:** Gating learns to handle LT distribution even though data is balanced

---

### 2️⃣ **Plugin Optimization** (`gse_balanced_plugin.py`)

**Current:** Uses balanced val (2,000 samples)
**Modification:** Apply weights during μ/α optimization

```python
# Load weights
weights_info = load_train_class_weights()
class_weights = weights_info['class_weights']

# During error computation:
sample_weights = get_sample_weights(labels, class_weights)

# Weighted error on accepted samples
weighted_error = (wrong_preds * accepted * sample_weights).sum() / (accepted * sample_weights).sum()
```

**Effect:** Optimization targets weighted error (reflects real deployment)

---

### 3️⃣ **Evaluation** (`eval_gse_plugin.py`)

**Current:** Uses balanced test (6,500 samples)
**Modification:** Compute weighted metrics

```python
# Load weights
weights_info = load_train_class_weights()
class_weights = weights_info['class_weights']

# Compute weighted accuracy
from src.data.reweighting_utils import compute_weighted_group_metrics

metrics = compute_weighted_group_metrics(
    preds, labels, class_weights, 
    weights_info['head_classes'],
    weights_info['tail_classes']
)

print(f"Weighted Accuracy: {metrics['weighted_acc']:.4f}")
print(f"Weighted Head Acc: {metrics['weighted_head_acc']:.4f}")  
print(f"Weighted Tail Acc: {metrics['weighted_tail_acc']:.4f}")

# Also report unweighted for comparison
print(f"Unweighted Accuracy: {metrics['unweighted_acc']:.4f}")
```

**Effect:** Metrics reflect expected performance on LT deployment distribution

---

## 📈 Mathematical Formulation

### Sample Weights

For a sample with label y:
```
w_y = n_y / N

where:
  n_y = number of samples of class y in training set
  N = total training samples
```

Example:
```
Class 0 (head): n_0 = 500, N = 10,847
  → w_0 = 500 / 10,847 = 0.0461

Class 99 (tail): n_99 = 5, N = 10,847
  → w_99 = 5 / 10,847 = 0.0005

Weight ratio: w_0 / w_99 = 100 (same as IF)
```

### Weighted Loss

```
L_weighted = (1 / Σw_i) * Σ(w_i * loss_i)

where:
  w_i = weight of sample i (based on its class)
  loss_i = cross-entropy loss for sample i
```

### Weighted Metrics

```
Accuracy_weighted = Σ(w_i * correct_i) / Σw_i

Error_weighted = 1 - Accuracy_weighted
```

---

## 🔄 Complete Pipeline Modifications

I'll now modify all your scripts to support evaluation reweighting. The changes will be:

### `train_expert.py`
- ✅ No changes needed (trains on LT train set as before)
- Exports logits on balanced val/tuneV/test

### `train_gating_only.py`
- Load train class weights
- Apply weighted loss during gating training
- Use balanced tuneV (1,500 samples)

### `gse_balanced_plugin.py`
- Load train class weights
- Apply weights during α/μ optimization
- Use balanced val (2,000 samples)

### `eval_gse_plugin.py`
- Load train class weights
- Compute weighted AURC, weighted accuracy
- Report both weighted and unweighted metrics
- Use balanced test (6,500 samples)

---

## ✅ Benefits of This Approach

| Aspect | Physical Replication | Evaluation Reweighting |
|--------|---------------------|----------------------|
| **Data usage** | Some duplication/waste | ALL unique samples used |
| **Val size** | 2,169 (with duplicates) | 2,000 (all unique) |
| **TuneV size** | 2,169 (with duplicates) | 1,500 (all unique) |
| **Test size** | 10,847 (with duplicates) | 6,500 (all unique) |
| **Tail diversity** | Limited (few unique) | Maximum (all available) |
| **Implementation** | Simple (just load data) | Moderate (apply weights) |
| **Flexibility** | Fixed distribution | Can adjust weights easily |
| **Research standard** | Used in some papers | **More common currently** |

---

## 🚀 Next Steps

I will now modify your pipeline scripts to implement evaluation reweighting. The modifications will:

1. ✅ Create `src/data/reweighting_utils.py` - utility functions (DONE)
2. ⏳ Modify `train_gating_only.py` - use weighted loss
3. ⏳ Modify `gse_balanced_plugin.py` - use weighted optimization
4. ⏳ Modify `eval_gse_plugin.py` - compute weighted metrics
5. ⏳ Create example showing both weighted and unweighted results

Ready to proceed with modifications?

