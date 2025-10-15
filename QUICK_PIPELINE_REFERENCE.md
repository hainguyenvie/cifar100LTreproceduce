# Quick Pipeline Reference

## Run Commands (In Order)

```bash
# 0️⃣ Create dataset splits
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"

# 1️⃣ Train experts & export logits
python -m src.train.train_expert

# 2️⃣ Pretrain gating
python -m src.train.train_gating_only --mode pretrain

# 3️⃣ Selective gating
python -m src.train.train_gating_only --mode selective

# 4️⃣ Plugin optimization
python run_improved_eg_outer.py

# 5️⃣ Evaluate
python -m src.train.eval_gse_plugin
```

---

## What Each Step Does

| Step | Script | Input | Output | Time |
|------|--------|-------|--------|------|
| **0** | `enhanced_datasets.py` | CIFAR-100 raw | 4 JSON splits + plots | ~1 min |
| **1** | `train_expert.py` | train/tuneV/val/test splits | 3 experts × 4 splits logits | ~2-4 hrs |
| **2** | `train_gating_only.py --mode pretrain` | tuneV logits | Pretrained gating | ~5-10 min |
| **3** | `train_gating_only.py --mode selective` | tuneV + val_lt logits | Selective gating checkpoint | ~15-30 min |
| **4** | `run_improved_eg_outer.py` | val_lt logits | Plugin checkpoint (α*, μ*, t*) | ~30-60 min |
| **5** | `eval_gse_plugin.py` | test_lt logits + plugin ckpt | Metrics + plots | ~5 min |

---

## Key Outputs

### After Step 0 (Dataset)
```
data/cifar100_lt_if100_splits/
├── *.json (4 splits)
└── dataset_statistics_comprehensive.png  👀 VIEW THIS
```

### After Step 1 (Experts)
```
outputs/logits/cifar100_lt_if100/
├── ce_baseline/*.pt
├── logitadjust_baseline/*.pt
└── balsoftmax_baseline/*.pt
```

### After Step 5 (Evaluation)
```
results_worst_eg_improved/cifar100_lt_if100/
├── metrics.json                    👀 KEY RESULTS
├── aurc_curves.png                 👀 VIEW THIS
└── rc_curve_comparison.pdf         👀 FOR PAPER
```

---

## Quick Checks

**After each step, verify:**

```bash
# Step 0: Check splits created
ls data/cifar100_lt_if100_splits/*.json
# Should see: train, tuneV, val_lt, test_lt

# Step 1: Check expert logits exported
ls outputs/logits/cifar100_lt_if100/*/tuneV_logits.pt
# Should see 3 files (ce, logitadjust, balsoftmax)

# Step 2: Check pretrain checkpoint
ls checkpoints/gating_pretrained/cifar100_lt_if100/gating_pretrained.ckpt

# Step 3: Check selective checkpoint
ls checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt

# Step 4: Check plugin checkpoint
ls checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt

# Step 5: Check results
ls results_worst_eg_improved/cifar100_lt_if100/metrics.json
```

---

## Configuration Summary

All scripts use:
- **Dataset:** `data/cifar100_lt_if100_splits/`
- **Experts:** `['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']`
- **Logits:** `outputs/logits/cifar100_lt_if100/`

**No manual configuration needed - everything is aligned!** ✅

---

## Troubleshooting

**Missing logits error?**
→ Run Step 1 (train_expert) first

**Wrong expert names?**
→ All fixed! Uses ce/logitadjust/balsoftmax consistently

**Missing splits?**
→ Run Step 0 (create_full_cifar100_lt_splits) first

**CUDA out of memory?**
→ Reduce batch size in respective CONFIG dict

---

## Full Documentation

- **Pipeline details:** `PIPELINE_ALIGNMENT_SUMMARY.md`
- **Data methodology:** `DATA_SPLITTING_METHODOLOGY.md`
- **Visualization guide:** `VISUALIZATION_OUTPUT_GUIDE.md`

