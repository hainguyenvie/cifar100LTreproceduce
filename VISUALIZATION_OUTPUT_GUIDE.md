# Dataset Visualization & Statistics Output Guide

## Quick Start

Run this command to create splits with comprehensive visualizations:

```bash
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"
```

## Output Files

All outputs are saved to `data/cifar100_lt_if100_splits/`:

### 📊 Visualizations

1. **`dataset_statistics_comprehensive.png`** (Main visualization)
   - High-resolution PNG (300 DPI)
   - Contains 7 subplots showing all statistics
   
2. **`dataset_statistics_comprehensive.pdf`**
   - Publication-quality PDF version
   - Ideal for papers/presentations

### 📄 CSV Statistics

3. **`split_summary_statistics.csv`**
   - Summary table with all key metrics per split
   - Columns: Total samples, head/medium/low/tail counts and ratios, imbalance factor, duplication stats

4. **`per_class_distribution.csv`**
   - Sample counts for each of 100 classes across all splits
   - Easy to analyze class-level distributions

### 📁 Split Indices (JSON)

5. **`train_indices.json`** - Training set indices
6. **`val_lt_indices.json`** - Validation set indices
7. **`tuneV_indices.json`** - TuneV (gating training) indices
8. **`test_lt_indices.json`** - Test set indices

## What's Visualized

The main comprehensive plot contains:

### Row 1: Overview Statistics
1. **Total Samples per Split** (bar chart)
   - Shows sample count for Train/Val/TuneV/Test
   - Labeled with exact counts

2. **Head/Medium/Low/Tail Distribution** (stacked bar)
   - Head: Classes 0-9
   - Medium: Classes 10-49
   - Low: Classes 50-89
   - Tail: Classes 90-99

3. **Group Proportions** (grouped bar)
   - Proportion of each group in each split
   - Confirms all splits match train distribution

### Row 2: Per-Class Distribution
4. **Per-Class Sample Distribution** (line plot, log scale)
   - All 100 classes on x-axis
   - Sample counts on y-axis (log scale)
   - Overlay of all splits showing they match

### Row 3: Detailed Analysis
5. **Imbalance Factors** (bar chart)
   - Max class count / Min class count for each split
   - Reference line at IF=100 (target)

6. **Duplication Statistics** (bar chart)
   - Average duplication factor for Val/TuneV/Test
   - Shows how much replication was needed per split

7. **Summary Table**
   - Quick reference with key numbers
   - Total samples, head%, tail%, imbalance factor

## Statistics Included

### For Each Split

- **Total samples**
- **Head/Medium/Low/Tail sample counts**
- **Head/Medium/Low/Tail proportions**
- **Imbalance factor** (max class / min class)
- **Max/Min class counts**

### For Val/TuneV/Test (with duplication)

- **Unique base samples** (before duplication)
- **Duplicated samples count**
- **Duplication ratio** (total / unique base)

## Example: Reading the Outputs

### In Python

```python
import pandas as pd

# Load summary statistics
summary = pd.read_csv('data/cifar100_lt_if100_splits/split_summary_statistics.csv')
print(summary)

# Load per-class distribution
per_class = pd.read_csv('data/cifar100_lt_if100_splits/per_class_distribution.csv')
print(per_class.head())

# Check train vs test distribution for class 0
print(f"Class 0 - Train: {per_class.loc[0, 'Train_count']}, Test: {per_class.loc[0, 'Test_count']}")
```

### Viewing Visualizations

**PNG (for quick viewing):**
```bash
# Windows
start data/cifar100_lt_if100_splits/dataset_statistics_comprehensive.png

# Mac
open data/cifar100_lt_if100_splits/dataset_statistics_comprehensive.png

# Linux
xdg-open data/cifar100_lt_if100_splits/dataset_statistics_comprehensive.png
```

**PDF (for papers):**
- Use the PDF version in LaTeX/Word documents
- Higher quality for publications

## What to Check

✅ **All splits have similar imbalance factors** (~100)  
✅ **Per-class distributions overlap** (in the line plot)  
✅ **Head/tail proportions match across splits**  
✅ **Duplication ratios are reasonable** (typically 2-5x for tail-heavy splits)  
✅ **No errors or warnings** during generation

## Customization

To change visualization parameters:

```python
from src.data.enhanced_datasets import create_full_cifar100_lt_splits

datasets, splits = create_full_cifar100_lt_splits(
    imb_factor=100,              # Training imbalance factor
    output_dir="custom_output",  # Custom output directory
    val_ratio=0.2,               # Validation base ratio
    tunev_ratio=0.15,            # TuneV base ratio
    seed=42                      # Random seed
)
```

## Troubleshooting

**Missing matplotlib/seaborn:**
```bash
pip install matplotlib seaborn pandas
```

**Plots look strange:**
- Check if all splits were created successfully
- Verify no errors during data splitting
- Try increasing figure DPI in code if text is too small

**Want different visualizations:**
- Modify `plot_comprehensive_statistics()` in `src/data/enhanced_datasets.py`
- Add custom plots as needed

## Path Summary

| Type | Path |
|------|------|
| Main visualization (PNG) | `data/cifar100_lt_if100_splits/dataset_statistics_comprehensive.png` |
| Publication PDF | `data/cifar100_lt_if100_splits/dataset_statistics_comprehensive.pdf` |
| Summary CSV | `data/cifar100_lt_if100_splits/split_summary_statistics.csv` |
| Per-class CSV | `data/cifar100_lt_if100_splits/per_class_distribution.csv` |
| Split indices | `data/cifar100_lt_if100_splits/*.json` |

All files are automatically generated when you run the data splitting command!

