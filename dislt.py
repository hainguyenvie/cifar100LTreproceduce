#!/usr/bin/env python3
"""
ImageNet-LT Distribution Analysis Script
Analyzes class distribution across train, validation, and test splits
for ImageNet-LT dataset with classes 0-999.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from typing import Dict, List, Tuple
import argparse


class ImageNetLTDistributionAnalyzer:
    """Analyzer for ImageNet-LT class distributions."""
    
    def __init__(self):
        self.train_dist = None
        self.val_dist = None
        self.test_dist = None
        self.combined_dist = None
        self.num_classes = 1000  # ImageNet has 1000 classes (0-999)
        
    def load_distribution(self, file_path: str) -> Dict[int, int]:
        """
        Load class distribution from a file.
        
        Args:
            file_path: Path to the ImageNet-LT split file
            
        Returns:
            Dictionary mapping class_id to count
        """
        print(f"Loading distribution from {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        class_counts = Counter()
        total_samples = 0
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                # Split by space, last element is the class label
                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                    
                try:
                    class_id = int(parts[-1])
                    if 0 <= class_id <= 999:
                        class_counts[class_id] += 1
                        total_samples += 1
                    else:
                        print(f"Warning: Invalid class ID {class_id} on line {line_num}")
                except ValueError:
                    print(f"Warning: Could not parse class ID on line {line_num}: {line}")
                    
        print(f"Loaded {total_samples} samples with {len(class_counts)} unique classes")
        return dict(class_counts)
    
    def analyze_split_distributions(self):
        """Analyze distributions for all splits."""
        files = {
            'train': 'ImageNet_LT_train.txt',
            'val': 'ImageNet_LT_val.txt', 
            'test': 'ImageNet_LT_test.txt'
        }
        
        distributions = {}
        for split_name, file_name in files.items():
            if os.path.exists(file_name):
                distributions[split_name] = self.load_distribution(file_name)
                setattr(self, f"{split_name}_dist", distributions[split_name])
            else:
                print(f"Warning: {file_name} not found, skipping {split_name} split")
                
        return distributions
    
    def get_combined_distribution(self) -> Dict[int, int]:
        """Get combined distribution across all splits."""
        if self.combined_dist is not None:
            return self.combined_dist
            
        combined = Counter()
        for dist in [self.train_dist, self.val_dist, self.test_dist]:
            if dist is not None:
                combined.update(dist)
                
        self.combined_dist = dict(combined)
        return self.combined_dist
    
    def get_distribution_statistics(self, distribution: Dict[int, int]) -> Dict:
        """Calculate comprehensive statistics for a distribution."""
        if not distribution:
            return {}
            
        counts = list(distribution.values())
        classes = list(distribution.keys())
        
        # Find classes with 0 samples (missing classes)
        all_classes = set(range(1000))
        present_classes = set(distribution.keys())
        missing_classes = sorted(all_classes - present_classes)
        
        # Find classes with 0 samples that are present in the distribution
        zero_sample_classes = sorted([cls for cls, count in distribution.items() if count == 0])
        
        stats = {
            'total_samples': sum(counts),
            'num_classes': len(distribution),
            'num_present_classes': len(present_classes),
            'num_missing_classes': len(missing_classes),
            'num_zero_sample_classes': len(zero_sample_classes),
            'min_samples': min(counts) if counts else 0,
            'max_samples': max(counts) if counts else 0,
            'mean_samples': np.mean(counts) if counts else 0,
            'median_samples': np.median(counts) if counts else 0,
            'std_samples': np.std(counts) if counts else 0,
            'min_class': min(classes) if classes else None,
            'max_class': max(classes) if classes else None,
            'missing_classes': missing_classes,
            'zero_sample_classes': zero_sample_classes,
            'classes_with_samples': sorted([cls for cls, count in distribution.items() if count > 0])
        }
        
        # Imbalance ratio (max/min, excluding zero samples)
        non_zero_counts = [c for c in counts if c > 0]
        if non_zero_counts and min(non_zero_counts) > 0:
            stats['imbalance_ratio'] = max(non_zero_counts) / min(non_zero_counts)
        else:
            stats['imbalance_ratio'] = float('inf')
            
        return stats
    
    def plot_distribution(self, distribution: Dict[int, int], title: str, 
                         save_path: str = None, log_scale: bool = True):
        """Plot class distribution."""
        if not distribution:
            print(f"No data to plot for {title}")
            return
            
        # Get statistics first
        stats = self.get_distribution_statistics(distribution)
        
        # Create comprehensive plot with zero-sample class information
        fig = plt.figure(figsize=(18, 14))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], width_ratios=[3, 1])
        
        # Main distribution plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Prepare data for all classes (0-999)
        all_classes = list(range(1000))
        all_counts = [distribution.get(cls, 0) for cls in all_classes]
        
        # Separate data for coloring
        zero_classes = [cls for cls in all_classes if distribution.get(cls, 0) == 0]
        non_zero_classes = [cls for cls in all_classes if distribution.get(cls, 0) > 0]
        non_zero_counts = [distribution.get(cls, 0) for cls in non_zero_classes]
        
        # Plot non-zero classes in blue
        if non_zero_classes:
            bars1 = ax1.bar(non_zero_classes, non_zero_counts, alpha=0.7, 
                           color='steelblue', label=f'Classes with samples ({len(non_zero_classes)})')
        
        # Plot zero classes in red (at y=1 for visibility in log scale)
        if zero_classes:
            bars2 = ax1.bar(zero_classes, [1] * len(zero_classes), alpha=0.8, 
                           color='red', label=f'Classes with 0 samples ({len(zero_classes)})')
        
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title(f'{title} - Class Distribution (Red bars = 0 samples)')
        ax1.legend()
        
        if log_scale:
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=0.5)  # Show zero-sample classes at y=1
        
        # Add detailed statistics text
        stats_text = f"Total Samples: {stats['total_samples']:,}\n"
        stats_text += f"Present Classes: {stats['num_present_classes']}\n"
        stats_text += f"Missing Classes: {stats['num_missing_classes']}\n"
        stats_text += f"Zero Sample Classes: {stats['num_zero_sample_classes']}\n"
        stats_text += f"Min Samples: {stats['min_samples']}\n"
        stats_text += f"Max Samples: {stats['max_samples']}\n"
        stats_text += f"Mean Samples: {stats['mean_samples']:.1f}\n"
        stats_text += f"Imbalance Ratio: {stats['imbalance_ratio']:.1f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Histogram of sample counts (excluding zeros)
        ax2 = fig.add_subplot(gs[1, :])
        non_zero_counts_for_hist = [c for c in all_counts if c > 0]
        if non_zero_counts_for_hist:
            ax2.hist(non_zero_counts_for_hist, bins=50, alpha=0.7, 
                    edgecolor='black', color='lightgreen')
            ax2.set_xlabel('Number of Samples per Class')
            ax2.set_ylabel('Number of Classes')
            ax2.set_title(f'{title} - Sample Count Distribution (Excluding Zero-Sample Classes)')
            if log_scale:
                ax2.set_yscale('log')
        
        # Zero-sample class details
        ax3 = fig.add_subplot(gs[2, 0])
        if stats['missing_classes'] or stats['zero_sample_classes']:
            zero_info = []
            if stats['missing_classes']:
                zero_info.append(f"Missing from dataset: {len(stats['missing_classes'])}")
                if len(stats['missing_classes']) <= 10:
                    zero_info.append(f"Classes: {stats['missing_classes'][:10]}")
                else:
                    zero_info.append(f"Classes: {stats['missing_classes'][:10]}...")
            
            if stats['zero_sample_classes']:
                zero_info.append(f"Present but 0 samples: {len(stats['zero_sample_classes'])}")
                if len(stats['zero_sample_classes']) <= 10:
                    zero_info.append(f"Classes: {stats['zero_sample_classes'][:10]}")
                else:
                    zero_info.append(f"Classes: {stats['zero_sample_classes'][:10]}...")
            
            ax3.text(0.05, 0.95, '\n'.join(zero_info), transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax3.set_title('Zero-Sample Classes Details')
            ax3.axis('off')
        else:
            ax3.text(0.5, 0.5, 'No zero-sample classes found!', 
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))
            ax3.set_title('Zero-Sample Classes Details')
            ax3.axis('off')
        
        # Summary pie chart
        ax4 = fig.add_subplot(gs[2, 1])
        if stats['num_missing_classes'] > 0 or stats['num_zero_sample_classes'] > 0:
            pie_data = []
            pie_labels = []
            pie_colors = []
            
            if stats['num_present_classes'] - stats['num_zero_sample_classes'] > 0:
                pie_data.append(stats['num_present_classes'] - stats['num_zero_sample_classes'])
                pie_labels.append('Classes with samples')
                pie_colors.append('lightgreen')
            
            if stats['num_zero_sample_classes'] > 0:
                pie_data.append(stats['num_zero_sample_classes'])
                pie_labels.append('Classes with 0 samples')
                pie_colors.append('orange')
            
            if stats['num_missing_classes'] > 0:
                pie_data.append(stats['num_missing_classes'])
                pie_labels.append('Missing classes')
                pie_colors.append('lightcoral')
            
            ax4.pie(pie_data, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Class Coverage')
        else:
            ax4.text(0.5, 0.5, 'All classes have samples!', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))
            ax4.set_title('Class Coverage')
            ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_comparison(self, save_path: str = None):
        """Plot comparison of all splits."""
        distributions = {
            'Train': self.train_dist,
            'Validation': self.val_dist,
            'Test': self.test_dist
        }
        
        # Filter out None distributions
        distributions = {k: v for k, v in distributions.items() if v is not None}
        
        if len(distributions) < 2:
            print("Need at least 2 splits to create comparison plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each split
        num_splits = len(distributions)
        fig, axes = plt.subplots(num_splits, 1, figsize=(15, 6 * num_splits))
        
        if num_splits == 1:
            axes = [axes]
            
        for i, (split_name, dist) in enumerate(distributions.items()):
            classes = sorted(dist.keys())
            counts = [dist[cls] for cls in classes]
            
            axes[i].bar(classes, counts, alpha=0.7)
            axes[i].set_xlabel('Class ID')
            axes[i].set_ylabel('Number of Samples')
            axes[i].set_title(f'{split_name} Split - Class Distribution')
            axes[i].set_yscale('log')
            
            # Add statistics
            stats = self.get_distribution_statistics(dist)
            stats_text = f"Total: {stats['total_samples']:,}, Classes: {stats['num_classes']}, "
            stats_text += f"Min: {stats['min_samples']}, Max: {stats['max_samples']}, "
            stats_text += f"Ratio: {stats['imbalance_ratio']:.1f}"
            
            axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(self) -> str:
        """Create a comprehensive summary report."""
        report = []
        report.append("=" * 80)
        report.append("ImageNet-LT Class Distribution Analysis Report")
        report.append("=" * 80)
        report.append("")
        
        # Analyze each split
        splits = [
            ('Train', self.train_dist),
            ('Validation', self.val_dist),
            ('Test', self.test_dist)
        ]
        
        for split_name, dist in splits:
            if dist is None:
                report.append(f"{split_name} Split: Not available")
                continue
                
            stats = self.get_distribution_statistics(dist)
            report.append(f"{split_name} Split:")
            report.append(f"  Total samples: {stats['total_samples']:,}")
            report.append(f"  Present classes: {stats['num_present_classes']}")
            report.append(f"  Missing classes: {stats['num_missing_classes']}")
            report.append(f"  Zero sample classes: {stats['num_zero_sample_classes']}")
            report.append(f"  Class range: {stats['min_class']} - {stats['max_class']}")
            report.append(f"  Samples per class - Min: {stats['min_samples']}, "
                         f"Max: {stats['max_samples']}, Mean: {stats['mean_samples']:.1f}")
            report.append(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}")
            
            if stats['missing_classes']:
                missing_count = len(stats['missing_classes'])
                report.append(f"  Missing classes details:")
                if missing_count <= 20:  # Show first 20 missing classes
                    report.append(f"    {stats['missing_classes'][:20]}")
                else:
                    report.append(f"    {stats['missing_classes'][:20]} ... and {missing_count-20} more")
            
            if stats['zero_sample_classes']:
                zero_count = len(stats['zero_sample_classes'])
                report.append(f"  Zero sample classes details:")
                if zero_count <= 20:  # Show first 20 zero sample classes
                    report.append(f"    {stats['zero_sample_classes'][:20]}")
                else:
                    report.append(f"    {stats['zero_sample_classes'][:20]} ... and {zero_count-20} more")
            
            report.append("")
        
        # Combined analysis
        if self.combined_dist:
            stats = self.get_distribution_statistics(self.combined_dist)
            report.append("Combined (All Splits):")
            report.append(f"  Total samples: {stats['total_samples']:,}")
            report.append(f"  Present classes: {stats['num_present_classes']}")
            report.append(f"  Missing classes: {stats['num_missing_classes']}")
            report.append(f"  Zero sample classes: {stats['num_zero_sample_classes']}")
            report.append(f"  Class range: {stats['min_class']} - {stats['max_class']}")
            report.append(f"  Samples per class - Min: {stats['min_samples']}, "
                         f"Max: {stats['max_samples']}, Mean: {stats['mean_samples']:.1f}")
            report.append(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}")
            
            if stats['missing_classes']:
                missing_count = len(stats['missing_classes'])
                report.append(f"  Missing classes: {missing_count}")
            
            if stats['zero_sample_classes']:
                zero_count = len(stats['zero_sample_classes'])
                report.append(f"  Zero sample classes: {zero_count}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def analyze_zero_sample_classes(self) -> Dict[str, List[int]]:
        """Analyze zero-sample classes across all splits."""
        zero_analysis = {}
        
        splits = [
            ('train', self.train_dist),
            ('val', self.val_dist),
            ('test', self.test_dist),
            ('combined', self.combined_dist)
        ]
        
        for split_name, dist in splits:
            if dist is None:
                zero_analysis[split_name] = {'missing': [], 'zero_samples': []}
                continue
                
            stats = self.get_distribution_statistics(dist)
            zero_analysis[split_name] = {
                'missing': stats['missing_classes'],
                'zero_samples': stats['zero_sample_classes']
            }
        
        return zero_analysis
    
    def print_zero_sample_summary(self):
        """Print a detailed summary of zero-sample classes."""
        zero_analysis = self.analyze_zero_sample_classes()
        
        print("\n" + "="*80)
        print("ZERO-SAMPLE CLASS ANALYSIS")
        print("="*80)
        
        for split_name, analysis in zero_analysis.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Missing classes (not in dataset): {len(analysis['missing'])}")
            if analysis['missing']:
                if len(analysis['missing']) <= 20:
                    print(f"    {analysis['missing']}")
                else:
                    print(f"    {analysis['missing'][:20]} ... and {len(analysis['missing'])-20} more")
            
            print(f"  Zero sample classes (in dataset but 0 samples): {len(analysis['zero_samples'])}")
            if analysis['zero_samples']:
                if len(analysis['zero_samples']) <= 20:
                    print(f"    {analysis['zero_samples']}")
                else:
                    print(f"    {analysis['zero_samples'][:20]} ... and {len(analysis['zero_samples'])-20} more")
        
        # Cross-split analysis
        print(f"\nCROSS-SPLIT ANALYSIS:")
        if zero_analysis['combined']['missing'] or zero_analysis['combined']['zero_samples']:
            all_zero_classes = set(zero_analysis['combined']['missing'] + zero_analysis['combined']['zero_samples'])
            print(f"  Total classes with no samples across all splits: {len(all_zero_classes)}")
            
            # Check which classes are missing in all splits
            all_missing = set(zero_analysis['train']['missing']) & set(zero_analysis['val']['missing']) & set(zero_analysis['test']['missing'])
            if all_missing:
                print(f"  Classes missing in ALL splits: {len(all_missing)}")
                if len(all_missing) <= 10:
                    print(f"    {sorted(all_missing)}")
                else:
                    print(f"    {sorted(all_missing)[:10]} ... and {len(all_missing)-10} more")
        else:
            print("  All classes (0-999) have samples in at least one split!")
        
        print("="*80)
    
    def save_distributions_to_csv(self, output_dir: str = "distribution_outputs"):
        """Save all distributions to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        splits = [
            ('train', self.train_dist),
            ('val', self.val_dist),
            ('test', self.test_dist),
            ('combined', self.combined_dist)
        ]
        
        for split_name, dist in splits:
            if dist is None:
                continue
                
            # Create DataFrame
            classes = sorted(dist.keys())
            counts = [dist[cls] for cls in classes]
            
            df = pd.DataFrame({
                'class_id': classes,
                'sample_count': counts
            })
            
            # Add missing classes with 0 count
            all_classes = set(range(1000))
            missing_classes = all_classes - set(classes)
            if missing_classes:
                missing_df = pd.DataFrame({
                    'class_id': sorted(missing_classes),
                    'sample_count': [0] * len(missing_classes)
                })
                df = pd.concat([df, missing_df], ignore_index=True)
                df = df.sort_values('class_id').reset_index(drop=True)
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f"{split_name}_distribution.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {split_name} distribution to {csv_path}")
    
    def run_full_analysis(self, save_plots: bool = True, save_csv: bool = True):
        """Run complete analysis pipeline."""
        print("Starting ImageNet-LT distribution analysis...")
        
        # Load distributions
        self.analyze_split_distributions()
        self.get_combined_distribution()
        
        # Print summary report
        print("\n" + self.create_summary_report())
        
        # Print zero-sample analysis
        self.print_zero_sample_summary()
        
        # Save plots
        if save_plots:
            os.makedirs("distribution_outputs", exist_ok=True)
            
            # Individual plots
            for split_name, dist in [('Train', self.train_dist), 
                                   ('Validation', self.val_dist),
                                   ('Test', self.test_dist)]:
                if dist is not None:
                    self.plot_distribution(dist, f"{split_name} Split", 
                                         f"distribution_outputs/{split_name.lower()}_distribution.png")
            
            # Combined plot
            if self.combined_dist:
                self.plot_distribution(self.combined_dist, "Combined Dataset",
                                     "distribution_outputs/combined_distribution.png")
            
            # Comparison plot
            self.plot_comparison("distribution_outputs/split_comparison.png")
        
        # Save CSV files
        if save_csv:
            self.save_distributions_to_csv()
        
        print("\nAnalysis complete!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze ImageNet-LT class distributions')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--no-csv', action='store_true', help='Skip saving CSV files')
    parser.add_argument('--output-dir', default='distribution_outputs', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ImageNetLTDistributionAnalyzer()
    analyzer.run_full_analysis(save_plots=not args.no_plots, save_csv=not args.no_csv)


if __name__ == "__main__":
    main()
