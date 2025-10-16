#!/usr/bin/env python3
"""
Utilities for evaluation reweighting in long-tail learning.
Provides functions to load train class weights and apply weighted loss/metrics.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

def load_train_class_weights(splits_dir: str = "data/cifar100_lt_if100_splits") -> Dict:
    """
    Load train class weights for evaluation reweighting.
    
    Returns:
        dict with keys:
        - 'class_counts': list of train counts per class
        - 'class_weights': list of proportions (sum to 1.0)
        - 'mode': 'evaluation_reweighting' or 'physical_replication'
        - 'head_classes': list of head class indices
        - 'tail_classes': list of tail class indices
    """
    weights_path = Path(splits_dir) / 'train_class_weights.json'
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Train class weights not found: {weights_path}\n"
            "Run dataset creation with use_evaluation_reweighting=True first."
        )
    
    with open(weights_path, 'r') as f:
        weights_info = json.load(f)
    
    return weights_info

def get_sample_weights(labels: torch.Tensor, class_weights: List[float], 
                      device: str = 'cpu') -> torch.Tensor:
    """
    Get per-sample weights based on class frequency from training.
    
    Args:
        labels: [B] tensor of class labels
        class_weights: list of class proportions from train (length=num_classes)
        device: torch device
        
    Returns:
        sample_weights: [B] tensor of weights for each sample
    """
    # Convert class weights to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    # Get weights for each sample based on its class
    sample_weights = class_weights_tensor[labels]
    
    return sample_weights

def weighted_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, 
                               class_weights: List[float], 
                               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute weighted cross-entropy loss.
    
    Args:
        logits: [B, C] model logits
        labels: [B] ground truth labels
        class_weights: list of class proportions from train
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        weighted loss
    """
    # Get sample weights
    sample_weights = get_sample_weights(labels, class_weights, logits.device)
    
    # Compute per-sample CE loss
    ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]
    
    # Apply sample weights
    weighted_loss = ce_loss * sample_weights  # [B]
    
    if reduction == 'mean':
        # Normalize by sum of weights
        return weighted_loss.sum() / sample_weights.sum()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:  # 'none'
        return weighted_loss

def weighted_accuracy(preds: torch.Tensor, labels: torch.Tensor, 
                     class_weights: List[float]) -> float:
    """
    Compute weighted accuracy.
    
    Args:
        preds: [N] predictions
        labels: [N] ground truth
        class_weights: list of class proportions from train
        
    Returns:
        weighted_accuracy: float
    """
    # Get sample weights
    sample_weights = get_sample_weights(labels, class_weights)
    
    # Compute correctness
    correct = (preds == labels).float()
    
    # Weighted accuracy
    weighted_acc = (correct * sample_weights).sum() / sample_weights.sum()
    
    return weighted_acc.item()

def compute_weighted_group_metrics(preds: torch.Tensor, labels: torch.Tensor,
                                   class_weights: List[float],
                                   head_classes: List[int],
                                   tail_classes: List[int]) -> Dict[str, float]:
    """
    Compute weighted metrics for head/tail groups.
    
    Args:
        preds: [N] predictions
        labels: [N] ground truth
        class_weights: list of class proportions from train  
        head_classes: list of head class indices
        tail_classes: list of tail class indices
        
    Returns:
        dict with weighted_acc, weighted_head_acc, weighted_tail_acc
    """
    sample_weights = get_sample_weights(labels, class_weights)
    correct = (preds == labels).float()
    
    # Overall weighted accuracy
    overall_weighted_acc = (correct * sample_weights).sum() / sample_weights.sum()
    
    # Head group
    head_mask = torch.tensor([l.item() in head_classes for l in labels])
    if head_mask.sum() > 0:
        head_weighted_acc = (correct[head_mask] * sample_weights[head_mask]).sum() / sample_weights[head_mask].sum()
    else:
        head_weighted_acc = torch.tensor(0.0)
    
    # Tail group
    tail_mask = torch.tensor([l.item() in tail_classes for l in labels])
    if tail_mask.sum() > 0:
        tail_weighted_acc = (correct[tail_mask] * sample_weights[tail_mask]).sum() / sample_weights[tail_mask].sum()
    else:
        tail_weighted_acc = torch.tensor(0.0)
    
    return {
        'weighted_acc': overall_weighted_acc.item(),
        'weighted_head_acc': head_weighted_acc.item(),
        'weighted_tail_acc': tail_weighted_acc.item(),
        'unweighted_acc': correct.mean().item(),
        'unweighted_head_acc': correct[head_mask].mean().item() if head_mask.sum() > 0 else 0.0,
        'unweighted_tail_acc': correct[tail_mask].mean().item() if tail_mask.sum() > 0 else 0.0,
    }

def apply_weighted_selective_metrics(preds: torch.Tensor, labels: torch.Tensor,
                                    accepted_mask: torch.Tensor,
                                    class_weights: List[float],
                                    class_to_group: torch.Tensor,
                                    K: int) -> Dict[str, float]:
    """
    Compute weighted selective classification metrics.
    
    Applies class frequency weights to error computation for realistic evaluation.
    
    Args:
        preds: [N] predictions
        labels: [N] ground truth
        accepted_mask: [N] boolean mask of accepted samples
        class_weights: list of class proportions from train
        class_to_group: [C] class to group mapping
        K: number of groups
        
    Returns:
        dict with weighted selective metrics
    """
    sample_weights = get_sample_weights(labels, class_weights)
    
    # Coverage (weighted by class frequency)
    weighted_coverage = (accepted_mask.float() * sample_weights).sum() / sample_weights.sum()
    
    # Error on accepted (weighted)
    if accepted_mask.sum() > 0:
        correct = (preds == labels).float()
        accepted_correct = correct[accepted_mask]
        accepted_weights = sample_weights[accepted_mask]
        
        weighted_acc_on_accepted = (accepted_correct * accepted_weights).sum() / accepted_weights.sum()
        weighted_error_on_accepted = 1.0 - weighted_acc_on_accepted.item()
    else:
        weighted_error_on_accepted = 1.0
    
    # Group-wise weighted errors
    y_groups = class_to_group[labels]
    group_errors = []
    
    for k in range(K):
        group_mask = (y_groups == k)
        group_accepted = accepted_mask & group_mask
        
        if group_accepted.sum() > 0:
            group_correct = (preds[group_accepted] == labels[group_accepted]).float()
            group_weights = sample_weights[group_accepted]
            
            group_weighted_acc = (group_correct * group_weights).sum() / group_weights.sum()
            group_errors.append(1.0 - group_weighted_acc.item())
        else:
            group_errors.append(1.0)
    
    return {
        'weighted_coverage': weighted_coverage.item(),
        'weighted_error': weighted_error_on_accepted,
        'weighted_balanced_error': float(np.mean(group_errors)),
        'weighted_worst_error': float(np.max(group_errors)),
        'group_errors': group_errors
    }

# Example usage:
if __name__ == "__main__":
    # Load weights
    weights_info = load_train_class_weights()
    
    print("Train Class Weights Info:")
    print(f"  Mode: {weights_info['mode']}")
    print(f"  Total train samples: {weights_info['total_samples']}")
    print(f"  IF: {weights_info['imbalance_factor']:.1f}")
    print(f"  Head classes (>20): {len(weights_info['head_classes'])}")
    print(f"  Tail classes (<=20): {len(weights_info['tail_classes'])}")
    print(f"\n  Class 0 weight: {weights_info['class_weights'][0]:.6f}")
    print(f"  Class 99 weight: {weights_info['class_weights'][99]:.6f}")
    print(f"  Weight ratio (head/tail): {weights_info['class_weights'][0] / weights_info['class_weights'][99]:.1f}")

