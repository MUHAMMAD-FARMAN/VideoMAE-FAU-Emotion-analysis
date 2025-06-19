import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def compute_f1_score(preds: torch.Tensor, targets: torch.Tensor):
    """
    Computes macro F1 score for multi-label or multi-class classification.
    Args:
        preds: (N, num_classes)
        targets: (N, num_classes)
    """
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    return f1_score(targets_np, preds_np, average="macro")


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor):
    """
    Computes overall accuracy.
    Supports both multi-label and multi-class (1D) inputs.
    """
    if preds.dim() > 1:
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        return (preds_np == targets_np).mean()
    else:
        return accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())


def compute_ccc(preds: torch.Tensor, targets: torch.Tensor):
    """
    Computes Concordance Correlation Coefficient (CCC).
    Assumes inputs are 1D (flattened frame-level predictions).
    """
    preds = preds.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()

    mean_pred = np.mean(preds)
    mean_gt = np.mean(targets)
    var_pred = np.var(preds)
    var_gt = np.var(targets)
    cov = np.mean((preds - mean_pred) * (targets - mean_gt))

    ccc = (2 * cov) / (var_pred + var_gt + (mean_pred - mean_gt) ** 2 + 1e-8)
    return ccc
