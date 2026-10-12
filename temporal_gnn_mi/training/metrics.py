import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute Accuracy, ROC-AUC, Average Precision."""
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    preds = (probs > 0.5).astype(int)
    labels = labels.cpu().numpy()
    
    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(labels, preds)
        if len(np.unique(labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(labels, probs)
            metrics['ap'] = average_precision_score(labels, probs)
        else:
            metrics['roc_auc'] = 0.5
            metrics['ap'] = 0.5
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error computing metrics: {e}")
        metrics['accuracy'] = 0.0
        metrics['roc_auc'] = 0.5
        metrics['ap'] = 0.5
        
    return metrics

def log_metrics(metrics: dict, epoch: int, prefix: str = 'train'):
    """Pretty-print metrics."""
    acc = metrics.get('accuracy', 0)
    roc = metrics.get('roc_auc', 0)
    ap = metrics.get('ap', 0)
    print(f"[{prefix.upper()}] Epoch {epoch:03d} | Acc: {acc:.4f} | ROC-AUC: {roc:.4f} | AP: {ap:.4f}")
