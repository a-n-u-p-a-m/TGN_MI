import torch
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import DataLoader
from temporal_gnn_mi.datasets.batch_utils import temporal_collate_fn
import os
import time
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from typing import Dict, Any

from temporal_gnn_mi.config import TrainingConfig
from .metrics import compute_metrics, log_metrics


def print_model_summary(model: nn.Module, device: torch.device):
    """Print detailed model architecture summary."""
    print("\n" + "=" * 60)
    print("  MODEL SUMMARY")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    print(f"  {'Module':<35} {'Shape':<25} {'Params':>10}")
    print(f"  {'-'*35} {'-'*25} {'-'*10}")
    
    for name, param in model.named_parameters():
        p_count = param.numel()
        total_params += p_count
        if param.requires_grad:
            trainable_params += p_count
        print(f"  {name:<35} {str(list(param.shape)):<25} {p_count:>10,}")
    
    print(f"  {'-'*35} {'-'*25} {'-'*10}")
    print(f"  {'Total Parameters':<35} {'':<25} {total_params:>10,}")
    print(f"  {'Trainable Parameters':<35} {'':<25} {trainable_params:>10,}")
    print(f"  {'Non-trainable Parameters':<35} {'':<25} {total_params - trainable_params:>10,}")
    print(f"  Device: {device}")
    
    # Buffer summary
    buffer_count = sum(1 for _ in model.buffers())
    buffer_elements = sum(b.numel() for b in model.buffers())
    print(f"  Registered Buffers: {buffer_count} ({buffer_elements:,} elements)")
    print("=" * 60 + "\n")
    
    return {'total_params': total_params, 'trainable_params': trainable_params}


class TrainingHistory:
    """Tracks all metrics across epochs for plotting."""
    
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.train_roc_auc = []
        self.train_ap = []
        self.val_loss = []
        self.val_acc = []
        self.val_roc_auc = []
        self.val_ap = []
        self.epoch_times = []
        self.epochs = []
    
    def record_train(self, epoch, metrics, elapsed):
        self.epochs.append(epoch)
        self.train_loss.append(metrics.get('loss', 0))
        self.train_acc.append(metrics.get('accuracy', 0))
        self.train_roc_auc.append(metrics.get('roc_auc', 0))
        self.train_ap.append(metrics.get('ap', 0))
        self.epoch_times.append(elapsed)
    
    def record_val(self, metrics):
        self.val_loss.append(metrics.get('loss', 0))
        self.val_acc.append(metrics.get('accuracy', 0))
        self.val_roc_auc.append(metrics.get('roc_auc', 0))
        self.val_ap.append(metrics.get('ap', 0))
    
    def to_dict(self):
        return {
            'epochs': self.epochs,
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'train_roc_auc': self.train_roc_auc,
            'train_ap': self.train_ap,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'val_roc_auc': self.val_roc_auc,
            'val_ap': self.val_ap,
            'epoch_times': self.epoch_times,
        }


def plot_training_curves(history: TrainingHistory, output_dir: str):
    """Generate and save all training metric plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    epochs = history.epochs
    has_val = len(history.val_loss) > 0
    
    # Use a clean style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ── Figure 1: Combined 2x2 dashboard ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history.train_loss, 'b-', label='Train Loss', linewidth=2)
    if has_val:
        ax.plot(epochs, history.val_loss, 'r--', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (BCEWithLogits)')
    ax.set_title('Loss Over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history.train_acc, 'b-', label='Train Acc', linewidth=2)
    if has_val:
        ax.plot(epochs, history.val_acc, 'r--', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Over Epochs')
    ax.axhline(y=0.98, color='green', linestyle=':', alpha=0.7, label='Target (98%)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC-AUC
    ax = axes[1, 0]
    ax.plot(epochs, history.train_roc_auc, 'b-', label='Train ROC-AUC', linewidth=2)
    if has_val:
        ax.plot(epochs, history.val_roc_auc, 'r--', label='Val ROC-AUC', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC Over Epochs')
    ax.axhline(y=0.99, color='green', linestyle=':', alpha=0.7, label='Target (0.99)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average Precision
    ax = axes[1, 1]
    ax.plot(epochs, history.train_ap, 'b-', label='Train AP', linewidth=2)
    if has_val:
        ax.plot(epochs, history.val_ap, 'r--', label='Val AP', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Precision')
    ax.set_title('Average Precision Over Epochs')
    ax.axhline(y=0.99, color='green', linestyle=':', alpha=0.7, label='Target (0.99)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, 'training_dashboard.png')
    fig.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {dashboard_path}")
    
    # ── Figure 2: Epoch timing ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(epochs, history.epoch_times, color='steelblue', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Per Epoch')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    timing_path = os.path.join(output_dir, 'epoch_timing.png')
    fig.savefig(timing_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {timing_path}")


class Trainer:
    """Training loop for TGN on temporal motif completion.
    
    Features:
    - Checkpoint saving at configurable intervals
    - Early stopping with patience
    - Train/val/test split evaluation
    - Metrics logging with plots
    - Per-epoch and total time tracking
    """
    def __init__(self, model: nn.Module, train_dataset: Any, val_dataset: Any, config: TrainingConfig, output_dir: str):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # num_workers=0 avoids multiprocessing issues on some platforms (Kaggle, Colab)
        # and prevents potential data-loading race conditions with custom collate fns.
        use_cuda = (config.device == 'cuda' or
                    (config.device == 'auto' and torch.cuda.is_available()))
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=temporal_collate_fn,
            num_workers=0,
            pin_memory=use_cuda,
        )
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=temporal_collate_fn,
                num_workers=0,
                pin_memory=use_cuda,
            )
        
        self.history = TrainingHistory()
            
    def train_epoch(self, completion_readout: bool = False) -> dict:
        self.model.train()
        total_loss = 0.0
        total_completion_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            labels = batch['y'].to(self.device).view(-1, 1)

            # Label Smoothing to prevent logit saturation
            smoothed_labels = labels * 0.9 + 0.05

            logits = self.model(batch, completion_readout=completion_readout)
            loss = self.criterion(logits, smoothed_labels)

            # Joint training of completion-time classifier when enabled
            if completion_readout and self.model.completion_logits is not None:
                comp_logits = self.model.completion_logits
                # Use the model's explicit m2 validity mask (not comp_logits != 0,
                # since a legitimate logit can theoretically equal zero)
                if hasattr(self.model, 'completion_captured') and self.model.completion_captured is not None:
                    has_m2 = self.model.completion_captured  # (B,) bool
                else:
                    # Fallback: check hook data
                    has_m2 = (comp_logits.abs() > 0).any(dim=-1)  # (B,) bool
                n_valid = has_m2.sum().item()
                if n_valid > 0:
                    # Compute loss ONLY on samples with valid m2
                    valid_comp_logits = comp_logits[has_m2]
                    valid_comp_labels = smoothed_labels[has_m2]
                    comp_loss = self.criterion(valid_comp_logits, valid_comp_labels)
                    comp_weight = self.config.completion_loss_weight
                    loss = loss + comp_weight * comp_loss
                    total_completion_loss += comp_loss.item() * n_valid

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())  # Keep original discrete labels for metrics

            if hasattr(self.model, 'clear_hooks'):
                self.model.clear_hooks()

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels)
        metrics['loss'] = total_loss / len(self.train_dataset)
        metrics['mean_abs_logit'] = all_logits.abs().mean().item()
        if completion_readout:
            metrics['completion_loss'] = total_completion_loss / len(self.train_dataset)
        return metrics

    def evaluate(self, dataset: Any, completion_readout: bool = False) -> dict:
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=temporal_collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        total_loss = 0.0
        all_logits = []
        all_labels = []
        comp_logits_list = []
        comp_labels_list = []
        total_comp_loss = 0.0
        total_comp_valid = 0

        with torch.no_grad():
            for batch in loader:
                labels = batch['y'].to(self.device).view(-1, 1)

                logits = self.model(batch, completion_readout=completion_readout)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                all_logits.append(logits)
                all_labels.append(labels)

                # Collect completion classifier outputs
                if completion_readout and self.model.completion_logits is not None:
                    comp_logits = self.model.completion_logits
                    if hasattr(self.model, 'completion_captured') and self.model.completion_captured is not None:
                        has_m2 = self.model.completion_captured
                    else:
                        has_m2 = (comp_logits.abs() > 0).any(dim=-1)
                    n_valid = has_m2.sum().item()
                    if n_valid > 0:
                        valid_comp_logits = comp_logits[has_m2]
                        valid_comp_labels = labels[has_m2]
                        comp_loss = self.criterion(valid_comp_logits, valid_comp_labels)
                        total_comp_loss += comp_loss.item() * n_valid
                        total_comp_valid += n_valid
                        comp_logits_list.append(valid_comp_logits)
                        comp_labels_list.append(valid_comp_labels)

                if hasattr(self.model, 'clear_hooks'):
                    self.model.clear_hooks()

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels)
        metrics['loss'] = total_loss / len(dataset)

        # Compute completion classifier metrics
        if completion_readout and comp_logits_list:
            all_comp_logits = torch.cat(comp_logits_list, dim=0)
            all_comp_labels = torch.cat(comp_labels_list, dim=0)
            comp_metrics = compute_metrics(all_comp_logits, all_comp_labels)
            metrics['completion_roc_auc'] = comp_metrics['roc_auc']
            metrics['completion_accuracy'] = comp_metrics['accuracy']
            metrics['completion_loss'] = total_comp_loss / max(total_comp_valid, 1)
            metrics['completion_valid_samples'] = total_comp_valid

        return metrics

    def train(self, completion_readout: bool = False) -> dict:
        """Run training loop.

        Args:
            completion_readout: if True, jointly train the completion-time
                classifier alongside the final-state classifier.
        """
        # Print model summary before training
        model_info = print_model_summary(self.model, self.device)

        best_val_roc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_metrics = {}
        logit_magnitudes = []
        total_start = time.time()

        # Adaptive checkpoint tracking
        prev_val_auc = 0.0
        prev_comp_auc = 0.0
        last_analysis_epoch = 0
        MIN_ADAPTIVE_GAP = 2     # don't save more often than every 2 epochs
        MAX_ANALYSIS_GAP = 10    # force a save at least every 10 epochs
        ADAPTIVE_DELTA = 0.01    # AUC change threshold for adaptive save

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            train_metrics = self.train_epoch(completion_readout=completion_readout)

            epoch_elapsed = time.time() - epoch_start
            self.history.record_train(epoch, train_metrics, epoch_elapsed)

            log_metrics(train_metrics, epoch, 'train')

            # Track logit magnitudes for diagnostics
            mean_abs_logit = train_metrics.get('mean_abs_logit', 0.0)
            logit_magnitudes.append(mean_abs_logit)
            if mean_abs_logit > 5.0:
                print(f"  ⚠ WARNING: mean |logit| = {mean_abs_logit:.2f} (>5.0) — logits may be exploding")

            if self.val_dataset:
                val_metrics = self.evaluate(self.val_dataset,
                                            completion_readout=completion_readout)
                log_metrics(val_metrics, epoch, 'val')
                # Log completion metrics if present
                if 'completion_roc_auc' in val_metrics:
                    print(f"    completion val: ROC-AUC={val_metrics['completion_roc_auc']:.4f}  "
                          f"acc={val_metrics['completion_accuracy']:.4f}  "
                          f"loss={val_metrics['completion_loss']:.4f}  "
                          f"(n={val_metrics['completion_valid_samples']})")
                self.history.record_val(val_metrics)

                if val_metrics['roc_auc'] > best_val_roc:
                    best_val_roc = val_metrics['roc_auc']
                    best_epoch = epoch
                    patience_counter = 0
                    best_metrics = val_metrics
                    self.save_checkpoint(epoch, val_metrics, name="best_model.pt",
                                        completion_readout=completion_readout)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Per-corruption evaluation every 10 epochs
                if epoch % 10 == 0:
                    self.evaluate_per_corruption(self.val_dataset)

                # ── Analysis checkpoint logic ──
                # Save lightweight analysis checkpoints at:
                #   - canonical milestones (always)
                #   - adaptive triggers (AUC jump > 0.01, or gap > 10 epochs)
                cur_val_auc = val_metrics.get('roc_auc', 0.0)
                cur_comp_auc = val_metrics.get('completion_roc_auc', 0.0)
                epochs_since_save = epoch - last_analysis_epoch

                should_save_analysis = (
                    epoch in self.MILESTONE_EPOCHS
                    or epochs_since_save >= MAX_ANALYSIS_GAP
                    or (epochs_since_save >= MIN_ADAPTIVE_GAP and (
                        abs(cur_val_auc - prev_val_auc) > ADAPTIVE_DELTA
                        or abs(cur_comp_auc - prev_comp_auc) > ADAPTIVE_DELTA
                    ))
                )
                if should_save_analysis:
                    self.save_analysis_checkpoint(epoch, val_metrics,
                                                 completion_readout=completion_readout)
                    last_analysis_epoch = epoch

                prev_val_auc = cur_val_auc
                prev_comp_auc = cur_comp_auc

            # Resume checkpoints at the configured interval
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, train_metrics, name=f"checkpoint_ep{epoch}.pt",
                                    completion_readout=completion_readout)
        
        # Save final-epoch model (separate from best)
        self.save_checkpoint(epoch, train_metrics, name="final_model.pt",
                            completion_readout=completion_readout)

        # At the very end of training, run the corruption eval on the best model to capture source_shuffle
        if self.val_dataset:
            self.load_checkpoint(os.path.join(self.output_dir, "best_model.pt"))
            corruption_results = self.evaluate_per_corruption(self.val_dataset)
            best_metrics['source_shuffle_acc'] = corruption_results.get('source_shuffle', 0.0)
            # Re-save best model with updated metrics (use actual best_epoch, not final epoch)
            self.save_checkpoint(best_epoch, best_metrics, name="best_model.pt",
                                completion_readout=completion_readout)

        total_elapsed = time.time() - total_start
        
        # Print timing summary
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total training time:    {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  Epochs completed:       {len(self.history.epochs)}")
        if self.history.epoch_times:
            avg_time = sum(self.history.epoch_times) / len(self.history.epoch_times)
            print(f"  Avg time per epoch:     {avg_time:.2f}s")
        print(f"  Best val ROC-AUC:       {best_val_roc:.4f}")
        if self.val_dataset:
            print(f"  Best source_shuffle acc:{best_metrics.get('source_shuffle_acc', 0.0):.4f}")
        print(f"  Total parameters:       {model_info['total_params']:,}")
        print(f"  Trainable parameters:   {model_info['trainable_params']:,}")
        print(f"{'='*60}")
        
        # Generate plots
        print("\nGenerating training plots...")
        plot_training_curves(self.history, self.output_dir)
        
        # Save training history as JSON
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history.to_dict(), f, indent=2)
        print(f"  Saved: {history_path}")

        # Save best metrics as standalone JSON for easy compilation
        metrics_path = os.path.join(self.output_dir, 'best_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in best_metrics.items()}, f, indent=2)
        print(f"  Saved: {metrics_path}")

        return best_metrics

    # Canonical milestone epochs — every seed saves these so cross-seed
    # circuit-formation curves align.
    MILESTONE_EPOCHS = {1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200}

    def save_checkpoint(self, epoch: int, metrics: dict, name: str = "checkpoint.pt",
                        completion_readout: bool = False):
        """Save a full resume checkpoint (includes optimizer state)."""
        path = os.path.join(self.output_dir, name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'completion_readout': completion_readout,
            'completion_loss_weight': self.config.completion_loss_weight,
        }, path)

    def save_analysis_checkpoint(self, epoch: int, metrics: dict,
                                 completion_readout: bool = False):
        """Save a lightweight analysis checkpoint (no optimizer state).

        These go into analysis_checkpoints/ and are used by T1/T3
        training-dynamics experiments.
        """
        ckpt_dir = os.path.join(self.output_dir, 'analysis_checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'completion_readout': completion_readout,
            'completion_loss_weight': self.config.completion_loss_weight,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path} (Epoch {checkpoint['epoch']})")

    def evaluate_per_corruption(self, dataset: Any) -> dict:
        """Evaluate model accuracy per corruption type.
        
        Groups samples by their `corruption` attribute, runs inference
        on each group, and prints a formatted summary.
        
        Returns:
            Dict mapping corruption type names to accuracy values.
        """
        self.model.eval()
        
        # Group samples by corruption type
        groups = defaultdict(list)
        for sample in dataset:
            corruption = getattr(sample, 'corruption', 'unknown')
            groups[corruption].append(sample)
        
        results = {}
        parts = []
        with torch.no_grad():
            for corruption_type, samples in sorted(groups.items()):
                loader = DataLoader(
                    samples,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    collate_fn=temporal_collate_fn,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )
                correct = 0
                total = 0
                for batch in loader:
                    labels = batch['y'].to(self.device).view(-1)
                    logits = self.model(batch).view(-1)
                    preds = (logits > 0).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    if hasattr(self.model, 'clear_hooks'):
                        self.model.clear_hooks()
                
                acc = correct / total if total > 0 else 0.0
                results[corruption_type] = acc
                parts.append(f"{corruption_type}: {acc:.2%}")
                
        print(f"  Per-corruption acc | {' | '.join(parts)}")
        
        return results
