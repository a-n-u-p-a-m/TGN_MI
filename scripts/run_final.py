"""
Final Training Run Script — Run 3: Known-Good Hyperparameters on Fixed Codebase.

Bypasses tuning entirely. Uses hardcoded hyperparameters from Run 1's Bayesian
search winner. After training completes, automatically runs the 6-baseline
shortcut audit and the intervention API test.

Bug Fix Verification (Tasks 2 & 3 from FINAL_RUN_PROMPT.md):
  ✓ Fix 1: Type-2 distractor is OUTSIDE the num_distractors loop
            (temporal_motif_generator.py L120-125)
  ✓ Fix 2: Degree-parity safeguard runs BEFORE _build_anti_shortcut_context()
            (temporal_motif_generator.py L318-344)
  ✓ Fix 3: evaluate_per_corruption uses correct/total, no undefined 'counts'
            (trainer.py L416-446)
  ✓ Label smoothing: smoothed_labels = labels * 0.9 + 0.05
            (trainer.py L237)

Usage:
    python -m scripts.run_final
    python -m scripts.run_final --seed 42 --train_samples 20000 --val_samples 3000
"""

import os
import sys
import json
import time
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_gnn_mi.config import DatasetConfig, ModelConfig, TrainingConfig
from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
from temporal_gnn_mi.datasets.batch_utils import temporal_collate_fn
from temporal_gnn_mi.models.tgn import TGN
from temporal_gnn_mi.training.trainer import Trainer, print_model_summary
from scripts.audit_model import run_audit
from experiments.utils import seed_everything


def test_intervention_api(model, dataset):
    """Verify that interventions actually change the output."""
    print("\n" + "=" * 60)
    print("  INTERVENTION API TEST")
    print("=" * 60)
    
    model.eval()
    sample = dataset.samples[0]
    batch = temporal_collate_fn([sample])
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Baseline: no intervention
        baseline_logit = model(batch).item()
        model.clear_hooks()
        
        # Find motif steps
        motif_steps = model.get_motif_step_indices(batch)
        m1, m2 = motif_steps[0]
        print(f"  Motif step indices: A→B at step {m1}, B→C at step {m2}")
        
        # Intervention: zero out mem_B before the B→C message is computed
        # This should change the logit because B's memory (which carries A's info)
        # is being erased right before it would be read by the B→C message
        def zero_mem(mem):
            return torch.zeros_like(mem)
        
        if m2 >= 0:
            interventions = {(m2, 'memory_pre_src'): zero_mem}
            patched_logit = model(batch, interventions=interventions).item()
            model.clear_hooks()
            
            diff = abs(baseline_logit - patched_logit)
            print(f"  Baseline logit:  {baseline_logit:.4f}")
            print(f"  Patched logit:   {patched_logit:.4f}")
            print(f"  Difference:      {diff:.4f}")
            
            if diff > 0.01:
                print("  ✓ Intervention API working — patching mem_B changes the output")
            else:
                print("  ✗ WARNING: Intervention had no effect — check implementation")
        else:
            print("  Skipping per-event intervention test (no motif edges in sample)")
        
        # Also test readout intervention: zero out mem_B at classification time
        interventions2 = {('readout', 'mem_B'): zero_mem}
        patched_logit2 = model(batch, interventions=interventions2).item()
        model.clear_hooks()
        
        diff2 = abs(baseline_logit - patched_logit2)
        print(f"  Readout-patched logit: {patched_logit2:.4f} (diff={diff2:.4f})")
        
        if diff2 > 0.01:
            print("  ✓ Readout intervention working")
        else:
            print("  ✗ WARNING: Readout intervention had no effect")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Final TGN Training Run (no tuning)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='runs/final_run')
    parser.add_argument('--train_samples', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=3000)
    parser.add_argument('--completion_readout', action='store_true',
                        help='Jointly train completion-time classifier '
                             '(reads memory at B→C, not end of sequence)')
    args = parser.parse_args()
    
    wall_clock_start = time.time()
    
    # ──────────────────────────────────────────────────
    #  Configuration (Run 1's Bayesian search winner)
    # ──────────────────────────────────────────────────
    
    seed_everything(args.seed)
    
    dataset_config = DatasetConfig(
        num_nodes=50,
        noise_multiplier=4,        # NOT the default 7
        time_window_delta=5.0,
        num_distractors=3,
        num_partial_motifs=2,
        seed=args.seed,
    )
    
    model_config = ModelConfig(
        memory_dim=128,
        message_dim=32,             # NOT 64 or 128
        time_dim=16,                # NOT 32 or 64
        embedding_dim=64,           # NOT 128
        dropout=0.108,              # from Run 1 tuning: 0.10771297372179586
    )
    
    train_config = TrainingConfig(
        epochs=200,
        lr=0.002,                   # from Run 1 tuning: 0.001989129673513745
        batch_size=32,              # NOT 16
        patience=20,
        checkpoint_interval=5,
    )
    
    # ──────────────────────────────────────────────────
    #  Setup output directory and save config
    # ──────────────────────────────────────────────────
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_dict = {
        "model": {
            "memory_dim": 128,
            "message_dim": 32,
            "time_dim": 16,
            "embedding_dim": 64,
            "lr": 0.002,
            "batch_size": 32,
            "dropout": 0.108,
            "noise_multiplier": 4
        },
        "training": {
            "epochs": 200,
            "batch_size": 32,
            "lr": 0.002,
            "patience": 20,
            "completion_readout": args.completion_readout,
            "completion_loss_weight": 0.5,
        },
        "dataset": {
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "noise_multiplier": 4
        }
    }
    
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"  Config saved to: {config_path}")
    
    # ──────────────────────────────────────────────────
    #  Header
    # ──────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("  TEMPORAL GNN MI — FINAL RUN (Run 3)")
    print("=" * 70)
    print(f"  Seed:           {args.seed}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Device:         {train_config.device}")
    print(f"  Train samples:  {args.train_samples}")
    print(f"  Val samples:    {args.val_samples}")
    print(f"  Epochs:         {train_config.epochs}")
    print(f"  Batch size:     {train_config.batch_size}")
    print(f"  Learning rate:  {train_config.lr}")
    print(f"  Noise mult:     {dataset_config.noise_multiplier}")
    print("=" * 70)
    
    # ──────────────────────────────────────────────────
    #  Generate datasets
    # ──────────────────────────────────────────────────
    
    print("\nGenerating datasets...")
    gen_start = time.time()
    
    train_dataset = TemporalMotifDataset(dataset_config, num_samples=args.train_samples, split='train')
    val_dataset = TemporalMotifDataset(dataset_config, num_samples=args.val_samples, split='val')
    
    gen_elapsed = time.time() - gen_start
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Generated in {gen_elapsed:.1f}s")
    
    # Dataset statistics
    pos_count = sum(1 for s in train_dataset.samples if s.y.item() == 1.0)
    neg_count = len(train_dataset) - pos_count
    avg_events = sum(s.src.size(0) for s in train_dataset.samples) / len(train_dataset)
    print(f"\n  Positive ratio:     {pos_count}/{len(train_dataset)} ({pos_count/len(train_dataset):.1%})")
    print(f"  Avg events/sample:  {avg_events:.1f}")
    
    # ──────────────────────────────────────────────────
    #  Initialize model
    # ──────────────────────────────────────────────────
    
    model = TGN(model_config, num_nodes=dataset_config.num_nodes, edge_feat_dim=16)
    
    # Print parameter summary
    print_model_summary(model, torch.device(train_config.device))
    
    # ──────────────────────────────────────────────────
    #  Train
    # ──────────────────────────────────────────────────
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=args.output_dir,
    )
    
    best_metrics = trainer.train(completion_readout=args.completion_readout)
    
    # ──────────────────────────────────────────────────
    #  Training summary
    # ──────────────────────────────────────────────────
    
    print("\n" + "═" * 58)
    print("  TRAINING COMPLETE")
    print("═" * 58)
    print(f"  Epochs completed:       {len(trainer.history.epochs)}")
    print(f"  Best Val ROC-AUC:       {best_metrics.get('roc_auc', 0.0):.4f}")
    print(f"  Best Val Accuracy:      {best_metrics.get('accuracy', 0.0):.4f}")
    print(f"  Best Val AP:            {best_metrics.get('ap', 0.0):.4f}")
    print("═" * 58)
    
    # ──────────────────────────────────────────────────
    #  Post-hoc shortcut audit
    # ──────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("  POST-TRAINING SHORTCUT AUDIT")
    print("=" * 70)
    
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        run_audit(checkpoint_path)
    else:
        print(f"  WARNING: No best_model.pt found at {checkpoint_path}")
    
    # ──────────────────────────────────────────────────
    #  Intervention API test
    # ──────────────────────────────────────────────────
    
    # Use a positive sample for the test (more interesting to patch)
    pos_samples = [s for s in val_dataset.samples if s.y.item() == 1.0]
    if pos_samples:
        test_dataset_for_intervention = type('Dataset', (), {'samples': pos_samples})()
        test_intervention_api(model, test_dataset_for_intervention)
    else:
        print("  WARNING: No positive samples found for intervention test")
    
    # ──────────────────────────────────────────────────
    #  Total wall-clock time
    # ──────────────────────────────────────────────────
    
    total_elapsed = time.time() - wall_clock_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    
    print("\n" + "=" * 70)
    print(f"  TOTAL WALL-CLOCK TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.0f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
