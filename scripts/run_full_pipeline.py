"""
Full Pipeline: Hyperparameter Tuning → Training → Smoke Test & Audit
=============================================================

Orchestrates the entire TGN MI pipeline in a single run:
  1. Runs Bayesian hyperparameter tuning (Optuna) on a smaller dataset
  2. Extracts the best hyperparameters and saves them to JSON
  3. Trains the full model using those optimal hyperparameters
  4. Runs the smoke test suite and baseline audit to verify correctness and absence of shortcuts
  5. Collects all results, metrics, and file paths into a final report

Usage:
    cd Project3_Codebase
    python -m scripts.run_full_pipeline

    # Override defaults:
    python -m scripts.run_full_pipeline --n_trials 10 --train_samples 5000 --epochs 100
"""

import argparse
import os
import sys
import json
import time
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_gnn_mi.config import DatasetConfig, ModelConfig, TrainingConfig
from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
from temporal_gnn_mi.datasets.utils import set_seed, compute_dataset_statistics
from temporal_gnn_mi.models.tgn import TGN
from temporal_gnn_mi.training.trainer import Trainer


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

class Logger:
    """Tee stdout to both console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def section(title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


# ──────────────────────────────────────────────────────────────
#  Phase 1: Hyperparameter Tuning
# ──────────────────────────────────────────────────────────────

def run_tuning(args, output_dir):
    """Run Bayesian hyperparameter tuning and return best config dict."""
    section("PHASE 1: HYPERPARAMETER TUNING (Bayesian / Optuna)")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tuning_dir = os.path.join(output_dir, "tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    # Generate a smaller dataset for faster tuning
    print(f"  Generating tuning datasets...")
    print(f"    Train: {args.tune_train_samples} samples")
    print(f"    Val:   {args.tune_val_samples} samples")
    dataset_config = DatasetConfig(seed=args.seed)
    tune_train = TemporalMotifDataset(dataset_config, num_samples=args.tune_train_samples, split='train')
    tune_val = TemporalMotifDataset(dataset_config, num_samples=args.tune_val_samples, split='val')
    print(f"  Done.\n")

    trial_counter = [0]
    all_trials = []

    def objective(trial):
        config = {
            'memory_dim':       trial.suggest_categorical('memory_dim', [32, 64, 128]),
            'message_dim':      trial.suggest_categorical('message_dim', [32, 64, 128]),
            'time_dim':         trial.suggest_categorical('time_dim', [16, 32, 64]),
            'embedding_dim':    trial.suggest_categorical('embedding_dim', [32, 64, 128]),
            'lr':               trial.suggest_float('lr', 1e-4, 2e-3, log=True),
            'batch_size':       trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout':          trial.suggest_float('dropout', 0.0, 0.2),
            'noise_multiplier': trial.suggest_categorical('noise_multiplier', [4, 7, 10]),
        }

        idx = trial_counter[0]
        trial_counter[0] += 1
        print(f"  [Trial {idx + 1}/{args.n_trials}] {config}")

        model_config = ModelConfig(
            memory_dim=config['memory_dim'],
            message_dim=config['message_dim'],
            time_dim=config['time_dim'],
            embedding_dim=config['embedding_dim'],
            dropout=config['dropout'],
        )
        train_config = TrainingConfig(
            epochs=args.tune_epochs,
            lr=config['lr'],
            batch_size=config['batch_size'],
            patience=10,
        )
        model = TGN(config=model_config, num_nodes=50, edge_feat_dim=16)
        trial_dir = os.path.join(tuning_dir, f"trial_{idx:03d}")
        trainer = Trainer(model, tune_train, tune_val, train_config, trial_dir)
        best_metrics = trainer.train()

        roc_auc = best_metrics.get('roc_auc', 0.5)
        print(f"    → Val ROC-AUC: {roc_auc:.4f}\n")

        all_trials.append({
            'trial': idx,
            'config': config,
            'roc_auc': roc_auc,
            'metrics': best_metrics,
        })
        return roc_auc

    print(f"  Starting {args.n_trials} Optuna trials ({args.tune_epochs} epochs each)...\n")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    best_params = study.best_params
    best_roc = study.best_value

    # Save tuning results
    tuning_results = {
        'best_params': best_params,
        'best_roc_auc': best_roc,
        'n_trials': args.n_trials,
        'tune_epochs': args.tune_epochs,
        'all_trials': all_trials,
    }
    results_path = os.path.join(tuning_dir, "tuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)

    section("TUNING COMPLETE")
    print(f"  Best ROC-AUC:   {best_roc:.4f}")
    print(f"  Best Params:    {json.dumps(best_params, indent=4)}")
    print(f"  Results saved:  {results_path}")

    return best_params


# ──────────────────────────────────────────────────────────────
#  Phase 2: Full Training
# ──────────────────────────────────────────────────────────────

def run_training(args, best_params, output_dir):
    """Train the full model using optimal hyperparameters."""
    section("PHASE 2: FULL TRAINING WITH OPTIMAL HYPERPARAMETERS")

    train_dir = os.path.join(output_dir, "training")
    os.makedirs(train_dir, exist_ok=True)

    # Save the config being used
    config_path = os.path.join(train_dir, "training_config.json")
    full_config = {
        'model': best_params,
        'training': {
            'epochs': args.epochs,
            'batch_size': best_params.get('batch_size', 32),
            'lr': best_params.get('lr', 0.002),
            'patience': args.patience,
        },
        'dataset': {
            'train_samples': args.train_samples,
            'val_samples': args.val_samples,
            'noise_multiplier': best_params.get('noise_multiplier', 4),
        },
    }
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    print(f"  Config saved: {config_path}")

    # Generate full datasets
    print(f"\n  Generating full datasets...")
    print(f"    Train: {args.train_samples} samples")
    print(f"    Val:   {args.val_samples} samples")
    dataset_config = DatasetConfig(
        seed=args.seed,
        noise_multiplier=best_params.get('noise_multiplier', 4),
    )
    train_dataset = TemporalMotifDataset(dataset_config, num_samples=args.train_samples, split='train')
    val_dataset = TemporalMotifDataset(dataset_config, num_samples=args.val_samples, split='val')

    stats = compute_dataset_statistics(train_dataset)
    print(f"    Positive ratio: {stats['positive_ratio']:.2f}")
    print(f"    Avg events/sample: {stats['avg_events']:.1f}")
    print()

    # Build model
    model_config = ModelConfig(
        memory_dim=best_params.get('memory_dim', 128),
        message_dim=best_params.get('message_dim', 32),
        time_dim=best_params.get('time_dim', 16),
        embedding_dim=best_params.get('embedding_dim', 64),
        dropout=best_params.get('dropout', 0.108),
    )
    train_config = TrainingConfig(
        epochs=args.epochs,
        lr=best_params.get('lr', 0.002),
        batch_size=best_params.get('batch_size', 32),
        patience=args.patience,
        checkpoint_interval=5,
    )

    model = TGN(config=model_config, num_nodes=dataset_config.num_nodes, edge_feat_dim=16)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    trainer = Trainer(model, train_dataset, val_dataset, train_config, train_dir)
    best_metrics = trainer.train()

    section("TRAINING COMPLETE")
    print(f"  Best Val ROC-AUC:  {best_metrics.get('roc_auc', 'N/A')}")
    print(f"  Best Val Accuracy: {best_metrics.get('accuracy', 'N/A')}")
    print(f"  Best Val AP:       {best_metrics.get('ap', 'N/A')}")
    print(f"  Best Val Loss:     {best_metrics.get('loss', 'N/A')}")

    return best_metrics, train_dir


# ──────────────────────────────────────────────────────────────
#  Phase 3: Verification (Smoke Test & Audit)
# ──────────────────────────────────────────────────────────────

class TeeTee:
    def __init__(self, secondary_file):
        self.primary = sys.stdout
        self.secondary = open(secondary_file, 'w', encoding='utf-8')
    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
    def flush(self):
        self.primary.flush()
        self.secondary.flush()
    def close(self):
        self.secondary.close()

def run_smoke_test(output_dir):
    """Run the smoke test suite and capture results."""
    section("PHASE 3A: SMOKE TEST VERIFICATION")
    
    smoke_results_path = os.path.join(output_dir, "smoke_test_results.txt")
    tee = TeeTee(smoke_results_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    # Import and run each test function from the smoke test module
    from tests.smoke_test import (
        test_dataset_generation,
        test_model_forward,
        test_hook_data,
        test_training_loop,
        test_checkpointing,
        test_causal_dependence,
    )

    try:
        dataset = test_dataset_generation()
        model = test_model_forward(dataset)
        test_hook_data(dataset)
        st_output_dir = test_training_loop()
        test_checkpointing(st_output_dir)
        test_causal_dependence(st_output_dir)
        print("\n  ✓ ALL 6 SMOKE TESTS PASSED")
        return True
    except Exception as e:
        print(f"\n  ✗ SMOKE TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"  Smoke test results saved to: {smoke_results_path}")

def run_audit_model(best_model_path, output_dir):
    """Run the baseline audit and capture results."""
    section("PHASE 3B: BASELINE AUDIT")
    
    audit_results_path = os.path.join(output_dir, "audit_results.txt")
    tee = TeeTee(audit_results_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    from scripts.audit_model import run_audit
    try:
        passed = run_audit(best_model_path)
        print("\n  ✓ AUDIT COMPLETE")
        return passed
    except Exception as e:
        print(f"\n  ✗ AUDIT FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"  Audit results saved to: {audit_results_path}")


# ──────────────────────────────────────────────────────────────
#  Phase 4: Final Report
# ──────────────────────────────────────────────────────────────

def generate_report(args, best_params, best_metrics, train_dir, smoke_passed, audit_passed, output_dir, total_time):
    """Generate the final summary report with all results and file locations."""
    section("PHASE 4: FINAL REPORT")

    report_path = os.path.join(output_dir, "pipeline_report.txt")

    # Collect all generated files
    generated_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            full = os.path.join(root, f)
            size = os.path.getsize(full)
            rel = os.path.relpath(full, output_dir)
            generated_files.append((rel, size))

    report_lines = [
        "=" * 70,
        "  TEMPORAL GNN MI — FULL PIPELINE REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "─── PIPELINE SUMMARY ───",
        f"  Total wall time:         {total_time:.1f}s ({total_time/60:.1f} min)",
        f"  Smoke test:              {'PASSED ✓' if smoke_passed else 'FAILED ✗'}",
        f"  Baseline audit:          {'PASSED ✓' if audit_passed else 'FAILED ✗'}",
        "",
        "─── PHASE 1: HYPERPARAMETER TUNING ───",
        f"  Strategy:                Bayesian (Optuna)",
        f"  Trials:                  {args.n_trials}",
        f"  Epochs per trial:        {args.tune_epochs}",
        f"  Tuning train samples:    {args.tune_train_samples}",
        f"  Tuning val samples:      {args.tune_val_samples}",
        "",
        "  Best Hyperparameters:",
    ]
    for k, v in best_params.items():
        report_lines.append(f"    {k:25s} = {v}")

    report_lines += [
        "",
        "─── PHASE 2: FULL TRAINING ───",
        f"  Train samples:           {args.train_samples}",
        f"  Val samples:             {args.val_samples}",
        f"  Max epochs:              {args.epochs}",
        f"  Early stopping patience: {args.patience}",
        "",
        "  Best Validation Metrics:",
        f"    ROC-AUC:               {best_metrics.get('roc_auc', 'N/A')}",
        f"    Accuracy:              {best_metrics.get('accuracy', 'N/A')}",
        f"    Average Precision:     {best_metrics.get('ap', 'N/A')}",
        f"    Loss:                  {best_metrics.get('loss', 'N/A')}",
        f"    Source Shuffle Acc:    {best_metrics.get('source_shuffle_acc', 'N/A')}",
        "",
        "─── GENERATED FILES ───",
    ]
    
    # Group files by directory
    dirs_seen = {}
    for rel, size in sorted(generated_files):
        parent = os.path.dirname(rel) or "."
        if parent not in dirs_seen:
            dirs_seen[parent] = []
            report_lines.append(f"\n  [{parent}/]")
        
        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
        if size >= 1024 * 1024:
            size_str = f"{size/(1024*1024):.1f} MB"
        report_lines.append(f"    {os.path.basename(rel):40s}  {size_str}")

    report_lines += [
        "",
        "─── KEY OUTPUT LOCATIONS ───",
        f"  Full report:             {report_path}",
        f"  Training dashboard:      {os.path.join(train_dir, 'training_dashboard.png')}",
        f"  Epoch timing:            {os.path.join(train_dir, 'epoch_timing.png')}",
        f"  Training history (JSON): {os.path.join(train_dir, 'training_history.json')}",
        f"  Best model checkpoint:   {os.path.join(train_dir, 'best_model.pt')}",
        f"  Tuning results:          {os.path.join(output_dir, 'tuning', 'tuning_results.json')}",
        "",
        "=" * 70,
        "  END OF REPORT",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Report saved to: {report_path}")

    return report_path


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Tuning → Training → Smoke Test → Report"
    )
    # Tuning args
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of Optuna tuning trials")
    parser.add_argument("--tune_epochs", type=int, default=20,
                        help="Epochs per tuning trial")
    parser.add_argument("--tune_train_samples", type=int, default=5000,
                        help="Training samples for tuning phase")
    parser.add_argument("--tune_val_samples", type=int, default=1000,
                        help="Validation samples for tuning phase")

    # Training args
    parser.add_argument("--train_samples", type=int, default=20000,
                        help="Training samples for full training")
    parser.add_argument("--val_samples", type=int, default=3000,
                        help="Validation samples for full training")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs for full training")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs/full_pipeline",
                        help="Root output directory for all pipeline artifacts")
    parser.add_argument("--skip_tuning", action="store_true",
                        help="Skip tuning and use default hyperparameters")
    parser.add_argument("--tuning_results", type=str, default=None,
                        help="Path to existing tuning_results.json to skip tuning")

    args = parser.parse_args()

    # Timestamp the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set up dual logging (console + file)
    log_path = os.path.join(output_dir, "pipeline_log.txt")
    logger = Logger(log_path)
    sys.stdout = logger

    set_seed(args.seed)
    total_start = time.time()

    section("TEMPORAL GNN MI — FULL PIPELINE")
    print(f"  Timestamp:   {timestamp}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Seed:        {args.seed}")
    print(f"  Device:      {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")

    try:
        # ── Phase 1: Tuning ──
        if args.tuning_results:
            print(f"\n  Loading existing tuning results from: {args.tuning_results}")
            with open(args.tuning_results, 'r') as f:
                tuning_data = json.load(f)
            best_params = tuning_data['best_params']
            # Copy the tuning results into our output dir
            tuning_dest = os.path.join(output_dir, "tuning")
            os.makedirs(tuning_dest, exist_ok=True)
            shutil.copy2(args.tuning_results, os.path.join(tuning_dest, "tuning_results.json"))
        elif args.skip_tuning:
            print("\n  Skipping tuning — using Run 1 Bayesian search winner.")
            best_params = {
                'memory_dim': 128, 'message_dim': 32, 'time_dim': 16,
                'embedding_dim': 64, 'lr': 0.002, 'batch_size': 32,
                'dropout': 0.108, 'noise_multiplier': 4,
            }
        else:
            best_params = run_tuning(args, output_dir)

        # ── Phase 2: Training ──
        best_metrics, train_dir = run_training(args, best_params, output_dir)

        # ── Phase 3: Smoke Test and Audit ──
        smoke_passed = run_smoke_test(output_dir)
        best_model_path = os.path.join(train_dir, "best_model.pt")
        audit_passed = run_audit_model(best_model_path, output_dir)

        # ── Phase 4: Report ──
        total_time = time.time() - total_start
        generate_report(args, best_params, best_metrics, train_dir, smoke_passed, audit_passed, output_dir, total_time)

    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n  PIPELINE ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n  Full console log saved to: {log_path}")


if __name__ == "__main__":
    main()
