"""Hyperparameter Tuning for TGN on Temporal Motif Completion.

Supports three search strategies:
  - random:   Random Search (fast, surprisingly effective)
  - grid:     Grid Search (exhaustive, slow)
  - bayesian: Bayesian Optimization via Optuna (smart, efficient)

Usage:
    cd Project3_Codebase
    python -m scripts.tune_hyperparams --strategy random --n_trials 20
    python -m scripts.tune_hyperparams --strategy grid
    python -m scripts.tune_hyperparams --strategy bayesian --n_trials 50
"""
import argparse
import os
import sys
import json
import random
import itertools
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_gnn_mi.config import DatasetConfig, ModelConfig, TrainingConfig
from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
from temporal_gnn_mi.datasets.utils import set_seed
from temporal_gnn_mi.models.tgn import TGN
from temporal_gnn_mi.training.trainer import Trainer


# ──────────────────────────────────────────────────────────
#  Hyperparameter Search Space
# ──────────────────────────────────────────────────────────

SEARCH_SPACE = {
    'memory_dim':       [32, 64, 128],
    'message_dim':      [32, 64, 128],
    'time_dim':         [16, 32, 64],
    'embedding_dim':    [32, 64, 128],
    'lr':               [1e-4, 5e-4, 1e-3, 2e-3],
    'batch_size':       [16, 32, 64],
    'dropout':          [0.0, 0.1, 0.2],
    'noise_multiplier': [4, 7, 10],
}


def train_single_config(config_dict, train_dataset, val_dataset, output_dir, trial_id):
    """Train a single configuration and return validation metrics."""
    
    model_config = ModelConfig(
        memory_dim=config_dict['memory_dim'],
        message_dim=config_dict['message_dim'],
        time_dim=config_dict['time_dim'],
        embedding_dim=config_dict['embedding_dim'],
        dropout=config_dict['dropout'],
    )
    train_config = TrainingConfig(
        epochs=50,  # Shorter for tuning — just enough to see trends
        lr=config_dict['lr'],
        batch_size=config_dict['batch_size'],
        patience=10,
    )
    
    model = TGN(
        config=model_config,
        num_nodes=50,
        edge_feat_dim=16
    )
    
    trial_dir = os.path.join(output_dir, f"trial_{trial_id:03d}")
    trainer = Trainer(model, train_dataset, val_dataset, train_config, trial_dir)
    
    best_metrics = trainer.train()
    return best_metrics


# ──────────────────────────────────────────────────────────
#  Strategy 1: Random Search
# ──────────────────────────────────────────────────────────

def random_search(n_trials, train_dataset, val_dataset, output_dir):
    """Sample n_trials random configurations from the search space."""
    print(f"\n{'='*60}")
    print(f"  RANDOM SEARCH: {n_trials} trials")
    print(f"{'='*60}\n")
    
    results = []
    for i in range(n_trials):
        config = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        print(f"\n--- Trial {i+1}/{n_trials} ---")
        print(f"  Config: {config}")
        
        metrics = train_single_config(config, train_dataset, val_dataset, output_dir, i)
        result = {'trial': i, 'config': config, 'metrics': metrics}
        results.append(result)
        
        print(f"  Val ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    
    return results


# ──────────────────────────────────────────────────────────
#  Strategy 2: Grid Search
# ──────────────────────────────────────────────────────────

def grid_search(train_dataset, val_dataset, output_dir):
    """Exhaustive search over all combinations (WARNING: can be very slow)."""
    
    # Use a reduced grid to avoid combinatorial explosion
    reduced_grid = {
        'memory_dim':    [64, 128],
        'message_dim':   [64, 128],
        'time_dim':      [32],
        'embedding_dim': [64],
        'lr':            [5e-4, 1e-3],
        'batch_size':    [32],
        'dropout':       [0.0, 0.1],
        'noise_multiplier': [7],
    }
    
    keys = list(reduced_grid.keys())
    combos = list(itertools.product(*[reduced_grid[k] for k in keys]))
    
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH: {len(combos)} combinations")
    print(f"{'='*60}\n")
    
    results = []
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"\n--- Trial {i+1}/{len(combos)} ---")
        print(f"  Config: {config}")
        
        metrics = train_single_config(config, train_dataset, val_dataset, output_dir, i)
        result = {'trial': i, 'config': config, 'metrics': metrics}
        results.append(result)
        
        print(f"  Val ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    
    return results


# ──────────────────────────────────────────────────────────
#  Strategy 3: Bayesian Optimization (Optuna)
# ──────────────────────────────────────────────────────────

def bayesian_search(n_trials, train_dataset, val_dataset, output_dir):
    """Bayesian optimization using Optuna's Tree-structured Parzen Estimator."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: Optuna not installed. Run: pip install optuna")
        print("Falling back to Random Search.")
        return random_search(n_trials, train_dataset, val_dataset, output_dir)
    
    print(f"\n{'='*60}")
    print(f"  BAYESIAN OPTIMIZATION (Optuna): {n_trials} trials")
    print(f"{'='*60}\n")
    
    trial_counter = [0]
    
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
        print(f"\n--- Optuna Trial {idx+1}/{n_trials} ---")
        print(f"  Config: {config}")
        
        metrics = train_single_config(config, train_dataset, val_dataset, output_dir, idx)
        roc_auc = metrics.get('roc_auc', 0.5)
        print(f"  Val ROC-AUC: {roc_auc:.4f}")
        return roc_auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL: #{study.best_trial.number}")
    print(f"  Best Val ROC-AUC: {study.best_value:.4f}")
    print(f"  Best Config: {study.best_params}")
    print(f"{'='*60}")
    
    return [{'trial': t.number, 'config': t.params, 'roc_auc': t.value} for t in study.trials]


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for TGN MI")
    parser.add_argument("--strategy", type=str, default="random",
                        choices=["random", "grid", "bayesian"],
                        help="Search strategy: random, grid, or bayesian")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials (for random/bayesian)")
    parser.add_argument("--train_samples", type=int, default=5000,
                        help="Training samples (smaller for tuning speed)")
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="Validation samples")
    parser.add_argument("--output_dir", type=str, default="runs/tuning",
                        help="Output directory for tuning results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Generate datasets once (shared across all trials)
    print("Generating datasets for tuning...")
    dataset_config = DatasetConfig(seed=args.seed)
    train_dataset = TemporalMotifDataset(dataset_config, num_samples=args.train_samples, split='train')
    val_dataset = TemporalMotifDataset(dataset_config, num_samples=args.val_samples, split='val')
    print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Run selected strategy
    if args.strategy == 'random':
        results = random_search(args.n_trials, train_dataset, val_dataset, args.output_dir)
    elif args.strategy == 'grid':
        results = grid_search(train_dataset, val_dataset, args.output_dir)
    elif args.strategy == 'bayesian':
        results = bayesian_search(args.n_trials, train_dataset, val_dataset, args.output_dir)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"tuning_results_{args.strategy}.json")
    
    # Find and print best result
    best = max(results, key=lambda r: r.get('metrics', r).get('roc_auc', 0))
    
    with open(results_path, 'w') as f:
        json.dump({'best': best, 'all_trials': results}, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    print(f"\nBEST CONFIG:")
    print(f"  {best.get('config', best.get('params', {}))}")
    roc = best.get('metrics', best).get('roc_auc', 'N/A')
    print(f"  Val ROC-AUC: {roc}")


if __name__ == "__main__":
    main()
