"""
Multi-Seed Master Pipeline

Runs the complete pipeline (training + all MI experiments + shortcut audit
+ Flat-GRU baseline) across multiple seeds for statistical robustness.

Designed for Kaggle T4 GPU (12-hour session limit, batch_size=32 max).

Usage:
    python -m scripts.run_all_seeds
    python -m scripts.run_all_seeds --seeds 42 43 44
    python -m scripts.run_all_seeds --skip_flatgru
    python -m scripts.run_all_seeds --seeds 42 --experiments_only
"""

import os
import sys
import time
import json
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_training(seed, output_dir, train_samples=20000, val_samples=3000,
                 completion_readout=False):
    """Train a TGN model with a specific seed."""
    import torch
    from temporal_gnn_mi.config import DatasetConfig, ModelConfig, TrainingConfig
    from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
    from temporal_gnn_mi.models.tgn import TGN
    from temporal_gnn_mi.training.trainer import Trainer, print_model_summary
    from experiments.utils import seed_everything

    seed_everything(seed)

    dataset_config = DatasetConfig(
        num_nodes=50,
        noise_multiplier=4,
        time_window_delta=5.0,
        num_distractors=3,
        num_partial_motifs=2,
        seed=seed,
    )

    model_config = ModelConfig(
        memory_dim=128,
        message_dim=32,
        time_dim=16,
        embedding_dim=64,
        dropout=0.108,
    )

    train_config = TrainingConfig(
        epochs=200,
        lr=0.002,
        batch_size=32,
        patience=20,
        checkpoint_interval=5,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save config with full environment metadata
    import datetime
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    config_dict = {
        "seed": seed,
        "model": {
            "memory_dim": 128, "message_dim": 32, "time_dim": 16,
            "embedding_dim": 64, "dropout": 0.108,
        },
        "training": {
            "epochs": 200, "batch_size": 32, "lr": 0.002, "patience": 20,
            "completion_readout": completion_readout,
            "completion_loss_weight": train_config.completion_loss_weight,
        },
        "dataset": {
            "train_samples": train_samples, "val_samples": val_samples,
            "noise_multiplier": 4, "num_nodes": 50,
            "time_window_delta": 5.0, "num_distractors": 3,
            "num_partial_motifs": 2,
        },
        "environment": env_info,
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    # Generate datasets
    print(f"  Generating datasets (seed={seed})...")
    train_dataset = TemporalMotifDataset(dataset_config, num_samples=train_samples, split='train')
    val_dataset = TemporalMotifDataset(dataset_config, num_samples=val_samples, split='val')

    # Initialize and train
    model = TGN(model_config, num_nodes=dataset_config.num_nodes, edge_feat_dim=16)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=output_dir,
    )
    best_metrics = trainer.train(completion_readout=completion_readout)
    return best_metrics


def run_flatgru_baseline(seed, output_dir, train_samples=20000, val_samples=3000):
    """Train a Flat-GRU baseline model with the same seed."""
    import torch
    from temporal_gnn_mi.config import DatasetConfig, ModelConfig, TrainingConfig
    from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
    from temporal_gnn_mi.datasets.batch_utils import temporal_collate_fn
    from temporal_gnn_mi.models.flat_gru import FlatGRU
    from temporal_gnn_mi.training.trainer import Trainer
    from torch.utils.data import DataLoader
    from experiments.utils import seed_everything
    import torch.nn as nn

    seed_everything(seed)

    dataset_config = DatasetConfig(
        num_nodes=50,
        noise_multiplier=4,
        time_window_delta=5.0,
        num_distractors=3,
        num_partial_motifs=2,
        seed=seed,
    )

    model_config = ModelConfig(
        memory_dim=128,
        message_dim=32,
        time_dim=16,
        embedding_dim=64,
        dropout=0.108,
    )

    train_config = TrainingConfig(
        epochs=200,
        lr=0.002,
        batch_size=32,
        patience=20,
        checkpoint_interval=5,
    )

    flat_dir = os.path.join(output_dir, 'flat_gru')
    os.makedirs(flat_dir, exist_ok=True)

    # Generate datasets (same seed = same data)
    print(f"  Generating datasets for Flat-GRU (seed={seed})...")
    train_dataset = TemporalMotifDataset(dataset_config, num_samples=train_samples, split='train')
    val_dataset = TemporalMotifDataset(dataset_config, num_samples=val_samples, split='val')

    # Initialize Flat-GRU
    model = FlatGRU(model_config, num_nodes=dataset_config.num_nodes, edge_feat_dim=16)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=flat_dir,
    )
    best_metrics = trainer.train()
    return best_metrics


def run_experiments_for_seed(seed, run_dir, experiment_dir, skip_exp7=False):
    """Run all MI experiments for a single seed."""
    from scripts.run_experiments import EXPERIMENTS, DEFAULT_ORDER, run_experiment

    model_path = os.path.join(run_dir, 'best_model.pt')
    checkpoint_dir = run_dir

    if not os.path.exists(model_path):
        print(f"  ✗ No best_model.pt found at {model_path}")
        return {}

    results = {}
    for key in DEFAULT_ORDER:
        if skip_exp7 and key == 'exp7':
            print(f"\n  Skipping {EXPERIMENTS[key]['name']} (use run_exp7_standalone)")
            results[key] = 'SKIPPED'
            continue
        exp = EXPERIMENTS[key]
        print(f"\n  Running {exp['name']}...")
        try:
            status = run_experiment(key, exp, model_path, checkpoint_dir,
                                     experiment_dir, 'cuda' if _has_cuda() else 'cpu', seed)
            results[key] = status
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            traceback.print_exc()
            results[key] = f'FAILED: {e}'

    return results


def run_permutation_test_for_seed(experiment_dir, seed):
    """Run the permutation test on exp2 results."""
    from experiments.static.exp2_composition import run_permutation_test

    results_path = os.path.join(experiment_dir, 'experiment_2_composition',
                                'experiment_2_composition.pt')
    if os.path.exists(results_path):
        print(f"  Running permutation test...")
        return run_permutation_test(results_path, seed=seed)
    else:
        print(f"  ✗ No exp2 results found for permutation test")
        return None


def run_audit_for_seed(run_dir):
    """Run shortcut audit for a single seed."""
    from scripts.audit_model import run_audit
    model_path = os.path.join(run_dir, 'best_model.pt')
    if os.path.exists(model_path):
        return run_audit(model_path)
    else:
        print(f"  ✗ No best_model.pt for audit")
        return None


def _has_cuda():
    import torch
    return torch.cuda.is_available()


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed pipeline for TGN MI experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45],
                        help='Seeds to run (default: 42 43 44 45)')
    parser.add_argument('--base_dir', type=str, default='runs',
                        help='Base output directory')
    parser.add_argument('--train_samples', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=3000)
    parser.add_argument('--skip_flatgru', action='store_true',
                        help='Skip Flat-GRU baseline training')
    parser.add_argument('--experiments_only', action='store_true',
                        help='Skip training, run experiments on existing checkpoints')
    parser.add_argument('--completion_readout', action='store_true', default=False,
                        help='Jointly train completion-time classifier '
                             '(reads memory at B→C, not end of sequence). '
                             'Requires counterfactual decision steps for negatives.')
    parser.add_argument('--no_completion_readout', action='store_false',
                        dest='completion_readout',
                        help='Disable completion-time classifier (use final-state only)')
    parser.add_argument('--compile_only', action='store_true',
                        help='Only run compile_results on existing experiment outputs')
    parser.add_argument('--skip_exp7', action='store_true',
                        help='Skip Exp7 (shuffled-label permutation test) during '
                             'the experiment phase. Run it separately post-training '
                             'with: python -m scripts.run_exp7_standalone')
    parser.add_argument('--skip_exp2_permutation', action='store_true',
                        help='Skip the Exp2 permutation test (10k permutations + '
                             '10k bootstrap) to save ~15-20 min per seed. '
                             'Run it separately afterward if needed.')
    args = parser.parse_args()

    total_start = time.time()
    seeds = args.seeds

    print("\n" + "█" * 70)
    print("  MULTI-SEED MI PIPELINE")
    print("█" * 70)
    print(f"  Seeds:          {seeds}")
    print(f"  Base dir:       {args.base_dir}")
    print(f"  Train samples:  {args.train_samples}")
    print(f"  Val samples:    {args.val_samples}")
    print(f"  Skip Flat-GRU:  {args.skip_flatgru}")
    print(f"  Completion RO:  {args.completion_readout}")
    print(f"  Device:         {'cuda' if _has_cuda() else 'cpu'}")
    print()

    if args.compile_only:
        print("  Compile-only mode: skipping training and experiments")
        _run_compile(args.base_dir, seeds)
        return

    seed_results = {}

    for seed_idx, seed in enumerate(seeds):
        seed_start = time.time()
        run_dir = os.path.join(args.base_dir, f'seed_{seed}')
        experiment_dir = os.path.join(run_dir, 'experiments')

        print(f"\n{'━' * 70}")
        print(f"  SEED {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'━' * 70}")

        seed_results[seed] = {}

        # ── Phase 1: Training ─────────────────────────────────
        if not args.experiments_only:
            print(f"\n  ── Phase 1: Training TGN (seed={seed}) ──")
            try:
                metrics = run_training(seed, run_dir, args.train_samples, args.val_samples,
                                       completion_readout=args.completion_readout)
                training_result = {
                    'status': 'OK',
                    'roc_auc': metrics.get('roc_auc', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'ap': metrics.get('ap', 0),
                }
                if 'completion_roc_auc' in metrics:
                    training_result['completion_roc_auc'] = metrics['completion_roc_auc']
                    training_result['completion_accuracy'] = metrics['completion_accuracy']
                    training_result['completion_loss'] = metrics['completion_loss']
                seed_results[seed]['training'] = training_result
            except Exception as e:
                print(f"  ✗ Training failed: {e}")
                traceback.print_exc()
                seed_results[seed]['training'] = {'status': f'FAILED: {e}'}
                continue  # Skip rest for this seed

        # ── Phase 2: Shortcut Audit ───────────────────────────
        if not args.experiments_only:
            print(f"\n  ── Phase 2: Shortcut Audit (seed={seed}) ──")
            try:
                audit_passed = run_audit_for_seed(run_dir)
                seed_results[seed]['audit'] = 'PASS' if audit_passed else 'FAIL'
            except Exception as e:
                print(f"  ✗ Audit failed: {e}")
                seed_results[seed]['audit'] = f'FAILED: {e}'

        # ── Phase 3: MI Experiments ───────────────────────────
        print(f"\n  ── Phase 3: MI Experiments (seed={seed}) ──")
        try:
            exp_results = run_experiments_for_seed(seed, run_dir, experiment_dir,
                                                     skip_exp7=args.skip_exp7)
            seed_results[seed]['experiments'] = exp_results
        except Exception as e:
            print(f"  ✗ Experiments failed: {e}")
            traceback.print_exc()
            seed_results[seed]['experiments'] = {'status': f'FAILED: {e}'}

        # ── Phase 3b: Permutation test ────────────────────────
        if args.skip_exp2_permutation:
            print(f"\n  ── Phase 3b: Permutation Test SKIPPED ──")
            seed_results[seed]['permutation_test'] = 'SKIPPED'
        else:
            print(f"\n  ── Phase 3b: Permutation Test (seed={seed}) ──")
            try:
                perm_results = run_permutation_test_for_seed(experiment_dir, seed)
                if perm_results:
                    seed_results[seed]['permutation_test'] = {
                        'p_value': perm_results['p_value'],
                        'observed_mean': perm_results['observed_mean'],
                        'ci_95': [perm_results['ci_95_lower'], perm_results['ci_95_upper']],
                    }
            except Exception as e:
                print(f"  ✗ Permutation test failed: {e}")

        # ── Phase 4: Flat-GRU Baseline ────────────────────────
        if not args.skip_flatgru and not args.experiments_only:
            print(f"\n  ── Phase 4: Flat-GRU Baseline (seed={seed}) ──")
            try:
                flat_metrics = run_flatgru_baseline(seed, run_dir,
                                                     args.train_samples, args.val_samples)
                seed_results[seed]['flat_gru'] = {
                    'status': 'OK',
                    'roc_auc': flat_metrics.get('roc_auc', 0),
                    'accuracy': flat_metrics.get('accuracy', 0),
                    'ap': flat_metrics.get('ap', 0),
                }
            except Exception as e:
                print(f"  ✗ Flat-GRU failed: {e}")
                traceback.print_exc()
                seed_results[seed]['flat_gru'] = {'status': f'FAILED: {e}'}

        seed_elapsed = time.time() - seed_start
        print(f"\n  Seed {seed} completed in {seed_elapsed/60:.1f} min")

    # ── Save cross-seed summary ──────────────────────────────────
    summary_path = os.path.join(args.base_dir, 'multi_seed_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(seed_results, f, indent=2, default=str)
    print(f"\n  ✓ Summary saved to: {summary_path}")

    # ── Phase 5: Cross-Seed Consistency (Exp 9) ────────────────────
    if len(seeds) >= 2:
        print(f"\n{'━' * 70}")
        print(f"  CROSS-SEED ANALYSIS (Experiment 9)")
        print(f"{'━' * 70}")
        try:
            from experiments.static.exp9_cross_seed_consistency import run as run_exp9
            run_exp9(args.base_dir, seeds, output_dir=args.base_dir)
        except Exception as e:
            print(f"  ⚠ Exp9 failed: {e}")
            traceback.print_exc()

    # ── Compile results ──────────────────────────────────────────
    _run_compile(args.base_dir, seeds)

    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    print(f"\n{'█' * 70}")
    print(f"  PIPELINE COMPLETE: {hours}h {minutes}m")
    print(f"{'█' * 70}")


def _run_compile(base_dir, seeds):
    """Run compile_results if the module is available."""
    try:
        from scripts.compile_results import compile_all
        compile_all(base_dir, seeds)
    except ImportError:
        print("  ⚠ compile_results.py not found — skipping compilation")
    except Exception as e:
        print(f"  ⚠ Compilation failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
