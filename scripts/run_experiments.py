"""
Unified MI Experiment Runner

Executes mechanistic interpretability experiments from the MI_EXPERIMENT_ROADMAP.
Experiments can be selected individually, by category, or run all at once.

Usage:
    # Run all experiments
    python -m scripts.run_experiments --model_path runs/.../best_model.pt --checkpoint_dir runs/.../training/

    # Run specific experiments
    python -m scripts.run_experiments --experiments exp1 exp4 T5 --model_path ...

    # Run by category
    python -m scripts.run_experiments --category static --model_path ...
    python -m scripts.run_experiments --category dynamics --checkpoint_dir ...

    # List available experiments
    python -m scripts.run_experiments --list
"""

import os
import sys
import time
import argparse
import traceback
import io

# Fix Windows console encoding for Unicode box-drawing characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════════
#  Experiment registry
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    # ── Part A: Static Circuit Analysis ────────────────────────────
    'exp1': {
        'name':        'Experiment 1 — Causal Importance Map',
        'category':    'static',
        'module':      'experiments.static.exp1_causal_map',
        'requires':    'model',
        'est_time':    '2–30 min',
        'description': 'Heatmap of causal importance across event steps and intervention points',
    },
    'exp2': {
        'name':        'Experiment 2 — Composition Test',
        'category':    'static',
        'module':      'experiments.static.exp2_composition',
        'requires':    'model',
        'est_time':    '~5 min',
        'description': 'Composition recovery ratio: does info flow A→B→C or independently?',
    },
    'exp3': {
        'name':        'Experiment 3A — Gate-Level Analysis',
        'category':    'static',
        'module':      'experiments.static.exp3_gate_analysis',
        'requires':    'model',
        'est_time':    '~2 min',
        'description': 'Per-dimension GRU gate discriminability (motif vs. noise updates)',
    },
    'exp4': {
        'name':        'Experiment 4 — Time-Window Mechanism',
        'category':    'static',
        'module':      'experiments.static.exp4_time_window',
        'requires':    'model',
        'est_time':    '~1 min',
        'description': 'Logit response curve across synthetic Δt sweep at B→C',
    },
    'exp5': {
        'name':        'Experiment 5 — Source-Shuffle Failure',
        'category':    'static',
        'module':      'experiments.static.exp5_source_shuffle',
        'requires':    'model',
        'est_time':    '~2 min',
        'description': 'Why source_shuffle accuracy is lower: memory similarity analysis',
    },

    'exp3b': {
        'name':        'Experiment 3B — Causal Gate Analysis',
        'category':    'static',
        'module':      'experiments.static.exp3_gate_analysis',
        'entry_point': 'run_3b',
        'requires':    'model',
        'est_time':    '~3 min',
        'description': 'Causal gate clamping: clamp z_t/r_t to noise mean at motif step',
    },
    'exp6': {
        'name':        'Experiment 6 — Ablation Method Comparison',
        'category':    'static',
        'module':      'experiments.static.exp6_ablation_comparison',
        'requires':    'model',
        'est_time':    '~10 min',
        'description': 'Mean-ablation vs zero-ablation: validates methodological choice in Exp 1',
    },
    'exp7': {
        'name':        'Experiment 7 — Shuffled-Label Null Control',
        'category':    'static',
        'module':      'experiments.static.exp7_null_control',
        'requires':    'model',
        'est_time':    '~15 min',
        'description': 'Null control for Exp 3A: shuffled motif/noise labels should show no signal',
    },
    'exp8': {
        'name':        'Experiment 8 — Memory Knockout',
        'category':    'static',
        'module':      'experiments.static.exp8_memory_knockout',
        'requires':    'model',
        'est_time':    '~3 min',
        'description': 'Zero memory at B→C step to test causal necessity of temporal memory',
    },

    # ── Part B: Training Dynamics ──────────────────────────────────
    'T5': {
        'name':        'Experiment T5 — Learned Frequencies',
        'category':    'dynamics',
        'module':      'experiments.dynamics.expT5_frequencies',
        'requires':    'checkpoints',
        'est_time':    'seconds',
        'description': 'Time encoder frequency evolution (no forward passes needed)',
    },
    'T2': {
        'name':        'Experiment T2 — Corruption Learning Order',
        'category':    'dynamics',
        'module':      'experiments.dynamics.expT2_corruption_order',
        'requires':    'checkpoints',
        'est_time':    '~5 min',
        'description': 'Per-corruption accuracy at each training checkpoint',
    },
    'T1': {
        'name':        'Experiment T1 — Circuit Formation',
        'category':    'dynamics',
        'module':      'experiments.dynamics.expT1_circuit_formation',
        'requires':    'checkpoints',
        'est_time':    '~2 min',
        'description': 'When does the compositional pathway become load-bearing?',
    },
    'T4': {
        'name':        'Experiment T4 — Time-Window Sharpness',
        'category':    'dynamics',
        'module':      'experiments.dynamics.expT4_time_sharpness',
        'requires':    'checkpoints',
        'est_time':    '~5 min',
        'description': 'Time-window sharpness evolution over training',
    },
    'T3': {
        'name':        'Experiment T3 — Gate Specialization',
        'category':    'dynamics',
        'module':      'experiments.dynamics.expT3_gate_specialization',
        'requires':    'checkpoints',
        'est_time':    '~2 min',
        'description': 'When do gate dimensions specialize for motif detection?',
    },
}

# Recommended execution order from the roadmap
DEFAULT_ORDER = ['T5', 'T2', 'exp4', 'exp1', 'exp2', 'exp5', 'T1', 'T4', 'T3', 'exp3', 'exp3b',
                 'exp6', 'exp7', 'exp8']


def _auto_detect_paths(base_dirs):
    """Try to auto-detect model and checkpoint paths."""
    search_dirs = list(base_dirs) + [
        'runs',
        os.path.join('Extracted_Results', 'runs'),
        os.path.join('Kaggle_Results', 'runs'),
    ]

    model_path = None
    checkpoint_dir = None

    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if f == 'best_model.pt' and model_path is None:
                    model_path = os.path.join(root, f)
                    # Checkpoint dir is the same directory
                    if checkpoint_dir is None:
                        checkpoint_dir = root

    return model_path, checkpoint_dir


def list_experiments():
    """Print a formatted table of all available experiments."""
    print("\n  Available MI Experiments:")
    print(f"  {'Key':<6s} {'Name':<45s} {'Category':<10s} {'Time':<12s}")
    print(f"  {'─' * 75}")

    for key in DEFAULT_ORDER:
        exp = EXPERIMENTS[key]
        print(f"  {key:<6s} {exp['name']:<45s} {exp['category']:<10s} {exp['est_time']:<12s}")

    print(f"\n  Categories: static (Part A), dynamics (Part B)")
    print(f"  Default order follows the MI Experiment Roadmap phases\n")


def run_experiment(key, exp_info, model_path, checkpoint_dir, output_dir,
                   device, seed):
    """Import and run a single experiment."""
    import importlib
    module = importlib.import_module(exp_info['module'])

    kwargs = {
        'output_dir': output_dir,
        'seed': seed,
        'device': device,
    }

    if exp_info['requires'] == 'model':
        if model_path is None:
            print(f"  ✗ Skipping {key}: --model_path not provided")
            return 'SKIPPED'
        kwargs['model_path'] = model_path
    elif exp_info['requires'] == 'checkpoints':
        if checkpoint_dir is None:
            print(f"  ✗ Skipping {key}: --checkpoint_dir not provided")
            return 'SKIPPED'
        kwargs['checkpoint_dir'] = checkpoint_dir

    try:
        entry_point = exp_info.get('entry_point', 'run')
        fn = getattr(module, entry_point)
        fn(**kwargs)
        return 'OK'
    except Exception as e:
        print(f"\n  ✗ FAILED: {key}")
        traceback.print_exc()
        return f'FAILED: {e}'


def main():
    parser = argparse.ArgumentParser(
        description='Run MI experiments on a trained TGN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_experiments --list
  python -m scripts.run_experiments --model_path runs/final_run/best_model.pt --checkpoint_dir runs/final_run/
  python -m scripts.run_experiments --experiments exp1 exp4 T5
  python -m scripts.run_experiments --category static
  python -m scripts.run_experiments --category dynamics
        """,
    )
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments and exit')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Experiment keys to run (e.g., exp1 exp4 T5). '
                             'Default: all experiments.')
    parser.add_argument('--category', type=str, default='all',
                        choices=['static', 'dynamics', 'all'],
                        help='Run experiments by category')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (best_model.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing epoch checkpoints')
    parser.add_argument('--output_dir', type=str, default='runs/experiments',
                        help='Base output directory for experiment results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (auto-detected if not set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    # ── Auto-detect device ────────────────────────────────────────
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Auto-detect paths if not provided ─────────────────────────
    if args.model_path is None or args.checkpoint_dir is None:
        auto_model, auto_ckpt = _auto_detect_paths(['.', '..'])
        if args.model_path is None and auto_model:
            args.model_path = auto_model
        if args.checkpoint_dir is None and auto_ckpt:
            args.checkpoint_dir = auto_ckpt

    # ── Determine which experiments to run ────────────────────────
    if args.experiments:
        # Validate keys
        for key in args.experiments:
            if key not in EXPERIMENTS:
                print(f"  ✗ Unknown experiment key: '{key}'")
                print(f"    Available: {', '.join(EXPERIMENTS.keys())}")
                return
        experiment_keys = args.experiments
    elif args.category != 'all':
        experiment_keys = [k for k in DEFAULT_ORDER
                           if EXPERIMENTS[k]['category'] == args.category]
    else:
        experiment_keys = list(DEFAULT_ORDER)

    # ── Print banner ──────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  MI EXPERIMENT RUNNER")
    print("█" * 70)
    print(f"  Device:         {args.device}")
    print(f"  Model path:     {args.model_path or '(not set)'}")
    print(f"  Checkpoint dir: {args.checkpoint_dir or '(not set)'}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Seed:           {args.seed}")
    print(f"  Experiments:    {len(experiment_keys)}")
    print()

    # Summary table
    print(f"  {'#':<4s} {'Key':<6s} {'Name':<45s} {'Est. Time':<12s}")
    print(f"  {'─' * 69}")
    for i, key in enumerate(experiment_keys):
        exp = EXPERIMENTS[key]
        print(f"  {i+1:<4d} {key:<6s} {exp['name']:<45s} {exp['est_time']:<12s}")
    print()

    # ── Warn about missing paths ──────────────────────────────────
    needs_model = any(EXPERIMENTS[k]['requires'] == 'model'
                      for k in experiment_keys)
    needs_ckpts = any(EXPERIMENTS[k]['requires'] == 'checkpoints'
                      for k in experiment_keys)

    if needs_model and args.model_path is None:
        print("  ⚠ WARNING: Some experiments require --model_path but it's not set.")
        print("    Those experiments will be skipped.")
    if needs_ckpts and args.checkpoint_dir is None:
        print("  ⚠ WARNING: Some experiments require --checkpoint_dir but it's not set.")
        print("    Those experiments will be skipped.")

    # ── Execute experiments ───────────────────────────────────────
    results = {}
    start_all = time.time()

    for i, key in enumerate(experiment_keys):
        exp = EXPERIMENTS[key]
        print(f"\n{'━' * 70}")
        print(f"  [{i+1}/{len(experiment_keys)}] {exp['name']}")
        print(f"  {exp['description']}")
        print(f"{'━' * 70}")

        start = time.time()
        status = run_experiment(key, exp, args.model_path, args.checkpoint_dir,
                                args.output_dir, args.device, args.seed)
        elapsed = time.time() - start
        results[key] = {'status': status, 'time': elapsed}

        print(f"\n  ⏱ {key} completed in {elapsed:.1f}s — {status}")

    total_time = time.time() - start_all

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n\n{'█' * 70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'█' * 70}")
    print(f"  {'Key':<6s} {'Status':<12s} {'Time':>10s}  {'Name'}")
    print(f"  {'─' * 70}")

    ok_count = 0
    for key in experiment_keys:
        r = results[key]
        status = r['status']
        emoji = '✓' if status == 'OK' else ('⊘' if status == 'SKIPPED' else '✗')
        time_str = f"{r['time']:.1f}s"
        name = EXPERIMENTS[key]['name']
        print(f"  {key:<6s} {emoji} {status:<10s} {time_str:>10s}  {name}")
        if status == 'OK':
            ok_count += 1

    print(f"  {'─' * 70}")
    print(f"  Total: {ok_count}/{len(experiment_keys)} succeeded in {total_time:.1f}s")
    print(f"  Results saved to: {os.path.abspath(args.output_dir)}")
    print()


if __name__ == '__main__':
    main()
