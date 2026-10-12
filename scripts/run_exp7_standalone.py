"""
Standalone Exp7 runner — decoupled from the training session.

Run this after training completes, in a separate Kaggle session or locally.
It loads existing checkpoints and runs the shuffled-label permutation test.

Usage:
    python -m scripts.run_exp7_standalone --base_dir runs --seeds 42 43 44 45
    python -m scripts.run_exp7_standalone --base_dir runs --seeds 42 --num_permutations 500
"""

import os
import sys
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description='Run Exp7 (shuffled-label null control) on existing checkpoints',
    )
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--base_dir', type=str, default='runs')
    parser.add_argument('--num_permutations', type=int, default=250,
                        help='Number of permutations (default: 250)')
    parser.add_argument('--num_samples', type=int, default=300)
    args = parser.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Permutations: {args.num_permutations}")

    for seed in args.seeds:
        run_dir = os.path.join(args.base_dir, f'seed_{seed}')
        model_path = os.path.join(run_dir, 'best_model.pt')
        experiment_dir = os.path.join(run_dir, 'experiments')

        if not os.path.exists(model_path):
            print(f"\n✗ No best_model.pt for seed {seed} at {model_path}")
            continue

        output_dir = experiment_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'━' * 60}")
        print(f"  Exp7 — Seed {seed}")
        print(f"{'━' * 60}")

        try:
            from experiments.static.exp7_null_control import run
            run(model_path, output_dir,
                num_samples=args.num_samples,
                num_permutations=args.num_permutations,
                seed=seed, device=device)
            print(f"  ✓ Seed {seed} complete")
        except Exception as e:
            print(f"  ✗ Seed {seed} failed: {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    main()
