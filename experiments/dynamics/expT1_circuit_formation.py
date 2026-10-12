"""
Experiment T1 — Circuit Formation Timeline

Question:  When does the compositional pathway (mem_B carrying A's info
           to the B→C decision) become causally load-bearing during training?

Method:    At each checkpoint, zero-ablate mem_B at the B→C step for fixed
           positive samples and measure causal effect, flip rate, and accuracy.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    load_model, prepare_eval_dataset, get_positive_samples, find_checkpoints,
    single_sample_batch, run_forward, get_motif_steps, setup_output_dir,
    setup_plot_style, save_figure, seed_everything,
)

import matplotlib.pyplot as plt


def run(checkpoint_dir, output_dir, num_samples=100, seed=42, device='cpu',
        **kwargs):
    """Run Experiment T1: Circuit Formation Timeline."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT T1 — Circuit Formation Timeline")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_T1_formation')

    # ── Find checkpoints ──────────────────────────────────────────
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"  ✗ No checkpoints found in {checkpoint_dir}")
        return None
    print(f"  Found {len(checkpoints)} checkpoints: epochs {[e for e, _ in checkpoints]}")

    # ── Generate fixed positive samples ───────────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} fixed positive samples")

    # ── Evaluate at each checkpoint ───────────────────────────────
    epochs = []
    mean_causal_effects = []
    flip_rates = []
    accuracies = []

    for ckpt_idx, (epoch, ckpt_path) in enumerate(checkpoints):
        print(f"  Checkpoint {ckpt_idx + 1}/{len(checkpoints)} (epoch {epoch})...")

        model, config, _ = load_model(ckpt_path, device=device)
        epochs.append(epoch)

        causal_effects = []
        flips = 0
        correct = 0
        valid = 0

        for sample in positives:
            batch = single_sample_batch(sample)
            motif_steps = get_motif_steps(model, batch)
            m1, m2 = motif_steps[0]

            if m1 < 0 or m2 < 0:
                continue

            valid += 1

            # Baseline
            baseline_logit = run_forward(model, batch).item()

            # Patched: zero mem_B at B→C step
            interventions = {
                (m2, 'memory_pre_src'): lambda x: torch.zeros_like(x)
            }
            patched_logit = run_forward(model, batch,
                                        interventions=interventions).item()

            # Causal effect
            effect = abs(baseline_logit - patched_logit)
            causal_effects.append(effect)

            # Flip: does zeroing mem_B change the prediction sign?
            if (baseline_logit > 0) != (patched_logit > 0):
                flips += 1

            # Accuracy: is the baseline prediction correct? (positive sample → logit > 0)
            if baseline_logit > 0:
                correct += 1

        if valid > 0:
            mean_causal_effects.append(np.mean(causal_effects))
            flip_rates.append(flips / valid)
            accuracies.append(correct / valid)
        else:
            mean_causal_effects.append(0.0)
            flip_rates.append(0.0)
            accuracies.append(0.0)

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'epochs': epochs,
        'mean_causal_effects': torch.tensor(mean_causal_effects),
        'flip_rates': torch.tensor(flip_rates),
        'accuracies': torch.tensor(accuracies),
        'num_samples': len(positives),
    }
    torch.save(results, os.path.join(out, 'experiment_T1_formation.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: accuracy and flip rate
    ax = axes[0]
    ax.plot(epochs, accuracies, color='#2ecc71', linewidth=2.5, marker='o',
            markersize=6, label='Accuracy (positive samples)')
    ax.plot(epochs, flip_rates, color='#e74c3c', linewidth=2.5, marker='s',
            markersize=6, label='Flip rate (mem_B zeroed)')
    ax.axhline(0.5, color='#bbb', linestyle=':', linewidth=1)
    ax.set_ylabel('Rate')
    ax.set_title('Circuit Formation — Accuracy & Prediction Flip Rate',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom: mean causal effect
    ax = axes[1]
    ax.plot(epochs, mean_causal_effects, color='#3498db', linewidth=2.5,
            marker='D', markersize=6, label='Mean |baseline − patched|')
    ax.fill_between(epochs, mean_causal_effects, alpha=0.15, color='#3498db')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean causal effect')
    ax.set_title('Causal Importance of mem_B at B→C', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_T1_formation.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Timeline:")
    print(f"  {'Epoch':>6s} {'Accuracy':>10s} {'Flip Rate':>10s} {'Causal Effect':>14s}")
    print(f"  {'─' * 44}")
    for i, ep in enumerate(epochs):
        print(f"  {ep:6d} {accuracies[i]:>10.1%} {flip_rates[i]:>10.1%} "
              f"{mean_causal_effects[i]:>14.4f}")

    # Identify formation epoch (first epoch where flip rate > 0.5)
    formation_epoch = None
    for i, fr in enumerate(flip_rates):
        if fr > 0.5:
            formation_epoch = epochs[i]
            break

    if formation_epoch:
        print(f"\n  → Circuit formation epoch (flip rate > 50%): epoch {formation_epoch}")
    else:
        print(f"\n  → No epoch found with flip rate > 50%")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment T1: Circuit Formation')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_dir, args.num_samples, args.seed, args.device)
