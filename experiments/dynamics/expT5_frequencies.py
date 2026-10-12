"""
Experiment T5 — Learned Frequencies

Question:  Does the time encoder learn frequencies that correspond to the
           task's time window Δ=5.0?

Method:    Load time_encoder.w from every checkpoint's state dict (no
           forward passes needed) and track how frequencies evolve during
           training.  Flag any frequencies whose implied period ≈ 5.0.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    find_checkpoints, load_state_dict_only, setup_output_dir,
    setup_plot_style, save_figure,
)

import matplotlib.pyplot as plt


def run(checkpoint_dir, output_dir, **kwargs):
    """Run Experiment T5: Learned Frequencies."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT T5 — Learned Frequencies")
    print("=" * 65)

    out = setup_output_dir(output_dir, 'experiment_T5_frequencies')

    # ── Find checkpoints ──────────────────────────────────────────
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"  ✗ No checkpoints found in {checkpoint_dir}")
        return None
    print(f"  Found {len(checkpoints)} checkpoints: epochs {[e for e, _ in checkpoints]}")

    # ── Extract frequencies from each checkpoint ──────────────────
    epochs = []
    freq_matrix = []  # list of (time_dim//2,) arrays

    for epoch, ckpt_path in checkpoints:
        state_dict, _ = load_state_dict_only(ckpt_path)
        w = state_dict['time_encoder.w'].cpu().numpy()  # (1, time_dim//2)
        freq_matrix.append(w.squeeze(0))
        epochs.append(epoch)

    freq_matrix = np.stack(freq_matrix)  # (num_checkpoints, time_dim//2)
    num_dims = freq_matrix.shape[1]

    # ── Reference frequency ───────────────────────────────────────
    ref_freq = 2 * np.pi / 5.0  # ≈ 1.257

    # ── Final learned frequencies ─────────────────────────────────
    final_freqs = freq_matrix[-1]
    implied_periods = 2 * np.pi / np.abs(final_freqs + 1e-10)

    # Flag task-aligned frequencies (period in 3.0-8.0 range)
    task_aligned = [(d, final_freqs[d], implied_periods[d])
                    for d in range(num_dims)
                    if 3.0 <= implied_periods[d] <= 8.0]

    # ── Save results ──────────────────────────────────────────────
    results = {
        'epochs': epochs,
        'freq_matrix': torch.tensor(freq_matrix),
        'final_frequencies': torch.tensor(final_freqs),
        'implied_periods': torch.tensor(implied_periods),
        'reference_frequency': ref_freq,
    }
    torch.save(results, os.path.join(out, 'experiment_T5_frequencies.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: frequency trajectory over training
    ax = axes[0]
    cmap = plt.cm.viridis
    for d in range(num_dims):
        color = cmap(d / num_dims)
        ax.plot(epochs, freq_matrix[:, d], color=color, alpha=0.7,
                linewidth=1.2, label=f'dim {d}' if d < 5 else None)

    ax.axhline(ref_freq, color='#e74c3c', linestyle='--', linewidth=2,
               alpha=0.8, label=f'2π/5.0 ≈ {ref_freq:.3f}')
    ax.axhline(-ref_freq, color='#e74c3c', linestyle='--', linewidth=2,
               alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learned frequency ω')
    ax.set_title('Frequency Evolution Over Training', fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Right: final frequencies with implied periods
    ax = axes[1]
    sorted_idx = np.argsort(np.abs(final_freqs))[::-1]
    bars = ax.bar(range(num_dims), np.abs(final_freqs[sorted_idx]),
                  color=['#e74c3c' if 3.0 <= implied_periods[d] <= 8.0 else '#3498db'
                         for d in sorted_idx],
                  edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axhline(ref_freq, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'2π/Δ ≈ {ref_freq:.3f} (period=5.0)')
    ax.set_xlabel('Dimension index (sorted by |ω|)')
    ax.set_ylabel('|ω| (frequency magnitude)')
    ax.set_title('Final Learned Frequencies', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add period annotations for top dims
    for rank in range(min(5, num_dims)):
        d = sorted_idx[rank]
        period = implied_periods[d]
        ax.text(rank, np.abs(final_freqs[d]) + 0.05, f'T={period:.1f}',
                ha='center', fontsize=7, color='#555')

    fig.suptitle('Experiment T5 — Time Encoder Frequency Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_T5_frequencies.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Final learned frequencies (sorted by |ω|):")
    print(f"  {'Dim':>4s} {'ω':>10s} {'|ω|':>10s} {'Period':>10s} {'Task-aligned':>14s}")
    print(f"  {'─' * 52}")
    for rank, d in enumerate(sorted_idx):
        aligned = '  ← ALIGNED' if 3.0 <= implied_periods[d] <= 8.0 else ''
        print(f"  {d:4d} {final_freqs[d]:+10.4f} {abs(final_freqs[d]):10.4f} "
              f"{implied_periods[d]:10.2f} {aligned}")

    if task_aligned:
        print(f"\n  ✓ Found {len(task_aligned)} task-aligned frequency dimension(s) "
              f"(period in [3.0, 8.0]):")
        for d, freq, period in task_aligned:
            print(f"    dim {d}: ω={freq:+.4f}, period={period:.2f}")
    else:
        print(f"\n  ✗ No frequency dimensions have period near 5.0")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment T5: Learned Frequencies')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_dir)
