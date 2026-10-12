"""
Experiment T4 — Time-Window Sharpness Over Training

Question:  Does the time-window detection curve sharpen during training,
           and if so, when?

Method:    At each checkpoint, inject a compact Δt sweep at the B→C step
           for fixed positive samples.  Compute a sharpness metric
           (logit@4.5 − logit@5.5) and track its evolution.
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

# Compact sweep for efficiency across many checkpoints
DELTA_T_SWEEP = [0.5, 2.0, 3.5, 4.5, 4.9, 5.1, 5.5, 7.0, 15.0]


def run(checkpoint_dir, output_dir, num_samples=50, seed=42, device='cpu',
        **kwargs):
    """Run Experiment T4: Time-Window Sharpness Over Training."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT T4 — Time-Window Sharpness Over Training")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_T4_sharpness')

    # ── Find checkpoints ──────────────────────────────────────────
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"  ✗ No checkpoints found in {checkpoint_dir}")
        return None
    print(f"  Found {len(checkpoints)} checkpoints: epochs {[e for e, _ in checkpoints]}")

    # ── Generate fixed positive samples ───────────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples, {len(DELTA_T_SWEEP)} Δt values")

    # ── Evaluate at each checkpoint ───────────────────────────────
    epochs = []
    # (num_checkpoints, num_dt) mean logits
    mean_logit_curves = []
    sharpness_values = []

    idx_4_5 = DELTA_T_SWEEP.index(4.5)
    idx_5_5 = DELTA_T_SWEEP.index(5.5)

    for ckpt_idx, (epoch, ckpt_path) in enumerate(checkpoints):
        print(f"  Checkpoint {ckpt_idx + 1}/{len(checkpoints)} (epoch {epoch})...")

        model, config, _ = load_model(ckpt_path, device=device)
        epochs.append(epoch)

        # Collect logits for each sample × Δt
        logit_matrix = []

        for sample in positives:
            batch = single_sample_batch(sample)
            motif_steps = get_motif_steps(model, batch)
            m1, m2 = motif_steps[0]

            if m1 < 0 or m2 < 0:
                continue

            sample_logits = []
            for synthetic_dt in DELTA_T_SWEEP:
                dt_tensor = torch.tensor([synthetic_dt], dtype=torch.float32,
                                         device=device)
                with torch.no_grad():
                    old_hooks = model.time_encoder.hook_data
                    model.time_encoder.hook_data = []
                    synthetic_enc = model.time_encoder(dt_tensor)
                    model.time_encoder.hook_data = old_hooks

                def inject_enc(x, _enc=synthetic_enc):
                    return _enc.to(x.device).expand_as(x)

                interventions = {(m2, 'time_encoding_src'): inject_enc}
                logit = run_forward(model, batch,
                                    interventions=interventions).item()
                sample_logits.append(logit)

            logit_matrix.append(sample_logits)

        logit_matrix = np.array(logit_matrix)
        mean_logits = logit_matrix.mean(axis=0)
        mean_logit_curves.append(mean_logits)

        # Sharpness metric
        sharpness = mean_logits[idx_4_5] - mean_logits[idx_5_5]
        sharpness_values.append(sharpness)

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'epochs': epochs,
        'delta_t_sweep': torch.tensor(DELTA_T_SWEEP),
        'mean_logit_curves': torch.tensor(np.stack(mean_logit_curves)),
        'sharpness_values': torch.tensor(sharpness_values),
        'num_samples': len(positives),
    }
    torch.save(results, os.path.join(out, 'experiment_T4_sharpness.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: family of Δt-vs-logit curves colored by epoch
    ax = axes[0]
    cmap = plt.cm.plasma
    for i, (epoch, curve) in enumerate(zip(epochs, mean_logit_curves)):
        color = cmap(i / max(len(epochs) - 1, 1))
        ax.plot(DELTA_T_SWEEP, curve, color=color, linewidth=1.8,
                marker='o', markersize=3, alpha=0.8,
                label=f'ep {epoch}')

    ax.axvline(5.0, color='#e74c3c', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Δ=5.0')
    ax.axhline(0, color='#7f8c8d', linestyle='-', linewidth=0.8,
               alpha=0.4)
    ax.set_xlabel('Synthetic Δt')
    ax.set_ylabel('Mean logit')
    ax.set_title('Δt Response Curves Over Training', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: sharpness metric over epochs
    ax = axes[1]
    ax.plot(epochs, sharpness_values, color='#e74c3c', linewidth=2.5,
            marker='D', markersize=7)
    ax.fill_between(epochs, sharpness_values, alpha=0.15, color='#e74c3c')
    ax.axhline(0, color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sharpness (logit@4.5 − logit@5.5)')
    ax.set_title('Time-Window Sharpness Over Training', fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment T4 — Time-Window Crystallization',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_T4_sharpness.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Sharpness timeline:")
    print(f"  {'Epoch':>6s} {'Sharpness':>12s}")
    print(f"  {'─' * 20}")
    for ep, sh in zip(epochs, sharpness_values):
        print(f"  {ep:6d} {sh:>12.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment T4: Time Sharpness')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_dir, args.num_samples, args.seed, args.device)
