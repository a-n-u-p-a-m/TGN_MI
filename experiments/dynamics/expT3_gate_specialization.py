"""
Experiment T3 — Gate Specialization Over Training

Question:  When do specific z_t dimensions become specialized for motif
           detection?

Method:    At each checkpoint, extract update gate z_t at motif vs. noise
           updates of B, compute per-dimension discriminability, and track
           how the specialization pattern evolves.
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


def _find_noise_b_update(mem_hooks, m1, m2, B_node, batch_idx=0):
    """Find a non-motif update of B.

    Excludes both motif-related B updates:
      - 2*m1 + 1: the A→B dst update of B
      - 2*m2:     the B→C src update of B
    """
    forbidden = {
        2 * m1 + 1,  # B update from A→B
        2 * m2,      # B update from B→C
    }

    for hi, h in enumerate(mem_hooks):
        if hi in forbidden:
            continue
        node_ids = h['node_ids']
        mask = h['mask']
        if mask[batch_idx].item() and node_ids[batch_idx].item() == B_node:
            return hi
    return -1


def run(checkpoint_dir, output_dir, num_samples=200, seed=42, device='cpu',
        **kwargs):
    """Run Experiment T3: Gate Specialization Over Training."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT T3 — Gate Specialization Over Training")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_T3_gate_specialization')

    # ── Find checkpoints ──────────────────────────────────────────
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"  ✗ No checkpoints found in {checkpoint_dir}")
        return None
    print(f"  Found {len(checkpoints)} checkpoints: epochs {[e for e, _ in checkpoints]}")

    # ── Generate fixed positive samples ───────────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Evaluate at each checkpoint ───────────────────────────────
    epochs = []
    discrim_matrix = []  # (num_checkpoints, memory_dim) discriminability
    num_specialized = []
    top5_dims_per_epoch = []

    # Descriptive threshold for visualization — NOT a statistically justified
    # specialization boundary.  Dimensions exceeding this are reported as
    # "exceeding the predefined discriminability threshold", not as "specialized".
    DISCRIM_THRESHOLD = 0.1

    for ckpt_idx, (epoch, ckpt_path) in enumerate(checkpoints):
        print(f"  Checkpoint {ckpt_idx + 1}/{len(checkpoints)} (epoch {epoch})...")

        model, config, _ = load_model(ckpt_path, device=device)
        memory_dim = config.memory_dim
        epochs.append(epoch)

        motif_z_list = []
        noise_z_list = []

        for sample in positives:
            batch = single_sample_batch(sample)
            motif_steps = get_motif_steps(model, batch)
            m1, m2 = motif_steps[0]

            if m1 < 0 or m2 < 0:
                continue

            _ = run_forward(model, batch)
            mem_hooks = model.memory.hook_data

            motif_hook_idx = 2 * m1 + 1
            if motif_hook_idx >= len(mem_hooks):
                continue

            candidates = batch['candidate_nodes'][0]
            B_node = candidates[1].item()

            motif_hook = mem_hooks[motif_hook_idx]
            if motif_hook['node_ids'][0].item() != B_node:
                continue

            noise_idx = _find_noise_b_update(mem_hooks, m1, m2, B_node)
            if noise_idx < 0:
                continue

            motif_z = motif_hook['gates']['z_t'][0].cpu().numpy()
            noise_z = mem_hooks[noise_idx]['gates']['z_t'][0].cpu().numpy()
            motif_z_list.append(motif_z)
            noise_z_list.append(noise_z)

        if len(motif_z_list) < 5:
            print(f"    Skipping: only {len(motif_z_list)} valid pairs")
            discrim_matrix.append(np.zeros(memory_dim))
            num_specialized.append(0)
            top5_dims_per_epoch.append([])
            continue

        motif_z = np.stack(motif_z_list)
        noise_z = np.stack(noise_z_list)

        discrim = np.abs(motif_z.mean(axis=0) - noise_z.mean(axis=0))
        discrim_matrix.append(discrim)

        n_spec = (discrim > DISCRIM_THRESHOLD).sum()
        num_specialized.append(int(n_spec))

        top5 = np.argsort(discrim)[::-1][:5].tolist()
        top5_dims_per_epoch.append(top5)

        print(f"    {len(motif_z_list)} pairs, {n_spec} dims exceeding threshold {DISCRIM_THRESHOLD}, "
              f"top dims: {top5}")

    discrim_matrix = np.stack(discrim_matrix)  # (num_checkpoints, memory_dim)

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'epochs': epochs,
        'discriminability_matrix': torch.tensor(discrim_matrix, dtype=torch.float32),
        'num_specialized': num_specialized,
        'top5_dims_per_epoch': top5_dims_per_epoch,
        'threshold': DISCRIM_THRESHOLD,
        'num_samples': len(positives),
    }
    torch.save(results, os.path.join(out, 'experiment_T3_gate_specialization.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    # Top: heatmap of discriminability over training
    ax = axes[0]
    im = ax.imshow(discrim_matrix, aspect='auto', cmap='hot',
                   interpolation='nearest',
                   vmin=0, vmax=max(discrim_matrix.max(), 0.01))
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels([str(e) for e in epochs])
    ax.set_ylabel('Epoch')
    ax.set_xlabel('Gate dimension')
    ax.set_title('Update Gate Discriminability Over Training',
                 fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='|mean_motif − mean_noise|',
                 shrink=0.8, pad=0.02)

    # Bottom: number of specialized dimensions
    ax = axes[1]
    ax.plot(epochs, num_specialized, color='#e74c3c', linewidth=2.5,
            marker='o', markersize=6)
    ax.fill_between(epochs, num_specialized, alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'# dims > {DISCRIM_THRESHOLD}')
    ax.set_title(f'Dimensions Exceeding Discriminability Threshold ({DISCRIM_THRESHOLD})',
                  fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_T3_gate_specialization.png'))

    # ── Dimension identity stability ──────────────────────────────
    if len(top5_dims_per_epoch) >= 2:
        print(f"\n  Dimension identity stability (top 5):")
        for i, ep in enumerate(epochs):
            dims = top5_dims_per_epoch[i]
            print(f"    Epoch {ep}: {dims}")

        # Overlap between consecutive epochs
        overlaps = []
        for i in range(1, len(top5_dims_per_epoch)):
            if top5_dims_per_epoch[i] and top5_dims_per_epoch[i-1]:
                overlap = len(set(top5_dims_per_epoch[i]) &
                              set(top5_dims_per_epoch[i-1]))
                overlaps.append(overlap)
        if overlaps:
            print(f"    Mean consecutive overlap: {np.mean(overlaps):.1f}/5 dims")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment T3: Gate Specialization')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_dir, args.num_samples, args.seed, args.device)
