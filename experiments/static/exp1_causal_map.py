"""
Experiment 1 — Causal Importance Map

Question:  At which event steps, and through which components, does the
           model's decision depend on the input?

Produces:  A heatmap of causal importance across all event steps and all
           6 intervention points — the "circuit fingerprint" of the model.

Method:    For every (event_step, intervention_point) pair, mean-ablate
           that activation (replace with population mean from a pilot set)
           and measure the change in logit relative to a clean forward pass.
           Using mean-ablation instead of zero-ablation ensures perturbation
           magnitudes are comparable across intervention points with different
           activation scales (e.g. bounded time encodings vs unbounded memory).
           Results are aligned across samples using motif-relative offsets
           and averaged.
"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    load_model, prepare_eval_dataset, get_positive_samples, single_sample_batch,
    run_forward, get_motif_steps, setup_output_dir, setup_plot_style, save_figure,
    seed_everything,
)

import matplotlib.pyplot as plt

INTERVENTION_POINTS = [
    'memory_pre_src',
    'memory_pre_dst',
    'message_src',
    'message_dst',
    'time_encoding_src',
    'time_encoding_dst',
]

POINT_LABELS = [
    'mem_src', 'mem_dst', 'msg_src', 'msg_dst', 'time_src', 'time_dst',
]


def _collect_activation_means(model, samples, device, num_pilot=20):
    """Collect per-intervention-point activation means from a pilot set.

    For efficiency, instruments a single forward pass per pilot sample by
    intercepting all intervention points at a few sampled event steps.
    Returns a dict mapping point_name -> mean activation tensor.
    """
    pilot = samples[:num_pilot]
    activation_sums = {}
    activation_counts = {}

    for sample in pilot:
        batch = single_sample_batch(sample)
        num_real = int(batch['mask'][0].sum().item())

        # Sample a few representative steps (beginning, middle, end)
        sample_steps = list(set([0, num_real // 4, num_real // 2,
                                 3 * num_real // 4, max(0, num_real - 1)]))

        for point_name in INTERVENTION_POINTS:
            for step_i in sample_steps:
                if step_i >= num_real:
                    continue
                captured = {}
                def record_fn(x, _cap=captured):
                    _cap['val'] = x.clone().detach()
                    return x  # Pass through unchanged

                interventions = {(step_i, point_name): record_fn}
                _ = run_forward(model, batch, interventions=interventions)

                if 'val' in captured:
                    val = captured['val'].squeeze(0)  # Remove batch dim
                    if point_name not in activation_sums:
                        activation_sums[point_name] = torch.zeros_like(val)
                        activation_counts[point_name] = 0
                    activation_sums[point_name] += val
                    activation_counts[point_name] += 1

    means = {}
    for point_name in INTERVENTION_POINTS:
        if point_name in activation_sums and activation_counts[point_name] > 0:
            means[point_name] = activation_sums[point_name] / activation_counts[point_name]
        else:
            means[point_name] = None

    return means


def run(model_path, output_dir, num_samples=200, seed=42, device='cpu'):
    """Run Experiment 1: Causal Importance Map."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1 — Causal Importance Map (Mean-Ablation)")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_1_causal_map')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep}, memory_dim={config.memory_dim})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Step 1: Collect population means from pilot set ───────────
    print("  Collecting activation means from pilot set (50 samples)...")
    activation_means = _collect_activation_means(model, positives, device, num_pilot=50)
    for pname, mean_val in activation_means.items():
        if mean_val is not None:
            print(f"    {pname}: mean_norm={mean_val.norm().item():.4f}, shape={list(mean_val.shape)}")

    # ── Step 2: Run causal sweep with mean-ablation ───────────────
    # For each sample: clean logit, then mean-ablate at every (step, point)
    all_effects = []     # list of dicts per sample
    all_m1 = []          # A→B step index per sample
    all_m2 = []          # B→C step index per sample
    all_num_real = []    # number of real events per sample

    for idx, sample in enumerate(positives):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(positives)}...")

        batch = single_sample_batch(sample)

        # Clean forward pass
        baseline_logit = run_forward(model, batch).item()

        # Get motif step indices
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]
        all_m1.append(m1)
        all_m2.append(m2)

        # Number of real events
        num_real = batch['mask'][0].sum().item()
        all_num_real.append(num_real)

        # Sweep all (step, point) pairs with mean-ablation
        sample_effects = {}
        for step_i in range(int(num_real)):
            for point_name in INTERVENTION_POINTS:
                mean_val = activation_means.get(point_name)
                if mean_val is not None:
                    def make_mean_ablation(mv=mean_val):
                        return lambda x: mv.unsqueeze(0).expand_as(x)
                    interventions = {
                        (step_i, point_name): make_mean_ablation()
                    }
                else:
                    # Fallback to zero-ablation if no mean available
                    interventions = {
                        (step_i, point_name): lambda x: torch.zeros_like(x)
                    }
                patched_logit = run_forward(model, batch, interventions=interventions).item()
                effect = baseline_logit - patched_logit
                sample_effects[(step_i, point_name)] = effect

        all_effects.append(sample_effects)

    # ── Align across samples using motif offsets ──────────────────
    # Offset = event_index - m1 (centers A→B at offset 0)
    max_real = max(all_num_real)

    # Compute the range of offsets across all samples
    min_offset = min(-m1 for m1 in all_m1)
    max_offset = max(nr - 1 - m1 for nr, m1 in zip(all_num_real, all_m1))
    offset_range = list(range(int(min_offset), int(max_offset) + 1))

    # Build aggregation arrays
    effect_sums = np.zeros((len(offset_range), len(INTERVENTION_POINTS)))
    effect_counts = np.zeros((len(offset_range), len(INTERVENTION_POINTS)))

    for sample_idx, (effects, m1) in enumerate(zip(all_effects, all_m1)):
        for (step_i, point_name), effect in effects.items():
            offset = step_i - m1
            offset_idx = offset - min_offset
            point_idx = INTERVENTION_POINTS.index(point_name)
            if 0 <= offset_idx < len(offset_range):
                effect_sums[offset_idx, point_idx] += effect
                effect_counts[offset_idx, point_idx] += 1

    # Mean effect per (offset, point)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_effects = np.where(effect_counts > 0,
                                effect_sums / effect_counts, 0.0)

    # ── Compute mean B→C offset ───────────────────────────────────
    mean_m2_offset = np.mean([m2 - m1 for m1, m2 in zip(all_m1, all_m2)
                              if m1 >= 0 and m2 >= 0])

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'mean_effects': torch.tensor(mean_effects, dtype=torch.float32),
        'offset_range': torch.tensor(offset_range, dtype=torch.long),
        'intervention_points': INTERVENTION_POINTS,
        'mean_m2_offset': mean_m2_offset,
        'num_samples': len(positives),
        'all_m1': all_m1,
        'all_m2': all_m2,
    }
    torch.save(results, os.path.join(out, 'experiment_1_causal_map.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot heatmap ──────────────────────────────────────────────
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    vmax = np.abs(mean_effects).max()
    vmin = -vmax

    im = ax.imshow(mean_effects.T, aspect='auto', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # Axis labels
    ax.set_yticks(range(len(POINT_LABELS)))
    ax.set_yticklabels(POINT_LABELS)
    ax.set_xlabel('Event offset (relative to A→B)')
    ax.set_title('Causal Importance Map — Mean effect of mean-ablation',
                 fontsize=14, fontweight='bold')

    # X-axis: show offset values
    tick_step = max(1, len(offset_range) // 20)
    tick_positions = list(range(0, len(offset_range), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(offset_range[i]) for i in tick_positions], fontsize=7)

    # Mark motif positions
    ab_idx = -min_offset  # offset 0 = A→B
    bc_idx = int(round(mean_m2_offset)) - min_offset  # mean B→C offset
    ax.axvline(ab_idx, color='#e74c3c', linestyle='--', linewidth=1.5,
               alpha=0.8, label='A→B')
    ax.axvline(bc_idx, color='#3498db', linestyle='--', linewidth=1.5,
               alpha=0.8, label='B→C (mean)')

    ax.legend(loc='upper right', fontsize=9)
    fig.colorbar(im, ax=ax, label='Causal effect (baseline − patched)',
                 shrink=0.8, pad=0.02)

    save_figure(fig, os.path.join(out, 'experiment_1_causal_map.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Summary:")
    print(f"    Samples: {len(positives)}")
    print(f"    Mean A→B step: {np.mean(all_m1):.1f}")
    print(f"    Mean B→C step: {np.mean(all_m2):.1f}")
    print(f"    Mean B→C offset from A→B: {mean_m2_offset:.1f}")
    print(f"    Max |causal effect|: {vmax:.4f}")

    # Top 5 most important (offset, point) pairs
    flat_idx = np.argsort(np.abs(mean_effects).ravel())[::-1]
    print(f"\n    Top 5 causally important (offset, point) pairs:")
    for rank, fi in enumerate(flat_idx[:5]):
        oi, pi = np.unravel_index(fi, mean_effects.shape)
        print(f"      {rank+1}. offset={offset_range[oi]:+d}, "
              f"point={INTERVENTION_POINTS[pi]}, "
              f"effect={mean_effects[oi, pi]:+.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 1: Causal Importance Map')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.model_path, args.output_dir, args.num_samples, args.seed, args.device)
