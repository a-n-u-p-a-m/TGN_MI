"""
Experiment 6 — Ablation Method Comparison

Question:  Is mean-ablation a better intervention method than zero-ablation
           for this model?  (Validates the methodological choice in Exp 1.)

Method:    For a shared set of positive samples, run the same causal sweep
           (across event offsets × intervention points) twice — once with
           mean-ablation and once with zero-ablation.  Compare:
             • Per-cell Pearson correlation of the two importance maps
             • Whether zero-ablation disproportionately affects high-scale
               activations (checking the scale-confound hypothesis)
             • SNR (variance of causal effects / mean squared effect) for
               each method

Produces:  Side-by-side heatmaps and a quantitative comparison summary.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    load_model, prepare_eval_dataset, get_positive_samples, single_sample_batch,
    run_forward, get_motif_steps, setup_output_dir, setup_plot_style, save_figure,
    seed_everything,
)

import matplotlib.pyplot as plt

# Intervention points — must match exp1_causal_map.py
INTERVENTION_POINTS = [
    'memory_pre_src',
    'memory_pre_dst',
    'message_src',
    'message_dst',
    'time_encoding_src',
    'time_encoding_dst',
]

POINT_LABELS = [
    'Mem (src)',
    'Mem (dst)',
    'Msg (src)',
    'Msg (dst)',
    'Time enc. (src)',
    'Time enc. (dst)',
]


def _run_ablation_sweep(model, positives, ablation_type, device, pilot_means=None):
    """Run the causal sweep with a specific ablation type.

    ablation_type: 'mean' or 'zero'
    pilot_means: dict of {(point_name,): mean_tensor} for mean-ablation
    """
    all_effects = []
    offsets_list = []
    m2_offsets = []

    for idx, sample in enumerate(positives):
        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]
        if m1 < 0 or m2 < 0:
            continue

        num_events = batch['src'].size(1)

        # Clean logit
        clean_logit = run_forward(model, batch).item()

        # Sweep
        min_offset = -m1
        max_offset = num_events - 1 - m1
        offset_range = list(range(min_offset, max_offset + 1))

        sample_effects = np.zeros((len(INTERVENTION_POINTS), len(offset_range)))

        for oi, offset in enumerate(offset_range):
            event_idx = m1 + offset
            if event_idx < 0 or event_idx >= num_events:
                continue

            for pi, point in enumerate(INTERVENTION_POINTS):
                if ablation_type == 'zero':
                    def intervene(x, _point=point):
                        return torch.zeros_like(x)
                else:
                    # Mean ablation
                    if pilot_means and point in pilot_means:
                        mean_val = pilot_means[point]
                        def intervene(x, _m=mean_val):
                            return _m.to(x.device).expand_as(x)
                    else:
                        def intervene(x):
                            return torch.zeros_like(x)

                interventions = {(event_idx, point): intervene}
                abl_logit = run_forward(model, batch, interventions=interventions).item()
                sample_effects[pi, oi] = clean_logit - abl_logit

        all_effects.append(sample_effects)
        offsets_list.append(offset_range)
        m2_offsets.append(m2 - m1)

    return all_effects, offsets_list, m2_offsets


def _compute_pilot_means(model, positives, num_pilot=50):
    """Compute population means for each intervention point from pilot samples."""
    pilot = positives[:num_pilot]
    accumulators = {p: [] for p in INTERVENTION_POINTS}

    for sample in pilot:
        batch = single_sample_batch(sample)
        num_events = batch['src'].size(1)

        for event_idx in range(num_events):
            for point in INTERVENTION_POINTS:
                captured = {}

                def capture(x, _c=captured):
                    _c['val'] = x.clone().detach()
                    return x

                interventions = {(event_idx, point): capture}
                run_forward(model, batch, interventions=interventions)
                if 'val' in captured:
                    accumulators[point].append(captured['val'].cpu())

    means = {}
    for point, vals in accumulators.items():
        if vals:
            means[point] = torch.stack(vals).mean(dim=0)
    return means


def _align_and_average(all_effects, offsets_list):
    """Align effect matrices by motif offset and average."""
    min_offset = min(o[0] for o in offsets_list)
    max_offset = max(o[-1] for o in offsets_list)
    full_range = list(range(min_offset, max_offset + 1))
    width = len(full_range)

    aligned = np.zeros((len(INTERVENTION_POINTS), width))
    counts = np.zeros(width)

    for effects, offsets in zip(all_effects, offsets_list):
        for oi, offset in enumerate(offsets):
            col = offset - min_offset
            aligned[:, col] += effects[:, oi]
            counts[col] += 1

    counts[counts == 0] = 1
    aligned /= counts[np.newaxis, :]
    return aligned, full_range


def run(model_path, output_dir, num_samples=100, seed=42, device='cpu'):
    """Run Experiment 6: Ablation Method Comparison."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 6 — Ablation Method Comparison")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_6_ablation_comparison')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Compute pilot means for mean-ablation ─────────────────────
    print("  Computing pilot means...")
    pilot_means = _compute_pilot_means(model, positives, num_pilot=min(50, len(positives)))

    # ── Run both ablation types ───────────────────────────────────
    print("  Running zero-ablation sweep...")
    zero_effects, zero_offsets, zero_m2 = _run_ablation_sweep(
        model, positives, 'zero', device)
    print(f"    Collected {len(zero_effects)} samples")

    print("  Running mean-ablation sweep...")
    mean_effects, mean_offsets, mean_m2 = _run_ablation_sweep(
        model, positives, 'mean', device, pilot_means=pilot_means)
    print(f"    Collected {len(mean_effects)} samples")

    # ── Align and average ─────────────────────────────────────────
    zero_map, zero_range = _align_and_average(zero_effects, zero_offsets)
    mean_map, mean_range = _align_and_average(mean_effects, mean_offsets)

    # Ensure same range for comparison
    min_r = max(zero_range[0], mean_range[0])
    max_r = min(zero_range[-1], mean_range[-1])
    z_start = min_r - zero_range[0]
    z_end = max_r - zero_range[0] + 1
    m_start = min_r - mean_range[0]
    m_end = max_r - mean_range[0] + 1
    zero_map = zero_map[:, z_start:z_end]
    mean_map = mean_map[:, m_start:m_end]
    common_range = list(range(min_r, max_r + 1))

    # ── Quantitative comparison ───────────────────────────────────
    # Flatten for correlation
    z_flat = zero_map.flatten()
    m_flat = mean_map.flatten()
    correlation = float(np.corrcoef(z_flat, m_flat)[0, 1])

    # Per-intervention-point scale comparison
    zero_scale = np.abs(zero_map).mean(axis=1)
    mean_scale = np.abs(mean_map).mean(axis=1)
    scale_ratio = zero_scale / (mean_scale + 1e-8)

    # Relative variance (coefficient of variation squared): σ²/μ²
    # Higher values indicate more variable (less consistent) causal effects.
    zero_rel_var = float(z_flat.var() / (z_flat.mean()**2 + 1e-8))
    mean_rel_var = float(m_flat.var() / (m_flat.mean()**2 + 1e-8))

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'zero_map': torch.tensor(zero_map, dtype=torch.float32),
        'mean_map': torch.tensor(mean_map, dtype=torch.float32),
        'offset_range': torch.tensor(common_range, dtype=torch.long),
        'correlation': correlation,
        'scale_ratio_per_point': torch.tensor(scale_ratio, dtype=torch.float32),
        'zero_rel_var': zero_rel_var,
        'mean_rel_var': mean_rel_var,
        'num_samples': len(zero_effects),
        'intervention_points': INTERVENTION_POINTS,
    }
    torch.save(results, os.path.join(out, 'experiment_6_ablation_comparison.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Zero-ablation heatmap
    vmax = max(np.abs(zero_map).max(), np.abs(mean_map).max())
    ax = axes[0]
    im = ax.imshow(zero_map, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_yticks(range(len(POINT_LABELS)))
    ax.set_yticklabels(POINT_LABELS)
    ax.set_xlabel('Event offset (relative to A→B)')
    ax.set_title('Zero-Ablation', fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Mean-ablation heatmap
    ax = axes[1]
    im = ax.imshow(mean_map, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_yticks(range(len(POINT_LABELS)))
    ax.set_yticklabels(POINT_LABELS)
    ax.set_xlabel('Event offset (relative to A→B)')
    ax.set_title('Mean-Ablation', fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Scale ratio bar chart
    ax = axes[2]
    bars = ax.bar(range(len(POINT_LABELS)), scale_ratio,
                  color=['#e74c3c' if r > 1.5 else '#3498db' for r in scale_ratio],
                  edgecolor='white')
    ax.set_xticks(range(len(POINT_LABELS)))
    ax.set_xticklabels(POINT_LABELS, rotation=45, ha='right')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('|zero| / |mean| scale ratio')
    ax.set_title('Scale Confound Check', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Ablation Method Comparison (r = {correlation:.3f})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, os.path.join(out, 'experiment_6_ablation_comparison.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Ablation Method Comparison:")
    print(f"    Pearson correlation (zero vs mean): {correlation:.4f}")
    print(f"    Zero-ablation relative variance: {zero_rel_var:.4f}")
    print(f"    Mean-ablation relative variance: {mean_rel_var:.4f}")
    print(f"    Scale ratio per intervention point:")
    for i, label in enumerate(POINT_LABELS):
        tag = " ← scale confound" if scale_ratio[i] > 1.5 else ""
        print(f"      {label:12s}: {scale_ratio[i]:.2f}{tag}")

    if correlation > 0.9:
        print("\n  → High correlation: both methods yield similar importance maps")
        print("    Mean-ablation preferred as a precaution against scale confounds")
    elif correlation > 0.7:
        print("\n  → Moderate correlation: methods agree on main features but differ in details")
        print("    Choice of ablation method materially affects the analysis")
    else:
        print("\n  → Low correlation: methods disagree substantially")
        print("    Results are sensitive to ablation method — interpret with caution")

    return results
