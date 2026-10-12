"""
Experiment 9 — Cross-Seed Consistency Analysis

Question:  Are the causal importance maps (Exp 1) stable across seeds, or
           are they seed-specific artifacts?

Method:    Load Exp 1 results from multiple seeds, compute pairwise Pearson
           correlation of the flattened importance maps, and report mean
           pairwise correlation.  High correlation (> 0.8) validates that
           the circuit structure is a stable property of the architecture
           and task, not a training fluke.

           Also checks gate analysis (Exp 3A) consistency: whether the same
           memory dimensions are flagged as significant across seeds.

Note:      This experiment is run as a post-hoc compilation step — it
           requires multiple seeds' results to already exist.  It is called
           by compile_results.py automatically when ≥ 2 seeds are available.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import setup_output_dir, setup_plot_style, save_figure

import matplotlib.pyplot as plt


def run(base_dir, seeds, output_dir=None, **kwargs):
    """Run Experiment 9: Cross-Seed Consistency.

    Args:
        base_dir: Base directory containing seed_* subdirectories
        seeds: List of seeds to compare
        output_dir: Where to save results (defaults to base_dir)
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 9 — Cross-Seed Consistency")
    print("=" * 65)

    if output_dir is None:
        output_dir = base_dir
    out = setup_output_dir(output_dir, 'experiment_9_cross_seed_consistency')

    # ═══════════════════════════════════════════════════════════════
    # Part A: Causal Map Consistency
    # ═══════════════════════════════════════════════════════════════

    print("\n  Part A: Causal Importance Map Consistency")
    maps = {}
    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_1_causal_map', 'experiment_1_causal_map.pt')
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu', weights_only=False)
            key = 'mean_effects' if 'mean_effects' in data else 'importance_map'
            if key in data:
                maps[seed] = data[key].numpy()

    map_correlation = None
    pairwise_corrs = []

    if len(maps) >= 2:
        seed_list = sorted(maps.keys())
        print(f"    Found maps for seeds: {seed_list}")

        # Compute pairwise correlations
        for i in range(len(seed_list)):
            for j in range(i + 1, len(seed_list)):
                s1, s2 = seed_list[i], seed_list[j]
                m1, m2 = maps[s1].flatten(), maps[s2].flatten()
                # Trim to common length
                min_len = min(len(m1), len(m2))
                r = float(np.corrcoef(m1[:min_len], m2[:min_len])[0, 1])
                pairwise_corrs.append({
                    'seed_a': s1, 'seed_b': s2, 'correlation': r
                })
                print(f"    Seed {s1} × {s2}: r = {r:.4f}")

        map_correlation = float(np.mean([c['correlation'] for c in pairwise_corrs]))
        print(f"    Mean pairwise correlation: {map_correlation:.4f}")
    else:
        print(f"    Need ≥ 2 seeds with Exp1 results; found {len(maps)}")

    # ═══════════════════════════════════════════════════════════════
    # Part B: Gate Dimension Consistency
    # ═══════════════════════════════════════════════════════════════

    print("\n  Part B: Gate Dimension Consistency")
    gate_data = {}
    for seed in seeds:
        for dirname in ['experiment_3a_gate_analysis', 'experiment_3_gate_analysis']:
            fname = dirname + '.pt'
            path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                                dirname, fname)
            if os.path.exists(path):
                data = torch.load(path, map_location='cpu', weights_only=False)
                z_key = 'z_rejected_bh' if 'z_rejected_bh' in data else None
                d_key = 'z_discriminability' if 'z_discriminability' in data else 'dprime_z'
                if z_key and z_key in data:
                    gate_data[seed] = {
                        'z_rejected': data[z_key].numpy().astype(bool),
                        'z_disc': data[d_key].numpy(),
                    }
                break

    gate_overlap = None
    gate_corrs = []

    if len(gate_data) >= 2:
        gate_seeds = sorted(gate_data.keys())
        print(f"    Found gate data for seeds: {gate_seeds}")

        # Jaccard similarity of significant dimensions
        jaccard_scores = []
        for i in range(len(gate_seeds)):
            for j in range(i + 1, len(gate_seeds)):
                s1, s2 = gate_seeds[i], gate_seeds[j]
                r1, r2 = gate_data[s1]['z_rejected'], gate_data[s2]['z_rejected']
                intersection = (r1 & r2).sum()
                union = (r1 | r2).sum()
                jaccard = float(intersection / union) if union > 0 else 0.0
                jaccard_scores.append({
                    'seed_a': s1, 'seed_b': s2, 'jaccard': jaccard
                })
                print(f"    Seed {s1} × {s2}: Jaccard(sig dims) = {jaccard:.4f}")

                # Discriminability vector correlation
                d1, d2 = gate_data[s1]['z_disc'], gate_data[s2]['z_disc']
                r = float(np.corrcoef(d1, d2)[0, 1])
                gate_corrs.append({
                    'seed_a': s1, 'seed_b': s2, 'disc_correlation': r
                })
                print(f"    Seed {s1} × {s2}: r(discriminability) = {r:.4f}")

        gate_overlap = float(np.mean([s['jaccard'] for s in jaccard_scores]))
        mean_gate_corr = float(np.mean([c['disc_correlation'] for c in gate_corrs]))
        print(f"    Mean Jaccard overlap: {gate_overlap:.4f}")
        print(f"    Mean discriminability correlation: {mean_gate_corr:.4f}")
    else:
        print(f"    Need ≥ 2 seeds with Exp3A results; found {len(gate_data)}")
        mean_gate_corr = None

    # ── Save results ──────────────────────────────────────────────
    results = {
        'seeds': seeds,
        'num_seeds_with_maps': len(maps),
        'num_seeds_with_gates': len(gate_data),
    }
    if map_correlation is not None:
        results['map_correlation'] = {
            'mean': map_correlation,
            'pairwise': pairwise_corrs,
        }
    if gate_overlap is not None:
        results['gate_consistency'] = {
            'mean_jaccard': gate_overlap,
            'jaccard_pairwise': [s['jaccard'] for s in jaccard_scores],
            'mean_disc_correlation': mean_gate_corr,
            'disc_correlations': [c['disc_correlation'] for c in gate_corrs],
        }
    torch.save(results, os.path.join(out, 'experiment_9_cross_seed_consistency.pt'))
    print(f"\n  ✓ Saved results")

    # ── Plot (if enough data) ─────────────────────────────────────
    if len(maps) >= 2 or len(gate_data) >= 2:
        setup_plot_style()
        n_plots = (1 if len(maps) >= 2 else 0) + (1 if len(gate_data) >= 2 else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        if len(maps) >= 2:
            ax = axes[plot_idx]
            corr_vals = [c['correlation'] for c in pairwise_corrs]
            labels = [f"{c['seed_a']}-{c['seed_b']}" for c in pairwise_corrs]
            ax.bar(range(len(corr_vals)), corr_vals,
                   color='#3498db', edgecolor='white')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Pearson r')
            ax.set_title('Causal Map Pairwise Correlation', fontweight='bold')
            ax.axhline(0.8, color='#2ecc71', linestyle='--', linewidth=1,
                       label='r = 0.8 threshold')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            plot_idx += 1

        if len(gate_data) >= 2:
            ax = axes[plot_idx]
            jacc_vals = [s['jaccard'] for s in jaccard_scores]
            disc_vals = [c['disc_correlation'] for c in gate_corrs]
            x = np.arange(len(jacc_vals))
            w = 0.35
            ax.bar(x - w/2, jacc_vals, w, color='#e74c3c', edgecolor='white',
                   label='Jaccard (sig dims)')
            ax.bar(x + w/2, disc_vals, w, color='#3498db', edgecolor='white',
                   label='r (discriminability)')
            labels = [f"{s['seed_a']}-{s['seed_b']}" for s in jaccard_scores]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel('Similarity')
            ax.set_title('Gate Analysis Consistency', fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        save_figure(fig, os.path.join(out, 'experiment_9_cross_seed_consistency.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Cross-Seed Consistency Summary:")
    if map_correlation is not None:
        if map_correlation > 0.8:
            print(f"    Causal maps: STABLE (mean r = {map_correlation:.4f})")
        elif map_correlation > 0.5:
            print(f"    Causal maps: MODERATELY STABLE (mean r = {map_correlation:.4f})")
        else:
            print(f"    Causal maps: UNSTABLE (mean r = {map_correlation:.4f})")
            print("    → Circuit structure may be seed-dependent")

    if gate_overlap is not None:
        print(f"    Gate dimensions: Jaccard = {gate_overlap:.4f}, "
              f"disc r = {mean_gate_corr:.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Cross-seed consistency analysis (run after all seeds complete)')
    parser.add_argument('--base_dir', type=str, default='runs',
                        help='Base directory containing seed_* subdirectories')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45],
                        help='Seeds to compare')
    args = parser.parse_args()
    run(args.base_dir, args.seeds)
