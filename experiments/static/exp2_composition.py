"""
Experiment 2 — Composition vs. Independent Accumulation

Question:  Does information from A reach C's classification through B's
           memory (compositional), or does the classifier head join three
           independently-evolved memories (independent)?

Produces:  A single number — the "composition recovery ratio" — that
           quantifies how much of the source-shuffle damage is mediated
           through B's memory.

Method:    For each positive sample, create a source_shuffle corruption
           (A→X), then surgically restore the clean mem_B at the B→C step
           and measure how much of the damage is recovered.
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
from temporal_gnn_mi.datasets.batch_utils import temporal_collate_fn

import matplotlib.pyplot as plt
import random


def _make_source_shuffle_batch(sample, batch, m1, num_nodes=50):
    """Create a corrupted batch where node A in the A→B edge is replaced with X.

    Returns the modified batch_dict (new tensor, not in-place).
    """
    candidates = batch['candidate_nodes'][0]  # (3,)
    A, B, C = candidates[0].item(), candidates[1].item(), candidates[2].item()

    # Pick X ≠ A, B, C
    exclude = {A, B, C}
    X_candidates = [n for n in range(num_nodes) if n not in exclude]
    X = random.choice(X_candidates)

    # Clone the batch and replace src at step m1 with X
    corrupt_batch = {k: v.clone() if isinstance(v, torch.Tensor) else list(v)
                     for k, v in batch.items()}
    corrupt_batch['src'] = batch['src'].clone()
    corrupt_batch['src'][0, m1] = X

    return corrupt_batch


def run(model_path, output_dir, num_samples=200, seed=42, device='cpu'):
    """Run Experiment 2: Composition vs. Independent Accumulation."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2 — Composition vs. Independent Accumulation")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_2_composition')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Run composition test ──────────────────────────────────────
    recovery_ratios = []
    clean_logits = []
    corrupt_logits = []
    patched_logits = []

    for idx, sample in enumerate(positives):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(positives)}...")

        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]

        if m1 < 0 or m2 < 0:
            continue  # skip samples with missing motif metadata

        # ── (1) Clean run — capture mem_B at B→C step ─────────
        captured = {}

        def capture_mem_b(x, _captured=captured):
            _captured['mem_B'] = x.clone().detach()
            return x  # pass through unchanged

        clean_interventions = {(m2, 'memory_pre_src'): capture_mem_b}
        clean_logit = run_forward(model, batch, interventions=clean_interventions).item()

        if 'mem_B' not in captured:
            continue  # safety check

        clean_mem_B = captured['mem_B']

        # ── (2) Corrupt run — source_shuffle (A→X) ────────────
        corrupt_batch = _make_source_shuffle_batch(sample, batch, m1,
                                                   num_nodes=model.num_nodes)
        corrupt_logit = run_forward(model, corrupt_batch).item()

        # ── (3) Patched run — inject clean mem_B at B→C ───────
        def inject_mem_b(x, _clean=clean_mem_B):
            return _clean.to(x.device)

        patch_interventions = {(m2, 'memory_pre_src'): inject_mem_b}
        patched_logit = run_forward(model, corrupt_batch,
                                    interventions=patch_interventions).item()

        # ── Compute recovery ratio ────────────────────────────
        denom = clean_logit - corrupt_logit
        if abs(denom) < 0.01:
            continue  # skip near-zero denominator

        ratio = (patched_logit - corrupt_logit) / denom
        recovery_ratios.append(ratio)
        clean_logits.append(clean_logit)
        corrupt_logits.append(corrupt_logit)
        patched_logits.append(patched_logit)

    # ── Results ───────────────────────────────────────────────────
    recovery_ratios = np.array(recovery_ratios)
    clean_arr = np.array(clean_logits)
    corrupt_arr = np.array(corrupt_logits)
    patched_arr = np.array(patched_logits)

    # Mean-of-ratios (per-sample, original metric — high variance)
    mean_ratio = recovery_ratios.mean()
    std_ratio = recovery_ratios.std()

    # Ratio-of-means (aggregate, more stable statistic)
    total_numerator = (patched_arr - corrupt_arr).sum()
    total_denominator = (clean_arr - corrupt_arr).sum()
    ratio_of_means = total_numerator / total_denominator if abs(total_denominator) > 1e-6 else 0.0

    # Raw gain analysis
    raw_gains = patched_arr - corrupt_arr
    mean_raw_gain = raw_gains.mean()

    # Paired bootstrap CI for ratio-of-means
    n_bootstrap = 10000
    np.random.seed(seed)
    boot_ratios = np.empty(n_bootstrap)
    N = len(clean_arr)
    for i in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        num = (patched_arr[idx] - corrupt_arr[idx]).sum()
        den = (clean_arr[idx] - corrupt_arr[idx]).sum()
        boot_ratios[i] = num / den if abs(den) > 1e-6 else 0.0
    ci_lower = float(np.percentile(boot_ratios, 2.5))
    ci_upper = float(np.percentile(boot_ratios, 97.5))

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'recovery_ratios': torch.tensor(recovery_ratios, dtype=torch.float32),
        'clean_logits': torch.tensor(clean_logits, dtype=torch.float32),
        'corrupt_logits': torch.tensor(corrupt_logits, dtype=torch.float32),
        'patched_logits': torch.tensor(patched_logits, dtype=torch.float32),
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'ratio_of_means': float(ratio_of_means),
        'ratio_of_means_ci95': [ci_lower, ci_upper],
        'mean_raw_gain': float(mean_raw_gain),
        'num_valid_samples': len(recovery_ratios),
    }
    torch.save(results, os.path.join(out, 'experiment_2_composition.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot distribution of recovery ratios ──────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(recovery_ratios, bins=30, color='#3498db', edgecolor='white',
            alpha=0.85, linewidth=0.5)
    ax.axvline(mean_ratio, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean = {mean_ratio:.3f}')
    ax.axvline(1.0, color='#2ecc71', linestyle=':', linewidth=1.5,
               label='Full recovery (1.0)')
    ax.set_xlabel('Recovery ratio')
    ax.set_ylabel('Count')
    ax.set_title('Composition Recovery Ratios', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Logit comparison
    ax = axes[1]
    x = np.arange(len(clean_logits))
    ax.scatter(x, clean_logits, s=8, alpha=0.5, label='Clean', color='#2ecc71')
    ax.scatter(x, corrupt_logits, s=8, alpha=0.5, label='Corrupt', color='#e74c3c')
    ax.scatter(x, patched_logits, s=8, alpha=0.5, label='Patched', color='#3498db')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Logit')
    ax.set_title('Clean / Corrupt / Patched Logits', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_2_composition.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  COMPOSITION RECOVERY ANALYSIS                   ║")
    print(f"  ║  Mean-of-ratios:  {mean_ratio:+.4f} ± {std_ratio:.4f}             ║")
    print(f"  ║  Ratio-of-means:  {ratio_of_means:+.4f}  [{ci_lower:+.4f}, {ci_upper:+.4f}]  ║")
    print(f"  ║  Mean raw gain:   {mean_raw_gain:+.4f}                        ║")
    print(f"  ║  Valid samples:   {len(recovery_ratios):>4d} / {len(positives):<4d}              ║")
    print(f"  ╚══════════════════════════════════════════════════╝")

    # Quantitative interpretation — descriptive, not causal
    if ratio_of_means > 0.5:
        print("  → Substantial recovery: restoring mem_B recovers majority of clean performance")
        print("    (B memory carries a large fraction of the information affected by source shuffle)")
    elif ratio_of_means > 0.1:
        print("  → Limited recovery: restoring mem_B recovers only a small fraction of the damage")
        print("    (B memory is not the sole carrier of the information affected by source shuffle)")
    elif ratio_of_means > -0.1:
        print("  → Negligible recovery: mem_B restoration has near-zero aggregate effect")
        print("    (source-shuffle effect may be distributed across A/B/C memories,")
        print("     or the intervention may not isolate mediation perfectly)")
    else:
        print("  → Negative recovery: restoring clean mem_B slightly hurts classification")
        print("    (model may rely on corrupted-state information)")

    return results


def run_permutation_test(results_path, output_dir=None, n_permutations=10000, seed=42):
    """Run a permutation test on the composition recovery ratio.

    Null hypothesis: the recovery ratio is indistinguishable from random
    (i.e., the mapping between clean and corrupt logits is arbitrary).

    Method: shuffle the pairing between (patched_logit, corrupt_logit)
    and clean_logit, recompute recovery ratios each time, and see how
    often the shuffled mean exceeds the observed mean.

    Args:
        results_path: path to experiment_2_composition.pt
        output_dir: where to save the permutation test results (defaults to
                    same directory as results_path)
        n_permutations: number of permutations (default 10,000)
        seed: random seed

    Returns:
        dict with p_value, observed_mean, null_distribution, ci_95
    """
    print("\n" + "=" * 65)
    print("  PERMUTATION TEST — Composition Recovery Ratio")
    print("=" * 65)

    np.random.seed(seed)

    # Load results
    results = torch.load(results_path, map_location='cpu', weights_only=False)
    clean = results['clean_logits'].numpy()
    corrupt = results['corrupt_logits'].numpy()
    patched = results['patched_logits'].numpy()
    observed_ratios = results['recovery_ratios'].numpy()
    observed_mean = observed_ratios.mean()

    N = len(clean)
    print(f"  Samples: {N}")
    print(f"  Observed mean recovery ratio: {observed_mean:.4f}")
    print(f"  Running {n_permutations} permutations...")

    # Permutation: shuffle which clean logit is paired with which
    # (patched, corrupt) pair. This breaks the real correspondence
    # while preserving marginal distributions.
    null_means = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_idx = np.random.permutation(N)
        perm_clean = clean[perm_idx]
        denom = perm_clean - corrupt
        valid = np.abs(denom) >= 0.01
        if valid.sum() == 0:
            null_means[i] = 0.0
            continue
        perm_ratios = (patched[valid] - corrupt[valid]) / denom[valid]
        null_means[i] = perm_ratios.mean()

    # p-value: fraction of null distribution >= observed
    p_value = (null_means >= observed_mean).mean()
    print(f"  p-value: {p_value:.6f}")

    # 95% CI via bootstrap on the observed ratios
    n_bootstrap = 10000
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_idx = np.random.choice(N, size=N, replace=True)
        bootstrap_means[i] = observed_ratios[boot_idx].mean()
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save results
    if output_dir is None:
        output_dir = os.path.dirname(results_path)

    perm_results = {
        'observed_mean': float(observed_mean),
        'p_value': float(p_value),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'null_distribution': torch.tensor(null_means, dtype=torch.float32),
        'n_permutations': n_permutations,
        'n_samples': N,
    }
    save_path = os.path.join(output_dir, 'experiment_2_permutation_test.pt')
    torch.save(perm_results, save_path)
    print(f"  ✓ Saved permutation test results")

    # Plot
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(null_means, bins=50, color='#95a5a6', edgecolor='white',
            linewidth=0.3, alpha=0.8, density=True, label='Null distribution')
    ax.axvline(observed_mean, color='#e74c3c', linewidth=2.5, linestyle='--',
               label=f'Observed mean = {observed_mean:.4f}')
    ax.axvline(ci_lower, color='#3498db', linewidth=1.5, linestyle=':',
               label=f'95% CI [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(ci_upper, color='#3498db', linewidth=1.5, linestyle=':')

    ax.set_xlabel('Mean recovery ratio')
    ax.set_ylabel('Density')
    ax.set_title(f'Permutation Test — p = {p_value:.4f} (N={n_permutations} permutations)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'experiment_2_permutation_test.png'))

    # Interpretation
    if p_value < 0.001:
        print(f"  → Highly significant (p < 0.001): recovery ratio is not due to chance")
    elif p_value < 0.05:
        print(f"  → Significant (p < 0.05): recovery ratio is unlikely due to chance")
    else:
        print(f"  → Not significant (p >= 0.05): cannot reject null hypothesis")

    return perm_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 2: Composition Test')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--permutation_test', action='store_true',
                        help='Also run permutation test on results')
    args = parser.parse_args()
    results = run(args.model_path, args.output_dir, args.num_samples, args.seed, args.device)
    if args.permutation_test and results is not None:
        results_file = os.path.join(args.output_dir, 'experiment_2_composition',
                                    'experiment_2_composition.pt')
        if os.path.exists(results_file):
            run_permutation_test(results_file, seed=args.seed)
