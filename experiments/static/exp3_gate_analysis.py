"""
Experiment 3A — Gate-Level Observational Analysis

Question:  Which dimensions of the GRU update gate z_t differ between
           motif and non-motif updates of node B?

Method:    For each positive sample, extract z_t and r_t values when B
           is updated by the A→B motif edge vs. when B is updated by a
           non-motif edge.  Compute per-dimension discriminability and
           statistical significance (paired t-test with Benjamini-Hochberg
           FDR correction for multiple comparisons).

Note:      This file contains both Part A (observational) and Part B (causal,
           which clamps gate values to their non-motif means).

Caveat:    The control (first non-motif B update) may differ from the motif
           update in elapsed time, incoming message, source identity, event
           position, and surrounding context. This analysis establishes that
           B's gates differ between motif and another B update, but does not
           by itself establish motif-specific specialization.
"""

import os
import sys
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    load_model, prepare_eval_dataset, get_positive_samples, single_sample_batch,
    run_forward, get_motif_steps, setup_output_dir, setup_plot_style, save_figure,
    seed_everything,
)

import matplotlib.pyplot as plt


def _benjamini_hochberg(pvals, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction.

    Returns (adjusted_pvals, rejection_mask).
    """
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_pvals[i] * n / (i + 1)
        )
    rejected = adjusted < alpha
    return adjusted, rejected


def _find_noise_b_update(hook_data, m1, m2, B_node, batch_idx=0):
    """Find a non-motif update of B.

    In the memory hook list:
      - Event step i produces hook indices 2*i (src update) and 2*i+1 (dst update)
      - The dst update at step m1 (hook index 2*m1 + 1) is the motif A→B update of B
      - The src update at step m2 (hook index 2*m2) is the motif B→C update of B

    Both motif-related B updates must be excluded so the returned hook is a
    genuine non-motif ("noise") update of node B.
    """
    forbidden = {
        2 * m1 + 1,  # B update from A→B
        2 * m2,      # B update from B→C
    }

    for hi, h in enumerate(hook_data):
        if hi in forbidden:
            continue

        node_ids = h['node_ids']
        mask = h['mask']

        if mask[batch_idx].item() and node_ids[batch_idx].item() == B_node:
            return hi

    return -1


def run(model_path, output_dir, num_samples=300, seed=42, device='cpu'):
    """Run Experiment 3A: Gate-Level Observational Analysis."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3A — Gate-Level Analysis (Observational)")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_3a_gate_analysis')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep}, memory_dim={config.memory_dim})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Collect gate activations ──────────────────────────────────
    motif_z_list = []
    motif_r_list = []
    noise_z_list = []
    noise_r_list = []
    skipped = 0

    for idx, sample in enumerate(positives):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(positives)}...")

        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]

        if m1 < 0 or m2 < 0:
            skipped += 1
            continue

        # Forward pass to populate hooks
        _ = run_forward(model, batch)

        # Get memory hooks
        mem_hooks = model.memory.hook_data

        # Motif update: B updated by A→B at dst hook index 2*m1 + 1
        motif_hook_idx = 2 * m1 + 1
        if motif_hook_idx >= len(mem_hooks):
            skipped += 1
            continue

        candidates = batch['candidate_nodes'][0]
        B_node = candidates[1].item()

        # Verify this hook is for the right node
        motif_hook = mem_hooks[motif_hook_idx]
        if motif_hook['node_ids'][0].item() != B_node:
            skipped += 1
            continue

        motif_z = motif_hook['gates']['z_t'][0].cpu().numpy()  # (memory_dim,)
        motif_r = motif_hook['gates']['r_t'][0].cpu().numpy()
        motif_z_list.append(motif_z)
        motif_r_list.append(motif_r)

        # Find noise update of B (excluding both motif-related B updates)
        noise_idx = _find_noise_b_update(mem_hooks, m1, m2, B_node)
        if noise_idx < 0:
            skipped += 1
            motif_z_list.pop()
            motif_r_list.pop()
            continue

        noise_hook = mem_hooks[noise_idx]
        noise_z = noise_hook['gates']['z_t'][0].cpu().numpy()
        noise_r = noise_hook['gates']['r_t'][0].cpu().numpy()
        noise_z_list.append(noise_z)
        noise_r_list.append(noise_r)

    print(f"  Collected {len(motif_z_list)} matched pairs (skipped {skipped})")

    if len(motif_z_list) == 0:
        print("  ✗ No valid samples — cannot compute gate analysis")
        return None

    # ── Per-dimension discriminability ────────────────────────────
    motif_z = np.stack(motif_z_list)  # (N, memory_dim)
    noise_z = np.stack(noise_z_list)
    motif_r = np.stack(motif_r_list)
    noise_r = np.stack(noise_r_list)

    memory_dim = motif_z.shape[1]

    # Update gate discriminability (paired t-test: same node B, same sequence)
    z_diff = np.abs(motif_z.mean(axis=0) - noise_z.mean(axis=0))
    z_pvals_raw = np.array([
        stats.ttest_rel(motif_z[:, d], noise_z[:, d]).pvalue
        for d in range(memory_dim)
    ])

    # Reset gate discriminability (paired t-test)
    r_diff = np.abs(motif_r.mean(axis=0) - noise_r.mean(axis=0))
    r_pvals_raw = np.array([
        stats.ttest_rel(motif_r[:, d], noise_r[:, d]).pvalue
        for d in range(memory_dim)
    ])

    # Benjamini-Hochberg FDR correction for multiple testing
    z_pvals_adj, z_rejected = _benjamini_hochberg(z_pvals_raw, alpha=0.05)
    r_pvals_adj, r_rejected = _benjamini_hochberg(r_pvals_raw, alpha=0.05)

    # Use adjusted p-values for all downstream analysis
    z_pvals = z_pvals_adj
    r_pvals = r_pvals_adj

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'motif_z': torch.tensor(motif_z),
        'noise_z': torch.tensor(noise_z),
        'motif_r': torch.tensor(motif_r),
        'noise_r': torch.tensor(noise_r),
        'z_discriminability': torch.tensor(z_diff),
        'z_pvalues_raw': torch.tensor(z_pvals_raw),
        'z_pvalues_bh': torch.tensor(z_pvals),
        'z_rejected_bh': torch.tensor(z_rejected),
        'r_discriminability': torch.tensor(r_diff),
        'r_pvalues_raw': torch.tensor(r_pvals_raw),
        'r_pvalues_bh': torch.tensor(r_pvals),
        'r_rejected_bh': torch.tensor(r_rejected),
        'num_samples': len(motif_z_list),
    }
    torch.save(results, os.path.join(out, 'experiment_3a_gate_analysis.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Update gate
    ax = axes[0]
    top20_z = np.argsort(z_diff)[::-1][:20]
    colors_z = ['#e74c3c' if z_rejected[d] else '#bbbbbb' for d in top20_z]
    ax.bar(range(20), z_diff[top20_z], color=colors_z, edgecolor='white',
           linewidth=0.5)
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(d) for d in top20_z], fontsize=8)
    ax.set_ylabel('|mean_motif − mean_noise|')
    ax.set_title('Update Gate (z_t) — Per-Dimension Discriminability',
                 fontweight='bold')
    sig_count_z = z_rejected.sum()
    ax.text(0.97, 0.92, f'red = BH-corrected q < 0.05 ({sig_count_z}/{memory_dim} dims)',
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3, axis='y')

    # Reset gate
    ax = axes[1]
    top20_r = np.argsort(r_diff)[::-1][:20]
    colors_r = ['#e74c3c' if r_rejected[d] else '#bbbbbb' for d in top20_r]
    ax.bar(range(20), r_diff[top20_r], color=colors_r, edgecolor='white',
           linewidth=0.5)
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(d) for d in top20_r], fontsize=8)
    ax.set_xlabel('Dimension index (ranked)')
    ax.set_ylabel('|mean_motif − mean_noise|')
    ax.set_title('Reset Gate (r_t) — Per-Dimension Discriminability',
                 fontweight='bold')
    sig_count_r = r_rejected.sum()
    ax.text(0.97, 0.92, f'red = BH-corrected q < 0.05 ({sig_count_r}/{memory_dim} dims)',
            transform=ax.transAxes, ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_3a_gate_analysis.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Summary:")
    print(f"    Update gate z_t: {sig_count_z}/{memory_dim} dimensions significant (BH q < 0.05)")
    print(f"    Reset gate r_t:  {sig_count_r}/{memory_dim} dimensions significant (BH q < 0.05)")
    print(f"    Fraction significant: z_t={sig_count_z/memory_dim:.1%}, r_t={sig_count_r/memory_dim:.1%}")

    print(f"\n    Top 10 z_t dimensions (by discriminability):")
    for rank, d in enumerate(top20_z[:10]):
        print(f"      {rank+1:2d}. dim {d:3d}: diff={z_diff[d]:.4f}, q={z_pvals[d]:.2e}, "
              f"motif_mean={motif_z[:, d].mean():.4f}, noise_mean={noise_z[:, d].mean():.4f}")

    # Direction check (purely descriptive)
    top5_z = top20_z[:5]
    motif_higher = sum(motif_z[:, d].mean() > noise_z[:, d].mean() for d in top5_z)
    print(f"\n    Direction (top 5 z_t dims): {motif_higher}/5 have higher mean for motif events")

    return results


def run_3b(model_path, output_dir, num_samples=200, seed=42, device='cpu'):
    """Run Experiment 3B: Causal Gate Analysis.

    For each positive sample, clamp z_t or r_t to their noise-condition mean
    at the motif A→B step (dst update of B) and measure the logit change.
    This tests whether the gate values are *causally* necessary, not just
    observationally different.

    Interventions:
      1. Clamp z_t → noise_mean_z at motif step (dst)
      2. Clamp r_t → noise_mean_r at motif step (dst)
      3. Clamp both z_t and r_t simultaneously
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3B — Causal Gate Analysis")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_3b_causal_gate')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep}, memory_dim={config.memory_dim})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Step 1: Collect noise-condition gate means ────────────────
    print("  Collecting noise-condition gate means...")
    noise_z_list = []
    noise_r_list = []
    skipped = 0

    for idx, sample in enumerate(positives):
        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]
        if m1 < 0 or m2 < 0:
            skipped += 1
            continue

        _ = run_forward(model, batch)
        mem_hooks = model.memory.hook_data
        motif_hook_idx = 2 * m1 + 1
        if motif_hook_idx >= len(mem_hooks):
            skipped += 1
            continue

        candidates = batch['candidate_nodes'][0]
        B_node = candidates[1].item()

        motif_hook = mem_hooks[motif_hook_idx]
        if motif_hook['node_ids'][0].item() != B_node:
            skipped += 1
            continue

        noise_idx = _find_noise_b_update(mem_hooks, m1, m2, B_node)
        if noise_idx < 0:
            skipped += 1
            continue

        noise_hook = mem_hooks[noise_idx]
        noise_z_list.append(noise_hook['gates']['z_t'][0].cpu().numpy())
        noise_r_list.append(noise_hook['gates']['r_t'][0].cpu().numpy())

    if len(noise_z_list) == 0:
        print("  ✗ No valid noise samples — cannot compute causal gate analysis")
        return None

    noise_mean_z = torch.tensor(np.stack(noise_z_list).mean(axis=0),
                                dtype=torch.float32, device=device)
    noise_mean_r = torch.tensor(np.stack(noise_r_list).mean(axis=0),
                                dtype=torch.float32, device=device)
    print(f"  Noise-condition means computed from {len(noise_z_list)} samples")

    # ── Step 1b: Collect motif-condition gate means (for shuffle control) ──
    motif_z_list = []
    for idx, sample in enumerate(positives):
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
        motif_z_list.append(motif_hook['gates']['z_t'][0].cpu().numpy())

    # Build shuffle control: randomly permute dimensions of motif-condition mean z_t.
    # This preserves the marginal distribution but destroys per-dimension structure.
    # If the shuffle control also causes large logit shifts, the 3B claim is confounded
    # (just disrupting the GRU). If only noise-clamping does, the claim is genuine.
    motif_mean_z = np.stack(motif_z_list).mean(axis=0) if motif_z_list else noise_mean_z.cpu().numpy()
    rng = np.random.RandomState(seed)
    shuffled_z = motif_mean_z.copy()
    rng.shuffle(shuffled_z)
    shuffle_mean_z = torch.tensor(shuffled_z, dtype=torch.float32, device=device)
    print(f"  Motif-condition means computed from {len(motif_z_list)} samples (shuffle control ready)")

    # ── Step 2: Causal clamping experiments ───────────────────────
    print("  Running causal clamping experiments...")
    logit_results = {'baseline': [], 'clamp_z': [], 'clamp_r': [], 'clamp_both': [],
                     'shuffle_z': []}
    valid_count = 0

    for idx, sample in enumerate(positives):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(positives)}...")

        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]
        if m1 < 0 or m2 < 0:
            continue

        # Baseline (no intervention)
        baseline_logit = run_forward(model, batch).item()
        logit_results['baseline'].append(baseline_logit)

        # Clamp z_t to noise-condition mean
        def clamp_z(z, _mean=noise_mean_z):
            return _mean.unsqueeze(0).expand_as(z)

        interventions_z = {(m1, 'gate_z_t_dst'): clamp_z}
        logit_z = run_forward(model, batch, interventions=interventions_z).item()
        logit_results['clamp_z'].append(logit_z)

        # Clamp r_t to noise-condition mean
        def clamp_r(r, _mean=noise_mean_r):
            return _mean.unsqueeze(0).expand_as(r)

        interventions_r = {(m1, 'gate_r_t_dst'): clamp_r}
        logit_r = run_forward(model, batch, interventions=interventions_r).item()
        logit_results['clamp_r'].append(logit_r)

        # Clamp both to noise-condition means
        interventions_both = {
            (m1, 'gate_z_t_dst'): clamp_z,
            (m1, 'gate_r_t_dst'): clamp_r,
        }
        logit_both = run_forward(model, batch, interventions=interventions_both).item()
        logit_results['clamp_both'].append(logit_both)

        # Shuffle control: clamp z_t to dimension-permuted motif mean
        # This disrupts per-dimension structure but preserves overall magnitude
        def clamp_shuffle_z(z, _mean=shuffle_mean_z):
            return _mean.unsqueeze(0).expand_as(z)

        interventions_shuffle = {(m1, 'gate_z_t_dst'): clamp_shuffle_z}
        logit_shuffle = run_forward(model, batch, interventions=interventions_shuffle).item()
        logit_results['shuffle_z'].append(logit_shuffle)

        valid_count += 1

    print(f"  Valid samples: {valid_count}")

    if valid_count == 0:
        print("  ✗ No valid samples for causal analysis")
        return None

    # ── Compute effect sizes ──────────────────────────────────────
    baseline_arr = np.array(logit_results['baseline'])
    effects = {}
    for cond in ['clamp_z', 'clamp_r', 'clamp_both', 'shuffle_z']:
        cond_arr = np.array(logit_results[cond])
        diff = baseline_arr - cond_arr
        effects[cond] = {
            'mean_diff': float(diff.mean()),
            'std_diff': float(diff.std()),
            'mean_abs_diff': float(np.abs(diff).mean()),
            'pct_sign_change': float(((baseline_arr > 0) != (cond_arr > 0)).mean()),
        }

    # ── Save results ──────────────────────────────────────────────
    results_3b = {
        'logit_results': {k: torch.tensor(v, dtype=torch.float32)
                          for k, v in logit_results.items()},
        'noise_mean_z': noise_mean_z.cpu(),
        'noise_mean_r': noise_mean_r.cpu(),
        'shuffle_mean_z': shuffle_mean_z.cpu(),
        'effects': effects,
        'num_valid': valid_count,
    }
    torch.save(results_3b, os.path.join(out, 'experiment_3b_causal_gate.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cond_labels = {'clamp_z': 'Clamp z_t', 'clamp_r': 'Clamp r_t',
                   'clamp_both': 'Clamp both', 'shuffle_z': 'Shuffle z_t (ctrl)'}
    cond_colors = {'clamp_z': '#e74c3c', 'clamp_r': '#3498db',
                   'clamp_both': '#9b59b6', 'shuffle_z': '#95a5a6'}

    # Panel 1: Logit shift distributions
    ax = axes[0]
    for cond in ['clamp_z', 'clamp_r', 'clamp_both', 'shuffle_z']:
        diff = baseline_arr - np.array(logit_results[cond])
        ax.hist(diff, bins=25, alpha=0.4, color=cond_colors[cond],
                label=f'{cond_labels[cond]} (μ={diff.mean():.3f})',
                edgecolor='white', linewidth=0.3, density=True)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Logit shift (baseline − clamped)')
    ax.set_ylabel('Density')
    ax.set_title('Causal Effect on Logit', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Baseline vs clamped scatter
    ax = axes[1]
    for cond in ['clamp_z', 'clamp_r', 'clamp_both', 'shuffle_z']:
        cond_arr = np.array(logit_results[cond])
        ax.scatter(baseline_arr, cond_arr, s=10, alpha=0.4,
                   color=cond_colors[cond], label=cond_labels[cond])
    lims = [min(baseline_arr.min(), -3), max(baseline_arr.max(), 3)]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Baseline logit')
    ax.set_ylabel('Clamped logit')
    ax.set_title('Baseline vs Clamped Logits', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Summary bar chart
    ax = axes[2]
    conds = ['clamp_z', 'clamp_r', 'clamp_both', 'shuffle_z']
    means = [effects[c]['mean_diff'] for c in conds]
    stds = [effects[c]['std_diff'] for c in conds]
    colors = [cond_colors[c] for c in conds]
    ax.bar(range(len(conds)), means, yerr=stds, capsize=5,
           color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([cond_labels[c] for c in conds], fontsize=9)
    ax.set_ylabel('Mean logit shift')
    ax.set_title('Causal Effect Magnitude', fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add sign-change percentages as text
    for i, c in enumerate(conds):
        pct = effects[c]['pct_sign_change'] * 100
        ax.text(i, means[i] + stds[i] + 0.05, f'{pct:.1f}% flip',
                ha='center', fontsize=8, color='#555')

    fig.suptitle('Experiment 3B — Causal Gate Analysis (Clamp to Noise Mean)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_3b_causal_gate.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Causal Gate Clamping Results (N={valid_count}):")
    print(f"  {'Condition':<20s} {'Mean Δlogit':>12s} {'Std':>8s} {'% Sign Flip':>12s}")
    print(f"  {'─' * 54}")
    for cond in conds:
        e = effects[cond]
        print(f"  {cond_labels[cond]:<20s} {e['mean_diff']:>12.4f} {e['std_diff']:>8.4f} "
              f"{e['pct_sign_change']*100:>11.1f}%")

    # Interpretation
    z_effect = abs(effects['clamp_z']['mean_diff'])
    r_effect = abs(effects['clamp_r']['mean_diff'])
    both_effect = abs(effects['clamp_both']['mean_diff'])
    shuffle_effect = abs(effects['shuffle_z']['mean_diff'])

    if z_effect > 0.5 and r_effect < 0.2:
        print("\n  → Update gate z_t has the larger functional effect on output")
    elif r_effect > 0.5 and z_effect < 0.2:
        print("\n  → Reset gate r_t has the larger functional effect on output")
    elif z_effect > 0.3 and r_effect > 0.3:
        print("\n  → Both gates have substantial functional effect on output")
    else:
        print("\n  → Gate clamping has limited effect — motif info may be in messages")

    if both_effect > z_effect + r_effect:
        print("  → Synergistic: combined effect exceeds sum of individual effects")
    elif both_effect < max(z_effect, r_effect):
        print("  → Redundant: combined effect is less than the larger individual effect")

    # Shuffle control interpretation
    # Note: A large shuffle effect means generic disruption changes output,
    # but this alone does not establish dimension-specific motif mechanism.
    if shuffle_effect > z_effect * 0.7:
        print(f"  ⚠ Shuffle control effect ({shuffle_effect:.4f}) is comparable to noise-clamp "
              f"({z_effect:.4f}) — disruption may be non-specific (generic gate perturbation)")
    else:
        print(f"  ✓ Shuffle control ({shuffle_effect:.4f}) << noise-clamp ({z_effect:.4f}) "
              f"— gate configuration matters more than generic disruption")

    return results_3b


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 3: Gate Analysis (A & B)')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--part', type=str, default='both', choices=['a', 'b', 'both'],
                        help='Which part to run: a (observational), b (causal), or both')
    args = parser.parse_args()
    if args.part in ('a', 'both'):
        run(args.model_path, args.output_dir, args.num_samples, args.seed, args.device)
    if args.part in ('b', 'both'):
        run_3b(args.model_path, args.output_dir, min(args.num_samples, 200), args.seed, args.device)
