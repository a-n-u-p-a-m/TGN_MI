"""
Experiment 7 — Shuffled-Label Null Control for Gate Analysis

Question:  Is the gate discriminability found in Exp 3A a genuine signal,
           or could a random motif/noise labeling produce similar results?

Method:    Repeat the Exp 3A gate analysis, but randomly shuffle which
           hook is designated "motif update" and which is "non-motif update"
           for each sample.  If the Exp 3A finding is real, this null
           distribution should show zero significant dimensions (after BH
           correction), validating that the original finding is not an
           artifact of the analysis pipeline.

Produces:  Null distribution of significant-dimension counts, compared
           against the real Exp 3A count.
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
from experiments.static.exp3_gate_analysis import _find_noise_b_update, _benjamini_hochberg

import matplotlib.pyplot as plt


def _extract_gate_vectors(model, positives, device, shuffle_labels=False, rng=None):
    """Extract z_t and r_t gate vectors for motif vs non-motif updates.

    If shuffle_labels=True, randomly assign which of the two updates is
    called 'motif' and which is 'non-motif', destroying the true signal.
    """
    motif_z_list, noise_z_list = [], []
    motif_r_list, noise_r_list = [], []

    for sample in positives:
        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]
        if m1 < 0:
            continue

        _ = run_forward(model, batch)
        hook_data = model.memory.hook_data

        B_node = batch['candidate_nodes'][0, 1].item()
        motif_hi = 2 * m1 + 1  # dst update at A→B step
        noise_hi = _find_noise_b_update(hook_data, m1, m2, B_node)
        if noise_hi < 0:
            continue

        # Optionally shuffle which is 'motif' vs 'noise'
        if shuffle_labels and rng is not None:
            if rng.random() < 0.5:
                motif_hi, noise_hi = noise_hi, motif_hi

        motif_z_list.append(hook_data[motif_hi]['gates']['z_t'][0].cpu().numpy())
        noise_z_list.append(hook_data[noise_hi]['gates']['z_t'][0].cpu().numpy())
        motif_r_list.append(hook_data[motif_hi]['gates']['r_t'][0].cpu().numpy())
        noise_r_list.append(hook_data[noise_hi]['gates']['r_t'][0].cpu().numpy())

    return (np.array(motif_z_list), np.array(noise_z_list),
            np.array(motif_r_list), np.array(noise_r_list))


def _count_significant(motif_z, noise_z, alpha=0.05):
    """Count BH-significant dimensions from paired t-test."""
    memory_dim = motif_z.shape[1]
    pvals = np.zeros(memory_dim)
    for d in range(memory_dim):
        _, p = stats.ttest_rel(motif_z[:, d], noise_z[:, d])
        pvals[d] = p

    _, rejected = _benjamini_hochberg(pvals, alpha=alpha)
    return int(rejected.sum()), pvals


def run(model_path, output_dir, num_samples=300, num_permutations=250,
        seed=42, device='cpu'):
    """Run Experiment 7: Shuffled-Label Null Control."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 7 — Shuffled-Label Null Control")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_7_null_control')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Real analysis (no shuffling) ──────────────────────────────
    print("  Running real gate analysis...")
    motif_z, noise_z, motif_r, noise_r = _extract_gate_vectors(
        model, positives, device, shuffle_labels=False)
    real_sig_z, _ = _count_significant(motif_z, noise_z)
    real_sig_r, _ = _count_significant(motif_r, noise_r)
    memory_dim = motif_z.shape[1]
    print(f"    Real: {real_sig_z}/{memory_dim} z_t dims, "
          f"{real_sig_r}/{memory_dim} r_t dims significant")

    # ── Null distribution (shuffled labels) ───────────────────────
    print(f"  Running {num_permutations} shuffled-label permutations...")
    null_sig_z = []
    null_sig_r = []
    rng = np.random.RandomState(seed)

    for i in range(num_permutations):
        if (i + 1) % 25 == 0:
            print(f"    Permutation {i + 1}/{num_permutations}...")

        shuf_motif_z, shuf_noise_z, shuf_motif_r, shuf_noise_r = \
            _extract_gate_vectors(model, positives, device,
                                  shuffle_labels=True, rng=rng)
        sig_z, _ = _count_significant(shuf_motif_z, shuf_noise_z)
        sig_r, _ = _count_significant(shuf_motif_r, shuf_noise_r)
        null_sig_z.append(sig_z)
        null_sig_r.append(sig_r)

    null_sig_z = np.array(null_sig_z)
    null_sig_r = np.array(null_sig_r)

    # Compute p-values with +1 correction (avoids exact-zero p-values)
    p_value_z = float((1 + (null_sig_z >= real_sig_z).sum()) / (1 + num_permutations))
    p_value_r = float((1 + (null_sig_r >= real_sig_r).sum()) / (1 + num_permutations))

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'real_sig_z': real_sig_z,
        'real_sig_r': real_sig_r,
        'null_sig_z': torch.tensor(null_sig_z),
        'null_sig_r': torch.tensor(null_sig_r),
        'p_value_z': p_value_z,
        'p_value_r': p_value_r,
        'num_permutations': num_permutations,
        'memory_dim': memory_dim,
        'num_samples': len(motif_z),
    }
    torch.save(results, os.path.join(out, 'experiment_7_null_control.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, null_vals, real_val, pval, gate_name in [
        (axes[0], null_sig_z, real_sig_z, p_value_z, 'z_t (update gate)'),
        (axes[1], null_sig_r, real_sig_r, p_value_r, 'r_t (reset gate)'),
    ]:
        ax.hist(null_vals, bins=max(10, int(null_vals.max()) + 1),
                color='#bbbbbb', edgecolor='white', alpha=0.8,
                label=f'Null distribution (n={num_permutations})')
        ax.axvline(real_val, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Real = {real_val} (p = {pval:.4f})')
        ax.set_xlabel('# significant dimensions (BH q < 0.05)')
        ax.set_ylabel('Count')
        ax.set_title(f'{gate_name}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Null Control: Shuffled-Label Gate Discriminability',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, os.path.join(out, 'experiment_7_null_control.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Null Control Results:")
    print(f"    z_t: real={real_sig_z}, null mean={null_sig_z.mean():.1f} "
          f"± {null_sig_z.std():.1f}, p={p_value_z:.4f}")
    print(f"    r_t: real={real_sig_r}, null mean={null_sig_r.mean():.1f} "
          f"± {null_sig_r.std():.1f}, p={p_value_r:.4f}")

    if p_value_z < 0.05 and p_value_r < 0.05:
        print("\n  → Both gates show significantly more discriminability than null")
        print("    Gate specialization finding (Exp 3A) is validated")
    elif p_value_z < 0.05 or p_value_r < 0.05:
        validated = "z_t" if p_value_z < 0.05 else "r_t"
        print(f"\n  → Only {validated} exceeds null; the other gate's signal is not clearly above chance")
    else:
        print("\n  → Neither gate exceeds null — gate discriminability may be artifactual")

    return results
