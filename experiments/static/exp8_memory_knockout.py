"""
Experiment 8 — Memory Knockout

Question:  Is temporal memory at the B→C step more functionally important
           than memory at a random control step?

Method:    For each positive sample:
             1. Run a clean forward pass → clean logit
             2. Zero out ALL node memory at the start of the B→C event step
                (i.e., all memory state is wiped just before the second
                motif edge is processed) → knockout logit
             3. Zero out memory at a random non-motif step → control logit

           If memory is causally necessary, the knockout should shift the
           logit toward the negative (incorrect) direction, while the
           random-step control should have a smaller effect.

Produces:  Paired comparison of clean vs knockout logits, with a
           random-step control for baseline comparison.
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


def run(model_path, output_dir, num_samples=200, seed=42, device='cpu'):
    """Run Experiment 8: Memory Knockout."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 8 — Memory Knockout")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_8_memory_knockout')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")

    # ── Run knockout experiment ───────────────────────────────────
    clean_logits = []
    knockout_logits = []
    control_logits = []
    rng = np.random.RandomState(seed)

    for idx, sample in enumerate(positives):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(positives)}...")

        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]
        if m1 < 0 or m2 < 0:
            continue

        num_events = batch['src'].size(1)

        # (1) Clean logit
        clean_logit = run_forward(model, batch).item()

        # (2) Knockout: zero ALL memory at B→C step (both src and dst)
        def zero_memory(x):
            return torch.zeros_like(x)

        knockout_interventions = {
            (m2, 'memory_pre_src'): zero_memory,
            (m2, 'memory_pre_dst'): zero_memory,
        }
        knockout_logit = run_forward(model, batch,
                                     interventions=knockout_interventions).item()

        # (3) Control: zero memory at a random non-motif step
        non_motif_steps = [s for s in range(num_events)
                          if s != m1 and s != m2 and s > 0]
        if non_motif_steps:
            ctrl_step = rng.choice(non_motif_steps)
            control_interventions = {
                (ctrl_step, 'memory_pre_src'): zero_memory,
                (ctrl_step, 'memory_pre_dst'): zero_memory,
            }
            control_logit = run_forward(model, batch,
                                        interventions=control_interventions).item()
        else:
            control_logit = clean_logit  # no non-motif steps available

        clean_logits.append(clean_logit)
        knockout_logits.append(knockout_logit)
        control_logits.append(control_logit)

    # ── Analysis ──────────────────────────────────────────────────
    clean_arr = np.array(clean_logits)
    knockout_arr = np.array(knockout_logits)
    control_arr = np.array(control_logits)

    # Effect sizes
    knockout_effect = clean_arr - knockout_arr  # positive = knockout hurt performance
    control_effect = clean_arr - control_arr

    mean_knockout_effect = float(knockout_effect.mean())
    std_knockout_effect = float(knockout_effect.std())
    mean_control_effect = float(control_effect.mean())
    std_control_effect = float(control_effect.std())

    # Classification accuracy change
    clean_correct = (clean_arr > 0).mean()
    knockout_correct = (knockout_arr > 0).mean()
    control_correct = (control_arr > 0).mean()

    # Paired t-test: knockout effect > control effect
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(knockout_effect, control_effect)
    # One-sided: knockout effect should be greater
    p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'clean_logits': torch.tensor(clean_arr, dtype=torch.float32),
        'knockout_logits': torch.tensor(knockout_arr, dtype=torch.float32),
        'control_logits': torch.tensor(control_arr, dtype=torch.float32),
        'mean_knockout_effect': mean_knockout_effect,
        'std_knockout_effect': std_knockout_effect,
        'mean_control_effect': mean_control_effect,
        'std_control_effect': std_control_effect,
        'clean_accuracy': float(clean_correct),
        'knockout_accuracy': float(knockout_correct),
        'control_accuracy': float(control_correct),
        't_stat': float(t_stat),
        'p_value_one_sided': float(p_one_sided),
        'num_samples': len(clean_arr),
    }
    torch.save(results, os.path.join(out, 'experiment_8_memory_knockout.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) Paired logit comparison
    ax = axes[0]
    ax.scatter(clean_arr, knockout_arr, s=15, alpha=0.5, color='#e74c3c',
               label='Knockout')
    ax.scatter(clean_arr, control_arr, s=15, alpha=0.5, color='#3498db',
               label='Control')
    lims = [min(clean_arr.min(), knockout_arr.min(), control_arr.min()) - 0.5,
            max(clean_arr.max(), knockout_arr.max(), control_arr.max()) + 0.5]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Clean logit')
    ax.set_ylabel('Intervened logit')
    ax.set_title('Clean vs Intervened Logits', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Effect size distribution
    ax = axes[1]
    ax.hist(knockout_effect, bins=25, color='#e74c3c', alpha=0.6,
            edgecolor='white', label=f'Knockout (μ={mean_knockout_effect:.3f})')
    ax.hist(control_effect, bins=25, color='#3498db', alpha=0.6,
            edgecolor='white', label=f'Control (μ={mean_control_effect:.3f})')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Logit reduction (clean − intervened)')
    ax.set_ylabel('Count')
    ax.set_title('Effect Size Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) Accuracy comparison
    ax = axes[2]
    conditions = ['Clean', 'Knockout\n(at B→C)', 'Control\n(random step)']
    accs = [clean_correct, knockout_correct, control_correct]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(conditions, accs, color=colors, edgecolor='white', width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', fontsize=10)
    ax.set_ylabel('Positive-class accuracy')
    ax.set_title('Accuracy Under Memory Knockout', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Memory Knockout (p = {p_one_sided:.4f}, knockout > control)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, os.path.join(out, 'experiment_8_memory_knockout.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Memory Knockout Results:")
    print(f"    Clean accuracy:    {clean_correct:.1%}")
    print(f"    Knockout accuracy: {knockout_correct:.1%}")
    print(f"    Control accuracy:  {control_correct:.1%}")
    print(f"    Knockout effect:   {mean_knockout_effect:+.4f} ± {std_knockout_effect:.4f}")
    print(f"    Control effect:    {mean_control_effect:+.4f} ± {std_control_effect:.4f}")
    print(f"    t-test (knockout > control): t={t_stat:.3f}, p={p_one_sided:.4f}")

    if p_one_sided < 0.05 and mean_knockout_effect > 0:
        print("\n  → B→C memory is more functionally important than memory at the sampled control step")
        print("    Knockout has significantly larger effect than random-step control (p < 0.05)")
        print("    Note: this establishes functional importance, not strict causal necessity")
    elif mean_knockout_effect > mean_control_effect:
        print("\n  → Knockout has larger effect than control but not statistically significant")
    else:
        print("\n  → No clear evidence that B→C memory is more important than the control step")

    return results
