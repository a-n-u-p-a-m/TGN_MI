"""
Experiment 4 — Time-Window Mechanism

Question:  How does the model implement "is Δt < 5.0"?
           Sharp threshold or gradual?  At exactly Δ=5.0 or offset?

Method:    For each positive sample, inject synthetic time encodings at
           the B→C step spanning a fine-grained Δt sweep and record the
           model's logit response.  The resulting curve reveals the
           learned temporal decision boundary.
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

# Dense near the task boundary Δ=5.0
DELTA_T_SWEEP = [
    0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
    4.0, 4.25, 4.5, 4.75, 4.9, 4.95, 5.0, 5.05, 5.1,
    5.25, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 25.0, 50.0,
]


def run(model_path, output_dir, num_samples=100, seed=42, device='cpu'):
    """Run Experiment 4: Time-Window Mechanism."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4 — Time-Window Mechanism")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_4_time_window')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep})")

    # ── Generate positive evaluation samples ──────────────────────
    dataset = prepare_eval_dataset(num_samples * 3, seed=seed)
    positives = get_positive_samples(dataset, max_samples=num_samples)
    print(f"  Using {len(positives)} positive samples")
    print(f"  Δt sweep: {len(DELTA_T_SWEEP)} values, range [{DELTA_T_SWEEP[0]}, {DELTA_T_SWEEP[-1]}]")

    # ── Run Δt sweep ──────────────────────────────────────────────
    # (num_samples, num_dt) matrix of logits
    logit_matrix = np.full((len(positives), len(DELTA_T_SWEEP)), np.nan)
    baseline_logits = []
    actual_deltas = []

    for idx, sample in enumerate(positives):
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  Processing sample {idx + 1}/{len(positives)}...")

        batch = single_sample_batch(sample)
        motif_steps = get_motif_steps(model, batch)
        m1, m2 = motif_steps[0]

        if m1 < 0 or m2 < 0:
            continue

        # Clean forward — get baseline logit
        baseline = run_forward(model, batch).item()
        baseline_logits.append(baseline)

        # Extract the actual Δt at B→C for reference
        # (the time encoding hooks capture delta_t)
        _ = run_forward(model, batch)
        te_hooks = model.time_encoder.hook_data
        # At event step m2, there are 2 time encoder calls:
        #   index m2*2 for src, m2*2+1 for dst
        # We want src (B's perspective)
        if m2 * 2 < len(te_hooks):
            actual_dt = te_hooks[m2 * 2]['delta_t'][0].item()
            actual_deltas.append(actual_dt)
        else:
            actual_deltas.append(float('nan'))

        # Sweep synthetic Δt values
        for dt_idx, synthetic_dt in enumerate(DELTA_T_SWEEP):
            # Compute the time encoding for this synthetic Δt
            dt_tensor = torch.tensor([synthetic_dt], dtype=torch.float32,
                                     device=device)

            # Pre-compute the encoding using the model's time encoder
            with torch.no_grad():
                # Temporarily clear time encoder hooks for this call
                old_hooks = model.time_encoder.hook_data
                model.time_encoder.hook_data = []
                synthetic_encoding = model.time_encoder(dt_tensor)
                model.time_encoder.hook_data = old_hooks

            def inject_time_enc(x, _enc=synthetic_encoding):
                return _enc.to(x.device).expand_as(x)

            interventions = {(m2, 'time_encoding_src'): inject_time_enc}
            patched_logit = run_forward(model, batch,
                                        interventions=interventions).item()
            logit_matrix[idx, dt_idx] = patched_logit

    # ── Filter out rows with NaN ──────────────────────────────────
    valid_mask = ~np.isnan(logit_matrix).any(axis=1)
    logit_matrix = logit_matrix[valid_mask]
    baseline_logits = np.array(baseline_logits)[valid_mask[:len(baseline_logits)]]
    print(f"  Valid samples: {logit_matrix.shape[0]}")

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'logit_matrix': torch.tensor(logit_matrix, dtype=torch.float32),
        'delta_t_sweep': torch.tensor(DELTA_T_SWEEP, dtype=torch.float32),
        'baseline_logits': torch.tensor(baseline_logits, dtype=torch.float32),
        'actual_deltas': actual_deltas,
        'num_valid': logit_matrix.shape[0],
    }
    torch.save(results, os.path.join(out, 'experiment_4_time_window.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    dt_array = np.array(DELTA_T_SWEEP)

    # Individual sample curves (thin, transparent)
    for i in range(logit_matrix.shape[0]):
        ax.plot(dt_array, logit_matrix[i], color='#3498db',
                alpha=0.08, linewidth=0.5)

    # Mean curve (bold)
    mean_logit = logit_matrix.mean(axis=0)
    ax.plot(dt_array, mean_logit, color='#2c3e50', linewidth=3,
            label='Mean', zorder=10)

    # ±1 std band
    std_logit = logit_matrix.std(axis=0)
    ax.fill_between(dt_array, mean_logit - std_logit, mean_logit + std_logit,
                    alpha=0.15, color='#3498db')

    # Task boundary
    ax.axvline(5.0, color='#e74c3c', linestyle='--', linewidth=2,
               alpha=0.8, label='Task boundary (Δt=5.0)')
    ax.axhline(0, color='#7f8c8d', linestyle='-', linewidth=1,
               alpha=0.5, label='Decision boundary (logit=0)')

    ax.set_xlabel('Synthetic Δt (injected at B→C)')
    ax.set_ylabel('Model logit')
    ax.set_title('Time-Window Mechanism — Logit Response to Δt',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(DELTA_T_SWEEP[0] * 0.5, DELTA_T_SWEEP[-1] * 1.5)

    save_figure(fig, os.path.join(out, 'experiment_4_time_window.png'))

    # ── Print summary ─────────────────────────────────────────────
    # Find the transition point (where mean logit crosses 0)
    sign_changes = np.where(np.diff(np.sign(mean_logit)))[0]
    if len(sign_changes) > 0:
        sc = sign_changes[0]
        transition_dt = dt_array[sc] + (dt_array[sc+1] - dt_array[sc]) * (
            -mean_logit[sc]) / (mean_logit[sc+1] - mean_logit[sc])
        print(f"\n  Transition point (logit crosses 0): Δt ≈ {transition_dt:.3f}")
        print(f"  Task boundary: Δt = 5.0")
        print(f"  Offset: {transition_dt - 5.0:+.3f}")
    else:
        print(f"\n  No clear sign transition found in mean curve")

    # Sharpness
    idx_4_5 = np.argmin(np.abs(dt_array - 4.5))
    idx_5_5 = np.argmin(np.abs(dt_array - 5.5))
    sharpness = mean_logit[idx_4_5] - mean_logit[idx_5_5]
    print(f"  Sharpness (logit@4.5 − logit@5.5): {sharpness:.4f}")

    # Note: Δt_TGN = t2 - t_B^last, not necessarily t2 - t1, because
    # intervening B events can shift the baseline. This result shows a
    # threshold in the model's internal temporal representation, not
    # necessarily exact computation of the planted motif interval.
    if sharpness > 2.0:
        print("  → Sharp sigmoid transition in the model's temporal representation")
    elif sharpness > 0.5:
        print("  → Moderate transition — soft temporal preference, not hard threshold")
    else:
        print("  → Gradual — continuous recency signal")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 4: Time-Window Mechanism')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.model_path, args.output_dir, args.num_samples, args.seed, args.device)
