"""
Experiment T2 — Per-Corruption Learning Order

Question:  Which corruption types does the model learn to detect first?

Method:    Evaluate a fixed test set at every checkpoint, grouping
           predictions by corruption type.  The resulting accuracy
           timelines reveal the model's learning curriculum.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    load_model, prepare_eval_dataset, find_checkpoints, make_dataloader,
    setup_output_dir, setup_plot_style, save_figure, seed_everything,
    run_forward, COLORS, CORRUPTION_ORDER, CORRUPTION_LABELS,
)

import matplotlib.pyplot as plt


def run(checkpoint_dir, output_dir, num_samples=500, seed=42, device='cpu',
        **kwargs):
    """Run Experiment T2: Per-Corruption Learning Order."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT T2 — Per-Corruption Learning Order")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_T2_corruption_order')

    # ── Find checkpoints ──────────────────────────────────────────
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"  ✗ No checkpoints found in {checkpoint_dir}")
        return None
    print(f"  Found {len(checkpoints)} checkpoints: epochs {[e for e, _ in checkpoints]}")

    # ── Generate fixed evaluation set ─────────────────────────────
    dataset = prepare_eval_dataset(num_samples, seed=seed)
    dataloader = make_dataloader(dataset.samples, batch_size=64)
    print(f"  Fixed eval set: {num_samples} samples")

    # ── Evaluate at each checkpoint ───────────────────────────────
    epochs = []
    per_corruption_accuracy = {}  # corruption_type -> list of accuracies

    for ckpt_idx, (epoch, ckpt_path) in enumerate(checkpoints):
        print(f"  Checkpoint {ckpt_idx + 1}/{len(checkpoints)} (epoch {epoch})...")

        model, config, _ = load_model(ckpt_path, device=device)
        epochs.append(epoch)

        # Collect predictions
        all_preds = []
        all_labels = []
        all_corruptions = []

        with torch.no_grad():
            for batch in dataloader:
                logits = run_forward(model, batch)
                preds = (logits > 0).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['y'].numpy())
                all_corruptions.extend(batch['corruption'])

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Per-corruption accuracy
        for corruption_type in set(all_corruptions):
            mask = np.array([c == corruption_type for c in all_corruptions])
            if mask.sum() == 0:
                continue
            acc = (all_preds[mask] == all_labels[mask]).mean()
            if corruption_type not in per_corruption_accuracy:
                per_corruption_accuracy[corruption_type] = []
            per_corruption_accuracy[corruption_type].append(acc)

        # Overall accuracy
        overall_acc = (all_preds == all_labels).mean()
        if 'overall' not in per_corruption_accuracy:
            per_corruption_accuracy['overall'] = []
        per_corruption_accuracy['overall'].append(overall_acc)

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'epochs': epochs,
        'per_corruption_accuracy': {
            k: torch.tensor(v) for k, v in per_corruption_accuracy.items()
        },
        'num_samples': num_samples,
    }
    torch.save(results, os.path.join(out, 'experiment_T2_corruption_timeline.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for corruption_type in CORRUPTION_ORDER:
        if corruption_type in per_corruption_accuracy:
            accs = per_corruption_accuracy[corruption_type]
            color = COLORS.get(corruption_type, '#888888')
            label = CORRUPTION_LABELS.get(corruption_type, corruption_type)
            ax.plot(epochs, accs, color=color, linewidth=2.2, marker='o',
                    markersize=5, label=f'{label}', alpha=0.85)

    # Overall (dashed)
    if 'overall' in per_corruption_accuracy:
        ax.plot(epochs, per_corruption_accuracy['overall'], color='#2c3e50',
                linewidth=2.5, linestyle='--', marker='s', markersize=4,
                label='Overall', alpha=0.9)

    ax.axhline(0.5, color='#bbb', linestyle=':', linewidth=1, alpha=0.6,
               label='Chance (50%)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Corruption Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    # Add threshold reference lines
    for threshold in [0.8, 0.9]:
        ax.axhline(threshold, color='#ddd', linestyle=':', linewidth=0.5)

    save_figure(fig, os.path.join(out, 'experiment_T2_corruption_timeline.png'))

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  Final per-corruption accuracy (epoch {epochs[-1]}):")
    print(f"  {'Corruption':<25s} {'Accuracy':>10s}")
    print(f"  {'─' * 37}")
    for corruption_type in CORRUPTION_ORDER:
        if corruption_type in per_corruption_accuracy:
            acc = per_corruption_accuracy[corruption_type][-1]
            label = CORRUPTION_LABELS.get(corruption_type, corruption_type)
            print(f"  {label:<25s} {acc:>10.1%}")
    if 'overall' in per_corruption_accuracy:
        print(f"  {'─' * 37}")
        print(f"  {'Overall':<25s} {per_corruption_accuracy['overall'][-1]:>10.1%}")

    # Learning order (by epoch where accuracy first exceeds 80%)
    print(f"\n  Learning order (epoch reaching 80% accuracy):")
    order = []
    for ct in CORRUPTION_ORDER:
        if ct in per_corruption_accuracy:
            accs = per_corruption_accuracy[ct]
            first_80 = next((epochs[i] for i, a in enumerate(accs) if a >= 0.8), None)
            label = CORRUPTION_LABELS.get(ct, ct)
            order.append((first_80 if first_80 is not None else float('inf'), label))
    order.sort()
    for rank, (ep, label) in enumerate(order):
        ep_str = f'epoch {ep}' if ep != float('inf') else 'never'
        print(f"    {rank + 1}. {label}: {ep_str}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment T2: Corruption Learning Order')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_dir, args.num_samples, args.seed, args.device)
