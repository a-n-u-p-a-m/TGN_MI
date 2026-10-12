"""
Experiment 5 — Source-Shuffle Failure Analysis

Question:  Why is source_shuffle accuracy only 77%?  What does the model
           "see" differently in samples it gets right vs. wrong?

Method:    Generate source_shuffle negatives, classify them, partition
           into correct / misclassified, and compare readout memories
           (mem_A, mem_B, mem_C) to the positive-sample reference
           distribution via cosine similarity.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.utils import (
    load_model, prepare_eval_dataset, get_positive_samples, get_negative_samples,
    single_sample_batch, run_forward, setup_output_dir, setup_plot_style,
    save_figure, seed_everything, make_dataloader,
)

import matplotlib.pyplot as plt


def _extract_readout_memories(model, samples, device='cpu'):
    """Extract readout memories (mem_A, mem_B, mem_C) for a list of samples.

    Returns (mem_A, mem_B, mem_C) tensors of shape (N, memory_dim).
    """
    all_mem_A, all_mem_B, all_mem_C = [], [], []
    all_logits = []

    for sample in samples:
        batch = single_sample_batch(sample)
        logit = run_forward(model, batch).item()
        all_logits.append(logit)

        # Extract readout memory from classifier hooks
        cls_hooks = model.classifier_hook_data
        if cls_hooks:
            readout = cls_hooks[-1]['readout_memory']  # (1, 3, mem_dim)
            all_mem_A.append(readout[0, 0].cpu())
            all_mem_B.append(readout[0, 1].cpu())
            all_mem_C.append(readout[0, 2].cpu())

    return (
        torch.stack(all_mem_A) if all_mem_A else torch.empty(0),
        torch.stack(all_mem_B) if all_mem_B else torch.empty(0),
        torch.stack(all_mem_C) if all_mem_C else torch.empty(0),
        np.array(all_logits),
    )


def _cosine_sim_to_mean(vectors, reference_mean):
    """Compute cosine similarity of each vector to a reference mean."""
    if vectors.numel() == 0 or reference_mean.numel() == 0:
        return np.array([])
    ref = reference_mean.unsqueeze(0)  # (1, dim)
    cos = torch.nn.functional.cosine_similarity(vectors, ref, dim=1)
    return cos.numpy()


def run(model_path, output_dir, num_samples=5000, seed=42, device='cpu'):
    """Run Experiment 5: Source-Shuffle Failure Analysis."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 5 — Source-Shuffle Failure Analysis")
    print("=" * 65)

    seed_everything(seed)
    out = setup_output_dir(output_dir, 'experiment_5_source_shuffle')

    # ── Load model ────────────────────────────────────────────────
    model, config, ep = load_model(model_path, device=device)
    print(f"  Model loaded (epoch {ep})")

    # ── Generate dataset with source_shuffle negatives ────────────
    # Generate a large dataset to get enough source_shuffle samples
    dataset = prepare_eval_dataset(num_samples, seed=seed)
    shuffle_negatives = get_negative_samples(dataset, corruption_type='source_shuffle')
    positive_samples = get_positive_samples(dataset, max_samples=200)
    print(f"  Source-shuffle negatives: {len(shuffle_negatives)}")
    print(f"  Positive reference samples: {len(positive_samples)}")

    if len(shuffle_negatives) < 10:
        print("  ✗ Too few source_shuffle samples — increase num_samples")
        return None

    # ── Extract readout memories for positives ────────────────────
    print("  Extracting positive reference memories...")
    pos_A, pos_B, pos_C, pos_logits = _extract_readout_memories(
        model, positive_samples, device)
    pos_mean_A = pos_A.mean(dim=0)
    pos_mean_B = pos_B.mean(dim=0)
    pos_mean_C = pos_C.mean(dim=0)

    # ── Classify source_shuffle negatives ─────────────────────────
    print("  Classifying source_shuffle negatives...")
    shuf_A, shuf_B, shuf_C, shuf_logits = _extract_readout_memories(
        model, shuffle_negatives, device)

    # Partition: correct (logit ≤ 0 → predicted negative) vs misclassified (logit > 0)
    correct_mask = shuf_logits <= 0
    misclassified_mask = shuf_logits > 0

    n_correct = correct_mask.sum()
    n_misclassified = misclassified_mask.sum()
    accuracy = n_correct / len(shuf_logits)
    print(f"  Correct: {n_correct}, Misclassified: {n_misclassified}")
    print(f"  Source-shuffle accuracy: {accuracy:.1%}")

    # ── Compute cosine similarities to positive mean ──────────────
    groups = {
        'positive': (pos_A, pos_B, pos_C),
        'correct_neg': (shuf_A[correct_mask], shuf_B[correct_mask], shuf_C[correct_mask]),
        'misclassified': (shuf_A[misclassified_mask], shuf_B[misclassified_mask], shuf_C[misclassified_mask]),
    }

    cos_sims = {}
    for group_name, (mem_A, mem_B, mem_C) in groups.items():
        cos_sims[group_name] = {
            'mem_A': _cosine_sim_to_mean(mem_A, pos_mean_A),
            'mem_B': _cosine_sim_to_mean(mem_B, pos_mean_B),
            'mem_C': _cosine_sim_to_mean(mem_C, pos_mean_C),
        }

    # ── Save raw results ──────────────────────────────────────────
    results = {
        'positive_mem': {'A': pos_A, 'B': pos_B, 'C': pos_C},
        'shuffle_mem': {'A': shuf_A, 'B': shuf_B, 'C': shuf_C},
        'shuffle_logits': torch.tensor(shuf_logits),
        'correct_mask': torch.tensor(correct_mask),
        'accuracy': accuracy,
        'cos_sims': {
            group: {node: torch.tensor(vals) for node, vals in sims.items()}
            for group, sims in cos_sims.items()
        },
    }
    torch.save(results, os.path.join(out, 'experiment_5_source_shuffle.pt'))
    print(f"  ✓ Saved raw results")

    # ── Plot ──────────────────────────────────────────────────────
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    node_names = ['mem_A', 'mem_B', 'mem_C']
    group_colors = {'positive': '#2ecc71', 'correct_neg': '#3498db', 'misclassified': '#e74c3c'}
    group_labels = {'positive': 'Positive', 'correct_neg': 'Correct neg', 'misclassified': 'Misclassified'}

    for ni, node in enumerate(node_names):
        ax = axes[ni]
        for group_name in ['positive', 'correct_neg', 'misclassified']:
            vals = cos_sims[group_name][node]
            if len(vals) > 0:
                ax.hist(vals, bins=25, alpha=0.55, color=group_colors[group_name],
                        label=f'{group_labels[group_name]} (μ={vals.mean():.3f})',
                        edgecolor='white', linewidth=0.3, density=True)
        ax.set_xlabel('Cosine similarity to positive mean')
        ax.set_ylabel('Density')
        ax.set_title(node, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Source-Shuffle Failure Analysis — Memory Similarity to Positive Reference',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'experiment_5_source_shuffle.png'))

    # ── Print summary table ───────────────────────────────────────
    print(f"\n  Cosine Similarity to Positive Mean:")
    print(f"  {'Group':<20s} {'mem_A':>10s} {'mem_B':>10s} {'mem_C':>10s}")
    print(f"  {'─' * 52}")
    for group_name in ['positive', 'correct_neg', 'misclassified']:
        vals_a = cos_sims[group_name]['mem_A']
        vals_b = cos_sims[group_name]['mem_B']
        vals_c = cos_sims[group_name]['mem_C']
        mean_a = vals_a.mean() if len(vals_a) > 0 else float('nan')
        mean_b = vals_b.mean() if len(vals_b) > 0 else float('nan')
        mean_c = vals_c.mean() if len(vals_c) > 0 else float('nan')
        print(f"  {group_labels[group_name]:<20s} {mean_a:>10.4f} {mean_b:>10.4f} {mean_c:>10.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 5: Source-Shuffle Analysis')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='runs/experiments')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    run(args.model_path, args.output_dir, args.num_samples, args.seed, args.device)
