"""
Shared utilities for MI experiments.

Provides model loading, dataset preparation, checkpoint discovery,
and consistent plotting helpers used across all experiment scripts.
"""

import os
import sys
import glob
import re
import json
import torch
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from temporal_gnn_mi.config import DatasetConfig, ModelConfig
from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
from temporal_gnn_mi.datasets.batch_utils import temporal_collate_fn
from temporal_gnn_mi.models.tgn import TGN
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def infer_model_config(state_dict):
    """Infer ModelConfig from checkpoint state dict weight shapes.

    Derivations:
      memory_dim  = memory.W_z.shape[1]               (128)
      message_dim = memory.W_z.shape[0] - memory_dim   (32)
      time_dim    = time_encoder.w.shape[1] * 2         (16)
      embedding_dim = classifier_hidden.0.weight.shape[0]  (64)
    """
    wz = state_dict['memory.W_z']
    memory_dim = wz.shape[1]
    message_dim = wz.shape[0] - memory_dim

    tw = state_dict['time_encoder.w']
    time_dim = tw.shape[1] * 2

    ch = state_dict['classifier_hidden.0.weight']
    embedding_dim = ch.shape[0]

    # Infer edge_feat_dim from message MLP input layer:
    #   input_dim = memory_dim*2 + time_dim + edge_feat_dim
    msg_input_dim = state_dict['message_function.mlp.0.weight'].shape[1]
    edge_feat_dim = msg_input_dim - memory_dim * 2 - time_dim

    return ModelConfig(
        memory_dim=memory_dim,
        message_dim=message_dim,
        time_dim=time_dim,
        embedding_dim=embedding_dim,
        dropout=0.108,
    ), edge_feat_dim


def load_model(checkpoint_path, device='cpu', num_nodes=50):
    """Load a trained TGN model from a checkpoint file.

    Returns:
        (model, config, epoch) — model is in eval mode on `device`.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', -1)
    else:
        state_dict = checkpoint
        epoch = -1

    config, edge_feat_dim = infer_model_config(state_dict)
    model = TGN(config, num_nodes=num_nodes, edge_feat_dim=edge_feat_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, epoch


def load_state_dict_only(checkpoint_path, device='cpu'):
    """Load just the state dict from a checkpoint (lighter weight)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint.get('epoch', -1)
    return checkpoint, -1


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset preparation
# ═══════════════════════════════════════════════════════════════════════════

def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_eval_dataset(num_samples, seed=42, noise_multiplier=4):
    """Generate a seeded evaluation dataset, isolated from training data.

    Uses the same DatasetConfig as the trained model (num_nodes=50,
    noise_multiplier=4, time_window_delta=5.0) but with a seed offset
    of +10000 to ensure MI experiments run on held-out data that does
    not overlap with the training set.
    """
    eval_seed = seed + 10000  # Isolate from training RNG stream
    seed_everything(eval_seed)
    dataset_config = DatasetConfig(
        num_nodes=50,
        noise_multiplier=noise_multiplier,
        time_window_delta=5.0,
        num_distractors=3,
        num_partial_motifs=2,
        seed=eval_seed,
    )
    dataset = TemporalMotifDataset(dataset_config, num_samples=num_samples, split='test')
    return dataset


def get_positive_samples(dataset, max_samples=None):
    """Filter dataset to only positive (label=1) samples."""
    positives = [s for s in dataset.samples if s.y.item() == 1.0]
    if max_samples is not None:
        positives = positives[:max_samples]
    return positives


def get_negative_samples(dataset, corruption_type=None, max_samples=None):
    """Filter dataset to negative samples, optionally by corruption type."""
    negatives = [s for s in dataset.samples if s.y.item() == 0.0]
    if corruption_type is not None:
        negatives = [s for s in negatives if getattr(s, 'corruption', '') == corruption_type]
    if max_samples is not None:
        negatives = negatives[:max_samples]
    return negatives


def make_dataloader(samples, batch_size=32):
    """Create a DataLoader from a list of TemporalData samples."""
    return DataLoader(samples, batch_size=batch_size,
                      collate_fn=temporal_collate_fn, shuffle=False)


def single_sample_batch(sample):
    """Wrap a single sample in a batch dict (batch_size=1)."""
    return temporal_collate_fn([sample])


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_checkpoints(run_dir):
    """Discover checkpoint files in a directory.

    Searches three locations and deduplicates by epoch:
      1. run_dir/checkpoint_ep*.pt          (fixed-interval training checkpoints)
      2. run_dir/analysis_checkpoints/epoch_*.pt  (smart + milestone checkpoints)
      3. run_dir/best_model.pt              (early-stopping best model)

    When the same epoch appears in both (1) and (2), the analysis checkpoint
    is preferred because it is lighter-weight (no optimizer state) and was
    saved specifically for MI experiments.

    Returns:
        Sorted list of (epoch, filepath) tuples.
    """
    seen = {}  # epoch -> filepath

    # 1. Fixed-interval training checkpoints
    pattern = os.path.join(run_dir, 'checkpoint_ep*.pt')
    for f in glob.glob(pattern):
        match = re.search(r'checkpoint_ep(\d+)', os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            seen[epoch] = f

    # 2. Smart / milestone analysis checkpoints (preferred over training ckpts)
    analysis_dir = os.path.join(run_dir, 'analysis_checkpoints')
    analysis_pattern = os.path.join(analysis_dir, 'epoch_*.pt')
    for f in glob.glob(analysis_pattern):
        match = re.search(r'epoch_(\d+)', os.path.basename(f))
        if match:
            epoch = int(match.group(1))
            seen[epoch] = f  # overwrites training ckpt for same epoch

    # 3. best_model.pt (only if its epoch isn't already covered)
    best = os.path.join(run_dir, 'best_model.pt')
    if os.path.exists(best):
        try:
            ckpt = torch.load(best, map_location='cpu', weights_only=False)
            ep = ckpt.get('epoch', -1)
            if ep >= 0 and ep not in seen:
                seen[ep] = best
        except Exception:
            pass

    results = sorted(seen.items(), key=lambda x: x[0])
    return results


def setup_output_dir(base_dir, experiment_name):
    """Create and return the output directory for an experiment."""
    out_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════
#  Forward-pass helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_forward(model, batch_dict, interventions=None):
    """Run a forward pass, return logits as a 1D tensor.

    Clears hooks before the pass to avoid accumulation.
    """
    model.clear_hooks()
    logits = model(batch_dict, interventions=interventions)
    return logits.squeeze(-1)  # (B,)


def get_motif_steps(model, batch_dict):
    """Get (m1, m2) motif step indices for each sample in the batch.

    Returns list of (m1, m2) tuples.
    """
    return model.get_motif_step_indices(batch_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

# Professional color palette
COLORS = {
    'topology_break':       '#e74c3c',
    'edge_deletion':        '#e67e22',
    'positive':             '#2ecc71',
    'none':                 '#2ecc71',
    'temporal_misordering': '#3498db',
    'timing_break':         '#9b59b6',
    'source_shuffle':       '#1abc9c',
}

CORRUPTION_ORDER = [
    'topology_break', 'edge_deletion', 'none',
    'temporal_misordering', 'timing_break', 'source_shuffle',
]

CORRUPTION_LABELS = {
    'topology_break':       'Topology Break',
    'edge_deletion':        'Edge Deletion',
    'none':                 'Positive',
    'positive':             'Positive',
    'temporal_misordering': 'Temporal Misordering',
    'timing_break':         'Timing Break',
    'source_shuffle':       'Source Shuffle',
}


def setup_plot_style():
    """Apply a clean, professional plotting style."""
    plt.rcParams.update({
        'figure.facecolor':  'white',
        'axes.facecolor':    '#fafafa',
        'axes.edgecolor':    '#cccccc',
        'axes.labelcolor':   '#333333',
        'axes.titlesize':    13,
        'axes.labelsize':    11,
        'text.color':        '#333333',
        'xtick.color':       '#555555',
        'ytick.color':       '#555555',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'grid.color':        '#e0e0e0',
        'grid.alpha':        0.7,
        'grid.linestyle':    '--',
        'figure.dpi':        150,
        'savefig.dpi':       150,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.15,
        'font.size':         11,
        'font.family':       'sans-serif',
        'legend.fontsize':   9,
        'legend.framealpha':  0.9,
        'legend.edgecolor':  '#cccccc',
        'lines.linewidth':   1.8,
    })


def save_figure(fig, path, name=None):
    """Save a figure and close it."""
    fig.savefig(path)
    plt.close(fig)
    display_name = name or os.path.basename(path)
    print(f"  ✓ Saved: {display_name}")
