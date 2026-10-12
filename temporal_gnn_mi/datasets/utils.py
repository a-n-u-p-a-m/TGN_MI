import torch
import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

def set_seed(seed: int):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sort_events_by_time(events: List[Tuple[int, int, float, torch.Tensor]]) -> List[Tuple[int, int, float, torch.Tensor]]:
    """Sorts a list of events by their timestamp."""
    return sorted(events, key=lambda x: x[2])

def visualize_temporal_graph(events: List[Tuple[int, int, float, torch.Tensor]], motif_edges: List[int] = None):
    """
    Visualizes the temporal graph, optionally highlighting motif edges.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = [e[2] for e in events]
    srcs = [e[0] for e in events]
    dsts = [e[1] for e in events]
    
    ax.scatter(times, srcs, c='lightgray', marker='o', label='Source', alpha=0.5)
    ax.scatter(times, dsts, c='gray', marker='x', label='Destination', alpha=0.5)
    
    for i, (u, v, t, _) in enumerate(events):
        color = 'red' if (motif_edges and i in motif_edges) else 'blue'
        alpha = 1.0 if (motif_edges and i in motif_edges) else 0.1
        ax.plot([t, t], [u, v], c=color, alpha=alpha)
        
    ax.set_xlabel('Time')
    ax.set_ylabel('Node ID')
    ax.set_title('Temporal Graph Events')
    plt.tight_layout()
    return fig

def compute_dataset_statistics(dataset) -> dict:
    """Computes basic statistics for the given dataset."""
    labels = []
    num_events = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(sample.y.item())
        num_events.append(sample.src.size(0))
            
    return {
        'num_samples': len(dataset),
        'positive_ratio': sum(labels) / len(labels) if labels else 0.0,
        'avg_events': sum(num_events) / len(num_events) if num_events else 0.0,
    }
