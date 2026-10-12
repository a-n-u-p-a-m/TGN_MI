import torch
import torch.nn as nn
from torch import Tensor
import typing

if typing.TYPE_CHECKING:
    from temporal_gnn_mi.config import ModelConfig


class GraphEmbedding(nn.Module):
    """Embedding module that computes final node representations.

    For the motif completion task, we use the MEMORY STATES directly
    (identity embedding) since the task tests memory-based temporal reasoning.
    Adding a graph attention layer here would complicate MI analysis
    without improving task performance.
    """
    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        
    def forward(self, memory_states: Tensor, *args, **kwargs) -> Tensor:
        return memory_states
