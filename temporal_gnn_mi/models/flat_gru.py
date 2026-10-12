"""
Flat-GRU Baseline Model

A standard GRU over the event sequence with learnable node embeddings.
NO per-node memory — all events feed sequentially into a single GRU,
and the final hidden state is used for classification.

This baseline tests whether the TGN's per-node memory architecture is
necessary, or whether a simpler sequence model can solve the task equally.
If Flat-GRU matches TGN performance, the per-node memory is not needed
and the MI findings about memory circuits are less interesting.

Architecture:
  1. Learnable node embeddings: (num_nodes, embedding_dim)
  2. Per-event input: concat(emb_src, emb_dst, time_encoding, edge_features)
  3. Standard nn.GRU processes events sequentially
  4. Final hidden state → classifier MLP → logit
"""

import torch
import torch.nn as nn
from torch import Tensor
from temporal_gnn_mi.config import ModelConfig


class FlatGRU(nn.Module):
    """Flat-GRU baseline: sequence model without per-node memory."""

    def __init__(self, config: ModelConfig, num_nodes: int, edge_feat_dim: int = 16):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim

        # Learnable node embeddings (substitute for per-node memory)
        self.node_embeddings = nn.Embedding(num_nodes + 1, config.embedding_dim,
                                            padding_idx=num_nodes)

        # Time encoding (same architecture as TGN)
        self.time_dim = config.time_dim
        self.time_w = nn.Parameter(torch.empty(1, config.time_dim // 2))
        nn.init.xavier_uniform_(self.time_w)

        # GRU input: emb_src + emb_dst + time_enc + edge_features
        gru_input_dim = config.embedding_dim * 2 + config.time_dim + edge_feat_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=config.memory_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Classifier: GRU hidden + embeddings of A, B, C → logit
        classifier_input_dim = config.memory_dim + config.embedding_dim * 3
        self.classifier_hidden = nn.Sequential(
            nn.Linear(classifier_input_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier_output = nn.Linear(config.embedding_dim, 1)

        # For compatibility with experiment utils
        self.hook_data = []
        self.classifier_hook_data = []

    def _time_encode(self, delta_t: Tensor) -> Tensor:
        """Learnable sinusoidal time encoding."""
        scaled = delta_t.unsqueeze(-1) * self.time_w  # (B, time_dim//2)
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

    def forward(self, batch_dict: dict, interventions: dict = None,
                completion_readout: bool = False) -> Tensor:
        """Process a batch of temporal graphs as flat sequences.

        Args:
            batch_dict: same format as TGN's input
            interventions: ignored (included for API compatibility)
            completion_readout: ignored (FlatGRU has no completion pathway)

        Returns:
            logits: (B, 1) tensor
        """
        device = next(self.parameters()).device

        src = batch_dict['src'].to(device)   # (B, max_events)
        dst = batch_dict['dst'].to(device)   # (B, max_events)
        t = batch_dict['t'].to(device)       # (B, max_events)
        msg = batch_dict['msg'].to(device)   # (B, max_events, F)
        mask = batch_dict['mask'].to(device) # (B, max_events)

        B, max_events = src.shape

        # Replace -1 padding with dummy node index
        src = torch.where(src == -1, torch.tensor(self.num_nodes, device=device), src)
        dst = torch.where(dst == -1, torch.tensor(self.num_nodes, device=device), dst)

        # Node embeddings
        emb_src = self.node_embeddings(src)  # (B, max_events, emb_dim)
        emb_dst = self.node_embeddings(dst)  # (B, max_events, emb_dim)

        # Time encoding: use absolute time differences from previous event
        # Shift times to get delta_t (time since previous event)
        t_shifted = torch.zeros_like(t)
        t_shifted[:, 1:] = t[:, 1:] - t[:, :-1]
        t_shifted[:, 0] = t[:, 0]  # first event uses absolute time
        time_enc = self._time_encode(t_shifted)  # (B, max_events, time_dim)

        # Concatenate all features into GRU input
        gru_input = torch.cat([emb_src, emb_dst, time_enc, msg], dim=-1)
        # (B, max_events, gru_input_dim)

        # Pack sequences using actual lengths for efficiency
        lengths = mask.sum(dim=1).cpu()  # (B,)
        # Clamp minimum length to 1 to avoid pack_padded_sequence errors
        lengths = lengths.clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            gru_input, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)  # h_n: (1, B, hidden_dim)
        h_final = h_n.squeeze(0)  # (B, hidden_dim)

        # Get embeddings of candidate nodes A, B, C
        candidates = batch_dict['candidate_nodes'].to(device)  # (B, 3)
        emb_A = self.node_embeddings(candidates[:, 0])  # (B, emb_dim)
        emb_B = self.node_embeddings(candidates[:, 1])
        emb_C = self.node_embeddings(candidates[:, 2])

        # Classifier input: GRU hidden + node embeddings
        cat_states = torch.cat([h_final, emb_A, emb_B, emb_C], dim=-1)
        hidden = self.classifier_hidden(cat_states)
        logits = self.classifier_output(hidden)  # (B, 1)

        return logits

    def get_motif_step_indices(self, batch_dict: dict) -> list:
        """Same interface as TGN for compatibility."""
        results = []
        B = batch_dict['src'].size(0)
        if 'motif_edges' in batch_dict:
            motif_edges = batch_dict['motif_edges']
            for b in range(B):
                results.append((motif_edges[b, 0].item(), motif_edges[b, 1].item()))
        else:
            for _ in range(B):
                results.append((-1, -1))
        return results

    def clear_hooks(self):
        self.hook_data = []
        self.classifier_hook_data = []

    def reset_memory(self, batch_size: int = 1, device: torch.device = None):
        """No-op for API compatibility."""
        pass
