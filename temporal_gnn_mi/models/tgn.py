import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Dict, Any, Optional, Callable

from .time_encoding import TimeEncoding
from .memory import TGNMemory
from .message import MessageFunction, MessageAggregator
from .embedding import GraphEmbedding
from temporal_gnn_mi.config import ModelConfig

class TGN(nn.Module):
    """Temporal Graph Network for temporal motif completion (batch-vectorized).
    
    Processes events sequentially by event index, but in parallel across all
    samples in the batch. Memory is (B, num_nodes+1, memory_dim) — each sample
    has its own independent memory buffer, so cross-sample vectorization
    preserves exact per-sample causal ordering.
    
    Hook Registry:
    - All intermediate activations stored in self.hook_data
    - Accessed via self.get_hook_data() after forward pass
    - Cleared via self.clear_hooks()
    
    Intervention API (Phase 3 — Activation Patching):
    - forward() accepts an optional `interventions` dict
    - Keys are (event_step_index, point_name) for per-event interventions
      or (string_tag, point_name) for classifier interventions
    - Values are callables: fn(tensor) -> tensor of same shape
    - When interventions=None (default), all if-blocks are skipped — zero overhead
    """
    
    def __init__(self, config: ModelConfig, num_nodes: int, edge_feat_dim: int = 16):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim
        
        self.memory = TGNMemory(num_nodes, config.memory_dim, config.message_dim)
        self.time_encoder = TimeEncoding(config.time_dim)
        
        self.message_function = MessageFunction(
            config.memory_dim, config.time_dim, edge_feat_dim, config.message_dim
        )
        # Aggregator kept for backward compat / single-sample MI analysis
        self.message_aggregator = MessageAggregator(method='mean')
        
        self.embedding = GraphEmbedding(config)
        
        # Classifier for motif completion: concat memory of nodes A, B, C
        self.classifier_hidden = nn.Sequential(
            nn.Linear(config.memory_dim * 3, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier_output = nn.Linear(config.embedding_dim, 1)

        # Completion-time classifier: same architecture, separate weights.
        # Reads memory of A, B, C immediately after the B→C event updates,
        # before any subsequent events can overwrite those memory states.
        self.completion_classifier_hidden = nn.Sequential(
            nn.Linear(config.memory_dim * 3, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.completion_classifier_output = nn.Linear(config.embedding_dim, 1)

        self.hook_data = []
        self.classifier_hook_data = []
        self.completion_logits = None
        self.completion_hook_data = []

    def forward(self, batch_dict: dict, interventions: dict = None,
                completion_readout: bool = False) -> Tensor:
        """Process a batch of temporal graphs in parallel.

        Args:
            batch_dict: dict from temporal_collate_fn with keys:
                src, dst, t, msg, mask, y, candidate_nodes, num_events
            interventions: optional dict mapping
                (event_step_index: int, point_name: str) -> callable for per-event interventions
                (tag: str, point_name: str) -> callable for classifier interventions
                Each callable receives the current tensor and returns a replacement of the same shape.
                When None (default), all intervention checks are skipped — zero overhead.
            completion_readout: if True, also classify from memory states captured
                immediately after each sample's B→C event (step m2).  The completion
                logits are stored in self.completion_logits (B, 1) and detailed hook
                data in self.completion_hook_data.  Samples whose m2 is missing (-1)
                get a completion logit of 0.  Default False for backward compat.

        Returns:
            logits: (B, 1) tensor — always from the final-state readout.
        """
        device = next(self.parameters()).device
        
        src  = batch_dict['src'].to(device)    # (B, max_events)
        dst  = batch_dict['dst'].to(device)    # (B, max_events)
        
        # Replace -1 padding with actual dummy node index
        src = torch.where(src == -1, torch.tensor(self.num_nodes, device=device), src)
        dst = torch.where(dst == -1, torch.tensor(self.num_nodes, device=device), dst)
        
        t    = batch_dict['t'].to(device)      # (B, max_events)
        msg  = batch_dict['msg'].to(device)    # (B, max_events, F)
        mask = batch_dict['mask'].to(device)   # (B, max_events)
        
        B = src.size(0)
        max_events = src.size(1)
        
        # Fresh memory for this batch — each sample starts from zero
        self.memory.reset(B, device)
        batch_idx = torch.arange(B, device=device)

        # ── Completion-time readout setup ────────────────────────────
        # Use decision_step (works for both positive and negative samples):
        #   Positive: decision_step = actual B→C event index
        #   Negative: state after the last event with t <= counterfactual t2
        # This lets the completion classifier train on both classes at a
        # comparable temporal decision point.
        completion_mem_snapshots = None
        if completion_readout:
            candidates = batch_dict['candidate_nodes'].to(device)  # (B, 3)
            if 'decision_step' in batch_dict:
                m2_per_sample = batch_dict['decision_step'].to(device)  # (B,)
            else:
                # Fallback to motif_edges for backward compatibility
                motif_indices = self.get_motif_step_indices(batch_dict)
                m2_per_sample = torch.tensor(
                    [mi[1] for mi in motif_indices], device=device, dtype=torch.long
                )
            # Pre-allocate snapshot storage: (B, 3, mem_dim)
            completion_mem_snapshots = torch.zeros(
                B, 3, self.config.memory_dim, device=device
            )
            completion_captured = torch.zeros(B, dtype=torch.bool, device=device)

        # Sequential loop over event index (NOT over samples)
        # ~30-50 iterations total, each doing GPU-parallel work across all B samples
        for i in range(max_events):
            step_src  = src[:, i]    # (B,)
            step_dst  = dst[:, i]    # (B,)
            step_t    = t[:, i]      # (B,)
            step_msg  = msg[:, i]    # (B, F)
            step_mask = mask[:, i]   # (B,) — False for padded events
            
            # READ: fetch current memory and compute time encoding
            mem_src = self.memory.get_memory(batch_idx, step_src)  # (B, mem_dim)
            mem_dst = self.memory.get_memory(batch_idx, step_dst)  # (B, mem_dim)
            
            # ── INTERVENTION POINT: memory_pre_src ──
            if interventions and (i, 'memory_pre_src') in interventions:
                mem_src = interventions[(i, 'memory_pre_src')](mem_src)
            
            # ── INTERVENTION POINT: memory_pre_dst ──
            if interventions and (i, 'memory_pre_dst') in interventions:
                mem_dst = interventions[(i, 'memory_pre_dst')](mem_dst)
            
            last_t_src = self.memory.last_update[batch_idx, step_src]  # (B,)
            last_t_dst = self.memory.last_update[batch_idx, step_dst]  # (B,)
            
            delta_t_src = step_t - last_t_src
            delta_t_dst = step_t - last_t_dst
            
            time_enc_src = self.time_encoder(delta_t_src)  # (B, time_dim)
            time_enc_dst = self.time_encoder(delta_t_dst)  # (B, time_dim)
            
            # ── INTERVENTION POINT: time_encoding_src ──
            if interventions and (i, 'time_encoding_src') in interventions:
                time_enc_src = interventions[(i, 'time_encoding_src')](time_enc_src)
            
            # ── INTERVENTION POINT: time_encoding_dst ──
            if interventions and (i, 'time_encoding_dst') in interventions:
                time_enc_dst = interventions[(i, 'time_encoding_dst')](time_enc_dst)
            
            # COMPUTE: calculate messages
            msg_for_src = self.message_function(mem_src, mem_dst, time_enc_src, step_msg)
            msg_for_dst = self.message_function(mem_dst, mem_src, time_enc_dst, step_msg)
            
            # ── INTERVENTION POINT: message_src ──
            if interventions and (i, 'message_src') in interventions:
                msg_for_src = interventions[(i, 'message_src')](msg_for_src)
            
            # ── INTERVENTION POINT: message_dst ──
            if interventions and (i, 'message_dst') in interventions:
                msg_for_dst = interventions[(i, 'message_dst')](msg_for_dst)
            
            # WRITE: update memory (masked — padding events are no-ops)
            # Build gate-level interventions for this event step if any are specified
            gate_int_src = None
            gate_int_dst = None
            if interventions:
                gate_int_src = {}
                if (i, 'gate_z_t_src') in interventions:
                    gate_int_src['gate_z_t'] = interventions[(i, 'gate_z_t_src')]
                if (i, 'gate_r_t_src') in interventions:
                    gate_int_src['gate_r_t'] = interventions[(i, 'gate_r_t_src')]
                if not gate_int_src:
                    gate_int_src = None

                gate_int_dst = {}
                if (i, 'gate_z_t_dst') in interventions:
                    gate_int_dst['gate_z_t'] = interventions[(i, 'gate_z_t_dst')]
                if (i, 'gate_r_t_dst') in interventions:
                    gate_int_dst['gate_r_t'] = interventions[(i, 'gate_r_t_dst')]
                if not gate_int_dst:
                    gate_int_dst = None

            self.memory.update(batch_idx, step_src, msg_for_src, step_t, step_mask,
                               gate_interventions=gate_int_src)
            self.memory.update(batch_idx, step_dst, msg_for_dst, step_t, step_mask,
                               gate_interventions=gate_int_dst)
            
            # Hook: capture event-level data with alignment metadata
            self.hook_data.append({
                'event_index': i,
                'src_nodes': step_src.clone().detach(),
                'dst_nodes': step_dst.clone().detach(),
                'mask': step_mask.clone().detach(),
            })

            # ── Completion-time snapshot ─────────────────────────────
            # After both src and dst memory updates at step i, snapshot
            # memory for any sample whose B→C event was this step.
            if completion_readout:
                just_completed = (m2_per_sample == i) & (~completion_captured)
                if just_completed.any():
                    jc_idx = just_completed.nonzero(as_tuple=True)[0]
                    for b_idx in jc_idx:
                        b = b_idx.item()
                        completion_mem_snapshots[b, 0] = self.memory.get_memory(
                            batch_idx[b:b+1], candidates[b:b+1, 0]
                        ).squeeze(0)
                        completion_mem_snapshots[b, 1] = self.memory.get_memory(
                            batch_idx[b:b+1], candidates[b:b+1, 1]
                        ).squeeze(0)
                        completion_mem_snapshots[b, 2] = self.memory.get_memory(
                            batch_idx[b:b+1], candidates[b:b+1, 2]
                        ).squeeze(0)
                    completion_captured[jc_idx] = True

        # CLASSIFY using final memory states of candidate nodes A, B, C
        candidates = batch_dict['candidate_nodes'].to(device)  # (B, 3)
        
        mem_A = self.memory.get_memory(batch_idx, candidates[:, 0])  # (B, mem_dim)
        mem_B = self.memory.get_memory(batch_idx, candidates[:, 1])  # (B, mem_dim)
        mem_C = self.memory.get_memory(batch_idx, candidates[:, 2])  # (B, mem_dim)
        
        # ── INTERVENTION POINT: readout_mem_A / B / C ──
        if interventions and ('readout', 'mem_A') in interventions:
            mem_A = interventions[('readout', 'mem_A')](mem_A)
        if interventions and ('readout', 'mem_B') in interventions:
            mem_B = interventions[('readout', 'mem_B')](mem_B)
        if interventions and ('readout', 'mem_C') in interventions:
            mem_C = interventions[('readout', 'mem_C')](mem_C)
        
        # Embedding (identity in current config)
        emb_A = self.embedding(mem_A)
        emb_B = self.embedding(mem_B)
        emb_C = self.embedding(mem_C)
        
        cat_states = torch.cat([emb_A, emb_B, emb_C], dim=-1)  # (B, 3*mem_dim)
        
        # ── INTERVENTION POINT: classifier_input ──
        if interventions and ('classifier', 'input') in interventions:
            cat_states = interventions[('classifier', 'input')](cat_states)
        
        # Classifier with hook on hidden layer
        hidden = self.classifier_hidden(cat_states)   # (B, embed_dim)
        
        # ── INTERVENTION POINT: classifier_hidden ──
        if interventions and ('classifier', 'hidden') in interventions:
            hidden = interventions[('classifier', 'hidden')](hidden)
        
        logits = self.classifier_output(hidden)        # (B, 1)
        
        # Hook: capture classifier data
        self.classifier_hook_data.append({
            'candidate_nodes': candidates.clone().detach(),
            'readout_memory': torch.stack([mem_A, mem_B, mem_C], dim=1).clone().detach(),  # (B, 3, mem_dim)
            'classifier_input': cat_states.clone().detach(),
            'classifier_hidden': hidden.clone().detach(),
            'logits': logits.clone().detach(),
        })

        # ── Completion-time classifier ───────────────────────────────
        # Runs the separately-parameterized classifier on memory states
        # captured immediately after B→C, before any subsequent events.
        # Stored alongside (not replacing) the final-state logits.
        if completion_readout:
            comp_mem_A = completion_mem_snapshots[:, 0]  # (B, mem_dim)
            comp_mem_B = completion_mem_snapshots[:, 1]
            comp_mem_C = completion_mem_snapshots[:, 2]

            comp_emb_A = self.embedding(comp_mem_A)
            comp_emb_B = self.embedding(comp_mem_B)
            comp_emb_C = self.embedding(comp_mem_C)

            comp_cat = torch.cat([comp_emb_A, comp_emb_B, comp_emb_C], dim=-1)
            comp_hidden = self.completion_classifier_hidden(comp_cat)
            comp_logits = self.completion_classifier_output(comp_hidden)

            # Zero out logits for samples where m2 was missing
            has_m2 = (m2_per_sample >= 0).unsqueeze(-1).float()
            comp_logits = comp_logits * has_m2

            self.completion_logits = comp_logits
            self.completion_captured = completion_captured

            self.completion_hook_data.append({
                'candidate_nodes': candidates.clone().detach(),
                'completion_memory': completion_mem_snapshots.clone().detach(),
                'completion_input': comp_cat.clone().detach(),
                'completion_hidden': comp_hidden.clone().detach(),
                'completion_logits': comp_logits.clone().detach(),
                'has_m2': has_m2.clone().detach(),
            })
        else:
            self.completion_logits = None
            self.completion_captured = None
            self.completion_hook_data = []

        return logits

    def get_motif_step_indices(self, batch_dict: dict) -> list:
        """For each sample in the batch, return the event-step indices
        of the two motif edges (A→B and B→C).
        
        Uses motif_edges metadata from the dataset to find which positions
        in the time-sorted event list correspond to the motif.
        
        Returns:
            List of (motif1_step, motif2_step) tuples, one per sample.
            Returns (-1, -1) for samples where motif_edges metadata is missing.
        """
        results = []
        B = batch_dict['src'].size(0)
        
        # motif_edges stores the indices within each sample's event list
        if 'motif_edges' in batch_dict:
            motif_edges = batch_dict['motif_edges']  # (B, 2)
            for b in range(B):
                idx1 = motif_edges[b, 0].item()
                idx2 = motif_edges[b, 1].item()
                results.append((idx1, idx2))
        else:
            # Fallback: find by matching src/dst against candidate nodes
            candidates = batch_dict['candidate_nodes']  # (B, 3)
            src = batch_dict['src']
            dst = batch_dict['dst']
            mask = batch_dict['mask']
            for b in range(B):
                A, B_node, C = candidates[b].tolist()
                m1, m2 = -1, -1
                for i in range(src.size(1)):
                    if not mask[b, i]:
                        continue
                    if src[b, i].item() == A and dst[b, i].item() == B_node:
                        m1 = i
                    if src[b, i].item() == B_node and dst[b, i].item() == C:
                        m2 = i
                results.append((m1, m2))
        
        return results

    def get_hook_data(self) -> dict:
        return {
            'memory_hooks':        self.memory.hook_data,
            'message_func_hooks':  self.message_function.hook_data,
            'time_encoding_hooks': self.time_encoder.hook_data,
            'classifier_hooks':    self.classifier_hook_data,
            'event_hooks':         self.hook_data,
        }

    def clear_hooks(self):
        self.memory.hook_data = []
        self.message_function.hook_data = []
        self.time_encoder.hook_data = []
        self.classifier_hook_data = []
        self.hook_data = []
        self.completion_logits = None
        self.completion_hook_data = []

    def reset_memory(self, batch_size: int = 1, device: torch.device = None):
        """Reset memory. Device defaults to model's parameter device."""
        if device is None:
            device = next(self.parameters()).device
        self.memory.reset(batch_size, device)
