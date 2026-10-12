import torch
import torch.nn as nn
from torch import Tensor

class TGNMemory(nn.Module):
    """Per-node memory module for TGN (batch-vectorized).
    
    Memory shape: (batch_size, num_nodes + 1, memory_dim)
    The +1 reserves a dummy node slot (index num_nodes) that absorbs
    writes from padded events harmlessly.
    
    Updates via manual GRU: s_v(t⁺) = GRU(m̄_v(t), s_v(t⁻))
    
    HOOK POINTS for MI:
    - pre_update: memory state before GRU update
    - post_update: memory state after GRU update
    - gates: GRU update gate z_t and reset gate r_t values
    """
    
    def __init__(self, num_nodes: int, memory_dim: int, message_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        
        # Memory and last_update are now created dynamically per batch via reset()
        # No register_buffer — they're created in reset() and live on the right device
        self.memory = None
        self.last_update = None
        
        # GRU parameters (manual implementation for hooks)
        self.W_z = nn.Parameter(torch.empty(message_dim + memory_dim, memory_dim))
        self.b_z = nn.Parameter(torch.zeros(memory_dim))
        
        self.W_r = nn.Parameter(torch.empty(message_dim + memory_dim, memory_dim))
        self.b_r = nn.Parameter(torch.zeros(memory_dim))
        
        self.W_h = nn.Parameter(torch.empty(message_dim + memory_dim, memory_dim))
        self.b_h = nn.Parameter(torch.zeros(memory_dim))
        
        # Initialization
        for w in [self.W_z, self.W_r, self.W_h]:
            nn.init.xavier_uniform_(w)
            
        self.hook_data = []

    def reset(self, batch_size: int, device: torch.device):
        """Create fresh memory for a batch of independent samples.
        
        Allocates num_nodes+1 slots: indices 0..num_nodes-1 for real nodes,
        index num_nodes as a dummy slot for padded events.
        """
        self.memory = torch.zeros(batch_size, self.num_nodes + 1, self.memory_dim, device=device)
        self.last_update = torch.zeros(batch_size, self.num_nodes + 1, device=device)
        self.hook_data = []

    def get_memory(self, batch_idx: Tensor, node_ids: Tensor) -> Tensor:
        """Read memory for specified nodes across the batch.
        
        Args:
            batch_idx: (B,) tensor, typically torch.arange(B)
            node_ids: (B,) tensor of node indices
        Returns:
            (B, memory_dim) tensor
        """
        return self.memory[batch_idx, node_ids]

    def update(self, batch_idx: Tensor, node_ids: Tensor, messages: Tensor,
               timestamps: Tensor, mask: Tensor,
               gate_interventions: dict = None) -> dict:
        """Batched GRU memory update with masking for padded events.

        Args:
            batch_idx: (B,) tensor
            node_ids: (B,) tensor of node indices to update
            messages: (B, message_dim) tensor
            timestamps: (B,) tensor
            mask: (B,) bool tensor — True for real events, False for padding
            gate_interventions: optional dict with keys 'gate_z_t' and/or 'gate_r_t',
                each mapping to a callable fn(tensor) -> tensor that replaces the gate value.
                Used for causal gate-level analysis (Experiment 3B).
        Returns:
            dict of hook data
        """
        if node_ids.size(0) == 0:
            return {}

        # Get current memory for these nodes
        mem = self.memory[batch_idx, node_ids]  # (B, memory_dim)

        # Pre-update hook
        hook_memory_pre_update = mem.clone().detach()

        # Manual GRU computation (identical math to original)
        cat_input = torch.cat([messages, mem], dim=-1)  # (B, message_dim + memory_dim)

        z_t = torch.sigmoid(cat_input @ self.W_z + self.b_z)  # (B, memory_dim)
        r_t = torch.sigmoid(cat_input @ self.W_r + self.b_r)  # (B, memory_dim)

        # ── INTERVENTION POINT: gate_z_t ──
        if gate_interventions and 'gate_z_t' in gate_interventions:
            z_t = gate_interventions['gate_z_t'](z_t)

        # ── INTERVENTION POINT: gate_r_t ──
        if gate_interventions and 'gate_r_t' in gate_interventions:
            r_t = gate_interventions['gate_r_t'](r_t)

        cat_input_h = torch.cat([messages, r_t * mem], dim=-1)
        h_tilde = torch.tanh(cat_input_h @ self.W_h + self.b_h)  # (B, memory_dim)

        memory_new = (1 - z_t) * mem + z_t * h_tilde  # (B, memory_dim)
        
        # DESIGN NOTE (BPTT): We write the graph-attached `memory_new`
        # tensor back into self.memory WITHOUT detaching, using a
        # clone-scatter pattern instead of in-place indexed assignment.
        # This ensures reliable autograd tracking on both CPU and CUDA,
        # working around a CUDA regression in some PyTorch 2.x versions
        # where `tensor[idx] = val` silently breaks the gradient chain
        # for sequential read-write memory patterns.
        mask_expanded = mask.unsqueeze(-1)  # (B, 1)
        memory_write = torch.where(mask_expanded, memory_new, mem)

        # Clone-scatter: creates explicit autograd nodes for BPTT
        B = batch_idx.size(0)
        idx = node_ids.unsqueeze(-1).expand(B, self.memory_dim).unsqueeze(1)  # (B, 1, D)
        src = memory_write.unsqueeze(1)  # (B, 1, D)
        self.memory = self.memory.clone().scatter_(1, idx, src)

        # Update last_update times (only for real events — no gradient needed)
        new_timestamps = torch.where(mask, timestamps, self.last_update[batch_idx, node_ids])
        self.last_update = self.last_update.clone()
        self.last_update[batch_idx, node_ids] = new_timestamps
        
        # Post-update hooks (detached for MI analysis)
        hook_memory_post_update = memory_write.clone().detach()
        hook_gru_gates = {'z_t': z_t.clone().detach(), 'r_t': r_t.clone().detach()}
        
        step_hook_data = {
            'node_ids': node_ids.clone().detach(),
            'mask': mask.clone().detach(),
            'pre_update': hook_memory_pre_update,
            'post_update': hook_memory_post_update,
            'gates': hook_gru_gates,
        }
        self.hook_data.append(step_hook_data)
        return step_hook_data

    def detach(self):
        """Detach memory from computation graph."""
        if self.memory is not None:
            self.memory.detach_()
