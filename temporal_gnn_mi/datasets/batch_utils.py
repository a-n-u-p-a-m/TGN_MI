import torch
from torch.utils.data import default_collate

# Fixed global padding target. Using a round number so torch.compile
# never sees varying loop lengths across batches.
GLOBAL_MAX_EVENTS = 50  # Safe upper bound; real samples have ~20-35 events

def temporal_collate_fn(data_list):
    """Stack TemporalData samples along a batch dimension with tail-padding.
    
    Padding uses a dummy node index = num_nodes (outside the real 0..num_nodes-1 range).
    All True entries in `mask` are a contiguous prefix (real events first, padding after).
    
    Returns:
        dict with keys: src, dst, t, msg, mask, y, candidate_nodes,
                        num_events, corruption, source_motif_t1, source_motif_t2,
                        motif_edges
    """
    B = len(data_list)
    max_events = GLOBAL_MAX_EVENTS
    
    # Infer dims from first sample
    feat_dim = data_list[0].msg.size(1) if data_list[0].msg.size(0) > 0 else 16
    dummy_node = -1
    
    src_padded  = torch.full((B, max_events), dummy_node, dtype=torch.long)
    dst_padded  = torch.full((B, max_events), dummy_node, dtype=torch.long)
    t_padded    = torch.zeros(B, max_events, dtype=torch.float)
    msg_padded  = torch.zeros(B, max_events, feat_dim, dtype=torch.float)
    mask_padded = torch.zeros(B, max_events, dtype=torch.bool)
    
    y_list = []
    candidate_nodes_list = []
    num_events_list = []
    corruption_list = []
    source_motif_t1_list = []
    source_motif_t2_list = []
    motif_edges_list = []
    decision_step_list = []
    
    for b, data in enumerate(data_list):
        n = data.src.size(0)
        assert n <= max_events, (
            f"Sample has {n} events but GLOBAL_MAX_EVENTS={max_events}. "
            f"Increase GLOBAL_MAX_EVENTS."
        )
        
        # Events are already time-sorted by the generator.
        # Padding goes at the tail so real events form a contiguous prefix.
        src_padded[b, :n]  = data.src
        dst_padded[b, :n]  = data.dst
        t_padded[b, :n]    = data.t
        msg_padded[b, :n]  = data.msg
        mask_padded[b, :n] = True
        
        y_list.append(data.y)
        candidate_nodes_list.append(data.candidate_nodes)
        num_events_list.append(n)
        corruption_list.append(getattr(data, 'corruption', 'unknown'))
        source_motif_t1_list.append(data.source_motif_t1)
        source_motif_t2_list.append(data.source_motif_t2)
        motif_edges_list.append(data.motif_edges)
        decision_step_list.append(
            data.decision_step if hasattr(data, 'decision_step')
            else torch.tensor([-1], dtype=torch.long)
        )
    
    # Verify contiguous prefix invariant
    for b in range(B):
        n = num_events_list[b]
        assert mask_padded[b, :n].all() and (not mask_padded[b, n:].any()), (
            f"Sample {b}: mask must be a contiguous True prefix"
        )
    
    # Stack motif_edges into (B, 2) tensor; pad empty (negative samples) with -1
    motif_edges_stacked = []
    for me in motif_edges_list:
        if me.numel() >= 2:
            motif_edges_stacked.append(me[:2])  # take first 2 indices
        else:
            motif_edges_stacked.append(torch.tensor([-1, -1], dtype=torch.long))
    
    return {
        'src': src_padded,
        'dst': dst_padded,
        't': t_padded,
        'msg': msg_padded,
        'mask': mask_padded,
        'y': torch.cat(y_list, dim=0),
        'candidate_nodes': torch.stack(candidate_nodes_list),
        'num_events': torch.tensor(num_events_list, dtype=torch.long),
        'corruption': corruption_list,
        'source_motif_t1': torch.cat(source_motif_t1_list, dim=0),
        'source_motif_t2': torch.cat(source_motif_t2_list, dim=0),
        'motif_edges': torch.stack(motif_edges_stacked),  # (B, 2)
        'decision_step': torch.cat(decision_step_list, dim=0),  # (B,)
    }
