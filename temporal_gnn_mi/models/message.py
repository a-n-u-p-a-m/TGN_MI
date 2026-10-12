import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class MessageFunction(nn.Module):
    """Computes raw messages from events.
    m_u(t) = MLP(concat(s_u(t⁻), s_v(t⁻), Φ(Δt), edge_features))
    
    HOOK POINT: self.hook_raw_messages
    """
    def __init__(self, memory_dim: int, time_dim: int, edge_feat_dim: int, message_dim: int):
        super().__init__()
        input_dim = memory_dim * 2 + time_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        self.hook_data = []

    def forward(self, src_memory: Tensor, dst_memory: Tensor, time_encoding: Tensor, edge_features: Tensor) -> Tensor:
        cat_input = torch.cat([src_memory, dst_memory, time_encoding, edge_features], dim=-1)
        raw_message = self.mlp(cat_input)
        
        self.hook_data.append(raw_message.clone().detach())
        return raw_message


class MessageAggregator(nn.Module):
    """Aggregates messages for each node.
    Supports 'mean' and 'last' aggregation.
    
    HOOK POINT: self.hook_aggregated_messages
    """
    def __init__(self, method: str = 'mean'):
        super().__init__()
        self.method = method
        self.hook_data = []

    def forward(self, node_ids: Tensor, messages: Tensor, timestamps: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns: (unique_node_ids, aggregated_messages)"""
        if len(node_ids) == 0:
            return node_ids, messages
            
        unique_nodes = torch.unique(node_ids)
        agg_msgs = []
        
        for n in unique_nodes:
            mask = (node_ids == n)
            msgs_n = messages[mask]
            
            if self.method == 'last':
                agg_msgs.append(msgs_n[-1])
            elif self.method == 'mean':
                agg_msgs.append(msgs_n.mean(dim=0))
            else:
                raise ValueError(f"Unknown aggregation method {self.method}")
                
        aggregated = torch.stack(agg_msgs)
        self.hook_data.append({'nodes': unique_nodes.clone(), 'messages': aggregated.clone().detach()})
        return unique_nodes, aggregated
